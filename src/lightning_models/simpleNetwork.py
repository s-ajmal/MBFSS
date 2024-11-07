import pytorch_lightning as pl
import timm
import torch
from torch import nn
from torch.nn import functional as F
from util import intersectionAndUnionGPU, batch_intersectionAndUnionGPU
from visu import make_episode_visualization

def masked_global_pooling(mask, Fs):
    # mask size = nway, kshot, 1, 56, 56
    mask = mask.float()
    mask = F.interpolate(mask, size=(Fs.shape[-2], Fs.shape[-1]))
    expanded_mask = mask.expand_as(Fs)
    masked_dog = Fs * expanded_mask
    out = torch.sum(masked_dog, dim=[-1, -2]) / (expanded_mask.sum(dim=[-1, -2]) + 1e-5)
    out = out.unsqueeze(-1)
    out = out.unsqueeze(-1)
    out = out.expand_as(Fs)
    return out


class SimpleNetwork(pl.LightningModule):
    def __init__(self, hparams, visualize=False):
        super(SimpleNetwork, self).__init__()
        print(hparams)
        self.save_hyperparameters()
        self.args = hparams
        self.args.visualize = self.hparams.visualize

        # -----------------------------ResNet 101 ---------------------------------
        Res_backbone_name = "resnet101d"
        self.Res_backbone = timm.create_model(Res_backbone_name, pretrained=self.args.pretrained)
        self.Res_layer0 = nn.Sequential(self.Res_backbone.conv1, self.Res_backbone.bn1, self.Res_backbone.act1,
                                        self.Res_backbone.maxpool)
        self.Res_layer1, self.Res_layer2 = self.Res_backbone.layer1, self.Res_backbone.layer2
        self.Res_layer3, self.Res_layer4 = self.Res_backbone.layer3, self.Res_backbone.layer4
        self.Res_feature_res = (50, 50)
        # ----------------------------------ResNeXt 101 --------------------------
        ResN_backbone_name = "resnext101_32x8d"
        self.ResN_backbone = ResN_backbone_name
        self.ResN_backbone = timm.create_model(ResN_backbone_name, pretrained=self.args.pretrained)
        self.ResN_layer0 = nn.Sequential(
            self.ResN_backbone.conv1,
            self.ResN_backbone.bn1,
            self.ResN_backbone.act1,
            self.ResN_backbone.maxpool
        )
        self.ResN_layer1 = self.ResN_backbone.layer1
        self.ResN_layer2 = self.ResN_backbone.layer2
        self.ResN_layer3 = self.ResN_backbone.layer3
        self.ResN_layer4 = self.ResN_backbone.layer4
        self.ResN_feature_res = (50, 50)
        # --------------------------------PVT Feature Extractor---------------------------------------

        self.pvt_backbone = timm.create_model(
            'pvt_v2_b5',
            pretrained=True,
            features_only=True,
        )
        self.pvt_backbone = self.pvt_backbone.eval()
        self.pvt_data_config = timm.data.resolve_model_data_config(self.pvt_backbone)
        self.pvt_transforms = timm.data.create_transform(**self.pvt_data_config, is_training=False)
        self.pvt_feature_res = (50, 50)
        # ----------------------------------pvt -------------------------------------
        # ---------------ResNeXt-------------
        for n, m in self.ResN_layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.ResN_layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        # ---------------------------Projections for ResNet & ResNeXT  --------------------------------------
        self.Res_projection1 = nn.Sequential(nn.Conv2d(in_channels=512,
                                                       out_channels=128, kernel_size=(3, 3),
                                                       padding=(1, 1)), nn.GroupNorm(4, 128), nn.ReLU())
        self.Res_projection2 = nn.Sequential(nn.Conv2d(in_channels=1024,
                                                       out_channels=128,
                                                       kernel_size=(3, 3),
                                                       padding=(1, 1)), nn.GroupNorm(4, 128), nn.ReLU())
        self.Res_projection3 = nn.Sequential(nn.Conv2d(in_channels=2048,
                                                       out_channels=128,
                                                       kernel_size=(3, 3),
                                                       padding=(1, 1)), nn.GroupNorm(4, 128), nn.ReLU())
        # ------------------------------------Update projection for pvt------------------------------
        # Assuming you have a pvt backbone with 112 channels in the input
        # projection for pvt
        self.pvt_projection1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(4, 128),
            nn.ReLU()
        )
        self.pvt_projection2 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=128, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(4, 128),
            nn.ReLU()
        )
        self.pvt_projection3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(4, 128),
            nn.ReLU()
        )
        # ------------------------------------Update projection for PVT-----------------------

        self.dense_conv = nn.Sequential(nn.Conv2d(in_channels=768, out_channels=128,
                                                  kernel_size=(3, 3),
                                                  padding=(1, 1)), nn.GroupNorm(4, 128), nn.ReLU())
        if not self.args.use_all_classes:
            self.val_class_IoU = [ClassIoUNew(self.args.num_classes_val)]
        else:
            self.val_class_IoU = [ClassIoUNew(self.args.num_classes_val),
                                  ClassIoUNew(self.args.num_classes_val),
                                  ClassIoUNew(self.args.num_classes_val),
                                  ClassIoUNew(self.args.num_classes_val)]

        self.decoder = nn.Sequential(nn.Conv2d(128, 128, (3, 3), padding=(1, 1), bias=True),
                                     nn.GroupNorm(4, 128),
                                     nn.ReLU(),
                                     nn.Conv2d(128, 128, (3, 3), padding=(1, 1), bias=True),
                                     nn.GroupNorm(4, 128),
                                     nn.ReLU(),
                                     nn.UpsamplingBilinear2d(scale_factor=2),
                                     nn.Conv2d(128, 64, (3, 3), padding=(1, 1), bias=True),
                                     nn.GroupNorm(4, 64),
                                     nn.ReLU(),
                                     nn.Conv2d(64, 64, (3, 3), padding=(1, 1), bias=True),
                                     nn.GroupNorm(4, 64),
                                     nn.ReLU(),
                                     nn.UpsamplingBilinear2d(scale_factor=2),
                                     nn.Conv2d(64, 32, (3, 3), padding=(1, 1), bias=True),
                                     nn.GroupNorm(4, 32),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 32, (3, 3), padding=(1, 1), bias=True),
                                     nn.GroupNorm(4, 32),
                                     nn.ReLU(),
                                     nn.UpsamplingBilinear2d(scale_factor=2),
                                     nn.Conv2d(32, 16, (3, 3), padding=(1, 1), bias=True),
                                     nn.GroupNorm(4, 16),
                                     nn.ReLU(),
                                     nn.Conv2d(16, 2, (1, 1), bias=False))

        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, support: torch.Tensor, smask: torch.Tensor, query: torch.Tensor):
        # Support Feature Extraction
        support = support.squeeze(1)
        with torch.no_grad():
            rfs = self.Res_layer0(support)
            rfs = self.Res_layer1(rfs)
            fs_2 = self.Res_layer2(rfs)
            nfs = self.ResN_layer0(support)
            nfs = self.ResN_layer1(nfs)
            nfs = self.ResN_layer2(nfs)
            fs_3 = self.ResN_layer3(nfs)
            
            output = self.pvt_backbone(support)  # unsqueeze single image into batch of 1
            for o in output:
                pass
                #print(o.shape)
            fs_4 = o
            
            # Query Feature Extraction
            rfq = self.Res_layer0(query)
            rfq = self.Res_layer1(rfq)
            fq_2 = self.Res_layer2(rfq)

            nfq = self.ResN_layer0(query)
            nfq = self.ResN_layer1(nfq)
            nfq = self.ResN_layer2(nfq)
            fq_3 = self.ResN_layer3(nfq)
            output = self.pvt_backbone(query)  # unsqueeze single image into batch of 1
            for o in output:
                pass
            fq_4 = o
            
        fs_2 = self.Res_projection1(fs_2)
        fq_2 = self.Res_projection1(fq_2)
        fs_3 = self.Res_projection2(fs_3)
        fq_3 = self.Res_projection2(fq_3)
        fq_4 = self.Res_projection3(fq_4)
        fs_4 = self.Res_projection3(fs_4)
        smask[smask == 255] = 0

        fq_2 = torch.cat([fq_2, masked_global_pooling(smask, fs_2)], dim=1)
        fq_3 = torch.cat([fq_3, masked_global_pooling(smask, fs_3)], dim=1)
        fq_4 = torch.cat([fq_4, masked_global_pooling(smask, fs_4)], dim=1)

        fq_4 = F.interpolate(fq_4, size=(fq_2.size(2), fq_2.size(3)), mode='bilinear', align_corners=False)
        fq = torch.cat([fq_2, fq_3, fq_4], dim=1)
        final = self.dense_conv(fq)
        result = self.decoder(final)
        return result

    def training_step(self, batch, batch_nb):
        qry_img, target, spprt_imgs, spprt_labels, subcls_list, support_image_path_list, image_paths = batch
        y_hat = self.forward(spprt_imgs, spprt_labels, qry_img)

        target = target.long()
        loss = self.criterion(y_hat, target)
        preds = torch.argmax(y_hat, dim=1)
        area_intersection, area_union, area_target = intersectionAndUnionGPU(preds, target, 2)
        miou = (area_intersection.float() / area_union.float())[1]

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_miou', miou, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx, dataset_nb=0):
        qry_img, target, spprt_imgs, spprt_labels, subcls_list, support_image_path_list, image_paths = batch
        y_hat = self.forward(spprt_imgs, spprt_labels, qry_img)
        target = target.long()

        loss = self.criterion(y_hat, target)
        preds = torch.argmax(y_hat, dim=1)
        area_intersection, area_union, area_target = intersectionAndUnionGPU(preds, target, 2)
        miou = (area_intersection.float() / area_union.float())[1]
        self.log("val_miou_old", miou, on_epoch=True, prog_bar=True, logger=True)
        y_hat = y_hat.unsqueeze(1)
        target = target.unsqueeze(1)
        intersection, union, _ = batch_intersectionAndUnionGPU(y_hat, target, 2)

        self.val_class_IoU[dataset_nb].update(intersection, union, subcls_list)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)


    def on_validation_epoch_end(self) -> None:
        if len(self.val_class_IoU) == 1:
            val_miou = self.val_class_IoU[0].compute()
            self.log('val_miou', val_miou, prog_bar=True, logger=True)
            self.val_class_IoU[0].reset()
        else:
            for i, calculator in enumerate(self.val_class_IoU):
                val_miou = calculator.compute()
                calculator.reset()
                self.log("val_miou_" + str(i), val_miou, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx, dataset_nb=0):
        qry_img, target, spprt_imgs, spprt_labels, subcls_list, support_image_path_list, image_paths = batch
        target = target.long()
        y_hat = self.forward(spprt_imgs, spprt_labels, qry_img)
        loss = self.criterion(y_hat, target)
        preds = torch.argmax(y_hat, dim=1)
        area_intersection, area_union, area_target = intersectionAndUnionGPU(preds, target, 2)
        miou = (area_intersection.float() / area_union.float())[1]
        self.log("test_miou_old", miou, on_epoch=True, prog_bar=True, logger=True)
        y_hat = y_hat.unsqueeze(1)
        target = target.unsqueeze(1)
        intersection, union, _ = batch_intersectionAndUnionGPU(y_hat, target, 2)

        self.val_class_IoU[dataset_nb].update(intersection, union, subcls_list)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        if self.args.visualize:
            for i in range(len(qry_img)):
                path = image_paths[0][i].split(".")[0].split("/")[-1]+"_"+str(dataset_nb)+"_"+str(batch_idx)+"_"+str(i)
                make_episode_visualization(spprt_imgs[i].cpu().numpy(),
                                           qry_img[i].cpu().numpy(),
                                           spprt_labels[i].cpu().numpy(),
                                           target[i, 0].cpu().numpy(),
                                           y_hat[i].cpu().numpy(),
                                           path)

    def on_test_epoch_end(self) -> None:
        test_miou = self.val_class_IoU[0].compute()
        self.log('test_miou', test_miou, prog_bar=True, logger=True)
        self.val_class_IoU[0].reset()

    def configure_optimizers(self):
        # updated by shahroz (adam to Nadam)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)
        return [optimizer]

    def segment_single_image(self,support_image, query_image, mask_image, model):
        result = self.forward(support_image, mask_image, query_image)
        return result


class ClassIoUNew:
    def __init__(self, class_size):
        self.class_size = class_size
        self.cls_iou = torch.zeros(self.class_size)
        self.cls_counts = torch.zeros(self.class_size)

    def update(self, intersection: torch.Tensor, union: torch.Tensor, classes: torch.Tensor):  # , batch_nb):
        for i, task_cls in enumerate(classes[0]):
            iou_score = intersection[i, 0, 1] / union[i, 0, 1]
            if union[i, 0, 1] != 0 and not torch.isnan(iou_score) and not torch.isinf(iou_score):
                self.cls_iou[(task_cls - 1) % self.class_size] += iou_score
                self.cls_counts[(task_cls - 1) % self.class_size] += 1

    def compute(self):
        print(self.cls_iou, self.cls_counts)
        return torch.mean(self.cls_iou[self.cls_counts != 0] / self.cls_counts[self.cls_counts != 0])

    def reset(self):
        self.cls_iou = torch.zeros(self.class_size)
        self.cls_counts = torch.zeros(self.class_size)
