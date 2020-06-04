import torch
from torch import nn
import numpy as np
from scipy import ndimage
from typing import Dict
from detectron2.layers import ShapeSpec
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from .solov2_head import SOLOV2_head
from .solov2_decode import SOLOV2_decode
from SOLOv2.utils import imrescale
from SOLOv2.losses import dice_loss, FocalLoss

__all__ = ['SOLOv2']


@PROPOSAL_GENERATOR_REGISTRY.register()
class SOLOv2(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        self.cfg = cfg
        self.in_features = cfg.MODEL.SOLOV2.IN_FEATURES
        self.scale_ranges = cfg.MODEL.SOLOV2.SCALE_RANGES
        self.strides = cfg.MODEL.SOLOV2.STRIDES
        self.seg_num_grids = cfg.MODEL.SOLOV2.NUM_GRIDS
        self.cate_out_channels = cfg.MODEL.SOLOV2.NUM_CLASSES
        self.sigma = cfg.MODEL.SOLOV2.SIGMA

        self.head = SOLOV2_head(cfg)
        self.decode = SOLOV2_decode(cfg)

        self.dice_loss = dice_loss
        self.ins_loss_weight = cfg.MODEL.LOSS.DICE_LOSS.WEIGHT
        self.focal_loss = FocalLoss(use_sigmoid=cfg.MODEL.LOSS.FOCAL_LOSS.USE_SIGMOID,
                                    gamma=cfg.MODEL.LOSS.FOCAL_LOSS.GAMMA,
                                    alpha=cfg.MODEL.LOSS.FOCAL_LOSS.ALPHA,
                                    loss_weight=cfg.MODEL.LOSS.FOCAL_LOSS.WEIGHT)

    def forward(self, images, features, gt_instances):
        ins_pred, cate_pred = self.head(features, eval=not self.training)

        if self.training:
            losses, _ = self.losses(ins_pred, cate_pred, gt_instances)
            return None, losses
        else:
            proposals = self.decode(images, ins_pred, cate_pred)
            return proposals, {}

    def _get_gt(self, gt_instances, featmap_sizes=None):
        ins_label_list, cate_label_list, ins_ind_label_list = [], [], []
        for im in range(len(gt_instances)):
            _ins_label_list, _cate_label_list, _ins_ind_label_list = self.get_gt_single(im, gt_instances,
                                                                                        featmap_sizes=featmap_sizes)

            ins_label_list.append(_ins_label_list)
            cate_label_list.append(_cate_label_list)
            ins_ind_label_list.append(_ins_ind_label_list)
        return ins_label_list, cate_label_list, ins_ind_label_list

    def get_gt_single(self, im, gt_instances, featmap_sizes=None):
        gt_bboxes_raw = gt_instances[im].gt_boxes.tensor
        gt_labels_raw = gt_instances[im].gt_classes
        gt_masks_raw = gt_instances[im].gt_masks.tensor

        device = gt_labels_raw[0].device
        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))

        ins_label_list = []
        cate_label_list = []
        ins_ind_label_list = []
        for (lower_bound, upper_bound), stride, featmap_size, num_grid \
                in zip(self.scale_ranges, self.strides, featmap_sizes, self.seg_num_grids):

            ins_label = torch.zeros([num_grid ** 2, featmap_size[0], featmap_size[1]], dtype=torch.uint8, device=device)
            cate_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device)
            ins_ind_label = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device)

            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
            if len(hit_indices) == 0:
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                ins_ind_label_list.append(ins_ind_label)
                continue
            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_labels = gt_labels_raw[hit_indices]
            gt_masks = gt_masks_raw[hit_indices.cpu().numpy(), ...].cpu().numpy()

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            output_stride = stride / 2

            for seg_mask, gt_label, half_h, half_w in zip(gt_masks, gt_labels, half_hs, half_ws):
                if seg_mask.sum() < 10:
                    continue
                # mass center
                upsampled_size = (featmap_sizes[0][0] * 4, featmap_sizes[0][1] * 4)
                center_h, center_w = ndimage.measurements.center_of_mass(seg_mask)
                coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))
                coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))

                # left, top, right, down
                top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))
                down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))
                left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))
                right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))

                top = max(top_box, coord_h - 1)
                down = min(down_box, coord_h + 1)
                left = max(coord_w - 1, left_box)
                right = min(right_box, coord_w + 1)

                cate_label[top:(down + 1), left:(right + 1)] = gt_label
                # ins
                seg_mask = imrescale(seg_mask, scale=1. / output_stride)
                seg_mask = torch.Tensor(seg_mask)
                for i in range(top, down + 1):
                    for j in range(left, right + 1):
                        label = int(i * num_grid + j)
                        ins_label[label, :seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask
                        ins_ind_label[label] = True

            ins_label_list.append(ins_label)
            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)
        return ins_label_list, cate_label_list, ins_ind_label_list

    def losses(self, ins_preds, cate_preds, gt_instances):
        featmap_sizes = [featmap.size()[-2:] for featmap in ins_preds]
        ins_label_list, cate_label_list, ins_ind_label_list = self._get_gt(gt_instances, featmap_sizes)

        # ins
        ins_labels = [torch.cat([ins_labels_level_img[ins_ind_labels_level_img, ...]
                                 for ins_labels_level_img, ins_ind_labels_level_img in
                                 zip(ins_labels_level, ins_ind_labels_level)], 0)
                      for ins_labels_level, ins_ind_labels_level in zip(zip(*ins_label_list), zip(*ins_ind_label_list))]

        ins_preds = [torch.cat([ins_preds_level_img[ins_ind_labels_level_img, ...]
                                for ins_preds_level_img, ins_ind_labels_level_img in
                                zip(ins_preds_level, ins_ind_labels_level)], 0)
                     for ins_preds_level, ins_ind_labels_level in zip(ins_preds, zip(*ins_ind_label_list))]

        ins_ind_labels = [
            torch.cat([ins_ind_labels_level_img.flatten()
                       for ins_ind_labels_level_img in ins_ind_labels_level])
            for ins_ind_labels_level in zip(*ins_ind_label_list)
        ]
        flatten_ins_ind_labels = torch.cat(ins_ind_labels)

        num_ins = flatten_ins_ind_labels.int().sum()

        # dice loss
        loss_ins = []
        for input, target in zip(ins_preds, ins_labels):
            if input.size()[0] == 0:
                continue
            input = torch.sigmoid(input)
            loss_ins.append(dice_loss(input, target))
        loss_ins = torch.cat(loss_ins).mean()
        loss_ins = loss_ins * self.ins_loss_weight

        # cate
        cate_labels = [
            torch.cat([cate_labels_level_img.flatten()
                       for cate_labels_level_img in cate_labels_level])
            for cate_labels_level in zip(*cate_label_list)
        ]
        flatten_cate_labels = torch.cat(cate_labels)  # [7767]

        cate_preds = [
            cate_pred.permute(0, 2, 3, 1).reshape(-1, self.cate_out_channels)
            for cate_pred in cate_preds
        ]
        flatten_cate_preds = torch.cat(cate_preds)  # [7767, 80]

        # prepare one_hot
        flatten_cate_labels_oh = torch.zeros_like(flatten_cate_preds)
        pos_ind = flatten_cate_labels.gt(0)
        flatten_cate_labels_oh[pos_ind, flatten_cate_labels[pos_ind].long()] = 1.

        loss_cate = self.focal_loss(flatten_cate_preds, flatten_cate_labels_oh, avg_factor=num_ins+1)

        return {
                   'loss_ins': loss_ins,
                   'loss_cate': loss_cate
               }, {}
