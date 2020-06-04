import torch
from torch import nn
import torch.nn.functional as F

from ..post_processing import matrix_nms
from detectron2.structures import Instances, Boxes


class SOLOV2_decode(nn.Module):
    def __init__(self, cfg):
        super(SOLOV2_decode, self).__init__()
        self.cfg = cfg

        self.nms_pre = cfg.MODEL.SOLOV2.NMS_PRE
        self.score_th = cfg.MODEL.SOLOV2.SCORE_TH
        self.mask_th = cfg.MODEL.SOLOV2.MASK_TH
        self.update_th = cfg.MODEL.SOLOV2.UPDATE_TH
        self.kernel = cfg.MODEL.SOLOV2.NMS_KERNEL
        self.sigma = cfg.MODEL.SOLOV2.NMS_SIGMA
        self.max_per_img = cfg.MODEL.SOLOV2.MAX_PER_IMG
        self.cate_out_channels = cfg.MODEL.SOLOV2.NUM_CLASSES
        self.seg_num_grids = cfg.MODEL.SOLOV2.NUM_GRIDS
        self.strides = cfg.MODEL.SOLOV2.STRIDES

    def forward(self, images, ins_pred, cate_pred):
        boxlists = self.get_seg(ins_pred, cate_pred, images)
        return boxlists

    def get_seg(self, seg_preds, cate_preds, images):
        assert len(seg_preds) == len(cate_preds)
        image_sizes = images.image_sizes
        num_levels = len(cate_preds)
        featmap_size = seg_preds[0].size()[-2:]  # 4

        boxlists = []
        for img_id in range(len(images)):
            cate_pred_list = [cate_preds[i][img_id].view(-1, self.cate_out_channels).detach()
                              for i in range(num_levels)]
            seg_pred_list = [seg_preds[i][img_id].detach() for i in range(num_levels)]

            cate_pred_list = torch.cat(cate_pred_list, dim=0)
            seg_pred_list = torch.cat(seg_pred_list, dim=0)

            boxlist = self.get_seg_single(cate_pred_list, seg_pred_list, featmap_size, image_sizes[img_id])
            boxlists.append(boxlist)
        return boxlists

    def get_seg_single(self, cate_preds, seg_preds, featmap_size, image_size):
        assert len(cate_preds) == len(seg_preds)

        h, w = image_size
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)

        inds = cate_preds.gt(self.score_th)
        cate_scores = cate_preds[inds]
        if len(cate_scores) == 0:
            boxlist = Instances(image_size)
            boxlist.scores = torch.tensor([])
            boxlist.pred_classes = torch.tensor([])
            boxlist.pred_masks = torch.tensor([])
            boxlist.pred_boxes = Boxes(torch.tensor([]))
            return boxlist

        # category labels.
        inds = inds.nonzero()
        cate_labels = inds[:, 1]

        # strides.
        size_trans = cate_labels.new_tensor(self.seg_num_grids).pow(2).cumsum(0)
        strides = cate_scores.new_ones(size_trans[-1])
        n_stage = len(self.seg_num_grids)
        strides[:size_trans[0]] *= self.strides[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_ - 1]:size_trans[ind_]] *= self.strides[ind_]
        strides = strides[inds[:, 0]]

        # masks.
        seg_preds = seg_preds[inds[:, 0]]
        seg_masks = seg_preds > self.mask_th
        sum_masks = seg_masks.sum((1, 2)).float()

        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            boxlist = Instances(image_size)
            boxlist.scores = torch.tensor([])
            boxlist.pred_classes = torch.tensor([])
            boxlist.pred_masks = torch.tensor([])
            boxlist.pred_boxes = Boxes(torch.tensor([]))
            return boxlist

        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # mask scoring.
        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores

        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > self.nms_pre:
            sort_inds = sort_inds[:self.nms_pre]
        seg_masks = seg_masks[sort_inds, :, :]
        seg_preds = seg_preds[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        # Matrix NMS
        cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores,
                                 kernel=self.kernel, sigma=self.sigma, sum_masks=sum_masks)

        # filter.
        keep = cate_scores >= self.update_th
        if keep.sum() == 0:
            boxlist = Instances(image_size)
            boxlist.scores = torch.tensor([])
            boxlist.pred_classes = torch.tensor([])
            boxlist.pred_masks = torch.tensor([])
            boxlist.pred_boxes = Boxes(torch.tensor([]))
            return boxlist

        seg_preds = seg_preds[keep, :, :]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > self.max_per_img:
            sort_inds = sort_inds[:self.max_per_img]
        seg_preds = seg_preds[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        seg_preds = F.interpolate(seg_preds.unsqueeze(0),
                                  size=upsampled_size_out,
                                  mode='bilinear')[:, :, :h, :w]
        seg_masks = seg_preds > self.mask_th
        seg_masks = seg_masks.float().permute(1, 0, 2, 3)

        boxlist = Instances(image_size)
        boxlist.scores = cate_scores
        boxlist.pred_classes = cate_labels
        boxlist.pred_masks = seg_masks

        # get bbox from mask
        pred_boxes = torch.zeros(seg_masks.size(0), 4)
        for i in range(seg_masks.size(0)):
            mask = seg_masks[i].squeeze()
            ys, xs = torch.where(mask > self.mask_th)
            pred_boxes[i] = torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()]).float()
        boxlist.pred_boxes = Boxes(pred_boxes)

        return boxlist
