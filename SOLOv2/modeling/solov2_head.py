import torch
from torch import nn
import torch.nn.functional as F
import math


def points_nms(heat, kernel=2):
    # kernel must be 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=1)
    keep = (hmax[:, :, :-1, :-1] == heat).float()
    return heat * keep


class SOLOV2_head(nn.Module):
    def __init__(self, cfg):
        super(SOLOV2_head, self).__init__()

        self.cfg = cfg
        self.num_classes = cfg.MODEL.SOLOV2.NUM_CLASSES
        self.cate_out_channels = self.num_classes  # 80
        self.seg_num_grids = cfg.MODEL.SOLOV2.NUM_GRIDS
        self.seg_feat_channels = cfg.MODEL.SOLOV2.SEG_FEAT_CHANNELS
        self.in_channels = cfg.MODEL.SOLOV2.IN_CHANNELS
        self.stacked_convs = cfg.MODEL.SOLOV2.STACKED_CONVS
        self.cate_down_pos = cfg.MODEL.SOLOV2.CATE_DOWN_POS

        self._init_layers()
        self.init_weight()

    def _init_layers(self):
        self.feature_convs = nn.ModuleList()
        self.kernel_convs = nn.ModuleList()
        self.cate_convs = nn.ModuleList()
        norm = nn.GroupNorm  # don't freeze

        # mask
        for i in range(4):
            convs_per_level = nn.Sequential()
            if i == 0:
                one_conv = nn.Sequential(
                    nn.Conv2d(self.in_channels, self.seg_feat_channels, 3, 1, 1, bias=False),
                    norm(num_groups=32, num_channels=self.seg_feat_channels),
                    nn.ReLU(True)
                )
                convs_per_level.add_module(f'conv{i}', one_conv)
                self.feature_convs.append(convs_per_level)
                continue

            for j in range(i):
                if j == 0:
                    if i == 3:
                        in_channel = self.in_channels + 2
                    else:
                        in_channel = self.in_channels
                    one_conv = nn.Sequential(
                        nn.Conv2d(in_channel, self.seg_feat_channels, 3, 1, 1, bias=False),
                        norm(32, self.seg_feat_channels),
                        nn.ReLU(True)
                    )
                    convs_per_level.add_module(f'conv{j}', one_conv)
                    one_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                    convs_per_level.add_module(f'upsample{j}', one_upsample)
                    continue

                one_conv = nn.Sequential(
                    nn.Conv2d(self.seg_feat_channels, self.seg_feat_channels, 3, 1, 1, bias=False),
                    norm(32, self.seg_feat_channels),
                    nn.ReLU(True)
                )
                convs_per_level.add_module(f'conv{j}', one_conv)
                one_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                convs_per_level.add_module(f'upsample{j}', one_upsample)
            self.feature_convs.append(convs_per_level)

        for i in range(self.stacked_convs):
            chn = self.in_channels + 2 if i == 0 else self.seg_feat_channels
            self.kernel_convs.append(nn.Sequential(
                nn.Conv2d(chn, self.seg_feat_channels, 3, 1, 1, bias=False),
                norm(32, self.seg_feat_channels),
                nn.ReLU(True)
            ))
            chn = self.in_channels if i == 0 else self.seg_feat_channels
            self.cate_convs.append(nn.Sequential(
                nn.Conv2d(chn, self.seg_feat_channels, 3, 1, 1, bias=False),
                norm(32, self.seg_feat_channels),
                nn.ReLU(True)
            ))

        self.solo_kernel = nn.Conv2d(self.seg_feat_channels, self.seg_feat_channels, 1, padding=0)
        self.solo_cate = nn.Conv2d(self.seg_feat_channels, self.cate_out_channels, 3, padding=1)
        self.solo_mask = nn.Sequential(
            nn.Conv2d(self.seg_feat_channels, self.seg_feat_channels, 1, padding=0, bias=False),
            norm(32, self.seg_feat_channels),
            nn.ReLU(True)
        )

    def init_weight(self):
        for m in self.feature_convs:
            s = len(m)
            for i in range(s):
                if i % 2 == 0:
                    torch.nn.init.normal_(m[i][0].weight, std=0.01)
                    torch.nn.init.constant_(m[i][0].weight, 0)
        for m in self.kernel_convs:
            torch.nn.init.normal_(m[0].weight, std=0.01)
            torch.nn.init.constant_(m[0].weight, 0)
        for m in self.cate_convs:
            torch.nn.init.normal_(m[0].weight, std=0.01)
            torch.nn.init.constant_(m[0].weight, 0)

        prior_prob = self.cfg.MODEL.SOLOV2.PRIOR_PROB
        bias_init = float(-math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.normal_(self.solo_cate.weight, std=0.01)
        torch.nn.init.constant_(self.solo_cate.bias, bias_init)

    def split_feats(self, feats):
        return (F.interpolate(feats['p2'], scale_factor=0.5, mode='bilinear'),
                feats['p3'],
                feats['p4'],
                feats['p5'],
                F.interpolate(feats['p6'], size=feats['p5'].shape[-2:], mode='bilinear'))

    def forward(self, feats, eval=False):
        new_feats = self.split_feats(feats)
        feats = tuple(feats.values())
        featmap_sizes = [featmap.size()[-2:] for featmap in new_feats]

        kernel_pred = []
        cate_pred = []
        for feat, idx in zip(new_feats, list(range(len(self.seg_num_grids)))):
            ret = self.forward_single(feat, idx, eval=eval)
            kernel_pred.append(ret[0])
            cate_pred.append(ret[1])

        # add coord for p5
        x_range = torch.linspace(-1, 1, feats[-2].shape[-1], device=feats[-2].device)
        y_range = torch.linspace(-1, 1, feats[-2].shape[-2], device=feats[-2].device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([feats[-2].shape[0], 1, -1, -1])
        x = x.expand([feats[-2].shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)

        feature_add_all_level = self.feature_convs[0](feats[0])  # p2
        for i in range(1, 3):  # [p3, p4]
            feature_add_all_level = feature_add_all_level + self.feature_convs[i](feats[i])
        feature_add_all_level = feature_add_all_level + self.feature_convs[3](torch.cat([feats[3], coord_feat], 1))

        feature_pred = self.solo_mask(feature_add_all_level)  # [4, ..., 4]
        N, c, h, w = feature_pred.shape
        feature_pred = feature_pred.view(-1, h, w).unsqueeze(0)
        ins_pred = []

        for i in range(5):
            kernel = kernel_pred[i].permute(0, 2, 3, 1).contiguous().view(-1, c).unsqueeze(-1).unsqueeze(-1)
            ins_i = F.conv2d(feature_pred, kernel, groups=N).view(N, self.seg_num_grids[i] ** 2, h, w)
            if not eval:
                ins_i = F.interpolate(ins_i, size=(featmap_sizes[i][0] * 2, featmap_sizes[i][1] * 2), mode='bilinear')
            if eval:
                ins_i = ins_i.sigmoid()
            ins_pred.append(ins_i)
        return ins_pred, cate_pred

    def forward_single(self, x, idx, eval=False):
        kernel_feat = x
        cate_feat = x

        x_range = torch.linspace(-1, 1, kernel_feat.shape[-1], device=kernel_feat.device)
        y_range = torch.linspace(-1, 1, kernel_feat.shape[-2], device=kernel_feat.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([kernel_feat.shape[0], 1, -1, -1])
        x = x.expand([kernel_feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)

        # kernel branch
        kernel_feat = torch.cat([kernel_feat, coord_feat], 1)
        for i, kernel_layer in enumerate(self.kernel_convs):
            if i == self.cate_down_pos:
                seg_num_grid = self.seg_num_grids[idx]
                kernel_feat = F.interpolate(kernel_feat, size=seg_num_grid, mode='bilinear')
            kernel_feat = kernel_layer(kernel_feat)
        kernel_pred = self.solo_kernel(kernel_feat)

        # cate branch
        for i, cate_layer in enumerate(self.cate_convs):
            if i == self.cate_down_pos:
                seg_num_grid = self.seg_num_grids[idx]
                cate_feat = F.interpolate(cate_feat, size=seg_num_grid, mode='bilinear')
            cate_feat = cate_layer(cate_feat)
        cate_pred = self.solo_cate(cate_feat)

        if eval:
            cate_pred = points_nms(cate_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1)
        return kernel_pred, cate_pred
