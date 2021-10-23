import collections
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# ---------------------------------------------------------------------------- #
# Functions for bolting FPN onto a backbone architectures
# ---------------------------------------------------------------------------- #
class fpn(nn.Module):
    """Add FPN connections based on the model described in the FPN paper.
    fpn_output_blobs is in reversed order: e.g [fpn5, fpn4, fpn3, fpn2]
    similarly for fpn_level_info.dims: e.g [2048, 1024, 512, 256]
    similarly for spatial_scale: e.g [1/32, 1/16, 1/8, 1/4]
    """
    def __init__(self, 
                 num_backbone_outs,
                 backbone_num_channels,
                 hidden_dim,
                 P2only=False, panet_buttomup=True, use_gn=False):
        super().__init__()
        self.P2only = P2only
        self.panet_buttomup = panet_buttomup

        self.dim_out = hidden_dim
        self.num_backbone_stages = num_backbone_outs

        backbone_num_channels.reverse()
        fpn_dim_lateral = backbone_num_channels
        self.spatial_scale = []  # a list of scales for FPN outputs

        #
        # Step 1: recursively build down starting from the coarsest backbone level
        #
        # For the coarest backbone level: 1x1 conv only seeds recursion
        self.conv_top = nn.Conv2d(fpn_dim_lateral[0], hidden_dim, 1, 1, 0)
        if use_gn:
            self.conv_top = nn.Sequential(
                nn.Conv2d(fpn_dim_lateral[0], hidden_dim, 1, 1, 0, bias=False),
                nn.GroupNorm(32, hidden_dim)
            )
        else:
            self.conv_top = nn.Conv2d(fpn_dim_lateral[0], hidden_dim, 1, 1, 0)
        self.topdown_lateral_modules = nn.ModuleList()
        self.posthoc_modules = nn.ModuleList()

        # For other levels add top-down and lateral connections
        for i in range(self.num_backbone_stages - 1):
            self.topdown_lateral_modules.append(
                topdown_lateral_module(hidden_dim, fpn_dim_lateral[i+1],use_gn=use_gn)
            )

        # Post-hoc scale-specific 3x3 convs
        for i in range(self.num_backbone_stages):
            if use_gn:
                self.posthoc_modules.append(nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False),
                    nn.GroupNorm(32, hidden_dim)
                ))
            else:
                self.posthoc_modules.append(
                    nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1)
                )

        # add for panet buttom-up path
        if self.panet_buttomup:
            self.panet_buttomup_conv1_modules = nn.ModuleList()
            self.panet_buttomup_conv2_modules = nn.ModuleList()
            for i in range(self.num_backbone_stages - 1):
                if use_gn:
                    self.panet_buttomup_conv1_modules.append(nn.Sequential(
                        nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1, bias=True),
                        nn.GroupNorm(32, hidden_dim),
                        nn.ReLU(inplace=True)
                    ))
                    self.panet_buttomup_conv2_modules.append(nn.Sequential(
                        nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=True),
                        nn.GroupNorm(32, hidden_dim),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    self.panet_buttomup_conv1_modules.append(
                        nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1)
                    )
                    self.panet_buttomup_conv2_modules.append(
                        nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1)
                    )

                #self.spatial_scale.append(fpn_level_info.spatial_scales[i])


        #
        # Step 2: build up starting from the coarsest backbone level
        #
        # Check if we need the P6 feature map
        if self.P2only:
            # use only the finest level
            self.spatial_scale = self.spatial_scale[-1]

        self._init_weights()


    def _init_weights(self):
        def init_func(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                
                #mynn.init.MSRAFill(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for child_m in self.children():
            if (not isinstance(child_m, nn.ModuleList) or
                not isinstance(child_m[0], topdown_lateral_module)):
                # topdown_lateral_module has its own init method
                child_m.apply(init_func)


    def forward(self, x):
        fpn_inner_blobs = [self.conv_top(x[-1])]
        for i in range(self.num_backbone_stages - 1):
            fpn_inner_blobs.append(
                self.topdown_lateral_modules[i](fpn_inner_blobs[-1], x[-(i+2)])
            )
        fpn_output_blobs = []
        if self.panet_buttomup:
            fpn_middle_blobs = []
        for i in range(self.num_backbone_stages):
            if not self.panet_buttomup:
                fpn_output_blobs.append(
                    self.posthoc_modules[i](fpn_inner_blobs[i])
                )
            else:
                fpn_middle_blobs.append(
                    self.posthoc_modules[i](fpn_inner_blobs[i])
                )
        if self.panet_buttomup:
            fpn_output_blobs.append(fpn_middle_blobs[-1])
            for i in range(2, self.num_backbone_stages + 1):
                fpn_tmp = self.panet_buttomup_conv1_modules[i - 2](fpn_output_blobs[0])
                #print(fpn_middle_blobs[self.num_backbone_stages - i].size())
                fpn_tmp = fpn_tmp + fpn_middle_blobs[self.num_backbone_stages - i]
                fpn_tmp = self.panet_buttomup_conv2_modules[i - 2](fpn_tmp)
                fpn_output_blobs.insert(0, fpn_tmp)        

        fpn_output_blobs.reverse()
        if self.P2only:
            # use only the finest level
            return fpn_output_blobs[-1]
        else:
            # use all levels
            return fpn_output_blobs


class topdown_lateral_module(nn.Module):
    """Add a top-down lateral module."""
    def __init__(self, dim_in_top, dim_in_lateral, use_gn=False):
        super().__init__()
        self.dim_in_top = dim_in_top
        self.dim_in_lateral = dim_in_lateral
        self.dim_out = dim_in_top
        self.use_gn = use_gn
        if use_gn:
            self.conv_lateral = nn.Sequential(
                nn.Conv2d(dim_in_lateral, self.dim_out, 1, 1, 0, bias=False),
                nn.GroupNorm(32, self.dim_out)
            )
        else:
            self.conv_lateral = nn.Conv2d(dim_in_lateral, self.dim_out, 1, 1, 0)

        self._init_weights()

    def _init_weights(self):
        if self.use_gn:
            conv = self.conv_lateral[0]
        else:
            conv = self.conv_lateral

        nn.init.xavier_uniform_(conv.weight, gain=1)
        if conv.bias is not None:
            init.constant_(conv.bias, 0)

    def forward(self, top_blob, lateral_blob):
        # Lateral 1x1 conv
        lat = self.conv_lateral(lateral_blob)
        feat_shape = lat.shape[-2:]
        # Top-down 2x upsampling
        td = F.interpolate(top_blob, size=feat_shape, mode="nearest")
        # td = F.upsample(top_blob, size=lat.size()[2:], mode='bilinear')
        # td = F.upsample(top_blob, scale_factor=2, mode='nearest')
        # Sum lateral and top-down
        return lat + td

