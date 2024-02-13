 # Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmdet.registry import MODELS
from mmdet.utils import OptMultiConfig



# SpatialAttention Module
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_map, gt_heatmap=None):
        avg_out = torch.mean(feature_map, dim=1, keepdim=True)
        max_out, _ = torch.max(feature_map, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.sigmoid(self.conv1(x))
        if gt_heatmap is not None:
            attention_map *= gt_heatmap
        return feature_map * attention_map

# AttentionFusion Module
class AttentionFusion(nn.Module):
    def __init__(self, in_channels):
        super(AttentionFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        attention = self.sigmoid(self.conv1(x))
        x = x * attention
        out = self.conv2(x)
        return out

@MODELS.register_module()
class CustomFusionNeck(BaseModule):
    def __init__(self, in_channels, num_deconv_filters, num_deconv_kernels, use_dcn=True, init_cfg=None):
        super(CustomFusionNeck, self).__init__(init_cfg=init_cfg)
        assert len(num_deconv_filters) == len(num_deconv_kernels)
        self.fp16_enabled = False
        self.use_dcn = use_dcn
        self.in_channels = in_channels
        

       
        # Initialize deconvolution layers
        self.deconv_layers = nn.ModuleList()
        for i, (in_channel, out_channel, kernel_size) in enumerate(zip(in_channels, num_deconv_filters, num_deconv_kernels)):
            self.deconv_layers.append(self._make_deconv_layer(in_channel, out_channel, kernel_size))


        # Initialize deconvolution layers
        #self.deconv_layers = nn.ModuleList()
        #for i, (out_channels, kernel_size) in enumerate(zip(num_deconv_filters, num_deconv_kernels)):
            #in_channels_layer = self.in_channels if i == 0 else num_deconv_filters[i-1]
            #self.deconv_layers.append(self._make_deconv_layer(in_channels_layer, out_channels, kernel_size))

        # Initialize spatial attention and attention fusion modules for each stage
        self.spatial_attentions = nn.ModuleList([SpatialAttention(out_channels) for out_channels in num_deconv_filters])
        self.attention_fusions = nn.ModuleList([AttentionFusion(out_channels) for out_channels in num_deconv_filters])
        self.channel_align_layers = nn.ModuleList([self._make_channel_align_layer(out_channel) for out_channel in num_deconv_filters])
    def _make_channel_align_layer(self, in_channels, out_channels=256):
        return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    def _make_deconv_layer(self, in_channels, out_channels, kernel_size):
        conv_module = ConvModule(
            in_channels,
            out_channels,
            3,
            padding=1,
            conv_cfg=dict(type='DCNv2') if self.use_dcn else None,
            norm_cfg=dict(type='BN'))
        
        upsample_module = ConvModule(
            out_channels,
            out_channels,
            kernel_size,
            stride=2,
            padding=1,
            conv_cfg=dict(type='deconv'),
            norm_cfg=dict(type='BN'))

        return nn.Sequential(conv_module, upsample_module)


    def init_weights(self) -> None:
        """Initialize the parameters."""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                # In order to be consistent with the source code,
                # reset the ConvTranspose2d initialization parameters
                m.reset_parameters()
                # Simulated bilinear upsampling kernel
                w = m.weight.data
                f = math.ceil(w.size(2) / 2)
                c = (2 * f - 1 - f % 2) / (2. * f)
                for i in range(w.size(2)):
                    for j in range(w.size(3)):
                        w[0, 0, i, j] = \
                            (1 - math.fabs(i / f - c)) * (
                                    1 - math.fabs(j / f - c))
                for c in range(1, w.size(0)):
                    w[c, 0, :, :] = w[0, 0, :, :]
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # self.use_dcn is False
            elif not self.use_dcn and isinstance(m, nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
                m.reset_parameters()


    def forward(self, feature_maps):
        #if len(feature_maps) != len(self.deconv_layers):
            #raise ValueError("Number of input feature maps does not match the number of deconvolution layers")
        assert isinstance(feature_maps, list)
        outs = []
        for i, (fmap, deconv_layer, spatial_attention, attention_fusion, align_layer) in enumerate(
            zip(feature_maps, self.deconv_layers, self.spatial_attentions, self.attention_fusions, self.channel_align_layers)):
        
            fmap = deconv_layer(fmap)
            fmap = spatial_attention(fmap)
            fmap = attention_fusion(fmap)
            fmap = align_layer(fmap)  # Corrected this line
            outs.append(fmap)
        #for fmap in outs:
            #print("fmap_shape-------------", fmap.shape)
        return outs

