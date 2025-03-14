# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmdet.registry import MODELS

# ------------------------------------------
# Multi-Frame Attention (MFA) with Adaptive Temporal Weights
# ------------------------------------------
class MultiFrameAttention(nn.Module):
    def __init__(self, channels):
        super(MultiFrameAttention, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = nn.Linear(channels, channels // 4)
        self.fc = nn.Linear(channels // 4, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_t, x_prev):
        """
        x_t: Current frame feature map (B, C, H, W)
        x_prev: Previous frame feature map (B, C, H, W)
        """
        # Global Average Pooling for Spatial Context
        s_t = self.global_pool(x_t).view(x_t.size(0), -1)
        s_prev = self.global_pool(x_prev).view(x_prev.size(0), -1)

        # Compute Adaptive Temporal Weights
        weights = self.bottleneck(s_t + s_prev)
        weights = self.fc(F.relu(weights))
        weights = self.sigmoid(weights).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)

        # Apply Temporal Attention to Merge Frames
        x_out = x_t + weights * x_prev  # Equation (4) in STF
        return x_out


# ------------------------------------------
# Single-Frame Attention (SFA) with Channel & Spatial Attention
# ------------------------------------------
class SingleFrameAttention(nn.Module):
    def __init__(self, in_channels):
        super(SingleFrameAttention, self).__init__()
        
        # Channel Attention
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Spatial Attention
        self.conv5x5 = nn.Conv2d(2, 1, kernel_size=5, padding=2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: Feature map of the current frame (B, C, H, W)
        """
        # Channel Attention
        avg_out = F.adaptive_avg_pool2d(x, 1)
        max_out = F.adaptive_max_pool2d(x, 1)
        ch_att = self.conv1x1(avg_out + max_out)
        ch_att = self.sigmoid(ch_att) * x

        # Spatial Attention
        avg_out_sp = torch.mean(ch_att, dim=1, keepdim=True)
        max_out_sp, _ = torch.max(ch_att, dim=1, keepdim=True)
        sp_att = self.conv5x5(torch.cat([avg_out_sp, max_out_sp], dim=1))
        sp_att = self.sigmoid(sp_att)

        return ch_att * sp_att


# ------------------------------------------
# Dual-Frame Fusion Module with Adaptive Feature Pooling (AFP) & Deformable Convolutions
# ------------------------------------------
class DualFrameFusion(nn.Module):
    def __init__(self, in_channels):
        super(DualFrameFusion, self).__init__()
        
        # Adaptive Feature Pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((32, 32))  # Target size for fusion
        
        # Deformable Convolution
        self.deform_conv = ConvModule(
            in_channels * 2, in_channels, kernel_size=3, padding=1, conv_cfg=dict(type='DCNv2'), norm_cfg=dict(type='BN')
        )

    def forward(self, x_mfa, x_sfa):
        """
        x_mfa: MFA-enhanced feature map (B, C, H, W)
        x_sfa: SFA-enhanced feature map (B, C, H, W)
        """
        # Adaptive Feature Pooling to match sizes
        x_sfa_resized = self.adaptive_pool(x_sfa)

        # Concatenation & Deformable Convolution
        x_fusion = torch.cat([x_mfa, x_sfa_resized], dim=1)
        x_fusion = self.deform_conv(x_fusion)

        return x_fusion


# ------------------------------------------
# CustomFusionNeck: Full STF Integration
# ------------------------------------------
@MODELS.register_module()
class CustomFusionNeck(BaseModule):
    def __init__(self, in_channels, use_dcn=True, init_cfg=None):
        super(CustomFusionNeck, self).__init__(init_cfg=init_cfg)
        self.use_dcn = use_dcn

        # STF Modules
        self.mfa = MultiFrameAttention(in_channels)
        self.sfa = SingleFrameAttention(in_channels)
        self.dual_fusion = DualFrameFusion(in_channels)

    def forward(self, x_t, x_prev):
        """
        x_t: Feature map of the current frame (B, C, H, W)
        x_prev: Feature map of the previous frame (B, C, H, W)
        """
        # Apply Multi-Frame Attention (MFA)
        x_mfa = self.mfa(x_t, x_prev)

        # Apply Single-Frame Attention (SFA)
        x_sfa = self.sfa(x_t)

        # Apply Dual-Frame Fusion Module
        x_fused = self.dual_fusion(x_mfa, x_sfa)

        return x_fused
