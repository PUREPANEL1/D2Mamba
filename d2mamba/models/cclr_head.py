"""CCLR decoder (Conv-LayerNorm2d-ReLU) used by D²Mamba."""

from typing import Sequence

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize
from mmseg.registry import MODELS


class LayerNorm2d(nn.Module):
    """LayerNorm for NCHW feature maps."""

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class ConvLNAct(nn.Sequential):
    """Conv + LayerNorm2d + activation used by the upsampling decoder."""

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, eps=1e-6):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=True),
            LayerNorm2d(out_channels, eps=eps),
            nn.ReLU(inplace=True),
        )


@MODELS.register_module()
class CCLRHead(BaseDecodeHead):
    """CCLR U-Net style decoder for DDR lesion segmentation.

    The D²Mamba backbone supplies four stages at roughly 1/4, 1/8, 1/16 and 1/32
    resolution. This head progressively upsamples the deepest feature and
    fuses shallower skip features before predicting at 1/4 resolution.
    """

    def __init__(self,
                 in_channels: Sequence[int],
                 channels: int = 256,
                 decoder_channels: Sequence[int] = (256, 192, 128),
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            input_transform='multiple_select',
            **kwargs)

        if len(in_channels) != 4:
            raise ValueError('CCLRHead expects four backbone feature maps.')
        if len(decoder_channels) != 3:
            raise ValueError('decoder_channels must have three values.')

        self.use_ln2d = self.norm_cfg is not None and self.norm_cfg.get('type') == 'LN2d'
        self.ln_eps = 1e-6 if not self.use_ln2d else self.norm_cfg.get('eps', 1e-6)

        self.lateral_convs = nn.ModuleList([
            self._make_conv_block(in_ch, channels, kernel_size=1)
            for in_ch in in_channels
        ])

        self.fuse_convs = nn.ModuleList()
        in_ch = channels
        for out_ch in decoder_channels:
            self.fuse_convs.append(
                nn.Sequential(
                    self._make_conv_block(
                        in_ch + channels,
                        out_ch,
                        kernel_size=3,
                        padding=1),
                    self._make_conv_block(
                        out_ch,
                        out_ch,
                        kernel_size=3,
                        padding=1)))
            in_ch = out_ch

        self.bottleneck = self._make_conv_block(
            channels, channels, kernel_size=3, padding=1)
        self.conv_seg = nn.Conv2d(decoder_channels[-1], self.num_classes, 1)

    def _make_conv_block(self, in_channels, out_channels, kernel_size, padding=0):
        if self.use_ln2d:
            return ConvLNAct(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                eps=self.ln_eps)
        return ConvModule(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        laterals = [
            lateral_conv(x)
            for lateral_conv, x in zip(self.lateral_convs, inputs)
        ]

        x = self.bottleneck(laterals[-1])
        for skip, fuse_conv in zip(reversed(laterals[:-1]), self.fuse_convs):
            x = resize(
                input=x,
                size=skip.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            x = fuse_conv(torch.cat([x, skip], dim=1))

        return self.cls_seg(x)
