"""
argos.model
===========

A small ResNeXt U-Net for precipitation retrieval.

The model fuses two inputs:

* ``geo``: the high-resolution (0.025-degree) geostationary observations, shape
  ``(batch, N_SLOTS, H, W)``.
* ``mw``: the reduced-resolution (0.05-degree) passive-microwave observations,
  shape ``(batch, N_MW_SLOTS, H // RESOLUTION_RATIO, W // RESOLUTION_RATIO)``.

The geostationary input is downsampled to the microwave/reference resolution,
fused with the microwave features, and passed through a ResNeXt U-Net whose
output is at the resolution of the microwave (reference) data, i.e.
``(batch, out_channels, H // RESOLUTION_RATIO, W // RESOLUTION_RATIO)``.

Either input may be omitted (``None``) for a whole batch; missing channels within
an otherwise present input should be encoded as ``NaN`` and are treated as zeros.
"""
from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from argos.data.dataset import N_MW_SLOTS, N_SLOTS, RESOLUTION_RATIO

__all__ = ["ResNeXtBlock", "ResNeXtUNet"]


class ResNeXtBlock(nn.Module):
    """
    A ResNeXt bottleneck block with grouped convolutions.

    Performs ``1x1 -> 3x3 (grouped) -> 1x1`` convolutions with a residual
    connection. The bottleneck width equals ``out_channels`` and must be
    divisible by ``cardinality``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cardinality: int = 8,
        stride: int = 1,
    ):
        super().__init__()
        if out_channels % cardinality != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by "
                f"cardinality ({cardinality})."
            )
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3,
            stride=stride, padding=1, groups=cardinality, bias=False,
        )
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.norm1(self.conv1(x)))
        out = self.act(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        return self.act(out + self.shortcut(x))


class ResNeXtUNet(nn.Module):
    """
    A small ResNeXt U-Net fusing geostationary and microwave observations.

    Args:
        geo_channels: Number of geostationary input channels.
        mw_channels: Number of microwave input channels.
        out_channels: Number of output channels (1 for surface precipitation).
        base_channels: Channel width of the first (full-resolution) stage.
        depth: Number of down-/up-sampling stages of the U-Net.
        cardinality: Number of groups in the ResNeXt blocks.
        geo_downsample: Factor by which the geostationary input is downsampled to
            reach the microwave/reference resolution.
    """

    def __init__(
        self,
        geo_channels: int = N_SLOTS,
        mw_channels: int = N_MW_SLOTS,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
        cardinality: int = 8,
        geo_downsample: int = RESOLUTION_RATIO,
    ):
        super().__init__()
        self.geo_channels = geo_channels
        self.mw_channels = mw_channels
        self.base_channels = base_channels
        self.geo_downsample = geo_downsample

        channels = base_channels

        # Input stems. The geo stem downsamples to the microwave resolution.
        self.geo_stem = nn.Sequential(
            nn.Conv2d(
                geo_channels, channels, 3,
                stride=geo_downsample, padding=1, bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            ResNeXtBlock(channels, channels, cardinality),
        )
        self.mw_stem = nn.Sequential(
            nn.Conv2d(mw_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            ResNeXtBlock(channels, channels, cardinality),
        )
        # Fuse the two (concatenated) feature maps.
        self.fuse = ResNeXtBlock(2 * channels, channels, cardinality)

        # Encoder: ``depth`` downsampling stages, doubling the channels each time.
        self.encoder = nn.ModuleList()
        skip_channels = [channels]
        for _ in range(depth):
            self.encoder.append(
                ResNeXtBlock(channels, channels * 2, cardinality, stride=2)
            )
            channels *= 2
            skip_channels.append(channels)

        # Decoder: ``depth`` upsampling stages with skip connections.
        self.upsamplers = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for stage in range(depth):
            self.upsamplers.append(
                nn.Conv2d(channels, channels // 2, 1, bias=False)
            )
            skip = skip_channels[depth - 1 - stage]
            self.decoder.append(
                ResNeXtBlock(channels // 2 + skip, channels // 2, cardinality)
            )
            channels //= 2

        self.head = nn.Conv2d(channels, out_channels, 1)

    def forward(
        self,
        geo: Optional[torch.Tensor] = None,
        mw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if geo is None and mw is None:
            raise ValueError("At least one of 'geo' or 'mw' must be provided.")

        # Encode each available input; missing channels (NaN) become zeros.
        mw_feat = None
        if mw is not None:
            mw_feat = self.mw_stem(torch.nan_to_num(mw))
        geo_feat = None
        if geo is not None:
            geo_feat = self.geo_stem(torch.nan_to_num(geo))

        # The output resolution follows the microwave/reference grid.
        reference = mw_feat if mw_feat is not None else geo_feat
        size = reference.shape[-2:]
        batch, device, dtype = (
            reference.shape[0], reference.device, reference.dtype,
        )
        if geo_feat is None:
            geo_feat = torch.zeros(
                batch, self.base_channels, *size, device=device, dtype=dtype
            )
        elif geo_feat.shape[-2:] != size:
            geo_feat = F.interpolate(
                geo_feat, size=size, mode="bilinear", align_corners=False
            )
        if mw_feat is None:
            mw_feat = torch.zeros(
                batch, self.base_channels, *size, device=device, dtype=dtype
            )

        x = self.fuse(torch.cat([geo_feat, mw_feat], dim=1))

        # Encoder, keeping the skip connections.
        skips: List[torch.Tensor] = [x]
        for block in self.encoder:
            x = block(x)
            skips.append(x)

        # Decoder with skip connections (the last skip is the bottleneck itself).
        for stage, (upsample, block) in enumerate(zip(self.upsamplers, self.decoder)):
            skip = skips[-2 - stage]
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            x = upsample(x)
            x = block(torch.cat([x, skip], dim=1))

        return self.head(x)
