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
from typing import Dict, List, Optional, Sequence, Union

from pytorch_retrieve.tensors.quantile_tensor import QuantileTensor
import torch
from torch import nn
from torch.nn import functional as F

from argos.data.dataset import N_MW_SLOTS, N_SLOTS, RESOLUTION_RATIO

__all__ = ["ResNeXtBlock", "ResNeXtUNet"]

quantiles = [
    0.01,
    0.03225806,
    0.06451613,
    0.09677419,
    0.12903226,
    0.16129032,
    0.19354839,
    0.22580645,
    0.25806452,
    0.29032258,
    0.32258065,
    0.35483871,
    0.38709677,
    0.41935484,
    0.4516129 ,
    0.48387097,
    0.51612903,
    0.5483871 ,
    0.58064516,
    0.61290323,
    0.64516129,
    0.67741935,
    0.70967742,
    0.74193548,
    0.77419355,
    0.80645161,
    0.83870968,
    0.87096774,
    0.90322581,
    0.93548387,
    0.96774194,
    0.99
]
import torch


def channel_dropout(
    x: torch.Tensor,
    p: float,
    training: bool = True,
) -> torch.Tensor:
    """Drop complete input channels independently for each sample."""
    if not training or p == 0.0:
        return x
    if not 0.0 <= p < 1.0:
        raise ValueError("p must satisfy 0 <= p < 1")

    mask_shape = x.shape[:2] + (1,) * (x.ndim - 2)
    mask = (
        p < torch.rand(mask_shape, device=x.device)
    )
    return torch.where(mask, x, torch.nan)


def batch_dropout(
    x: torch.Tensor,
    p: float,
    training: bool = True,
) -> torch.Tensor:
    """Drop complete input channels independently for each sample."""
    if not training or p == 0.0:
        return x
    if not 0.0 <= p < 1.0:
        raise ValueError("p must satisfy 0 <= p < 1")

    mask_shape = x.shape[:1] + (1,) * (x.ndim - 1)
    mask = (
        p < torch.rand(mask_shape, device=x.device)
    )
    return torch.where(mask, x, torch.nan)


class TemporalFusion(nn.Module):
    def __init__(
            self,
            chans_in: int,
            times_in: int,
            times_out: int,
            channel_expansion: int = 1,
            time_expansion: int = 2
    ):
        super().__init__()
        self.in_channels = chans_in
        self.hidden_channels = channel_expansion * self.in_channels
        self.in_times = times_in
        self.hidden_times = times_out * time_expansion
        self.out_times = times_out

        self.channel_expansion = nn.Linear(self.in_channels, self.hidden_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.in_times, self.hidden_times),
            nn.LayerNorm(self.hidden_times),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_times, self.out_times),
        )
        self.channel_mlp = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.LayerNorm(self.hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_channels, self.in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        BT, C, H, W = x.shape
        x = x.reshape(BT // self.in_times, self.in_times, C, H, W)
        shortcut = x

        # Order: B, T, H, W, C
        x = x.permute((0, 1, 3, 4, 2))
        x = self.channel_expansion(x)

        # Order: B, C, H, W, T
        x = x.permute((0, 4, 2, 3, 1))
        x = self.time_mlp(x)

        # Order: B, T, H, W, C
        x = x.permute((0, 4, 2, 3, 1))
        x = self.channel_mlp(x)

        # Order: B, T, H, W, C
        x = x.permute((0, 1, 4, 2, 3))

        x = x + shortcut[:, :self.out_times]
        return x.flatten(0, 1)


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
            times_in: Optional[int] = None,
            times_out: Optional[int] = None
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

        self.times_in = times_in
        self.times_out = times_out
        if times_in is not None:
            self.temporal_fusion = TemporalFusion(out_channels, times_in, times_out)
        else:
            self.temporal_fusion = None


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.norm1(self.conv1(x)))
        out = self.act(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))

        x = self.act(out + self.shortcut(x))

        if self.temporal_fusion is not None:
            x = self.temporal_fusion(x)

        return x


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
        blocks_per_stage: Number of ResNeXt blocks in each encoder stage. Either
            an ``int`` (same for every stage) or a sequence of length ``depth``
            (one entry per stage, from finest to coarsest). Defaults to
            ``(1, 2, ..., depth)``, i.e. more blocks at the coarser stages.
    """

    def __init__(
        self,
        geo_channels: int = N_SLOTS,
        mw_channels: int = N_MW_SLOTS,
        base_channels: int = 48,
        depth: int = 4,
        cardinality: int = 8,
        geo_downsample: int = RESOLUTION_RATIO,
        blocks_per_stage: Optional[Union[int, Sequence[int]]] = None,
        times_in: Optional[int] = None
    ):
        super().__init__()
        self.geo_channels = geo_channels
        self.mw_channels = mw_channels
        self.base_channels = base_channels
        self.geo_downsample = geo_downsample
        self.times_in = times_in
        out_channels = len(quantiles)


        # Number of blocks in each (increasingly coarse) encoder stage.
        if blocks_per_stage is None:
            stage_blocks = list(range(1, depth + 1))
        elif isinstance(blocks_per_stage, int):
            stage_blocks = [blocks_per_stage] * depth
        else:
            stage_blocks = list(blocks_per_stage)
            if len(stage_blocks) != depth:
                raise ValueError(
                    f"blocks_per_stage must have length depth ({depth}), got "
                    f"{len(stage_blocks)}."
                )
        self.blocks_per_stage = stage_blocks

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
        # Each stage downsamples with its first block and then stacks
        # ``stage_blocks`` blocks at the (coarser) stage resolution.
        self.encoder = nn.ModuleList()
        skip_channels = [channels]

        times_out = times_in

        for n_blocks in stage_blocks:
            self.encoder.append(
                self._make_stage(
                    channels,
                    channels * 2,
                    n_blocks,
                    cardinality,
                    stride=2,
                    times_in=times_in,
                    times_out=times_out
                )
            )
            channels *= 2
            if times_in is not None:
                times_in = times_out
                times_out = max(1, times_out // 2)
            skip_channels.append(channels)

        # Decoder: ``depth`` upsampling stages with skip connections, mirroring the
        # encoder so that each stage has the same number of blocks as the encoder
        # stage at the matching resolution (reversed order).
        self.upsamplers = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for stage, n_blocks in enumerate(reversed(stage_blocks)):
            self.upsamplers.append(
                nn.Conv2d(channels, channels // 2, 1, bias=False)
            )
            skip = skip_channels[depth - 1 - stage]
            self.decoder.append(
                self._make_stage(
                    channels // 2 + skip, channels // 2, n_blocks, cardinality
                )
            )
            channels //= 2

        self.quantiles = torch.tensor(quantiles)
        self.head = nn.Conv2d(channels, out_channels, 1)

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        n_blocks: int,
        cardinality: int,
        stride: int = 1,
        times_in: Optional[int] = None,
        times_out: Optional[int] = None
    ) -> nn.Sequential:
        """A (possibly strided) ResNeXt block followed by ``n_blocks - 1`` blocks."""
        blocks = [ResNeXtBlock(
            in_channels,
            out_channels,
            cardinality,
            stride=stride,
            times_in=times_in,
            times_out=times_out
        )]
        blocks += [
            ResNeXtBlock(
                out_channels,
                out_channels,
                cardinality,
                times_in=times_out,
                times_out=times_out
            )
            for _ in range(n_blocks - 1)
        ]
        return nn.Sequential(*blocks)

    def forward(
        self,
        inpt: Dict[str, torch.Tensor]
    ) -> torch.Tensor:

        times_in = self.times_in

        has_time = False
        x_geo = inpt["geo"]
        if x_geo.ndim == 5:
            T = x_geo.shape[2]
            x_geo = x_geo.permute((0, 2, 1, 3, 4)).flatten(0, 1)
            has_time = True

        x_mw = inpt["mw"]
        if x_mw.ndim == 5:
            x_mw = x_mw.permute((0, 2, 1, 3, 4)).flatten(0, 1)

        geo = channel_dropout(
            x_geo,
            0.1,
            training=self.training
        )
        geo = batch_dropout(
            geo,
            0.1,
            training=self.training
        )
        mw = channel_dropout(
            x_mw,
            0.1,
            training=self.training,
        )
        mw = batch_dropout(
            mw,
            0.1,
            training=self.training,
        )
        mw_feat = self.mw_stem(torch.nan_to_num(mw, nan=-1.5))
        geo_feat = self.geo_stem(torch.nan_to_num(geo, nan=-1.5))

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
        if has_time:
            skips: List[torch.Tensor] = [x[T - 1::T]]
        else:
            skips: List[torch.Tensor] = [x]

        for block in self.encoder:
            x = block(x)
            skips.append(x if not has_time else x[T - 1::T])
            if times_in is not None:
                T = max(T // 2, 1)

        if has_time:
            x = x[T - 1::T]

        # Decoder with skip connections (the last skip is the bottleneck itself).
        for stage, (upsample, block) in enumerate(zip(self.upsamplers, self.decoder)):
            skip = skips[-2 - stage]
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            x = upsample(x)
            x = block(torch.cat([x, skip], dim=1))

        return {
            "surface_precip": QuantileTensor(
                self.head(x),
                tau=self.quantiles.to(dtype=x.dtype, device=x.device)
            )
        }
