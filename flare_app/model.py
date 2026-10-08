"""Train a ConvNeXt-T classifier with wavelet convolution and wavelet pooling.

The network keeps the ConvNeXt-T stage widths/depths [96,192,384,768] and
[3,3,9,3]. Standard 7x7 depthwise convolution is replaced by multi-level Haar
WTConv, and every spatial downsampling operation uses a Haar wavelet filter bank
followed by learnable sub-band mixing.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x: torch.Tensor, probability: float, training: bool) -> torch.Tensor:
    if probability == 0.0 or not training:
        return x
    keep = 1.0 - probability
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
    return x.div(keep) * random_tensor.floor()


def haar_filters() -> torch.Tensor:
    # Orthonormal analysis/synthesis bank: LL, LH, HL, HH.
    return torch.tensor([
        [[1.0, 1.0], [1.0, 1.0]],
        [[-1.0, -1.0], [1.0, 1.0]],
        [[-1.0, 1.0], [-1.0, 1.0]],
        [[1.0, -1.0], [-1.0, 1.0]],
    ], dtype=torch.float32) / 2.0


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.probability, self.training)


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).square().mean(1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class HaarDWT2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        weight = haar_filters()[:, None, :, :].repeat(channels, 1, 1, 1)
        self.register_buffer("weight", weight, persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        original = (x.shape[-2], x.shape[-1])
        pad_h, pad_w = original[0] % 2, original[1] % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        coefficients = F.conv2d(x, self.weight.to(dtype=x.dtype), stride=2, groups=self.channels)
        b, _, h, w = coefficients.shape
        return coefficients.view(b, self.channels, 4, h, w), original


class HaarIDWT2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        weight = haar_filters()[:, None, :, :].repeat(channels, 1, 1, 1)
        self.register_buffer("weight", weight, persistent=False)

    def forward(self, coefficients: torch.Tensor,
                output_shape: tuple[int, int]) -> torch.Tensor:
        b, c, bands, h, w = coefficients.shape
        if c != self.channels or bands != 4:
            raise ValueError("Invalid Haar coefficient tensor.")
        flat = coefficients.reshape(b, 4 * c, h, w)
        result = F.conv_transpose2d(flat, self.weight.to(dtype=flat.dtype), stride=2,
                                    groups=self.channels)
        return result[:, :, :output_shape[0], :output_shape[1]]


class ChannelScale(nn.Module):
    def __init__(self, channels: int, initial: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), initial))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class WaveletDepthwiseConv(nn.Module):
    """Multi-level WTConv-style depthwise operator with Haar reconstruction."""

    def __init__(self, channels: int, levels: int = 2, kernel_size: int = 5) -> None:
        super().__init__()
        if levels < 1:
            raise ValueError("Wavelet levels must be positive.")
        self.levels = levels
        self.dwt = HaarDWT2d(channels)
        self.idwt = HaarIDWT2d(channels)
        self.base_conv = nn.Conv2d(channels, channels, 7, padding=3,
                                   groups=channels, bias=True)
        self.base_scale = ChannelScale(channels, 1.0)
        self.wavelet_convs = nn.ModuleList([
            nn.Conv2d(4 * channels, 4 * channels, kernel_size,
                      padding=kernel_size // 2, groups=4 * channels, bias=False)
            for _ in range(levels)
        ])
        self.wavelet_scales = nn.ModuleList([
            ChannelScale(4 * channels, 0.1) for _ in range(levels)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current = x
        processed_low, processed_high, shapes = [], [], []
        for conv, scale in zip(self.wavelet_convs, self.wavelet_scales):
            coefficients, shape = self.dwt(current)
            b, c, _, h, w = coefficients.shape
            filtered = scale(conv(coefficients.reshape(b, 4 * c, h, w)))
            filtered = filtered.view(b, c, 4, h, w)
            processed_low.append(filtered[:, :, 0])
            processed_high.append(filtered[:, :, 1:])
            shapes.append(shape)
            current = coefficients[:, :, 0]

        reconstructed: Optional[torch.Tensor] = None
        for low, high, shape in zip(reversed(processed_low), reversed(processed_high), reversed(shapes)):
            if reconstructed is not None:
                low = low + reconstructed
            coefficients = torch.cat([low.unsqueeze(2), high], dim=2)
            reconstructed = self.idwt(coefficients, shape)
        if reconstructed is None:
            raise RuntimeError("Wavelet reconstruction received no levels.")
        return self.base_scale(self.base_conv(x)) + reconstructed


class WaveletPool(nn.Module):
    """2x downsampling via all four Haar sub-bands and learnable mixing."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.dwt = HaarDWT2d(in_channels)
        initial = torch.tensor([1.0, 0.5, 0.5, 0.25], dtype=torch.float32)
        self.band_scale = nn.Parameter(initial.view(1, 1, 4, 1, 1))
        self.proj = nn.Conv2d(4 * in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coefficients, _ = self.dwt(x)
        coefficients = coefficients * self.band_scale
        b, c, _, h, w = coefficients.shape
        return self.proj(coefficients.reshape(b, 4 * c, h, w))


class WaveletStem(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        hidden = out_channels // 2
        self.pool1 = WaveletPool(3, hidden)
        self.act = nn.GELU()
        self.norm1 = LayerNorm2d(hidden)
        self.pool2 = WaveletPool(hidden, out_channels)
        self.norm2 = LayerNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(self.act(self.pool1(x)))
        return self.norm2(self.pool2(x))


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, wavelet_levels: int, drop_probability: float,
                 layer_scale_init: float = 1e-6) -> None:
        super().__init__()
        self.dwconv = WaveletDepthwiseConv(dim, levels=wavelet_levels)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.drop_path = DropPath(drop_probability) if drop_probability > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv2(self.act(self.pwconv1(x)))
        x = x * self.gamma
        x = x.permute(0, 3, 1, 2)
        return residual + self.drop_path(x)


class WaveletConvNeXtTiny(nn.Module):
    def __init__(self, num_classes: int, wavelet_levels: int = 2,
                 drop_path_rate: float = 0.1) -> None:
        super().__init__()
        depths = (3, 3, 9, 3)
        dims = (96, 192, 384, 768)
        self.downsample_layers = nn.ModuleList([WaveletStem(dims[0])])
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm2d(dims[i]), WaveletPool(dims[i], dims[i + 1])
            ))
        rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        offset = 0
        self.stages = nn.ModuleList()
        for stage, (depth, dim) in enumerate(zip(depths, dims)):
            blocks = [ConvNeXtBlock(dim, wavelet_levels, rates[offset + j])
                      for j in range(depth)]
            self.stages.append(nn.Sequential(*blocks))
            offset += depth
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for downsample, stage in zip(self.downsample_layers, self.stages):
            x = stage(downsample(x))
        return self.norm(x.mean(dim=(-2, -1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))
