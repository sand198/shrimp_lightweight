# feathernetx_fixed.py
# Fixed lightweight architecture - maintains original parameters/GFLOPs

import math
from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Utilities
# ----------------------------

def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class DropPath(nn.Module):
    """Stochastic Depth per sample (when training)."""
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# ----------------------------
# Core building blocks (Lightweight)
# ----------------------------

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, g=1, act=nn.SiLU, bias=False):
        super().__init__()
        if p is None:
            p = (k - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = act() if act is not None else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class GhostModule(nn.Module):
    """
    Ghost module (lightweight pointwise feature generation).
    """
    def __init__(self, in_ch, out_ch, ratio=2, kernel_size=3, dw_act=True):
        super().__init__()
        init_channels = math.ceil(out_ch / ratio)
        new_channels = out_ch - init_channels
        self.primary = ConvBNAct(in_ch, init_channels, k=1, act=nn.SiLU)
        self.cheap = ConvBNAct(
            init_channels,
            new_channels,
            k=kernel_size,
            s=1,
            p=(kernel_size - 1) // 2,
            g=init_channels,
            act=(nn.SiLU if dw_act else None),
        )

    def forward(self, x):
        x1 = self.primary(x)
        x2 = self.cheap(x1)
        return torch.cat([x1, x2], dim=1)

class ECALite(nn.Module):
    """
    Efficient Channel Attention (fixed version)
    """
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(1, 2)
        y = self.conv(y)
        y = y.transpose(1, 2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y

class FeatherBlock(nn.Module):
    """
    Lightweight block with residual scaling to combat underfitting
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        expand_ratio: float = 3.0,
        kernel_size: int = 3,
        ghost_ratio: int = 2,
        eca_k: int = 3,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.stride = stride
        hidden = _make_divisible(in_ch * expand_ratio, 8)

        self.expand = GhostModule(in_ch, hidden, ratio=ghost_ratio, kernel_size=3, dw_act=False)
        self.dw = ConvBNAct(hidden, hidden, k=kernel_size, s=stride, g=hidden, act=nn.SiLU)
        self.project = GhostModule(hidden, out_ch, ratio=ghost_ratio, kernel_size=3, dw_act=False)
        self.eca = ECALite(out_ch, k_size=eca_k)

        self.has_res = (stride == 1 and in_ch == out_ch)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        # Learnable residual scaling to combat underfitting
        if self.has_res:
            self.res_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        out = self.expand(x)
        out = self.dw(out)
        out = self.project(out)
        out = self.eca(out)
        
        if self.has_res:
            out = x + self.res_scale * self.drop_path(out)
        
        return out

class CSPStage(nn.Module):
    """
    Cross Stage Partial stage
    """
    def __init__(self, in_ch, out_ch, num_blocks, stride, expand_ratio, drop_path_rate):
        super().__init__()
        self.down = ConvBNAct(in_ch, out_ch, k=3, s=stride, act=nn.SiLU)

        mid = out_ch // 2
        self.part1 = ConvBNAct(out_ch, mid, k=1, s=1, act=nn.SiLU)
        self.part2 = ConvBNAct(out_ch, mid, k=1, s=1, act=nn.SiLU)

        blocks = []
        for i in range(num_blocks):
            dp = drop_path_rate * float(i) / max(1, num_blocks - 1)
            blocks.append(FeatherBlock(mid, mid, stride=1, expand_ratio=expand_ratio, drop_path=dp))
        self.blocks = nn.Sequential(*blocks)

        self.fuse = ConvBNAct(mid * 2, out_ch, k=1, s=1, act=nn.SiLU)

    def forward(self, x):
        x = self.down(x)
        x1 = self.part1(x)
        x2 = self.part2(x)
        x2 = self.blocks(x2)
        out = torch.cat([x1, x2], dim=1)
        out = self.fuse(out)
        return out


# ----------------------------
# FeatherNetX (Fixed)
# ----------------------------

class FeatherNetX(nn.Module):
    """
    Lightweight CNN with underfitting fixes
    """
    def __init__(
        self,
        num_classes: int = 1000,
        stem_channels: int = 24,
        channels: Tuple[int, int, int, int] = (48, 96, 192, 256),
        layers: Tuple[int, int, int, int] = (2, 3, 5, 2),
        expand_ratio: float = 3.0,
        drop_path_rate: float = 0.0,
        width_mult: float = 1.0,
        norm_head: bool = True
    ):
        super().__init__()

        stem_channels = _make_divisible(stem_channels * width_mult, 8)
        channels = tuple(_make_divisible(c * width_mult, 8) for c in channels)

        # Stem
        self.stem = nn.Sequential(
            ConvBNAct(3, stem_channels, k=3, s=2, act=nn.SiLU),
            ConvBNAct(stem_channels, stem_channels, k=3, s=1, g=stem_channels, act=nn.SiLU),
            ConvBNAct(stem_channels, stem_channels, k=1, s=1, act=nn.SiLU),
        )

        # Stages
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]
        idx = 0
        self.stage1 = CSPStage(stem_channels, channels[0], layers[0], stride=2, 
                              expand_ratio=expand_ratio, drop_path_rate=dp_rates[idx] if layers[0] > 0 else 0.0)
        idx += layers[0]
        self.stage2 = CSPStage(channels[0], channels[1], layers[1], stride=2, 
                              expand_ratio=expand_ratio, drop_path_rate=dp_rates[idx] if layers[1] > 0 else 0.0)
        idx += layers[1]
        self.stage3 = CSPStage(channels[1], channels[2], layers[2], stride=2, 
                              expand_ratio=expand_ratio, drop_path_rate=dp_rates[idx] if layers[2] > 0 else 0.0)
        idx += layers[2]
        self.stage4 = CSPStage(channels[2], channels[3], layers[3], stride=2, 
                              expand_ratio=expand_ratio, drop_path_rate=dp_rates[idx] if layers[3] > 0 else 0.0)

        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.BatchNorm1d(channels[3]) if norm_head else nn.Identity(),
            nn.Linear(channels[3], num_classes, bias=True),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.head(x)
        return x


# ----------------------------
# Model factories
# ----------------------------

def feathernetx_tiny(num_classes=1000):
    """Lightweight version ~1M params"""
    return FeatherNetX(
        num_classes=num_classes,
        stem_channels=16,
        channels=(48, 96, 160, 224),
        layers=(2, 3, 4, 2),
        expand_ratio=3.0,
        drop_path_rate=0.0,
        width_mult=0.9,
    )


