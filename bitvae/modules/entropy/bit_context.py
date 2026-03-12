"""Bit-level entropy model for BSQ codes.

This module provides a lightweight conditional masked CNN (PixelCNN-style) that
predicts per-bit logits for q in {0,1} at a given scale, conditioned on
previous-scale information that is already available at the decoder.

It is meant for training-time entropy estimation (rate term) and can also be
used for actual arithmetic coding if paired with a range coder.

Design goals (for fine-tuning the tokenizer):
- Provide differentiable rate proxy: R = -log2 p(q | context)
- Encourage less redundancy across tokens by penalizing predictable bits
- Allow weighting later (finer) scales more heavily to push information upward

Notes:
- We model each channel as an independent Bernoulli bit.
- Masking enforces raster-scan causality within the current scale.
- Conditioning is non-causal across scales (previous scales are assumed known).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class MaskedConv2d(nn.Module):
    """Masked convolution for raster-scan autoregressive modeling.

    mask_type:
      - 'A': excludes the center pixel (no access to current position)
      - 'B': includes the center pixel
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        mask_type: str = 'A',
        stride: int = 1,
        padding: Optional[int] = None,
        bias: bool = True,
    ):
        super().__init__()
        assert mask_type in ('A', 'B')
        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        kh, kw = self.conv.weight.shape[-2:]
        mask = torch.ones_like(self.conv.weight)
        yc, xc = kh // 2, kw // 2

        mask[:, :, yc + 1 :, :] = 0
        mask[:, :, yc, xc + 1 :] = 0
        if mask_type == 'A':
            mask[:, :, yc, xc] = 0

        self.register_buffer('mask', mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv.weight * self.mask
        return F.conv2d(
            x,
            w,
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.net(x)
        return identity + out


class ChannelNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.weight is not None:
            x = x * self.weight + self.bias
        return x


@dataclass
class BitwiseMaskedCNNConfig:
    bits_channels: int
    cond_channels: int
    hidden_channels: int = 192
    num_res_blocks: int = 6
    kernel_size: int = 5
    use_scale_embedding: bool = True
    max_scales: int = 32
    scale_emb_dim: int = 16

    # new
    fusion_mode: str = "gated_concat"   # add | concat | gated_concat | film
    cond_norm: bool = True
    cond_gate_init: float = 0.0


class BitwiseMaskedCNN(nn.Module):
    """Conditional masked CNN producing per-bit logits.

    Inputs:
      - y_bits: (B, D, H, W) float in [0,1] (or {-1,1} is also ok; will be mapped)
      - cond:   (B, Cc, H, W) float
      - scale_idx: optional int tensor of shape (B,) or scalar

    Output:
      - logits: (B, D, H, W) logits for bit=1
    """

    def __init__(self, cfg: BitwiseMaskedCNNConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.fusion_mode in ("add", "concat", "gated_concat", "film")

        in_ch = cfg.bits_channels
        if cfg.use_scale_embedding:
            self.scale_emb = nn.Embedding(cfg.max_scales, cfg.scale_emb_dim)
            cond_in = cfg.cond_channels + cfg.scale_emb_dim
        else:
            self.scale_emb = None
            cond_in = cfg.cond_channels

        self.bits_norm = ChannelNorm2d(in_ch) if cfg.cond_norm else nn.Identity()
        self.cond_norm = ChannelNorm2d(cond_in) if cfg.cond_norm else nn.Identity()

        self.in_conv = MaskedConv2d(
            in_ch, cfg.hidden_channels, kernel_size=cfg.kernel_size, mask_type='A', padding=cfg.kernel_size // 2
        )

        if cfg.fusion_mode == "add":
            self.cond_proj = nn.Conv2d(cond_in, cfg.hidden_channels, 1)
            self.fuse = None
            self.film_gamma = None
            self.film_beta = None
            self.cond_gate = None

        elif cfg.fusion_mode == "concat":
            self.cond_proj = nn.Conv2d(cond_in, cfg.hidden_channels, 1)
            self.fuse = nn.Sequential(
                nn.Conv2d(cfg.hidden_channels * 2, cfg.hidden_channels, 1),
                nn.ReLU(inplace=False),
            )
            self.film_gamma = None
            self.film_beta = None
            self.cond_gate = None

        elif cfg.fusion_mode == "gated_concat":
            self.cond_proj = nn.Conv2d(cond_in, cfg.hidden_channels, 1)
            self.fuse = nn.Sequential(
                nn.Conv2d(cfg.hidden_channels * 2, cfg.hidden_channels, 1),
                nn.ReLU(inplace=False),
            )
            self.cond_gate = nn.Parameter(
                torch.full((1, cfg.hidden_channels, 1, 1), float(cfg.cond_gate_init))
            )
            self.film_gamma = None
            self.film_beta = None

        else:  # film
            self.cond_proj = None
            self.fuse = None
            self.cond_gate = None
            self.film_gamma = nn.Conv2d(cond_in, cfg.hidden_channels, 1)
            self.film_beta = nn.Conv2d(cond_in, cfg.hidden_channels, 1)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(cfg.hidden_channels) for _ in range(cfg.num_res_blocks)]
        )

        self.out_net = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(cfg.hidden_channels, cfg.hidden_channels, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(cfg.hidden_channels, cfg.bits_channels, 1),
        )

    def _prepare_cond(
        self,
        y_bits: torch.Tensor,
        cond: Optional[torch.Tensor],
        scale_idx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if cond is None:
            cond = torch.zeros(
                (y_bits.shape[0], self.cfg.cond_channels, y_bits.shape[2], y_bits.shape[3]),
                device=y_bits.device,
                dtype=y_bits.dtype,
            )

        if self.scale_emb is not None:
            if scale_idx is None:
                scale_idx = torch.zeros((y_bits.shape[0],), device=y_bits.device, dtype=torch.long)
            elif scale_idx.ndim == 0:
                scale_idx = scale_idx.expand(y_bits.shape[0])
            scale_idx = scale_idx.clamp(0, self.cfg.max_scales - 1)
            emb = self.scale_emb(scale_idx).to(dtype=cond.dtype)
            emb = emb[:, :, None, None].expand(-1, -1, cond.shape[2], cond.shape[3])
            cond = torch.cat([cond, emb], dim=1)

        cond = self.cond_norm(cond)
        return cond

    def forward(
        self,
        y_bits: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        scale_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert y_bits.ndim == 4, f"y_bits must be (B,D,H,W), got {y_bits.shape}"

        if y_bits.min() < 0:
            y_bits = (y_bits + 1.0) * 0.5

        y_bits = self.bits_norm(y_bits)
        h = self.in_conv(y_bits)

        cond = self._prepare_cond(y_bits, cond, scale_idx)

        if self.cfg.fusion_mode == "add":
            h = h + self.cond_proj(cond)

        elif self.cfg.fusion_mode == "concat":
            c = self.cond_proj(cond)
            h = self.fuse(torch.cat([h, c], dim=1))

        elif self.cfg.fusion_mode == "gated_concat":
            c = self.cond_proj(cond)
            g = torch.sigmoid(self.cond_gate)
            h = self.fuse(torch.cat([h, g * c], dim=1))

        else:  # film
            gamma = self.film_gamma(cond)
            beta = self.film_beta(cond)
            h = h * (1.0 + torch.tanh(gamma)) + beta

        h = self.res_blocks(h)
        logits = self.out_net(h)
        return logits


def _pick_gn_groups(channels: int, max_groups: int = 8) -> int:
    g = max(1, min(max_groups, channels))
    while channels % g != 0 and g > 1:
        g -= 1
    return g


def _build_2d_sincos_pos_embed(h: int, w: int, dim: int, device, dtype):
    if dim % 4 != 0:
        raise ValueError(f"pos dim must be divisible by 4, got {dim}")
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=torch.float32),
        indexing='ij'
    )
    omega = torch.arange(dim // 4, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(1, dim // 4 - 1)))

    y = y.reshape(-1, 1) * omega.reshape(1, -1)
    x = x.reshape(-1, 1) * omega.reshape(1, -1)

    pe = torch.cat([torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)], dim=1)
    return pe.to(dtype=dtype)  # (H*W, dim)


class PriorTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(round(dim * mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


@dataclass
class OneShotScaleCausalPriorConfig:
    bits_channels: int
    cond_channels: int
    model_dim: int = 256
    depth: int = 4
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    max_scales: int = 32
    use_scale_embedding: bool = True
    use_pos2d: bool = True
    checkpoint_blocks: bool = False
    cond_norm: bool = True


class OneShotScaleCausalPrior(nn.Module):
    """
    One-shot all-scale prior:
      - input: list of cond maps [F_{0}, F_{1}, ..., F_{K-1}] resized to each current scale
      - each scale -> tokens
      - concat all scales into one sequence
      - apply scale-causal attention mask (scale i can see <= i)
      - output logits for all scales in one forward
    """

    def __init__(self, cfg: OneShotScaleCausalPriorConfig):
        super().__init__()
        self.cfg = cfg
        self.cond_norm = cfg.cond_norm
        self.cond_gain = nn.Parameter(torch.tensor(1.0))

        gn = _pick_gn_groups(cfg.model_dim, 8)
        self.in_proj = nn.Sequential(
            nn.Conv2d(cfg.cond_channels, cfg.model_dim, kernel_size=3, padding=1),
            nn.GroupNorm(gn, cfg.model_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(cfg.model_dim, cfg.model_dim, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )

        self.scale_emb = nn.Embedding(cfg.max_scales, cfg.model_dim) if cfg.use_scale_embedding else None

        self.blocks = nn.ModuleList([
            PriorTransformerBlock(
                dim=cfg.model_dim,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                dropout=cfg.dropout
            )
            for _ in range(cfg.depth)
        ])

        self.norm = nn.LayerNorm(cfg.model_dim)
        self.head = nn.Linear(cfg.model_dim, cfg.bits_channels)

        # cache bool masks on CPU by lengths tuple
        self._mask_cache: Dict[Tuple[int, ...], torch.Tensor] = {}

    def _get_attn_mask(self, lengths: List[int], device) -> torch.Tensor:
        key = tuple(int(x) for x in lengths)
        if key not in self._mask_cache:
            lvl = []
            for si, L in enumerate(lengths):
                lvl.append(torch.full((L,), si, dtype=torch.long))
            lvl = torch.cat(lvl, dim=0)  # (L_total,)
            q = lvl[:, None]
            k = lvl[None, :]
            # True means masked / not allowed
            mask = (q < k)  # future scales are masked
            self._mask_cache[key] = mask.cpu()
        return self._mask_cache[key].to(device=device)

    def forward(
        self,
        cond_maps: List[torch.Tensor],
        scale_ids: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        if len(cond_maps) == 0:
            return []

        if scale_ids is None:
            scale_ids = list(range(len(cond_maps)))

        B = cond_maps[0].shape[0]
        tokens = []
        lengths = []
        metas = []

        for li, cond in enumerate(cond_maps):
            assert cond.ndim == 4, f"cond must be (B,C,H,W), got {cond.shape}"

            if self.cond_norm:
                rms = cond.pow(2).mean(dim=1, keepdim=True).add(1e-6).sqrt()
                cond = cond / rms
            cond = cond * self.cond_gain.to(dtype=cond.dtype)

            feat = self.in_proj(cond)  # (B,C,H,W)
            _, C, H, W = feat.shape

            tok = feat.flatten(2).transpose(1, 2).contiguous()  # (B, H*W, C)

            if self.cfg.use_pos2d:
                pos = _build_2d_sincos_pos_embed(H, W, C, tok.device, tok.dtype)  # (L,C)
                tok = tok + pos.unsqueeze(0)

            if self.scale_emb is not None:
                sid = int(scale_ids[li])
                sid = max(0, min(self.cfg.max_scales - 1, sid))
                tok = tok + self.scale_emb.weight[sid].to(dtype=tok.dtype).view(1, 1, C)

            tokens.append(tok)
            lengths.append(H * W)
            metas.append((H, W))

        x = torch.cat(tokens, dim=1)  # (B, L_total, C)
        attn_mask = self._get_attn_mask(lengths, device=x.device)  # (L_total, L_total), bool

        for blk in self.blocks:
            if self.cfg.checkpoint_blocks and self.training:
                x = checkpoint(lambda inp, mask: blk(inp, mask), x, attn_mask, use_reentrant=False)
            else:
                x = blk(x, attn_mask)

        x = self.norm(x)
        logits_all = self.head(x)  # (B, L_total, D)

        outs = []
        ptr = 0
        D = self.cfg.bits_channels
        for L, (H, W) in zip(lengths, metas):
            cur = logits_all[:, ptr:ptr + L, :]  # (B,L,D)
            cur = cur.transpose(1, 2).contiguous().view(B, D, H, W)  # (B,D,H,W)
            outs.append(cur)
            ptr += L

        return outs

def bernoulli_nll_bits_from_logits(logits: torch.Tensor, target_bits: torch.Tensor, reduce: str = 'sum') -> torch.Tensor:
    """Return -log2 p(target_bits | logits) (bits), where p = sigmoid(logits)."""
    nll_nats = F.binary_cross_entropy_with_logits(logits, target_bits, reduction='none')
    nll_bits = nll_nats / torch.log(torch.tensor(2.0, device=logits.device, dtype=logits.dtype))
    if reduce == 'sum':
        return nll_bits.sum()
    if reduce == 'mean':
        return nll_bits.mean()
    return nll_bits