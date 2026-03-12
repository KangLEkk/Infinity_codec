from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


TextCondTuple = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]


def masked_mean_pool_text(text_cond_tuple: TextCondTuple) -> torch.Tensor:
    kv_compact, lens, cu_seqlens_k, _ = text_cond_tuple
    if lens.numel() == 0:
        return kv_compact.new_zeros((0, kv_compact.shape[-1]))

    pooled = []
    for i in range(lens.numel()):
        st = int(cu_seqlens_k[i].item())
        ed = int(cu_seqlens_k[i + 1].item())
        if ed <= st:
            pooled.append(kv_compact.new_zeros((kv_compact.shape[-1],)))
        else:
            pooled.append(kv_compact[st:ed].mean(dim=0))
    return torch.stack(pooled, dim=0)


class FiLMResBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.cond = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, channels * 4),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        g1, b1, g2, b2 = self.cond(cond).chunk(4, dim=-1)
        h = self.norm1(x)
        h = h * (1.0 + g1[:, :, None, None]) + b1[:, :, None, None]
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = h * (1.0 + g2[:, :, None, None]) + b2[:, :, None, None]
        h = F.silu(h)
        h = self.conv2(h)
        return x + h


class TinyEntropyStudent(nn.Module):
    """
    Lightweight scale-conditional entropy model.

    Input:
      prefix_map: [B, D, H, W]   previous cumulative latent, downsampled to current scale
      text_cond_tuple: compact text KV tuple used by Infinity
      scale_id: int or LongTensor [B]

    Output:
      bit logits [B, D, H, W]
    """
    def __init__(
        self,
        codebook_dim: int,
        text_dim: int = 2048,
        hidden_dim: int = 256,
        depth: int = 4,
        max_scales: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.codebook_dim = int(codebook_dim)
        self.text_dim = int(text_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_scales = int(max_scales)

        self.in_proj = nn.Conv2d(self.codebook_dim, self.hidden_dim, kernel_size=3, padding=1)
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.scale_emb = nn.Embedding(self.max_scales, self.hidden_dim)
        self.blocks = nn.ModuleList([FiLMResBlock(self.hidden_dim, self.hidden_dim) for _ in range(depth)])
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.GroupNorm(8, self.hidden_dim)
        self.out = nn.Conv2d(self.hidden_dim, self.codebook_dim, kernel_size=1)

    def forward(
        self,
        prefix_map: torch.Tensor,
        text_cond_tuple: TextCondTuple,
        scale_id: int | torch.Tensor,
    ) -> torch.Tensor:
        assert prefix_map.ndim == 4, f"prefix_map must be BCHW, got {prefix_map.shape}"
        B = prefix_map.shape[0]
        if isinstance(scale_id, int):
            scale_id = torch.full((B,), scale_id, device=prefix_map.device, dtype=torch.long)
        elif scale_id.ndim == 0:
            scale_id = scale_id.expand(B).long().to(prefix_map.device)
        else:
            scale_id = scale_id.long().to(prefix_map.device)

        pooled_text = masked_mean_pool_text(text_cond_tuple)
        cond = self.text_proj(pooled_text) + self.scale_emb(scale_id.clamp_(0, self.max_scales - 1))

        h = self.in_proj(prefix_map)
        for blk in self.blocks:
            h = blk(h, cond)
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.dropout(h)
        return self.out(h)


def teacher_logits_to_bit_logits_per_scale(
    logits_BLV: torch.Tensor,
    scale_schedule: Sequence[Tuple[int, int, int]],
    codebook_dim: int,
) -> List[torch.Tensor]:
    """
    Converts Infinity bit-label logits [B, L, 2*D] into a list of [B, D, H, W] bit logits.
    """
    B, L, V = logits_BLV.shape
    assert V == codebook_dim * 2, f"Expected V=2*D={codebook_dim * 2}, got {V}"
    logits = logits_BLV.view(B, L, codebook_dim, 2)
    bit_logits = logits[..., 1] - logits[..., 0]  # [B, L, D]

    out: List[torch.Tensor] = []
    ptr = 0
    for (pt, ph, pw) in scale_schedule:
        if pt != 1:
            raise NotImplementedError("TinyEntropyStudent currently assumes image scales with pt=1.")
        cur = ph * pw
        z = bit_logits[:, ptr:ptr + cur]  # [B, HW, D]
        z = z.view(B, ph, pw, codebook_dim).permute(0, 3, 1, 2).contiguous()  # [B, D, H, W]
        out.append(z)
        ptr += cur
    assert ptr == L, f"Split mismatch: consumed {ptr}, total {L}"
    return out


def bits_list_to_bdhw(all_bit_indices: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    out = []
    for bits in all_bit_indices:
        if bits is None:
            continue
        assert bits.ndim == 5, f"Expected [B,T,H,W,D], got {bits.shape}"
        B, T, H, W, D = bits.shape
        if T != 1:
            raise NotImplementedError("TinyEntropyStudent currently assumes image scales with T=1.")
        out.append(bits[:, 0].permute(0, 3, 1, 2).contiguous().float())
    return out
