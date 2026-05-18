import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def pair_logits_to_single_bit_logits(logits_BLD2: torch.Tensor) -> torch.Tensor:
    return logits_BLD2[..., 1] - logits_BLD2[..., 0]


def single_bit_logits_to_pair_logits(bit_logits: torch.Tensor) -> torch.Tensor:
    return torch.stack((-0.5 * bit_logits, 0.5 * bit_logits), dim=-1)


def split_flat_logits_to_bit_logits_per_scale(logits_BLV: torch.Tensor, scale_schedule: List[Tuple[int, int, int]]) -> List[torch.Tensor]:
    B, _, _ = logits_BLV.shape
    pair = logits_BLV.reshape(B, logits_BLV.shape[1], -1, 2)
    bit_logits = pair_logits_to_single_bit_logits(pair)
    out = []
    ptr = 0
    for pt, ph, pw in scale_schedule:
        cur_l = int(pt * ph * pw)
        cur = bit_logits[:, ptr:ptr + cur_l]
        if pt != 1:
            cur = cur.reshape(B, pt, ph, pw, -1).permute(0, 4, 1, 2, 3).reshape(B, -1, ph, pw)
        else:
            cur = cur.reshape(B, ph, pw, -1).permute(0, 3, 1, 2).contiguous()
        out.append(cur)
        ptr += cur_l
    return out


def compute_normalized_uncertainty(bit_logits_BDHW: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(bit_logits_BDHW)
    entropy = -(probs * torch.log(probs.clamp_min(eps)) + (1.0 - probs) * torch.log((1.0 - probs).clamp_min(eps)))
    entropy = entropy / math.log(2.0)
    return entropy.mean(dim=1, keepdim=True).clamp_(0.0, 1.0)


def compute_neighborhood_context(bit_probs_BDHW: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    pad = kernel_size // 2
    return F.avg_pool2d(bit_probs_BDHW, kernel_size=kernel_size, stride=1, padding=pad)


class DepthwiseSeparableResBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=max(1, min(8, channels // 8 or 1)), num_channels=channels)
        self.act = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dw(x)
        h = self.pw(h)
        h = self.norm(h)
        h = self.act(h)
        h = self.dropout(h)
        return x + h


class SameScaleRefinementHead(nn.Module):
    """A lightweight learned-gate probability calibrator for VAR bit logits.

    The module predicts a bounded residual in logit space and a soft gate that
    decides how much of that residual to apply. This keeps the sender/receiver
    protocol deterministic while avoiding hand-picked high-uncertainty masks.
    """

    def __init__(
        self,
        codebook_dim: int,
        text_dim: int = 4096,
        hidden_state_dim: int = 0,
        hidden_dim: int = 64,
        depth: int = 2,
        max_scales: int = 32,
        dropout: float = 0.0,
        neighborhood_kernel: int = 3,
        max_delta: float = 4.0,
        gate_bias_init: float = -2.0,
    ):
        super().__init__()
        self.codebook_dim = int(codebook_dim)
        self.hidden_dim = int(hidden_dim)
        self.neighborhood_kernel = int(neighborhood_kernel)
        self.max_delta = float(max_delta)
        self.gate_bias_init = float(gate_bias_init)

        # prefix, base logits, base probs, local mean, local variance, entropy,
        # and a scalar confidence margin.
        in_ch = self.codebook_dim * 5 + 2
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, kernel_size=1),
            nn.GroupNorm(num_groups=max(1, min(8, hidden_dim // 8 or 1)), num_channels=hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[DepthwiseSeparableResBlock(hidden_dim, dropout=dropout) for _ in range(max(1, depth))])
        self.out_proj = nn.Conv2d(hidden_dim, self.codebook_dim * 2, kernel_size=1)

        self.scale_embed = nn.Embedding(max_scales, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim) if text_dim and text_dim > 0 else None
        self.hidden_proj = nn.Linear(hidden_state_dim, hidden_dim) if hidden_state_dim and hidden_state_dim > 0 else None
        self._init_calibrator_output()

    def _init_calibrator_output(self):
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.out_proj.bias)
        with torch.no_grad():
            self.out_proj.bias[self.codebook_dim:].fill_(self.gate_bias_init)

    def forward(
        self,
        prefix_feat: torch.Tensor,
        bit_logits: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
        neighbor_context: Optional[torch.Tensor] = None,
        text_summary: Optional[torch.Tensor] = None,
        scale_hidden: Optional[torch.Tensor] = None,
        scale_id: int = 0,
        return_gate: bool = False,
    ) -> torch.Tensor:
        if uncertainty is None:
            uncertainty = compute_normalized_uncertainty(bit_logits)
        bit_probs = torch.sigmoid(bit_logits)
        if neighbor_context is None:
            neighbor_context = compute_neighborhood_context(bit_probs, kernel_size=self.neighborhood_kernel)
        local_var = compute_neighborhood_context((bit_probs - neighbor_context).pow(2), kernel_size=self.neighborhood_kernel)
        margin = torch.tanh(bit_logits.abs()).mean(dim=1, keepdim=True)
        x = torch.cat([prefix_feat, bit_logits, bit_probs, neighbor_context, local_var, uncertainty, margin], dim=1)
        h = self.in_proj(x)

        cond = self.scale_embed.weight[scale_id].unsqueeze(0).expand(h.shape[0], -1)
        if self.text_proj is not None and text_summary is not None:
            cond = cond + self.text_proj(text_summary)
        if self.hidden_proj is not None and scale_hidden is not None:
            cond = cond + self.hidden_proj(scale_hidden)
        h = h + cond[:, :, None, None]
        h = self.blocks(h)
        delta_raw, gate_logits = self.out_proj(h).split(self.codebook_dim, dim=1)
        bounded_delta = self.max_delta * torch.tanh(delta_raw)
        gate = torch.sigmoid(gate_logits)
        calibrated_delta = gate * bounded_delta
        if return_gate:
            return calibrated_delta, gate, bounded_delta
        return calibrated_delta
