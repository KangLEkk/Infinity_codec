"""Entropy-map based sparse token selection.

Given an entropy map H(x,y), we select the most uncertain locations.
This is the typical "spend bits where the prior is weak" policy.

We provide:
- topk indices selection (deterministic)
- fixed ratio selection

All selections operate on a single image (no batch) for simplicity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class SelectionResult:
    # Flattened indices in row-major order: idx = y * W + x
    flat_idx: torch.Tensor  # [K], int64
    H: int
    W: int
    # Optional: entropy values at selected positions
    scores: torch.Tensor  # [K], float


def topk_from_entropy(entropy_map: torch.Tensor, k: int) -> SelectionResult:
    """Select top-k positions.

    Args:
        entropy_map: [H, W] or [1, H, W]
        k: number of positions

    Returns:
        SelectionResult
    """
    if entropy_map.dim() == 3:
        entropy_map = entropy_map.squeeze(0)
    assert entropy_map.dim() == 2, f"expected [H,W], got {entropy_map.shape}"

    H, W = entropy_map.shape
    k = int(min(max(k, 0), H * W))
    flat = entropy_map.reshape(-1)
    if k == 0:
        idx = torch.zeros((0,), dtype=torch.long, device=entropy_map.device)
        scores = torch.zeros((0,), dtype=entropy_map.dtype, device=entropy_map.device)
        return SelectionResult(flat_idx=idx, H=H, W=W, scores=scores)

    scores, idx = torch.topk(flat, k=k, largest=True, sorted=True)
    return SelectionResult(flat_idx=idx.to(torch.long), H=H, W=W, scores=scores)


def ratio_from_entropy(entropy_map: torch.Tensor, ratio: float) -> SelectionResult:
    """Select a fixed ratio of locations by entropy ranking."""
    if entropy_map.dim() == 3:
        entropy_map = entropy_map.squeeze(0)
    H, W = entropy_map.shape
    k = int(round(H * W * float(ratio)))
    return topk_from_entropy(entropy_map, k)


def gather_tokens(token_map: torch.Tensor, flat_idx: torch.Tensor) -> torch.Tensor:
    """Gather token values at selected positions.

    Args:
        token_map: [H, W] (int)
        flat_idx: [K]

    Returns:
        values: [K]
    """
    assert token_map.dim() == 2
    flat = token_map.reshape(-1)
    return flat.index_select(0, flat_idx)


def scatter_tokens(base: torch.Tensor, flat_idx: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Scatter sparse values into a base token map.

    Args:
        base: [H,W] token map to be updated (copied internally)
        flat_idx: [K]
        values: [K]

    Returns:
        updated: [H,W]
    """
    out = base.clone()
    flat = out.reshape(-1)
    flat[flat_idx] = values
    return out
