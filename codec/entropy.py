"""Entropy / uncertainty utilities.

We compute an **entropy map** from model outputs and use it to decide which token
positions should be transmitted.

Supported output forms:

1) Categorical logits (softmax):
    logits_cat: Tensor[B, H, W, K]

2) Bitwise logits (sigmoid):
    logits_bit: Tensor[B, H, W, d]

Return:
    entropy: Tensor[B, H, W], measured in **bits**.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_EPS = 1e-12


def entropy_from_categorical_logits(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax entropy in bits.

    Args:
        logits: [B, H, W, K]
        dim: category dimension

    Returns:
        entropy: [B, H, W] in bits
    """
    probs = F.softmax(logits, dim=dim)
    logp = torch.log(probs.clamp_min(_EPS))
    ent_nats = -(probs * logp).sum(dim=dim)
    ent_bits = ent_nats / torch.log(torch.tensor(2.0, device=logits.device))
    return ent_bits


def entropy_from_bit_logits(bit_logits: torch.Tensor) -> torch.Tensor:
    """Bitwise entropy in bits.

    Args:
        bit_logits: [B, H, W, d], each entry is a logit for a Bernoulli bit.

    Returns:
        entropy: [B, H, W], sum of per-bit entropies (bits).
    """
    p = torch.sigmoid(bit_logits)
    # H(p) = -p log2 p - (1-p) log2 (1-p)
    p = p.clamp(_EPS, 1.0 - _EPS)
    h = -(p * torch.log2(p) + (1.0 - p) * torch.log2(1.0 - p))
    return h.sum(dim=-1)


def entropy_map_from_model_output(output: torch.Tensor, kind: str) -> torch.Tensor:
    """Unified helper.

    Args:
        output: logits tensor
        kind: 'categorical' or 'bitwise'

    Returns:
        entropy: [B, H, W]
    """
    kind = kind.lower()
    if kind in ("categorical", "softmax", "class"):
        return entropy_from_categorical_logits(output)
    if kind in ("bitwise", "bits", "bernoulli"):
        return entropy_from_bit_logits(output)
    raise ValueError(f"Unknown entropy kind: {kind}")
