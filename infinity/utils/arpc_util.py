"""Utility helpers for the ARPC (Autoregressive-based Progressive Coding) pipeline.

This repo is a research codebase and Infinity upstream does not natively expose
GM-BMSRQ / SRD hyper-parameters via the global CLI Args.

We keep all ARPC knobs in this small helper module so that:
  - the *default* Infinity training/inference is not affected;
  - ARPC scripts can opt-in by passing `--arpc_gm_bits default` etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


def arpc_default_active_bits(codebook_dim: int) -> List[int]:
    """Default three-group masking (GM-BMSRQ) from the paper.

    The paper reports channel dims {8,12,16} for the three groups (for c=16)
    and applies SRD from the 4th scale. We map this to "active bits".
    """
    if codebook_dim <= 0:
        raise ValueError(f"codebook_dim must be > 0, got {codebook_dim}")

    if codebook_dim == 16:
        return [8] * 4 + [12] * 5 + [16] * 4
    if codebook_dim == 32:
        return [16] * 4 + [24] * 5 + [32] * 4

    # fallback: no masking
    return [codebook_dim] * 13


def parse_active_bits_spec(spec: str, codebook_dim: int) -> Optional[List[int]]:
    """Parse a GM active-bits spec.

    Supported formats:
      - "" / "none" -> None
      - "default" -> paper default mapping for 16/32-dim BSQ
      - comma-separated list, e.g. "8,8,8,8,12,12,12,12,12,16,16,16,16"
      - run-length encoding, e.g. "8x4,12x5,16x4"
    """
    if spec is None:
        return None
    s = str(spec).strip().lower()
    if s in ("", "none", "null", "0"):
        return None
    if s in ("default", "paper", "arpc"):
        return arpc_default_active_bits(codebook_dim)

    out: List[int] = []
    parts = [p.strip() for p in s.split(',') if p.strip()]
    for p in parts:
        if 'x' in p:
            v, n = p.split('x', 1)
            out.extend([int(v)] * int(n))
        else:
            out.append(int(p))
    return out


@dataclass
class ARPCVAEKwargs:
    """Arguments forwarded into the VAE quantizer for ARPC."""

    gm_active_bits_per_scale: Optional[List[int]] = None
    srd_prob: float = 0.0
    srd_start_scale: int = 3
    srd_mode: str = "level"
