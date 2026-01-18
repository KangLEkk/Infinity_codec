"""Infinity adapter layer.

Why this exists
---------------
Infinity is a full text-to-image system (tokenizer + multi-scale VAR transformer).
This repo focuses on **progressive compression logic**, so we keep a thin adapter
that abstracts Infinity's internals.

You should be able to drop-in any Infinity checkpoint or model variant as long as
you can provide the following primitives:

- encode_image_to_tokens(image)  -> List[LongTensor[Hs, Ws]]  (coarse to fine)
- decode_tokens_to_image(tokens) -> Tensor[3, H, W] in [0,1] (or PIL)
- predict_scale_logits(prompt, known_tokens, scale_idx, known_mask) -> logits
- sample_scale(prompt, known_tokens, scale_idx, known_mask, seed) -> token_map

This file provides:
- `InfinityAdapter` : abstract base class
- `DummyInfinityAdapter`: a toy adapter you can run on CPU for sanity checks

To connect to the official repo:
- clone https://github.com/FoundationVision/Infinity
- `pip install -r requirements.txt` in their repo
- `pip install -e .` so that `import infinity` works
- implement a small subclass that calls their `predict.py` or model forward.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import torch


class InfinityAdapter(ABC):
    """Abstract interface needed by the progressive codec."""

    @abstractmethod
    def encode_image_to_tokens(self, image: torch.Tensor) -> List[torch.Tensor]:
        """Encode image into a list of 2D token maps (coarse→fine)."""

    @abstractmethod
    def decode_tokens_to_image(self, tokens: List[torch.Tensor]) -> torch.Tensor:
        """Decode token maps back to image tensor [3,H,W] in [0,1]."""

    @abstractmethod
    def predict_scale_logits(
        self,
        prompt: str,
        known_tokens: List[torch.Tensor],
        scale_idx: int,
        known_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict logits for a given scale.

        Args:
            prompt: text condition
            known_tokens: tokens for scales <= scale_idx (some positions may be placeholders)
            scale_idx: which scale to predict
            known_mask: [H,W] bool mask where tokens at this scale are already known

        Returns:
            logits: either [1,H,W,K] categorical or [1,H,W,d] bitwise
        """

    @abstractmethod
    def sample_scale(
        self,
        prompt: str,
        known_tokens: List[torch.Tensor],
        scale_idx: int,
        known_mask: Optional[torch.Tensor],
        seed: int,
    ) -> torch.Tensor:
        """Sample / decode the missing tokens at a given scale."""

    @property
    @abstractmethod
    def output_kind(self) -> str:
        """'categorical' or 'bitwise' for entropy computation."""


class DummyInfinityAdapter(InfinityAdapter):
    """CPU-friendly dummy adapter for development/testing.

    - Tokens: random uint16 maps with 2 scales
    - Predict logits: simple noisy logits; uncertainty higher around edges
    - Sampling: fill unknown with argmax

    This is NOT a real compressor; it just makes the pipeline runnable.
    """

    def __init__(self, image_hw: Tuple[int, int] = (256, 256), scales: int = 2, vocab_size: int = 1024):
        self._H, self._W = image_hw
        self._scales = scales
        self._vocab = vocab_size

    @property
    def output_kind(self) -> str:
        return "categorical"

    def encode_image_to_tokens(self, image: torch.Tensor) -> List[torch.Tensor]:
        # pretend we have two scales: 1/16 and 1/8 resolution token maps
        device = image.device
        tokens: List[torch.Tensor] = []
        for s in range(self._scales):
            hs = self._H // (16 // (2**s))
            ws = self._W // (16 // (2**s))
            t = torch.randint(0, self._vocab, (hs, ws), device=device, dtype=torch.long)
            tokens.append(t)
        return tokens

    def decode_tokens_to_image(self, tokens: List[torch.Tensor]) -> torch.Tensor:
        # dummy decode: upscale coarse tokens to RGB pattern
        t0 = tokens[0].float()
        img = (t0 / float(self._vocab)).clamp(0, 1)
        img = img.unsqueeze(0).repeat(3, 1, 1)
        img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(self._H, self._W), mode="nearest")
        return img.squeeze(0)

    def predict_scale_logits(
        self,
        prompt: str,
        known_tokens: List[torch.Tensor],
        scale_idx: int,
        known_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # output logits [1,H,W,K]
        t = known_tokens[scale_idx]
        H, W = t.shape
        logits = torch.randn((1, H, W, self._vocab), device=t.device) * 0.5
        # make center more confident (lower entropy), edges less confident
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=t.device),
            torch.linspace(-1, 1, W, device=t.device),
            indexing="ij",
        )
        rad = (xx**2 + yy**2).sqrt()
        conf = (1.2 - rad).clamp(0.2, 1.2)  # center higher
        logits = logits * conf.view(1, H, W, 1)
        return logits

    def sample_scale(
        self,
        prompt: str,
        known_tokens: List[torch.Tensor],
        scale_idx: int,
        known_mask: Optional[torch.Tensor],
        seed: int,
    ) -> torch.Tensor:
        logits = self.predict_scale_logits(prompt, known_tokens, scale_idx, known_mask)
        pred = logits.argmax(dim=-1).squeeze(0)  # [H,W]
        if known_mask is None:
            return pred
        out = pred
        out = out.clone()
        out[known_mask.bool()] = known_tokens[scale_idx][known_mask.bool()]
        return out
