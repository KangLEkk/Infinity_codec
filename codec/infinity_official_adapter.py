"""Template adapter for the official FoundationVision/Infinity repo.

Because this sandbox cannot vendor the whole Infinity codebase, this file gives
a *drop-in* adapter skeleton.

How to use
----------
1) Clone Infinity in your environment:
   git clone https://github.com/FoundationVision/Infinity.git
   cd Infinity
   pip install -r requirements.txt
   pip install -e .

2) Download checkpoints mentioned in the official README.

3) Implement the TODO blocks below using Infinity's tokenizer/model classes.

After that, replace `DummyInfinityAdapter` in scripts/ with `InfinityOfficialAdapter`.

Notes
-----
- Infinity uses **bitwise classifier (IVC)** and a **multi-scale residual tokenizer**.
- For entropy-map coding, you want per-position uncertainty:
    entropy(x,y) = sum_{b=1..d} H( sigmoid(logit_b) )
  if the model outputs bit logits.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from .infinity_adapter import InfinityAdapter


class InfinityOfficialAdapter(InfinityAdapter):
    def __init__(self, vae_ckpt: str, model_ckpt: str, device: str = "cuda"):
        self.device = device

        # TODO: import and load tokenizer / vae
        # from infinity.xxx import ...
        # self.tokenizer = ...

        # TODO: import and load transformer model
        # self.model = ...

        self._output_kind = "bitwise"  # likely bitwise for Infinity

    @property
    def output_kind(self) -> str:
        return self._output_kind

    def encode_image_to_tokens(self, image: torch.Tensor) -> List[torch.Tensor]:
        """Return a list of 2D token maps (coarse->fine)."""
        # TODO: tokenizer.encode(image) or similar
        raise NotImplementedError

    def decode_tokens_to_image(self, tokens: List[torch.Tensor]) -> torch.Tensor:
        """Decode token maps back to RGB image in [0,1]."""
        # TODO: tokenizer.decode(tokens)
        raise NotImplementedError

    def predict_scale_logits(
        self,
        prompt: str,
        known_tokens: List[torch.Tensor],
        scale_idx: int,
        known_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return logits for token prediction at this scale.

        For bitwise:
          logits shape should be [1, H, W, d]
        For categorical:
          logits shape should be [1, H, W, K]

        """
        # TODO: call Infinity model forward to get logits at scale_idx.
        # - You likely need prompt embeddings (Flan-T5-XL per README)
        # - You need to represent known_mask/known_tokens so the model knows fixed positions.
        raise NotImplementedError

    def sample_scale(
        self,
        prompt: str,
        known_tokens: List[torch.Tensor],
        scale_idx: int,
        known_mask: Optional[torch.Tensor],
        seed: int,
    ) -> torch.Tensor:
        """Fill missing tokens at this scale.

        Minimal strategy:
          - run predict_scale_logits(...)
          - take argmax (categorical) or threshold bits (bitwise)
          - overwrite known positions with transmitted values

        Stronger strategy:
          - run iterative masked refinement (MaskGIT-style)
          - or Infinity's own sampling loop
        """
        raise NotImplementedError
