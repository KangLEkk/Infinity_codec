"""Progressive Entropy Codec.

This module implements the main logic requested:

- B2-style entropy-map transmission
- progressive coarse-to-fine reconstruction

Workflow
--------
compress(image, prompt):
  1) encode image -> multi-scale tokens t[0..S-1]
  2) write t[0] fully
  3) for s>=1:
        * compute model entropy map at scale s
        * select top-K positions
        * transmit those ground-truth tokens only

decompress(bitstream):
  1) receive t0
  2) for s>=1:
        * initialize unknown token map
        * fill known sparse tokens
        * ask model to sample remaining tokens
  3) decode tokens -> final image

Important
---------
- This code is **model-agnostic**, driven by `InfinityAdapter`.
- For real use with Infinity, implement an adapter that calls the official model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import torch
from PIL import Image

from .infinity_adapter import InfinityAdapter
from .entropy import entropy_map_from_model_output
from .selection import topk_from_entropy, gather_tokens
from .bitstream import StreamHeader, StreamPayload, ScaleMeta, write_bitstream, read_bitstream


def _pil_to_tensor(pil: Image.Image) -> torch.Tensor:
    arr = np.asarray(pil.convert("RGB")).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return t


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.detach().cpu().clamp(0, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


@dataclass
class CodecConfig:
    bits_per_token: int = 32
    # How many tokens to send per scale (scale0 will be ignored -> full send)
    tokens_per_scale: Optional[List[int]] = None
    seed: int = 0


class ProgressiveEntropyCodec:
    def __init__(self, adapter: InfinityAdapter, cfg: CodecConfig):
        self.adapter = adapter
        self.cfg = cfg

    @torch.no_grad()
    def compress(self, image: Image.Image, prompt: str, out_path: str, device: str = "cuda") -> None:
        """Encode image and write a progressive bitstream."""
        x = _pil_to_tensor(image).to(device)
        tokens = self.adapter.encode_image_to_tokens(x)
        num_scales = len(tokens)

        if self.cfg.tokens_per_scale is None:
            # default: send 5% tokens for each refine scale
            self.cfg.tokens_per_scale = [0] + [max(1, int(t.numel() * 0.05)) for t in tokens[1:]]
        if len(self.cfg.tokens_per_scale) != num_scales:
            raise ValueError(f"tokens_per_scale length mismatch: {len(self.cfg.tokens_per_scale)} vs {num_scales}")

        # header meta
        scales_meta: List[ScaleMeta] = []
        for s, t in enumerate(tokens):
            H, W = t.shape
            if s == 0:
                scales_meta.append(ScaleMeta(H=H, W=W, full=True, K=0))
            else:
                K = int(self.cfg.tokens_per_scale[s])
                scales_meta.append(ScaleMeta(H=H, W=W, full=False, K=K))

        header = StreamHeader(
            num_scales=num_scales,
            bits_per_token=int(self.cfg.bits_per_token),
            seed=int(self.cfg.seed),
            prompt=prompt,
            scales=scales_meta,
        )

        # payload build
        scale0_full = tokens[0].reshape(-1).detach().cpu().to(torch.long).numpy().astype(np.uint64)
        sparse_idx: List[np.ndarray] = []
        sparse_val: List[np.ndarray] = []

        # We simulate "decoder-side" entropy prediction using current known tokens.
        # For simplicity we always compute entropy against the GT token placeholders.
        known_tokens: List[torch.Tensor] = [tokens[0]]
        for s in range(1, num_scales):
            # placeholder map (unknown positions can be -1)
            H, W = tokens[s].shape
            placeholder = torch.full((H, W), -1, device=device, dtype=torch.long)
            known_tokens.append(placeholder)

            logits = self.adapter.predict_scale_logits(prompt=prompt, known_tokens=known_tokens, scale_idx=s, known_mask=None)
            ent = entropy_map_from_model_output(logits, kind=self.adapter.output_kind).squeeze(0)

            K = int(self.cfg.tokens_per_scale[s])
            sel = topk_from_entropy(ent, k=K)
            vals = gather_tokens(tokens[s], sel.flat_idx).detach().cpu().to(torch.long).numpy().astype(np.uint64)

            sparse_idx.append(sel.flat_idx.detach().cpu().numpy().astype(np.uint32))
            sparse_val.append(vals)

            # update known tokens with transmitted subset (helps later scales)
            placeholder_flat = known_tokens[s].reshape(-1)
            placeholder_flat[sel.flat_idx] = tokens[s].reshape(-1)[sel.flat_idx]
            known_tokens[s] = placeholder_flat.reshape(H, W)

        payload = StreamPayload(scale0_full=scale0_full, sparse_idx=sparse_idx, sparse_val=sparse_val)
        write_bitstream(out_path, header, payload)

    @torch.no_grad()
    def decompress(self, bitstream_path: str, out_image_path: str, device: str = "cuda") -> None:
        header, payload = read_bitstream(bitstream_path)
        prompt = header.prompt
        num_scales = header.num_scales

        # reconstruct known tokens list
        tokens: List[torch.Tensor] = []

        # scale0 full
        H0 = header.scales[0].H
        W0 = header.scales[0].W
        t0 = torch.from_numpy(payload.scale0_full.astype(np.int64)).reshape(H0, W0).to(device)
        tokens.append(t0)

        for s in range(1, num_scales):
            H = header.scales[s].H
            W = header.scales[s].W
            K = header.scales[s].K
            idx = torch.from_numpy(payload.sparse_idx[s - 1].astype(np.int64)).to(device)
            val = torch.from_numpy(payload.sparse_val[s - 1].astype(np.int64)).to(device)

            # placeholder unknown
            t = torch.full((H, W), -1, device=device, dtype=torch.long)
            flat = t.reshape(-1)
            flat[idx] = val
            t = flat.reshape(H, W)

            known_mask = t.ge(0)
            tokens.append(t)

            # Let model fill the remaining missing tokens at this scale
            sampled = self.adapter.sample_scale(prompt=prompt, known_tokens=tokens, scale_idx=s, known_mask=known_mask, seed=header.seed)
            tokens[s] = sampled

        img_t = self.adapter.decode_tokens_to_image(tokens)
        pil = _tensor_to_pil(img_t)
        pil.save(out_image_path)
