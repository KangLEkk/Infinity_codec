#!/usr/bin/env python
"""Save intermediate reconstructions after each transmitted layer.

This is useful to *visually verify* the progressive behavior.

Example:
  python scripts/progressive_decode.py \
    --stream out.stream \
    --out-dir out_frames
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from PIL import Image

from codec.bitstream import read_bitstream
from codec.infinity_adapter import DummyInfinityAdapter
from codec.codec import _tensor_to_pil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    header, payload = read_bitstream(args.stream)

    # TODO: Replace with real Infinity adapter.
    adapter = DummyInfinityAdapter(image_hw=(256, 256), scales=header.num_scales, vocab_size=2048)

    # scale0
    H0, W0 = header.scales[0].H, header.scales[0].W
    t0 = torch.from_numpy(payload.scale0_full.astype(np.int64)).reshape(H0, W0).to(args.device)

    tokens = [t0]

    # save base reconstruction
    img0 = adapter.decode_tokens_to_image(tokens)
    _tensor_to_pil(img0).save(os.path.join(args.out_dir, "stage0.png"))

    for s in range(1, header.num_scales):
        H, W = header.scales[s].H, header.scales[s].W
        idx = torch.from_numpy(payload.sparse_idx[s - 1].astype(np.int64)).to(args.device)
        val = torch.from_numpy(payload.sparse_val[s - 1].astype(np.int64)).to(args.device)

        t = torch.full((H, W), -1, device=args.device, dtype=torch.long)
        flat = t.reshape(-1)
        flat[idx] = val
        t = flat.reshape(H, W)
        tokens.append(t)

        known_mask = t.ge(0)
        sampled = adapter.sample_scale(prompt=header.prompt, known_tokens=tokens, scale_idx=s, known_mask=known_mask, seed=header.seed)
        tokens[s] = sampled

        img = adapter.decode_tokens_to_image(tokens)
        _tensor_to_pil(img).save(os.path.join(args.out_dir, f"stage{s}.png"))

    print(f"[OK] wrote intermediate images to: {args.out_dir}")


if __name__ == "__main__":
    main()
