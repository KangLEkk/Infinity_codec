#!/usr/bin/env python
"""CLI: compress an image with entropy-map progressive coding.

Example:
  python scripts/compress.py \
    --image path/to/img.png \
    --prompt "a photo of a cat" \
    --out out.stream \
    --tokens-per-scale 0,256,1024

Note:
  - `--tokens-per-scale` is a comma-separated list with length = #scales.
  - scale0 is ignored (always full).
"""

from __future__ import annotations

import argparse
from PIL import Image
import torch

from codec.codec import ProgressiveEntropyCodec, CodecConfig
from codec.infinity_adapter import DummyInfinityAdapter


def parse_tokens_per_scale(s: str):
    if not s:
        return None
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--prompt", type=str, default="")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokens-per-scale", type=str, default="")
    args = ap.parse_args()

    image = Image.open(args.image)

    # TODO: Replace DummyInfinityAdapter with a real Infinity adapter.
    adapter = DummyInfinityAdapter(image_hw=image.size[::-1], scales=2, vocab_size=2048)

    cfg = CodecConfig(tokens_per_scale=parse_tokens_per_scale(args.tokens_per_scale), seed=args.seed)
    codec = ProgressiveEntropyCodec(adapter=adapter, cfg=cfg)

    codec.compress(image=image, prompt=args.prompt, out_path=args.out, device=args.device)
    print(f"[OK] wrote bitstream: {args.out}")


if __name__ == "__main__":
    main()
