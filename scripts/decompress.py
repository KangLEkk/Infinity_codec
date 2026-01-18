#!/usr/bin/env python
"""CLI: decompress a progressive bitstream.

Example:
  python scripts/decompress.py --stream out.stream --out recon.png
"""

from __future__ import annotations

import argparse

from codec.codec import ProgressiveEntropyCodec, CodecConfig
from codec.infinity_adapter import DummyInfinityAdapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    # TODO: Replace DummyInfinityAdapter with a real Infinity adapter.
    adapter = DummyInfinityAdapter(image_hw=(256, 256), scales=2, vocab_size=2048)

    codec = ProgressiveEntropyCodec(adapter=adapter, cfg=CodecConfig())
    codec.decompress(bitstream_path=args.stream, out_image_path=args.out, device=args.device)
    print(f"[OK] wrote image: {args.out}")


if __name__ == "__main__":
    main()
