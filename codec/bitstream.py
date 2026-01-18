"""Bitstream format.

We store a progressive token stream as:

Header:
  - magic 'IECB'
  - uint8 version
  - uint16 num_scales
  - uint16 bits_per_token (default 32)
  - uint32 seed
  - uint32 prompt_length + prompt_bytes (utf8)
  - per-scale metadata:
      * uint16 H, uint16 W
      * uint32 full_count      (H*W if full else 0)
      * uint32 sparse_count    (K)

Payload:
  - scale0: full token map values as uint64, length H*W
  - scale>=1: sparse (flat_idx uint32, value uint64) * K

Notes:
  * For research/debug clarity, we always store uint64 values.
  * For large scale experiments, you can switch to uint32 if d<=32.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np

MAGIC = b"IECB"
VERSION = 1


@dataclass
class ScaleMeta:
    H: int
    W: int
    full: bool
    K: int  # sparse count for this scale


@dataclass
class StreamHeader:
    num_scales: int
    bits_per_token: int
    seed: int
    prompt: str
    scales: List[ScaleMeta]


@dataclass
class StreamPayload:
    # scale0 full map: np.ndarray[uint64] shape (H*W,)
    scale0_full: np.ndarray
    # for scale>=1
    sparse_idx: List[np.ndarray]  # list of uint32 arrays of length K
    sparse_val: List[np.ndarray]  # list of uint64 arrays of length K


def write_bitstream(path: str, header: StreamHeader, payload: StreamPayload) -> None:
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<B", VERSION))
        f.write(struct.pack("<H", header.num_scales))
        f.write(struct.pack("<H", header.bits_per_token))
        f.write(struct.pack("<I", int(header.seed) & 0xFFFFFFFF))
        prompt_bytes = header.prompt.encode("utf-8")
        f.write(struct.pack("<I", len(prompt_bytes)))
        f.write(prompt_bytes)

        assert len(header.scales) == header.num_scales
        for s in range(header.num_scales):
            meta = header.scales[s]
            f.write(struct.pack("<HH", int(meta.H), int(meta.W)))
            full_count = int(meta.H * meta.W) if meta.full else 0
            f.write(struct.pack("<I", full_count))
            f.write(struct.pack("<I", int(meta.K)))

        # payload
        payload.scale0_full.astype(np.uint64).tofile(f)
        for s in range(1, header.num_scales):
            idx = payload.sparse_idx[s - 1].astype(np.uint32)
            val = payload.sparse_val[s - 1].astype(np.uint64)
            idx.tofile(f)
            val.tofile(f)


def read_bitstream(path: str) -> tuple[StreamHeader, StreamPayload]:
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"Bad magic: {magic}")
        version = struct.unpack("<B", f.read(1))[0]
        if version != VERSION:
            raise ValueError(f"Unsupported version: {version}")

        num_scales = struct.unpack("<H", f.read(2))[0]
        bits_per_token = struct.unpack("<H", f.read(2))[0]
        seed = struct.unpack("<I", f.read(4))[0]
        prompt_len = struct.unpack("<I", f.read(4))[0]
        prompt = f.read(prompt_len).decode("utf-8")

        scales: List[ScaleMeta] = []
        full_counts: List[int] = []
        for _ in range(num_scales):
            H, W = struct.unpack("<HH", f.read(4))
            full_count = struct.unpack("<I", f.read(4))[0]
            K = struct.unpack("<I", f.read(4))[0]
            full = full_count > 0
            scales.append(ScaleMeta(H=H, W=W, full=full, K=K))
            full_counts.append(full_count)

        # payload
        scale0_count = full_counts[0]
        scale0_full = np.fromfile(f, dtype=np.uint64, count=scale0_count)

        sparse_idx: List[np.ndarray] = []
        sparse_val: List[np.ndarray] = []
        for s in range(1, num_scales):
            K = scales[s].K
            idx = np.fromfile(f, dtype=np.uint32, count=K)
            val = np.fromfile(f, dtype=np.uint64, count=K)
            sparse_idx.append(idx)
            sparse_val.append(val)

        header = StreamHeader(
            num_scales=num_scales,
            bits_per_token=bits_per_token,
            seed=seed,
            prompt=prompt,
            scales=scales,
        )
        payload = StreamPayload(scale0_full=scale0_full, sparse_idx=sparse_idx, sparse_val=sparse_val)
        return header, payload
