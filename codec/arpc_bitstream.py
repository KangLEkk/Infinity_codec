"""ARPC bitstream container.

This file defines a small, self-contained container for progressive, arithmetic-coded
bitwise BSQ tokens.

We support **version 2** which adds:
- seed
- entropy-mask strategy name
- entropy-mask parameters (JSON string)

Format (little-endian)
----------------------
magic          : 4 bytes  b'ARPC'
version        : uint8    currently 2
num_scales K   : uint16
bit_depth d    : uint16
k_transmit     : uint16
seed           : uint32
prompt_len     : uint32
prompt_utf8    : bytes
mask_name_len  : uint16
mask_name_utf8 : bytes
mask_json_len  : uint32
mask_json_utf8 : bytes  (JSON; may be empty)
active_bits    : K * uint8
scale_shapes   : K * (uint16 H, uint16 W)
payload_len    : uint32
payload        : bytes

Note
----
We deliberately keep the container simple to make it easy to inspect and extend.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


MAGIC = b"ARPC"
VERSION_V2 = 2


@dataclass
class ARPCHeader:
    version: int = VERSION_V2
    num_scales: int = 0
    bit_depth: int = 0
    k_transmit: int = 0
    seed: int = 0
    prompt: str = ""
    active_bits: List[int] = field(default_factory=list)
    scale_shapes: List[Tuple[int, int]] = field(default_factory=list)  # (H, W)

    # entropy-mask (optional)
    mask_strategy: str = "none"
    mask_params: Dict = field(default_factory=dict)

    def validate(self):
        assert self.version in (1, 2), f"unsupported header version {self.version}"
        assert self.num_scales == len(self.active_bits), "active_bits length mismatch"
        assert self.num_scales == len(self.scale_shapes), "scale_shapes length mismatch"
        assert 0 < self.k_transmit <= self.num_scales, "invalid k_transmit"
        for b in self.active_bits:
            assert 0 < int(b) <= int(self.bit_depth), f"invalid active bit width {b}"
        for (h, w) in self.scale_shapes:
            assert 0 < int(h) < 65536 and 0 < int(w) < 65536, "invalid scale shape"
        assert isinstance(self.mask_strategy, str)
        assert isinstance(self.mask_params, dict)


def save_arpc_bitstream(path: str, header: ARPCHeader, payload: bytes) -> None:
    """Write header+payload to disk."""
    header.validate()

    prompt_bytes = (header.prompt or "").encode("utf-8")
    mask_name_bytes = (header.mask_strategy or "none").encode("utf-8")
    mask_json_bytes = json.dumps(header.mask_params or {}, ensure_ascii=False).encode("utf-8")

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<B", int(VERSION_V2)))
        f.write(struct.pack("<HHH", int(header.num_scales), int(header.bit_depth), int(header.k_transmit)))
        f.write(struct.pack("<I", int(header.seed) & 0xFFFFFFFF))

        f.write(struct.pack("<I", len(prompt_bytes)))
        f.write(prompt_bytes)

        f.write(struct.pack("<H", len(mask_name_bytes)))
        f.write(mask_name_bytes)

        f.write(struct.pack("<I", len(mask_json_bytes)))
        f.write(mask_json_bytes)

        # active bits
        f.write(bytes(int(x) & 0xFF for x in header.active_bits))

        # shapes
        for (h, w) in header.scale_shapes:
            f.write(struct.pack("<HH", int(h), int(w)))

        # payload
        f.write(struct.pack("<I", len(payload)))
        f.write(payload)


def load_arpc_bitstream(path: str) -> tuple[ARPCHeader, bytes]:
    """Read header+payload from disk."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"Bad magic: {magic!r}")
        (version,) = struct.unpack("<B", f.read(1))

        if version == 1:
            # Legacy v1 (no seed/mask fields)
            num_scales, bit_depth, k_transmit = struct.unpack("<HHH", f.read(6))
            (prompt_len,) = struct.unpack("<I", f.read(4))
            prompt = f.read(prompt_len).decode("utf-8")

            active_bits = list(f.read(num_scales))
            scale_shapes: List[Tuple[int, int]] = []
            for _ in range(num_scales):
                h, w = struct.unpack("<HH", f.read(4))
                scale_shapes.append((int(h), int(w)))

            (payload_len,) = struct.unpack("<I", f.read(4))
            payload = f.read(payload_len)
            if len(payload) != payload_len:
                raise ValueError("Truncated payload")

            header = ARPCHeader(
                version=1,
                num_scales=int(num_scales),
                bit_depth=int(bit_depth),
                k_transmit=int(k_transmit),
                seed=0,
                prompt=prompt,
                active_bits=active_bits,
                scale_shapes=scale_shapes,
                mask_strategy="none",
                mask_params={},
            )
            header.validate()
            return header, payload

        if version != VERSION_V2:
            raise ValueError(f"Unsupported version: {version}")

        num_scales, bit_depth, k_transmit = struct.unpack("<HHH", f.read(6))
        (seed,) = struct.unpack("<I", f.read(4))

        (prompt_len,) = struct.unpack("<I", f.read(4))
        prompt = f.read(prompt_len).decode("utf-8")

        (mask_name_len,) = struct.unpack("<H", f.read(2))
        mask_strategy = f.read(mask_name_len).decode("utf-8")

        (mask_json_len,) = struct.unpack("<I", f.read(4))
        mask_json = f.read(mask_json_len).decode("utf-8")
        try:
            mask_params = json.loads(mask_json) if mask_json.strip() else {}
        except Exception:
            mask_params = {}

        active_bits = list(f.read(num_scales))

        scale_shapes: List[Tuple[int, int]] = []
        for _ in range(num_scales):
            h, w = struct.unpack("<HH", f.read(4))
            scale_shapes.append((int(h), int(w)))

        (payload_len,) = struct.unpack("<I", f.read(4))
        payload = f.read(payload_len)
        if len(payload) != payload_len:
            raise ValueError("Truncated payload")

    header = ARPCHeader(
        version=int(version),
        num_scales=int(num_scales),
        bit_depth=int(bit_depth),
        k_transmit=int(k_transmit),
        seed=int(seed),
        prompt=prompt,
        active_bits=active_bits,
        scale_shapes=scale_shapes,
        mask_strategy=mask_strategy or "none",
        mask_params=mask_params or {},
    )
    header.validate()
    return header, payload


# Backward-compatible aliases
write_arpc = save_arpc_bitstream
read_arpc = load_arpc_bitstream
