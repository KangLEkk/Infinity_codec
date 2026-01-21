"""Binary arithmetic (range) coder.

This module is used by the ARPC codec implementation.

Design goals
------------
- Deterministic (encoder/decoder match bit-exactly)
- No external dependencies
- Binary alphabet {0,1}

We quantize floating probabilities into integer frequencies with a fixed total.
Encoder and decoder MUST use the same quantization rule.

This is *not* an optimized implementation; it is intended as a reference
implementation that is easy to read and debug.
"""

from __future__ import annotations

from dataclasses import dataclass


# Frequency total for probability quantization.
# 2**15 keeps good precision while avoiding overflow in 32-bit arithmetic.
TOT_FREQ = 1 << 15


def prob_to_freq1(p1: float, tot: int = TOT_FREQ) -> int:
    """Convert probability of symbol '1' into integer frequency.

    Ensures freq1 in [1, tot-1] to avoid degenerate intervals.
    """
    if p1 != p1:  # NaN
        p1 = 0.5
    if p1 <= 0.0:
        return 1
    if p1 >= 1.0:
        return tot - 1
    # Round-to-nearest for stability.
    f1 = int(p1 * tot + 0.5)
    if f1 <= 0:
        f1 = 1
    elif f1 >= tot:
        f1 = tot - 1
    return f1


@dataclass
class RangeEncoder:
    """Binary range encoder."""

    tot: int = TOT_FREQ

    def __post_init__(self):
        self.low = 0
        self.high = 0xFFFFFFFF
        self.pending_bits = 0
        self._out_bytes = bytearray()
        self._bit_buffer = 0
        self._bit_count = 0

    def _emit_bit(self, bit: int):
        self._bit_buffer = (self._bit_buffer << 1) | (bit & 1)
        self._bit_count += 1
        if self._bit_count == 8:
            self._out_bytes.append(self._bit_buffer & 0xFF)
            self._bit_buffer = 0
            self._bit_count = 0

    def _flush_pending(self, bit: int):
        # Output `bit`, then output the opposite for each pending underflow.
        self._emit_bit(bit)
        inv = bit ^ 1
        while self.pending_bits > 0:
            self._emit_bit(inv)
            self.pending_bits -= 1

    def encode_bit(self, bit: int, p1: float):
        """Encode a single bit given P(bit=1)."""
        bit = 1 if bit else 0
        f1 = prob_to_freq1(float(p1), self.tot)
        f0 = self.tot - f1

        # CDF for binary
        # symbol 0 -> [0, f0)
        # symbol 1 -> [f0, tot)
        lo_cum = 0 if bit == 0 else f0
        hi_cum = f0 if bit == 0 else self.tot

        rng = self.high - self.low + 1
        self.high = self.low + (rng * hi_cum // self.tot) - 1
        self.low = self.low + (rng * lo_cum // self.tot)

        # Renormalize
        HALF = 1 << 31
        QUARTER = 1 << 30
        THREE_QUARTER = 3 << 30
        while True:
            if self.high < HALF:
                self._flush_pending(0)
            elif self.low >= HALF:
                self._flush_pending(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTER:
                self.pending_bits += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break

            self.low = (self.low << 1) & 0xFFFFFFFF
            self.high = ((self.high << 1) | 1) & 0xFFFFFFFF

    def finish(self) -> bytes:
        """Finalize the stream and return bytes."""
        # Termination: emit one extra bit to disambiguate.
        QUARTER = 1 << 30
        self.pending_bits += 1
        if self.low < QUARTER:
            self._flush_pending(0)
        else:
            self._flush_pending(1)

        # Flush remaining bits to full byte.
        if self._bit_count:
            self._bit_buffer <<= (8 - self._bit_count)
            self._out_bytes.append(self._bit_buffer & 0xFF)
            self._bit_buffer = 0
            self._bit_count = 0

        return bytes(self._out_bytes)


@dataclass
class RangeDecoder:
    """Binary range decoder."""

    data: bytes
    tot: int = TOT_FREQ

    def __post_init__(self):
        self.low = 0
        self.high = 0xFFFFFFFF
        self.code = 0
        self._data = self.data
        self._byte_pos = 0
        self._bit_mask = 0
        self._curr_byte = 0

        # Initialize with 32 bits
        for _ in range(32):
            self.code = ((self.code << 1) | self._read_bit()) & 0xFFFFFFFF

    def _read_bit(self) -> int:
        if self._bit_mask == 0:
            if self._byte_pos < len(self._data):
                self._curr_byte = self._data[self._byte_pos]
                self._byte_pos += 1
            else:
                self._curr_byte = 0
            self._bit_mask = 0x80
        bit = 1 if (self._curr_byte & self._bit_mask) else 0
        self._bit_mask >>= 1
        return bit

    def decode_bit(self, p1: float) -> int:
        """Decode one bit given P(bit=1)."""
        f1 = prob_to_freq1(float(p1), self.tot)
        f0 = self.tot - f1

        rng = self.high - self.low + 1
        # Determine scaled value within [0, tot)
        scaled = ((self.code - self.low + 1) * self.tot - 1) // rng

        # CDF split at f0
        if scaled < f0:
            bit = 0
            lo_cum, hi_cum = 0, f0
        else:
            bit = 1
            lo_cum, hi_cum = f0, self.tot

        self.high = self.low + (rng * hi_cum // self.tot) - 1
        self.low = self.low + (rng * lo_cum // self.tot)

        # Renormalize
        HALF = 1 << 31
        QUARTER = 1 << 30
        THREE_QUARTER = 3 << 30
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.code -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTER:
                self.low -= QUARTER
                self.high -= QUARTER
                self.code -= QUARTER
            else:
                break

            self.low = (self.low << 1) & 0xFFFFFFFF
            self.high = ((self.high << 1) | 1) & 0xFFFFFFFF
            self.code = ((self.code << 1) | self._read_bit()) & 0xFFFFFFFF

        return bit
