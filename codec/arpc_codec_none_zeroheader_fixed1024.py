"""ARPC (Infinity VAR) ultra-low bitrate codec — NONE masking + ZERO header.

This is a specialized variant for your setup:

- mask_strategy = 'none' (transmit ALL active bits)
- active_bits   = [d] * K (default)
- token_coding  = 'direct'
- K, d, scale_schedule are shared defaults between encoder/decoder
- seed / version / prompt are NOT stored in the bitstream
- Bitstream contains ONLY the arithmetic-coded payload bytes (0-byte header)

Assumptions (hard-coded defaults):
- Image resolution is fixed to 1024x1024.
- k_transmit is fixed (must be the same on both encoder/decoder sides).

You can still pass `prompt` out-of-band to both compress() and decompress().

Drop-in usage: instantiate `ARPCNoneZeroHeaderFixed1024` with your
(vae, model, text_tokenizer, text_encoder) and call compress/decompress.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# Expect these to exist in your project (same as the original ARPC script).
from codec.arithmetic import RangeDecoder, RangeEncoder


_EPS = 1e-12


def _softmax_last2(logits_2: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits_2, dim=-1)


# def _logits_to_p1(logits_BlV: torch.Tensor, bit_depth: int) -> torch.Tensor:
#     """Convert Infinity VAR logits -> p(bit=1) for each bit.

#     logits_BlV: [B, L, V] where V=2 for bit {0,1}.
#     Returns: p1 [B, L, d] where d==bit_depth, matching the original ARPC layout.

#     NOTE: In the original ARPC script, logits may already be shaped [B,L,d,2]
#           or [B,L,2] depending on the VAR implementation.
#           Here we try to handle both common cases safely.
#     """
#     if logits_BlV.ndim == 4 and logits_BlV.shape[-1] == 2:
#         # [B, L, d, 2] -> [B, L, d]
#         probs = _softmax_last2(logits_BlV)  # [B,L,d,2]
#         p1 = probs[..., 1]
#         return p1

#     if logits_BlV.ndim == 3 and logits_BlV.shape[-1] == 2:
#         # [B, L, 2] -> [B, L, 1] then broadcast to d
#         probs = _softmax_last2(logits_BlV)  # [B,L,2]
#         p1 = probs[..., 1:2]
#         return p1.expand(-1, -1, bit_depth)

    raise ValueError(f"Unexpected logits shape: {tuple(logits_BlV.shape)}")

def _logits_to_p1(logits_BlV: torch.Tensor, bit_depth: int) -> torch.Tensor:
    """Convert Infinity logits [B,L,2*d] into p(1) [B,L,d]."""
    assert logits_BlV.ndim == 3, f"expected [B,L,V], got {tuple(logits_BlV.shape)}"
    B, L, V = logits_BlV.shape
    assert V == 2 * bit_depth, f"V={V} != 2*d={2*bit_depth}"
    logits = logits_BlV.view(B, L, bit_depth, 2).float()
    probs = _softmax_last2(logits)
    return probs[..., 1].clamp(_EPS, 1.0 - _EPS)


def _encode_stream_bits(enc: RangeEncoder, bits_1d: np.ndarray, p1_1d: np.ndarray) -> None:
    """Arithmetic-code b ~ Bernoulli(p1). (direct coding only)"""
    bits_1d = np.asarray(bits_1d, dtype=np.uint8).reshape(-1)
    p1_1d = np.asarray(p1_1d, dtype=np.float32).reshape(-1)
    for b, p in zip(bits_1d.tolist(), p1_1d.tolist()):
        enc.encode_bit(int(b), float(p))


def _decode_stream_bits(dec: RangeDecoder, p1_1d: np.ndarray) -> np.ndarray:
    """Arithmetic-decode a 1D bit stream. (direct coding only)"""
    p1_1d = np.asarray(p1_1d, dtype=np.float32).reshape(-1)
    out = np.zeros((p1_1d.size,), dtype=np.uint8)
    for i, p in enumerate(p1_1d.tolist()):
        out[i] = dec.decode_bit(float(p))
    return out


def _save_payload_only(path: str, payload: bytes) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(payload)


def _load_payload_only(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


@dataclass
class ARPCNoneZeroHeaderFixed1024:
    """Specialized ARPC codec: NONE masking + ZERO header + fixed 1024x1024."""

    vae: torch.nn.Module
    model: torch.nn.Module
    text_tokenizer: object
    text_encoder: torch.nn.Module

    device: str = "cuda"
    tlen: int = 512

    # Hard constraints (shared between encoder/decoder)
    fixed_hw: Tuple[int, int] = (1024, 1024)
    k_transmit: int = 5
    seed: int = 0  # not transmitted; kept fixed for reproducibility

    # Cached shared defaults (computed once)
    _cached_scale_schedule: Optional[List[Tuple[int, int, int]]] = None
    _cached_Kd: Optional[Tuple[int, int]] = None

    # -------------------------
    # Text conditioning
    # -------------------------
    @torch.no_grad()
    def _prepare_text_cond_tuple(self, prompt: str):
        """Create the compact text conditioning tuple (kv_compact, lens, cu_seqlens, maxlen)."""
        prompt = prompt or ""
        prompts = [prompt]
        tok = self.text_tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tlen,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tok.input_ids.to(self.device)
        attn_mask = tok.attention_mask.to(self.device)

        feats = self.text_encoder(input_ids=input_ids, attention_mask=attn_mask)
        last_hidden = feats["last_hidden_state"] if isinstance(feats, dict) else feats.last_hidden_state

        lens = attn_mask.sum(dim=-1).int()  # [B]
        max_seqlen_k = int(lens.max().item())
        cu_seqlens_k = torch.zeros((lens.shape[0] + 1,), device=self.device, dtype=torch.int32)
        cu_seqlens_k[1:] = torch.cumsum(lens, dim=0)
        kv_compact = torch.cat([last_hidden[i, :l, :] for i, l in enumerate(lens.tolist())], dim=0)
        return kv_compact, lens, cu_seqlens_k, max_seqlen_k

    # -------------------------
    # Shared defaults (K, d, schedule)
    # -------------------------
    @torch.no_grad()
    def _get_shared_schedule_K_d(self) -> Tuple[List[Tuple[int, int, int]], int, int]:
        """Compute and cache (scale_schedule, K, d) for the fixed resolution.

        We use a dummy zero-image ONLY to infer latent H/W and the multi-scale schedule.
        This does NOT transmit anything and is deterministic given the fixed resolution and VAE.
        """
        if self._cached_scale_schedule is not None and self._cached_Kd is not None:
            K, d = self._cached_Kd
            return self._cached_scale_schedule, K, d

        H, W = self.fixed_hw
        dummy = torch.zeros((1, 3, H, W), device=self.device, dtype=torch.float32)

        # Same logic as the original ARPC script: derive schedule from the tokenizer/quantizer.
        h, _, _ = self.vae.encode_for_raw_features(dummy, scale_schedule=None)
        if h.ndim == 5:
            _, _, T, hH, hW = h.shape
        else:
            T = 1
            _, _, hH, hW = h.shape

        from infinity.models.bsq_vae.multiscale_bsq import get_latent2scale_schedule

        schedule = get_latent2scale_schedule(T, int(hH), int(hW), mode=self.vae.quantizer.schedule_mode)
        # schedule elements should already be (T, H, W)
        scale_schedule = [(int(t), int(hh), int(ww)) for (t, hh, ww) in schedule]

        # Infer K and d:
        # We still run quantizer ONCE to get d reliably (matches your training config).
        _, _, all_bit_indices, _, _, _ = self.vae.quantizer(h, scale_schedule=schedule)
        d = int(all_bit_indices[0].shape[-1])
        K = len(all_bit_indices)

        self._cached_scale_schedule = scale_schedule
        self._cached_Kd = (K, d)
        return scale_schedule, K, d

    # -------------------------
    # VAR logits per scale (same as original)
    # -------------------------
    @torch.no_grad()
    def _get_logits_for_scale(self, text_cond_tuple, scale_schedule, prev_bits: List[torch.Tensor], si: int, seed: int):
        """Get logits for scale si conditioned on already reconstructed prev_bits (0..si-1)."""
        cfg_list = [1.0] * len(scale_schedule)
        tau_list = [1.0] * len(scale_schedule)

        gt_list = list(prev_bits)
        while len(gt_list) <= si:
            _, H, W = scale_schedule[len(gt_list)]
            d = int(getattr(self.model, "codebook_dim", 32))
            dummy = torch.zeros((1, 1, H, W, d), device=self.device, dtype=torch.uint8)
            gt_list.append(dummy)

        out = self.model.autoregressive_infer_cfg(
            vae=self.vae,
            scale_schedule=scale_schedule,
            label_B_or_BLT=text_cond_tuple,
            B=1,
            g_seed=None,
            cfg_sc=3,
            cfg_list=cfg_list,
            tau_list=tau_list,
            top_k=0,
            top_p=0.0,
            returns_vemb=1,
            gumbel=0,
            ret_img=False,
            trunk_scale=int(si + 1),
            gt_leak=int(si + 1),
            gt_ls_Bl=gt_list,
            inference_mode=True,
            sampling_per_bits=1,
            vae_type=1,
            return_logits_per_scale=True,
            do_sample=False,
        )

        # The original script expects logits per scale at out[0] or out[-1] depending on Infinity version.
        # Try to robustly pick logits for this scale.
        # logits_per_scale = out[0] if isinstance(out, (list, tuple)) else out
        # if isinstance(logits_per_scale, list):
        #     logits = logits_per_scale[si]
        # else:
        #     logits = logits_per_scale

        # logits_per_scale = out[0] if isinstance(out, (list, tuple)) else out

        # # dict wrapper (some forks)
        # if isinstance(logits_per_scale, dict):
        #     logits_per_scale = logits_per_scale.get("logits_per_scale", logits_per_scale.get("logits", logits_per_scale))

        # if isinstance(logits_per_scale, (list, tuple)):
        #     if len(logits_per_scale) == 0:
        #         raise RuntimeError("return_logits_per_scale=True but got empty logits list.")
        #     if si < len(logits_per_scale):
        #         logits = logits_per_scale[si]
        #     else:
        #         # most common: list length == 1 (only current scale logits)
        #         logits = logits_per_scale[-1]
        # else:
        #     logits = logits_per_scale
        logits = out[-1][si]
        return logits

    # -------------------------
    # Tokenization (GT bit indices)
    # -------------------------
    @torch.no_grad()
    def encode_image_to_bitidx_and_schedule(self, img_B3HW: torch.Tensor):
        """Encode image -> bit indices per scale, plus scale schedule and bit depth d."""
        assert img_B3HW.ndim == 4 and img_B3HW.shape[0] == 1, "B=1 only"
        img_B3HW = img_B3HW.to(self.device)

        h, _, _ = self.vae.encode_for_raw_features(img_B3HW, scale_schedule=None)
        if h.ndim == 5:
            _, _, T, H, W = h.shape
        else:
            T = 1
            _, _, H, W = h.shape

        from infinity.models.bsq_vae.multiscale_bsq import get_latent2scale_schedule

        schedule = get_latent2scale_schedule(T, int(H), int(W), mode=self.vae.quantizer.schedule_mode)
        _, _, all_bit_indices, _, _, _ = self.vae.quantizer(h, scale_schedule=schedule)

        bitidx_per_scale: List[torch.Tensor] = []
        for bits in all_bit_indices:
            # bits: [B,T,h,w,d]
            if bits.ndim == 5:
                bits = bits[:, 0:1]
            elif bits.ndim == 4:
                bits = bits.unsqueeze(1)
            bitidx_per_scale.append(bits.to(torch.uint8))

        d = int(bitidx_per_scale[0].shape[-1])
        # Use bit-index shapes as the effective scale_schedule (same as the original ARPC script)
        scale_schedule = [(1, int(b.shape[2]), int(b.shape[3])) for b in bitidx_per_scale]
        return bitidx_per_scale, scale_schedule, d

    # -------------------------
    # Public API: compress / decompress (0 header)
    # -------------------------
    @torch.no_grad()
    def compress(self, img_B3HW: torch.Tensor, out_stream_path: str, prompt: str = "") -> bytes:
        """Compress a single fixed-1024 image -> payload-only bitstream file (0-byte header)."""
        H, W = int(img_B3HW.shape[2]), int(img_B3HW.shape[3])
        if (H, W) != tuple(self.fixed_hw):
            raise ValueError(f"Fixed-resolution codec expects {self.fixed_hw}, got {(H, W)}")

        bitidx_gt, scale_schedule, d = self.encode_image_to_bitidx_and_schedule(img_B3HW)
        K = len(scale_schedule)
        if self.k_transmit > K:
            raise ValueError(f"k_transmit={self.k_transmit} > K={K}")

        text_cond_tuple = self._prepare_text_cond_tuple(prompt)

        enc = RangeEncoder()
        reconstructed_bits: List[torch.Tensor] = []

        for si in range(self.k_transmit):
            bits_gt = bitidx_gt[si].to(self.device)  # [1,1,Hs,Ws,d]
            Hs, Ws = int(bits_gt.shape[2]), int(bits_gt.shape[3])
            L = Hs * Ws

            logits = self._get_logits_for_scale(text_cond_tuple, scale_schedule, reconstructed_bits, si, self.seed)
            p1 = _logits_to_p1(logits, bit_depth=d)  # [1,L,d]

            # Flatten and encode ALL bits (none masking, active_bits=[d]*K)
            bits_1d = bits_gt[:, 0].reshape(1, L, d).reshape(-1).detach().cpu().numpy().astype(np.uint8)
            p1_1d = p1.reshape(-1).detach().cpu().numpy().astype(np.float32)

            _encode_stream_bits(enc, bits_1d, p1_1d)

            # For "none", we reconstruct previous scales as GT (lossless)
            reconstructed_bits.append(bits_gt)

        payload = enc.finish()
        _save_payload_only(out_stream_path, payload)
        return payload

    @torch.no_grad()
    def decompress(self, stream_path: str, out_path: str, prompt: str = "") -> torch.Tensor:
        """Decompress payload-only bitstream -> reconstructed image uint8 tensor (B,H,W,3)."""
        # Shared defaults for fixed resolution
        scale_schedule, K, d = self._get_shared_schedule_K_d()
        if self.k_transmit > K:
            raise ValueError(f"k_transmit={self.k_transmit} > K={K}")

        # Forcing the same effective scale shapes used by tokenizer:
        # We recompute the *effective* scale_schedule via the quantizer shapes (safer).
        H, W = self.fixed_hw
        dummy = torch.zeros((1, 3, H, W), device=self.device, dtype=torch.float32)
        _, effective_scale_schedule, effective_d = self.encode_image_to_bitidx_and_schedule(dummy)
        scale_schedule = effective_scale_schedule
        d = effective_d
        K = len(scale_schedule)

        payload = _load_payload_only(stream_path)
        text_cond_tuple = self._prepare_text_cond_tuple(prompt)

        dec = RangeDecoder(payload)
        reconstructed_bits: List[torch.Tensor] = []

        # Decode first k scales losslessly
        for si in range(self.k_transmit):
            _, Hs, Ws = scale_schedule[si]
            L = int(Hs * Ws)

            logits = self._get_logits_for_scale(text_cond_tuple, scale_schedule, reconstructed_bits, si, self.seed)
            p1 = _logits_to_p1(logits, bit_depth=d)  # [1,L,d]

            p1_1d = p1.reshape(-1).detach().cpu().numpy().astype(np.float32)
            bits_1d = _decode_stream_bits(dec, p1_1d)

            rec_bits = torch.from_numpy(bits_1d).to(self.device).to(torch.uint8)
            rec_bits = rec_bits.view(1, L, d)  # [1,L,d]

            rec_full = torch.zeros((1, 1, Hs, Ws, d), device=self.device, dtype=torch.uint8)
            rec_full[:, 0].reshape(1, L, d)[:, :, :d] = rec_bits
            reconstructed_bits.append(rec_full)

        # Full decode with forced early scales (same as original ARPC)
        forced_bits: List[Optional[torch.Tensor]] = [None] * K
        forced_mask: List[Optional[torch.Tensor]] = [None] * K
        for si in range(self.k_transmit):
            forced_bits[si] = reconstructed_bits[si]
            _, hh, ww = scale_schedule[si]
            forced_mask[si] = torch.ones((1, hh, ww), device=self.device, dtype=torch.bool)

        cfg_list = [1.0] * len(scale_schedule)
        tau_list = [1.0] * len(scale_schedule)

        out = self.model.autoregressive_infer_cfg(
            vae=self.vae,
            scale_schedule=scale_schedule,
            label_B_or_BLT=text_cond_tuple,
            B=1,
            g_seed=None,
            cfg_sc=3,
            cfg_list=cfg_list,
            tau_list=tau_list,
            top_k=0,
            top_p=0.0,
            returns_vemb=1,
            gumbel=0,
            ret_img=True,
            trunk_scale=int(K),
            inference_mode=True,
            sampling_per_bits=1,
            vae_type=1,
            forced_bitidx_per_scale=forced_bits,
            forced_mask_per_scale=forced_mask,
            return_logits_per_scale=False,
        )

        img_u8 = out[2]
        if isinstance(img_u8, list):
            img_u8 = img_u8[-1]

        # Infinity returns uint8 in (B,H,W,C) and flips channel order (BGR).
        if isinstance(img_u8, torch.Tensor) and img_u8.ndim == 4 and img_u8.shape[-1] == 3:
            img_u8 = img_u8.flip(dims=(3,))

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        out_lower = out_path.lower()
        if out_lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
            Image.fromarray(img_u8[0].detach().cpu().numpy()).save(out_path)
        else:
            torch.save(img_u8.detach().cpu(), out_path)

        return img_u8


# -------------------------
# Optional CLI
# -------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser("ARPC NONE + ZERO header (fixed 1024)")
    p.add_argument("--mode", type=str, choices=["enc", "dec"], required=True)
    p.add_argument("--in_path", type=str, required=True, help="input image (enc) or payload file (dec)")
    p.add_argument("--out_path", type=str, required=True, help="payload file (enc) or output image (dec)")
    p.add_argument("--prompt", type=str, default="", help="out-of-band text prompt")
    p.add_argument("--k_transmit", type=int, default=5, help="must match on both sides (not stored)")
    args = p.parse_args()

    raise SystemExit(
        "This file is a library module. Instantiate ARPCNoneZeroHeaderFixed1024 in your project.\n"
        "CLI wiring depends on your Infinity model/vae loading code."
    )
