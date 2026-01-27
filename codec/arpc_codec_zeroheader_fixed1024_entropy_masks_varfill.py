"""
ARPC (Infinity VAR) ultra-low bitrate codec — ZERO header + entropy masking strategies
(Reference-aligned: entropy_channel / entropy_scale / entropy_spatial)

What is stored in the bitstream?
--------------------------------
- NOTHING BUT the arithmetic-coded payload bytes (0-byte header).

Therefore BOTH encoder & decoder MUST share these settings out-of-band:
- fixed resolution (default: 1024x1024)
- k_transmit (default: 5)
- mask_strategy + mask_params (default: "none")

Mask strategies (paper-style, no training)
------------------------------------------
We encode only a subset of bits/tokens and fill the rest with deterministic MAP (argmax),
so the arithmetic coder remains decodable without storing masks.

Supported:
- "none"            : transmit all bits in the first k scales (baseline ARPC)
- "entropy_channel" : keep only high-entropy bit-planes (channels) across ALL spatial positions
- "entropy_scale"   : per-scale keep ratio derived from mean entropy, then channel selection
- "entropy_spatial" : keep only high-entropy spatial positions, transmit ALL active bits at those positions

Important notes for ZERO-header mode
------------------------------------
- mask decisions are derived from VAR probabilities only (deterministic at decoder)
- we always fill "dropped" parts with MAP bits for conditioning & for forcing
- forced_mask_per_scale is set to ALL-ONES for the first k scales, because:
  * entropy_channel drops bit-planes (cannot be represented by spatial forced_mask)
  * we need deterministic filled bits for the full [H,W,d] tensor anyway

This file is tailored to your Infinity fork where:
  logits = out[-1][si]
  logits shape = [B, L, 2*d]
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from codec.arithmetic import RangeDecoder, RangeEncoder

_EPS = 1e-12


# -----------------------------
# Utilities
# -----------------------------
def _softmax_last2(logits_2: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits_2, dim=-1)


def _logits_to_p1(logits_BlV: torch.Tensor, bit_depth: int) -> torch.Tensor:
    """Convert Infinity logits [B,L,2*d] into p(1) [B,L,d]."""
    assert logits_BlV.ndim == 3, f"expected [B,L,V], got {tuple(logits_BlV.shape)}"
    B, L, V = logits_BlV.shape
    assert V == 2 * bit_depth, f"V={V} != 2*d={2*bit_depth}"
    logits = logits_BlV.view(B, L, bit_depth, 2).float()
    probs = _softmax_last2(logits)
    return probs[..., 1].clamp(_EPS, 1.0 - _EPS)


def _binary_entropy(p1: torch.Tensor) -> torch.Tensor:
    """Binary entropy in bits, elementwise."""
    p1 = p1.clamp(_EPS, 1.0 - _EPS)
    return -(p1 * torch.log2(p1) + (1.0 - p1) * torch.log2(1.0 - p1))


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


def _topk_indices(scores_1d: torch.Tensor, k: int) -> torch.Tensor:
    """Deterministic topk with tie-break using index; returns sorted indices (stable for coding)."""
    L = int(scores_1d.numel())
    k = int(max(0, min(int(k), L)))
    if k == 0:
        return torch.zeros((0,), dtype=torch.long, device=scores_1d.device)

    idx = torch.arange(L, device=scores_1d.device, dtype=scores_1d.dtype)
    adj = scores_1d + (-idx) * 1e-7  # tie-break: prefer smaller index
    _, top_idx = torch.topk(adj, k=k, largest=True, sorted=True)
    return torch.sort(top_idx.to(torch.long))[0]


def _select_channels_by_entropy(
    p1_Bl_d: torch.Tensor,
    d_eff: int,
    keep_ratio: Optional[float] = None,
    entropy_thr: Optional[float] = None,
) -> List[int]:
    """Return kept channel indices within [0, d_eff)."""
    assert p1_Bl_d.ndim == 3 and p1_Bl_d.shape[0] == 1
    p1 = p1_Bl_d[:, :, :d_eff]
    H = _binary_entropy(p1)        # [1,L,d_eff]
    Hc = H.mean(dim=1)[0]          # [d_eff]

    if entropy_thr is not None:
        thr = float(entropy_thr)
        keep = (Hc >= thr)
        kept = torch.nonzero(keep, as_tuple=False).view(-1).tolist()
        if len(kept) == 0:
            kept = [int(torch.argmax(Hc).item())]
        return sorted(set(int(i) for i in kept))

    r = 1.0 if keep_ratio is None else float(keep_ratio)
    r = max(0.0, min(1.0, r))
    m = int(math.ceil(d_eff * r))
    m = max(1, min(m, d_eff))

    idx = torch.arange(d_eff, device=Hc.device, dtype=Hc.dtype)
    adj = Hc + (-idx) * 1e-7
    _, top = torch.topk(adj, k=m, largest=True, sorted=True)
    kept = torch.sort(top.to(torch.long))[0].tolist()
    return [int(i) for i in kept]


def _select_positions_by_entropy(
    p1_Bl_d: torch.Tensor,
    H: int,
    W: int,
    d_eff: int,
    keep_ratio: Optional[float] = None,
    entropy_thr: Optional[float] = None,
) -> torch.Tensor:
    """Return kept flat positions indices [K] in [0, L)."""
    assert p1_Bl_d.ndim == 3 and p1_Bl_d.shape[0] == 1
    L = int(H * W)
    p1 = p1_Bl_d[:, :, :d_eff]
    Hb = _binary_entropy(p1)       # [1,L,d_eff]
    Hpos = Hb.mean(dim=2)[0]       # [L]

    if entropy_thr is not None:
        thr = float(entropy_thr)
        keep = (Hpos >= thr)
        kept = torch.nonzero(keep, as_tuple=False).view(-1)
        if kept.numel() == 0:
            kept = _topk_indices(Hpos, k=1)
        return kept.to(torch.long)

    r = 1.0 if keep_ratio is None else float(keep_ratio)
    r = max(0.0, min(1.0, r))
    k = int(round(L * r))
    k = max(1, min(k, L))
    return _topk_indices(Hpos, k=k)


def _flatten_for_channel_mask(
    bits_B1HWd: torch.Tensor,
    p1_Bl_d: torch.Tensor,
    kept_channels: List[int],
    d_eff: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten only kept channels. Order: pos-major, channel-major."""
    B, one, H, W, d = bits_B1HWd.shape
    assert B == 1 and one == 1
    L = int(H * W)
    bits = bits_B1HWd[:, 0].reshape(B, L, d)[:, :, :d_eff]  # [1,L,d_eff]
    p1 = p1_Bl_d[:, :, :d_eff]

    kept = torch.tensor(kept_channels, device=bits.device, dtype=torch.long)
    bits_k = bits.index_select(2, kept)  # [1,L,m]
    p1_k = p1.index_select(2, kept)

    bits_1d = bits_k.reshape(-1).detach().cpu().numpy().astype(np.uint8)
    p1_1d = p1_k.reshape(-1).detach().cpu().numpy().astype(np.float32)
    return bits_1d, p1_1d


def _flatten_for_spatial_mask(
    bits_B1HWd: torch.Tensor,
    p1_Bl_d: torch.Tensor,
    kept_pos: torch.Tensor,
    d_eff: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten only kept positions. Order: increasing flat index, then bit index."""
    B, one, H, W, d = bits_B1HWd.shape
    assert B == 1 and one == 1
    L = int(H * W)
    bits = bits_B1HWd[:, 0].reshape(B, L, d)[:, :, :d_eff]  # [1,L,d_eff]
    p1 = p1_Bl_d[:, :, :d_eff]

    kept_pos = kept_pos.to(bits.device).to(torch.long)
    bits_k = bits.index_select(1, kept_pos)  # [1,K,d_eff]
    p1_k = p1.index_select(1, kept_pos)

    bits_1d = bits_k.reshape(-1).detach().cpu().numpy().astype(np.uint8)
    p1_1d = p1_k.reshape(-1).detach().cpu().numpy().astype(np.float32)
    return bits_1d, p1_1d


def _save_payload_only(path: str, payload: bytes) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(payload)


def _load_payload_only(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


# -----------------------------
# Main codec (0-header)
# -----------------------------
@dataclass
class ARPCZeroHeaderFixed1024:
    vae: torch.nn.Module
    model: torch.nn.Module
    text_tokenizer: object
    text_encoder: torch.nn.Module

    device: str = "cuda"
    tlen: int = 512

    fixed_hw: Tuple[int, int] = (1024, 1024)
    k_transmit: int = 5
    seed: int = 0

    # ZERO-header strategy config (must be the same on encoder/decoder)
    mask_strategy: str = "none"
    mask_params: Optional[Dict[str, Any]] = None

    # active bits per scale (ZERO-header, shared)
    active_bits_spec: str = "all"  # "all" -> d for every scale; can also be single int like "16"

    # cache
    _cached_scale_schedule: Optional[List[Tuple[int, int, int]]] = None
    _cached_Kd: Optional[Tuple[int, int]] = None

    # -------------------------
    # text cond
    # -------------------------
    @torch.no_grad()
    def _prepare_text_cond_tuple(self, prompt: str):
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
    # shared defaults
    # -------------------------
    @torch.no_grad()
    def _get_shared_schedule_K_d(self) -> Tuple[List[Tuple[int, int, int]], int, int]:
        if self._cached_scale_schedule is not None and self._cached_Kd is not None:
            K, d = self._cached_Kd
            return self._cached_scale_schedule, K, d

        H, W = self.fixed_hw
        dummy = torch.zeros((1, 3, H, W), device=self.device, dtype=torch.float32)

        h, _, _ = self.vae.encode_for_raw_features(dummy, scale_schedule=None)
        if h.ndim == 5:
            _, _, T, hH, hW = h.shape
        else:
            T = 1
            _, _, hH, hW = h.shape

        from infinity.models.bsq_vae.multiscale_bsq import get_latent2scale_schedule

        schedule = get_latent2scale_schedule(T, int(hH), int(hW), mode=self.vae.quantizer.schedule_mode)
        scale_schedule = [(int(t), int(hh), int(ww)) for (t, hh, ww) in schedule]

        _, _, all_bit_indices, _, _, _ = self.vae.quantizer(h, scale_schedule=schedule)
        d = int(all_bit_indices[0].shape[-1])
        K = len(all_bit_indices)

        self._cached_scale_schedule = scale_schedule
        self._cached_Kd = (K, d)
        return scale_schedule, K, d

    # -------------------------
    # tokenizer
    # -------------------------
    @torch.no_grad()
    def encode_image_to_bitidx_and_schedule(self, img_B3HW: torch.Tensor):
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
            if bits.ndim == 5:
                bits = bits[:, 0:1]
            elif bits.ndim == 4:
                bits = bits.unsqueeze(1)
            bitidx_per_scale.append(bits.to(torch.uint8))

        d = int(bitidx_per_scale[0].shape[-1])
        scale_schedule_eff = [(1, int(b.shape[2]), int(b.shape[3])) for b in bitidx_per_scale]
        return bitidx_per_scale, scale_schedule_eff, d

    # -------------------------
    # logits for scale (your fork)
    # -------------------------
    @torch.no_grad()
    def _get_logits_for_scale(self, text_cond_tuple, scale_schedule, prev_bits: List[torch.Tensor], si: int, seed: int):
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
        return out[-1][si]

    # -------------------------
    # active bits (ZERO-header shared)
    # -------------------------
    def _active_bits_per_scale(self, K: int, d: int) -> List[int]:
        spec = (self.active_bits_spec or "all").strip().lower()
        if spec in ("all", "full", "d", "default"):
            return [int(d)] * int(K)

        try:
            v = int(spec)
            v = max(1, min(v, d))
            return [int(v)] * int(K)
        except Exception:
            # fallback
            return [int(d)] * int(K)

    # -------------------------
    # compress / decompress
    # -------------------------
    @torch.no_grad()
    def compress(self, img_B3HW: torch.Tensor, out_stream_path: str, prompt: str = "") -> bytes:
        H, W = int(img_B3HW.shape[2]), int(img_B3HW.shape[3])
        if (H, W) != tuple(self.fixed_hw):
            raise ValueError(f"Fixed-resolution codec expects {self.fixed_hw}, got {(H, W)}")

        bitidx_gt, scale_schedule, d = self.encode_image_to_bitidx_and_schedule(img_B3HW)
        K = len(scale_schedule)
        if self.k_transmit > K:
            raise ValueError(f"k_transmit={self.k_transmit} > K={K}")

        active_bits = self._active_bits_per_scale(K, d)
        text_cond_tuple = self._prepare_text_cond_tuple(prompt)

        enc = RangeEncoder()
        reconstructed_bits: List[torch.Tensor] = []
        mask_strategy = (self.mask_strategy or "none").lower()
        mask_params = dict(self.mask_params or {})

        for si in range(self.k_transmit):
            bits_gt = bitidx_gt[si].to(self.device)  # [1,1,Hs,Ws,d]
            Hs, Ws = int(bits_gt.shape[2]), int(bits_gt.shape[3])
            L = int(Hs * Ws)
            d_eff = int(active_bits[si])

            logits = self._get_logits_for_scale(text_cond_tuple, scale_schedule, reconstructed_bits, si, self.seed)
            p1 = _logits_to_p1(logits, bit_depth=d)  # [1,L,d]
            pred = (p1[:, :, :d_eff] >= 0.5).to(torch.uint8)  # [1,L,d_eff]

            if mask_strategy == "none":
                bits_1d = bits_gt[:, 0].reshape(1, L, d)[:, :, :d_eff].reshape(-1).detach().cpu().numpy().astype(np.uint8)
                p1_1d = p1[:, :, :d_eff].reshape(-1).detach().cpu().numpy().astype(np.float32)
                _encode_stream_bits(enc, bits_1d, p1_1d)

                # reconstruct = GT for active bits
                rec_bits = bits_gt[:, 0].reshape(1, L, d)[:, :, :d_eff].contiguous()

            elif mask_strategy == "entropy_channel":
                kept_channels = _select_channels_by_entropy(
                    p1_Bl_d=p1,
                    d_eff=d_eff,
                    keep_ratio=mask_params.get("keep_ratio", 0.5),
                    entropy_thr=mask_params.get("entropy_thr", None),
                )
                bits_1d, p1_1d = _flatten_for_channel_mask(bits_gt, p1, kept_channels, d_eff)
                _encode_stream_bits(enc, bits_1d, p1_1d)

                # reconstruct = MAP fill, overwrite kept channels with GT
                rec_bits = pred.clone()
                gt_flat = bits_gt[:, 0].reshape(1, L, d)[:, :, :d_eff]
                kept = torch.tensor(kept_channels, device=self.device, dtype=torch.long)
                rec_bits.index_copy_(2, kept, gt_flat.index_select(2, kept))

            elif mask_strategy == "entropy_scale":
                Hb = _binary_entropy(p1[:, :, :d_eff])  # [1,L,d_eff]
                Hmean = float(Hb.mean().item())          # in bits (0..1)

                min_r = float(mask_params.get("min_keep_ratio", 0.2))
                max_r = float(mask_params.get("max_keep_ratio", 1.0))
                gamma = float(mask_params.get("gamma", 1.0))
                r = min_r + (max_r - min_r) * (max(0.0, min(1.0, Hmean)) ** gamma)
                r = max(0.0, min(1.0, r))

                kept_channels = _select_channels_by_entropy(p1_Bl_d=p1, d_eff=d_eff, keep_ratio=r, entropy_thr=None)
                bits_1d, p1_1d = _flatten_for_channel_mask(bits_gt, p1, kept_channels, d_eff)
                _encode_stream_bits(enc, bits_1d, p1_1d)

                rec_bits = pred.clone()
                gt_flat = bits_gt[:, 0].reshape(1, L, d)[:, :, :d_eff]
                kept = torch.tensor(kept_channels, device=self.device, dtype=torch.long)
                rec_bits.index_copy_(2, kept, gt_flat.index_select(2, kept))

            elif mask_strategy == "entropy_spatial":
                kept_pos = _select_positions_by_entropy(
                    p1_Bl_d=p1,
                    H=Hs,
                    W=Ws,
                    d_eff=d_eff,
                    keep_ratio=mask_params.get("keep_ratio", 0.25),
                    entropy_thr=mask_params.get("entropy_thr", None),
                )
                bits_1d, p1_1d = _flatten_for_spatial_mask(bits_gt, p1, kept_pos, d_eff)
                _encode_stream_bits(enc, bits_1d, p1_1d)

                rec_bits = pred.clone()
                gt_flat = bits_gt[:, 0].reshape(1, L, d)[:, :, :d_eff]
                rec_bits.index_copy_(1, kept_pos.to(self.device), gt_flat.index_select(1, kept_pos.to(self.device)))

            else:
                raise ValueError(f"Unknown mask_strategy: {mask_strategy}")

            # Pack reconstructed bits into [1,1,H,W,d] (bits beyond d_eff are zero)
            rec_full = torch.zeros((1, 1, Hs, Ws, d), device=self.device, dtype=torch.uint8)
            rec_full[:, 0].reshape(1, L, d)[:, :, :d_eff] = rec_bits
            reconstructed_bits.append(rec_full)

        payload = enc.finish()
        _save_payload_only(out_stream_path, payload)
        return payload

    @torch.no_grad()
    def decompress(self, stream_path: str, out_path: str, prompt: str = "") -> torch.Tensor:
        # Get effective schedule/d via dummy encoding (shape-safe)
        H, W = self.fixed_hw
        dummy = torch.zeros((1, 3, H, W), device=self.device, dtype=torch.float32)
        _, scale_schedule, d = self.encode_image_to_bitidx_and_schedule(dummy)
        K = len(scale_schedule)
        if self.k_transmit > K:
            raise ValueError(f"k_transmit={self.k_transmit} > K={K}")

        active_bits = self._active_bits_per_scale(K, d)
        payload = _load_payload_only(stream_path)
        text_cond_tuple = self._prepare_text_cond_tuple(prompt)

        dec = RangeDecoder(payload)
        reconstructed_bits: List[torch.Tensor] = []
        # Keep-mask per decoded scale (spatial).
        # For entropy_spatial we will force ONLY transmitted positions, allowing VAR to predict the rest.
        keep_masks: List[torch.Tensor] = []
        mask_strategy = (self.mask_strategy or "none").lower()
        mask_params = dict(self.mask_params or {})

        for si in range(self.k_transmit):
            _, Hs, Ws = scale_schedule[si]
            L = int(Hs * Ws)
            d_eff = int(active_bits[si])

            logits = self._get_logits_for_scale(text_cond_tuple, scale_schedule, reconstructed_bits, si, self.seed)
            p1 = _logits_to_p1(logits, bit_depth=d)  # [1,L,d]
            pred = (p1[:, :, :d_eff] >= 0.5).to(torch.uint8)  # [1,L,d_eff]

            if mask_strategy == "none":
                p1_1d = p1[:, :, :d_eff].reshape(-1).detach().cpu().numpy().astype(np.float32)
                bits_1d = _decode_stream_bits(dec, p1_1d)
                rec_bits = torch.from_numpy(bits_1d.reshape(1, L, d_eff)).to(self.device).to(torch.uint8)
                keep_mask = torch.ones((1, int(Hs), int(Ws)), device=self.device, dtype=torch.bool)

            elif mask_strategy == "entropy_channel":
                kept_channels = _select_channels_by_entropy(
                    p1_Bl_d=p1,
                    d_eff=d_eff,
                    keep_ratio=mask_params.get("keep_ratio", 0.5),
                    entropy_thr=mask_params.get("entropy_thr", None),
                )
                kept = torch.tensor(kept_channels, device=self.device, dtype=torch.long)

                p1_stream = p1[:, :, :d_eff].index_select(2, kept).reshape(-1).detach().cpu().numpy().astype(np.float32)
                bits_stream = _decode_stream_bits(dec, p1_stream)  # [L*m]
                decoded = torch.from_numpy(bits_stream.reshape(1, L, len(kept_channels))).to(self.device).to(torch.uint8)

                rec_bits = pred.clone()
                keep_mask = torch.ones((1, int(Hs), int(Ws)), device=self.device, dtype=torch.bool)
                rec_bits.index_copy_(2, kept, decoded)

            elif mask_strategy == "entropy_scale":
                Hb = _binary_entropy(p1[:, :, :d_eff])
                Hmean = float(Hb.mean().item())

                min_r = float(mask_params.get("min_keep_ratio", 0.2))
                max_r = float(mask_params.get("max_keep_ratio", 1.0))
                gamma = float(mask_params.get("gamma", 1.0))
                r = min_r + (max_r - min_r) * (max(0.0, min(1.0, Hmean)) ** gamma)
                r = max(0.0, min(1.0, r))

                kept_channels = _select_channels_by_entropy(p1_Bl_d=p1, d_eff=d_eff, keep_ratio=r, entropy_thr=None)
                kept = torch.tensor(kept_channels, device=self.device, dtype=torch.long)

                p1_stream = p1[:, :, :d_eff].index_select(2, kept).reshape(-1).detach().cpu().numpy().astype(np.float32)
                bits_stream = _decode_stream_bits(dec, p1_stream)
                decoded = torch.from_numpy(bits_stream.reshape(1, L, len(kept_channels))).to(self.device).to(torch.uint8)

                rec_bits = pred.clone()
                keep_mask = torch.ones((1, int(Hs), int(Ws)), device=self.device, dtype=torch.bool)
                rec_bits.index_copy_(2, kept, decoded)

            elif mask_strategy == "entropy_spatial":
                kept_pos = _select_positions_by_entropy(
                    p1_Bl_d=p1,
                    H=Hs,
                    W=Ws,
                    d_eff=d_eff,
                    keep_ratio=mask_params.get("keep_ratio", 0.25),
                    entropy_thr=mask_params.get("entropy_thr", None),
                )
                p1_stream = p1[:, :, :d_eff].index_select(1, kept_pos.to(self.device)).reshape(-1).detach().cpu().numpy().astype(np.float32)
                bits_stream = _decode_stream_bits(dec, p1_stream)
                decoded = torch.from_numpy(bits_stream.reshape(1, int(kept_pos.numel()), d_eff)).to(self.device).to(torch.uint8)

                rec_bits = pred.clone()
                keep_mask = torch.zeros((1, int(Hs), int(Ws)), device=self.device, dtype=torch.bool)
                keep_mask.view(-1)[kept_pos.to(self.device)] = True
                rec_bits.index_copy_(1, kept_pos.to(self.device), decoded)

            else:
                raise ValueError(f"Unknown mask_strategy: {mask_strategy}")

            keep_masks.append(keep_mask)
            rec_full = torch.zeros((1, 1, Hs, Ws, d), device=self.device, dtype=torch.uint8)
            rec_full[:, 0].reshape(1, L, d)[:, :, :d_eff] = rec_bits
            reconstructed_bits.append(rec_full)

        # Final VAR reconstruction with forced early scales (ALL ones mask)
        forced_bitidx_per_scale: List[Optional[torch.Tensor]] = [None] * K
        forced_mask_per_scale: List[Optional[torch.Tensor]] = [None] * K
        for si in range(self.k_transmit):
            forced_bitidx_per_scale[si] = reconstructed_bits[si]
            _, Hs, Ws = scale_schedule[si]
            if mask_strategy == "entropy_spatial":
                # Force ONLY transmitted spatial positions; let VAR predict the rest.
                forced_mask_per_scale[si] = keep_masks[si]
            else:
                # For other strategies we force all positions of the early scales.
                forced_mask_per_scale[si] = torch.ones((1, int(Hs), int(Ws)), device=self.device, dtype=torch.bool)

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
            forced_bitidx_per_scale=forced_bitidx_per_scale,
            forced_mask_per_scale=forced_mask_per_scale,
            return_logits_per_scale=False,
        )

        img_u8 = out[2]
        if isinstance(img_u8, list):
            img_u8 = img_u8[-1]

        # Infinity returns uint8 (B,H,W,C) and may be BGR
        if isinstance(img_u8, torch.Tensor) and img_u8.ndim == 4 and img_u8.shape[-1] == 3:
            img_u8 = img_u8.flip(dims=(3,))

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        out_lower = out_path.lower()
        if out_lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
            Image.fromarray(img_u8[0].detach().cpu().numpy()).save(out_path)
        else:
            torch.save(img_u8.detach().cpu(), out_path)

        return img_u8


# Backward-compatible alias used by your CLI/eval scripts
ARPCNoneZeroHeaderFixed1024 = ARPCZeroHeaderFixed1024
