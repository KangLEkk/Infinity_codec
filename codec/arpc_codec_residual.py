"""ARPC: Autoregressive-based Progressive Coding for ultra-low bitrate image compression.

This codec builds on Infinity's bitwise VAR model as an entropy estimator.

Core (paper-aligned) idea
------------------------
- A BSQ-VAE tokenizer produces multi-scale **bitwise** tokens.
- For the first k scales, we losslessly entropy-code tokens using a binary arithmetic coder,
  where token probabilities come from the VAR model.
- Remaining scales are reconstructed via autoregressive inference.

Extension (no-training entropy masking)
--------------------------------------
We optionally *do not transmit* low-entropy parts (easy-to-predict) and let the VAR fill them
with deterministic argmax predictions.

Mask strategies:
- 'none'              : transmit all active bits (baseline ARPC)
- 'entropy_channel'   : keep top channels (bit-planes) by entropy
- 'entropy_scale'     : per-scale keep ratio derived from mean entropy (then channel selection)
- 'entropy_spatial'   : keep top spatial positions by entropy (transmit all active bits at kept positions)

Important correctness note
--------------------------
We must ensure the arithmetic coder is decodable:
  * Encoder and decoder compute identical probabilities and identical mask decisions.
  * Therefore, mask decisions are derived ONLY from VAR probabilities conditioned on
    already reconstructed previous scales.

This implementation supports batch size B=1 for simplicity.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from codec.arithmetic import RangeDecoder, RangeEncoder
from codec.arpc_bitstream import ARPCHeader, load_arpc_bitstream, save_arpc_bitstream


# -----------------------------
# Utilities
# -----------------------------

_EPS = 1e-12


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



def _enc_raw_bit(enc: RangeEncoder, bit: int) -> None:
    """Encode a raw bit with p=0.5 (i.e., no probabilistic compression)."""
    enc.encode_bit(int(bit), 0.5)


def _dec_raw_bit(dec: RangeDecoder) -> int:
    """Decode a raw bit with p=0.5."""
    return int(dec.decode_bit(0.5))


def _encode_uint_elias_gamma(enc: RangeEncoder, n: int) -> None:
    """Encode unsigned integer n>=1 using Elias gamma code (bitwise)."""
    assert n >= 1
    L = int(n.bit_length())
    # prefix: L-1 zeros
    for _ in range(L - 1):
        _enc_raw_bit(enc, 0)
    # write n in binary (L bits, MSB->LSB)
    for i in reversed(range(L)):
        _enc_raw_bit(enc, (n >> i) & 1)


def _decode_uint_elias_gamma(dec: RangeDecoder) -> int:
    """Decode Elias gamma code, returning n>=1."""
    zeros = 0
    while True:
        b = _dec_raw_bit(dec)
        if b == 1:
            break
        zeros += 1
        if zeros > 60:
            raise RuntimeError("Elias gamma decode failed: too many leading zeros")
    # We already consumed the leading '1' (MSB). Read remaining zeros bits.
    n = 1 << zeros
    for i in reversed(range(zeros)):
        bit = _dec_raw_bit(dec)
        n |= (bit << i)
    return int(n)


def _choose_rice_k_from_perr(p_err_1d: np.ndarray) -> int:
    """Heuristic Rice parameter from mean error prob. Deterministic on both sides."""
    pe = float(np.clip(np.mean(p_err_1d), _EPS, 1.0 - _EPS))
    mean_gap = (1.0 - pe) / pe  # expected zeros between ones (geometric)
    k = int(max(0.0, math.floor(math.log2(mean_gap + 1.0))))
    return int(max(0, min(12, k)))


def _encode_rice(enc: RangeEncoder, x: int, k: int) -> None:
    """Encode non-negative integer x using Rice(k) coding."""
    assert x >= 0 and k >= 0
    q = x >> k
    r = x & ((1 << k) - 1) if k > 0 else 0
    # unary for q: q zeros then one
    for _ in range(q):
        _enc_raw_bit(enc, 0)
    _enc_raw_bit(enc, 1)
    # remainder
    if k > 0:
        for i in reversed(range(k)):
            _enc_raw_bit(enc, (r >> i) & 1)


def _decode_rice(dec: RangeDecoder, k: int) -> int:
    """Decode Rice(k) coding, return non-negative integer."""
    assert k >= 0
    q = 0
    while True:
        b = _dec_raw_bit(dec)
        if b == 1:
            break
        q += 1
        if q > 1_000_000:
            raise RuntimeError("Rice decode failed: quotient too large")
    r = 0
    if k > 0:
        for _ in range(k):
            r = (r << 1) | _dec_raw_bit(dec)
    return (q << k) | r


def _encode_residual_sparse(enc: RangeEncoder, bits_1d: np.ndarray, p1_1d: np.ndarray) -> None:
    """Encode residual bits using sparse Rice gap coding (experimental).

    We encode:
      - K+1 using Elias gamma (so K can be 0)
      - K gaps (gap before each 1), Rice(k) coded
    Decoder knows stream length N = len(p1_1d), so tail zeros are implicit.
    """
    bits_1d = np.asarray(bits_1d, dtype=np.uint8).reshape(-1)
    p1_1d = np.asarray(p1_1d, dtype=np.float32).reshape(-1)
    N = int(bits_1d.size)
    if N == 0:
        _encode_uint_elias_gamma(enc, 1)  # K+1
        return

    pred = (p1_1d >= 0.5).astype(np.uint8)
    r = (bits_1d ^ pred).astype(np.uint8)

    K = int(r.sum())
    _encode_uint_elias_gamma(enc, K + 1)  # allow K=0

    if K == 0:
        return

    # deterministic Rice parameter from p_err distribution
    p_err = np.where(pred == 0, p1_1d, 1.0 - p1_1d).astype(np.float32)
    k = _choose_rice_k_from_perr(p_err)

    ones = np.flatnonzero(r)
    prev = -1
    for idx in ones.tolist():
        gap = int(idx - prev - 1)
        _encode_rice(enc, gap, k)
        prev = idx


def _decode_residual_sparse(dec: RangeDecoder, p1_1d: np.ndarray) -> np.ndarray:
    """Decode residual_sparse stream into original bits b (not residual)."""
    p1_1d = np.asarray(p1_1d, dtype=np.float32).reshape(-1)
    N = int(p1_1d.size)
    if N == 0:
        _ = _decode_uint_elias_gamma(dec)  # K+1
        return np.zeros((0,), dtype=np.uint8)

    pred = (p1_1d >= 0.5).astype(np.uint8)
    # deterministic Rice parameter (must match encoder)
    p_err = np.where(pred == 0, p1_1d, 1.0 - p1_1d).astype(np.float32)
    k = _choose_rice_k_from_perr(p_err)

    K_plus_1 = _decode_uint_elias_gamma(dec)
    K = int(K_plus_1 - 1)

    r = np.zeros((N,), dtype=np.uint8)
    if K <= 0:
        return (r ^ pred).astype(np.uint8)

    pos = -1
    for _ in range(K):
        gap = _decode_rice(dec, k)
        pos = pos + gap + 1
        if pos < 0 or pos >= N:
            # stream corruption; clamp and break to avoid crash
            break
        r[pos] = 1

    return (r ^ pred).astype(np.uint8)

def _encode_stream_bits(
    enc: RangeEncoder,
    bits_1d: np.ndarray,
    p1_1d: np.ndarray,
    token_coding: str = "direct",
) -> None:
    """Arithmetic-code a 1D bit stream.

    token_coding:
      - 'direct'        : encode b ~ Bernoulli(p1)
      - 'residual'      : encode r = b XOR argmax(p1), using p_err
      - 'residual_sparse': encode residual via sparse gap (Rice) coding (no extra training)
    """
    token_coding = (token_coding or "direct").lower()
    if token_coding not in ("direct", "residual", "residual_sparse"):
        raise ValueError(f"unknown token_coding: {token_coding}")

    bits_1d = np.asarray(bits_1d, dtype=np.uint8).reshape(-1)
    p1_1d = np.asarray(p1_1d, dtype=np.float32).reshape(-1)

    if token_coding == "direct":
        for b, p in zip(bits_1d.tolist(), p1_1d.tolist()):
            enc.encode_bit(int(b), float(p))
        return

    if token_coding == "residual_sparse":
        # Sparse Rice gap coding over residual r=b XOR argmax(p1)
        _encode_residual_sparse(enc, bits_1d, p1_1d)
        return

    # residual coding

    pred = (p1_1d >= 0.5).astype(np.uint8)
    r = (bits_1d ^ pred).astype(np.uint8)
    p_err = np.where(pred == 0, p1_1d, 1.0 - p1_1d).astype(np.float32)
    for rb, p in zip(r.tolist(), p_err.tolist()):
        enc.encode_bit(int(rb), float(p))


def _decode_stream_bits(
    dec: RangeDecoder,
    p1_1d: np.ndarray,
    token_coding: str = "direct",
) -> np.ndarray:
    """Arithmetic-decode a 1D bit stream.

    Returns decoded *original* bits b (not residual bits).
    """
    token_coding = (token_coding or "direct").lower()
    if token_coding not in ("direct", "residual", "residual_sparse"):
        raise ValueError(f"unknown token_coding: {token_coding}")

    p1_1d = np.asarray(p1_1d, dtype=np.float32).reshape(-1)

    if token_coding == "direct":
        out = np.zeros((p1_1d.size,), dtype=np.uint8)
        for i, p in enumerate(p1_1d.tolist()):
            out[i] = dec.decode_bit(float(p))
        return out

    if token_coding == "residual_sparse":
        return _decode_residual_sparse(dec, p1_1d)

    # residual coding

    pred = (p1_1d >= 0.5).astype(np.uint8)
    p_err = np.where(pred == 0, p1_1d, 1.0 - p1_1d).astype(np.float32)

    r = np.zeros((p1_1d.size,), dtype=np.uint8)
    for i, p in enumerate(p_err.tolist()):
        r[i] = dec.decode_bit(float(p))
    b = (r ^ pred).astype(np.uint8)
    return b



def _parse_active_bits_spec(spec: str, K: int, d: int) -> List[int]:
    """Parse active bits per scale.

    Supported:
      - 'default'                -> paper default for (K=13,d=16) else all d
      - '16,16,16,...'           -> explicit list length K
      - '8x3,12x5,16x5'          -> run-length encoding
      - single int '16'          -> apply to all scales
    """
    spec = (spec or "default").strip().lower()

    def paper_default() -> List[int]:
        if K == 13 and d == 16:
            return [8] * 4 + [12] * 5 + [16] * 4
        return [d] * K

    if spec in ("default", "paper"):
        return paper_default()

    if "x" in spec:
        out: List[int] = []
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        for p in parts:
            if "x" not in p:
                raise ValueError(f"bad run-length segment: {p}")
            v, n = p.split("x", 1)
            v = int(v)
            n = int(n)
            out.extend([v] * n)
        if len(out) != K:
            raise ValueError(f"active_bits_spec expands to {len(out)} != K={K}")
        return [max(1, min(int(v), d)) for v in out]

    if "," in spec:
        vals = [int(x.strip()) for x in spec.split(",") if x.strip()]
        if len(vals) != K:
            raise ValueError(f"active_bits_spec length {len(vals)} != K={K}")
        return [max(1, min(int(v), d)) for v in vals]

    # single integer
    try:
        v = int(spec)
        v = max(1, min(v, d))
        return [v] * K
    except Exception as e:
        raise ValueError(f"unrecognized active_bits_spec: {spec}") from e


def _topk_indices(scores_1d: torch.Tensor, k: int) -> torch.Tensor:
    """Deterministic-ish topk with tie-break using index."""
    L = scores_1d.numel()
    k = int(max(0, min(k, L)))
    if k == 0:
        return torch.zeros((0,), dtype=torch.long, device=scores_1d.device)
    idx = torch.arange(L, device=scores_1d.device, dtype=scores_1d.dtype)
    # tie-break: prefer smaller index
    adj = scores_1d + (-idx) * 1e-7
    _, top_idx = torch.topk(adj, k=k, largest=True, sorted=True)
    # to make order stable for coding, sort by flat index
    top_idx = torch.sort(top_idx.to(torch.long))[0]
    return top_idx


def _select_channels_by_entropy(
    p1_Bl_d: torch.Tensor,
    d_eff: int,
    keep_ratio: Optional[float] = None,
    entropy_thr: Optional[float] = None,
) -> List[int]:
    """Return kept channel indices within [0, d_eff)."""
    assert p1_Bl_d.ndim == 3 and p1_Bl_d.shape[0] == 1
    p1 = p1_Bl_d[:, :, :d_eff]
    H = _binary_entropy(p1)  # [1,L,d_eff]
    Hc = H.mean(dim=1)[0]  # [d_eff]

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

    # top-m channels by Hc with tie-break
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
    L = H * W
    p1 = p1_Bl_d[:, :, :d_eff]
    Hb = _binary_entropy(p1)  # [1,L,d_eff]
    Hpos = Hb.mean(dim=2)[0]  # [L]

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
    L = H * W
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
    L = H * W
    bits = bits_B1HWd[:, 0].reshape(B, L, d)[:, :, :d_eff]  # [1,L,d_eff]
    p1 = p1_Bl_d[:, :, :d_eff]

    kept_pos = kept_pos.to(bits.device).to(torch.long)
    bits_k = bits.index_select(1, kept_pos)  # [1,K,d_eff]
    p1_k = p1.index_select(1, kept_pos)

    bits_1d = bits_k.reshape(-1).detach().cpu().numpy().astype(np.uint8)
    p1_1d = p1_k.reshape(-1).detach().cpu().numpy().astype(np.float32)
    return bits_1d, p1_1d


# -----------------------------
# Main codec
# -----------------------------


@dataclass
class ARPCCodec:
    vae: torch.nn.Module
    model: torch.nn.Module
    text_tokenizer: object
    text_encoder: torch.nn.Module
    tlen: int = 512
    device: str = "cuda"

    # -------------------------
    # Text conditioning
    # -------------------------

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

        with torch.no_grad():
            feats = self.text_encoder(input_ids=input_ids, attention_mask=attn_mask)
            last_hidden = feats["last_hidden_state"] if isinstance(feats, dict) else feats.last_hidden_state

        lens = attn_mask.sum(dim=-1).int()  # [B]
        max_seqlen_k = int(lens.max().item())
        cu_seqlens_k = torch.zeros((lens.shape[0] + 1,), device=self.device, dtype=torch.int32)
        cu_seqlens_k[1:] = torch.cumsum(lens, dim=0)
        kv_compact = torch.cat([last_hidden[i, :l, :] for i, l in enumerate(lens.tolist())], dim=0)
        return kv_compact, lens, cu_seqlens_k, max_seqlen_k

    # -------------------------
    # Tokenization
    # -------------------------

    @torch.no_grad()
    def encode_image_to_bitidx_and_schedule(self, img_B3HW: torch.Tensor):
        """Encode image -> bit indices per scale, plus scale schedule."""
        assert img_B3HW.ndim == 4 and img_B3HW.shape[0] == 1, "B=1 only"
        img_B3HW = img_B3HW.to(self.device)

        h, hs, hs_mid = self.vae.encode_for_raw_features(img_B3HW, scale_schedule=None)
        if h.ndim == 5:
            _, _, T, H, W = h.shape
        else:
            T = 1
            _, _, H, W = h.shape

        from infinity.models.bsq_vae.multiscale_bsq import get_latent2scale_schedule

        schedule = get_latent2scale_schedule(T, H, W, mode=self.vae.quantizer.schedule_mode)
        z, all_indices, all_bit_indices, _, _, _ = self.vae.quantizer(h, scale_schedule=schedule)

        bitidx_per_scale: List[torch.Tensor] = []
        for bits in all_bit_indices:
            # bits: [B,T,h,w,d]
            if bits.ndim == 5:
                bits = bits[:, 0:1]
            elif bits.ndim == 4:
                bits = bits.unsqueeze(1)
            bitidx_per_scale.append(bits.to(torch.uint8))

        d = int(bitidx_per_scale[0].shape[-1])
        scale_schedule = [(1, int(b.shape[2]), int(b.shape[3])) for b in bitidx_per_scale]
        return bitidx_per_scale, scale_schedule, d

    # -------------------------
    # VAR logits per scale
    # -------------------------

    @torch.no_grad()
    def _get_logits_for_scale(self, text_cond_tuple, scale_schedule, prev_bits: List[torch.Tensor], si: int, seed: int):
        """Get logits for scale si conditioned on already reconstructed prev_bits (0..si-1)."""
        cfg_list = [1.0] * len(scale_schedule)
        tau_list = [1.0] * len(scale_schedule)

        gt_list = list(prev_bits)
        while len(gt_list) <= si:
            _, H, W = scale_schedule[len(gt_list)]
            # bit depth = 32（优先从模型读取）
            d = int(getattr(self.model, "codebook_dim", 32))

            dummy = torch.zeros((1, 1, H, W, d), device=self.device, dtype=torch.uint8)
            gt_list.append(dummy)
        out = self.model.autoregressive_infer_cfg(
            vae=self.vae,
            scale_schedule=scale_schedule,
            label_B_or_BLT=text_cond_tuple,
            B=1,
            g_seed=int(seed),
            cfg_sc=3,
            cfg_list=cfg_list,
            tau_list=tau_list,
            top_k=0,
            top_p=0.0,
            returns_vemb=1,
            gumbel=0,
            ret_img=False,
            trunk_scale=int(si + 1),
            # gt_leak=int(si),
            # gt_ls_Bl=prev_bits,
            gt_leak=int(si + 1),
            gt_ls_Bl=gt_list,
            inference_mode=True,
            sampling_per_bits=1,
            vae_type=1,
            return_logits_per_scale=True,
            do_sample=False
        )
        logits = out[-1][si]
        return logits

    # -------------------------
    # Public API
    # -------------------------

    @torch.no_grad()
    def compress(
        self,
        img_B3HW: torch.Tensor,
        prompt: str,
        out_path: str,
        k_transmit: int,
        active_bits_spec: str = "default",
        mask_strategy: str = "none",
        mask_params: Optional[Dict] = None,
        token_coding: str = "direct",
        seed: int = 0,
    ) -> ARPCHeader:
        """Compress a single image to an ARPC bitstream."""
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        bitidx_gt, scale_schedule, d = self.encode_image_to_bitidx_and_schedule(img_B3HW)
        # print(f"dimention: {d}")
        K = len(bitidx_gt)
        k_transmit = int(max(1, min(int(k_transmit), K)))

        active_bits = _parse_active_bits_spec(active_bits_spec, K=K, d=d)

        mask_strategy = (mask_strategy or "none").lower()
        mask_params = dict(mask_params or {})

        token_coding = str(mask_params.get("token_coding", token_coding or "direct")).lower()
        if token_coding not in ("direct", "residual", "residual_sparse"):
            raise ValueError(f"Unknown token_coding: {token_coding}")
        # persist into header for decoder
        mask_params["token_coding"] = token_coding

        text_cond_tuple = self._prepare_text_cond_tuple(prompt)

        enc = RangeEncoder()
        reconstructed_bits: List[torch.Tensor] = []

        for si in range(k_transmit):
            bits_gt = bitidx_gt[si].to(self.device)  # [1,1,H,W,d]
            H, W = int(bits_gt.shape[2]), int(bits_gt.shape[3])
            L = H * W
            d_eff = int(active_bits[si])

            logits = self._get_logits_for_scale(text_cond_tuple, scale_schedule, reconstructed_bits, si, seed)
            p1 = _logits_to_p1(logits, bit_depth=d)  # [1,L,d]

            # argmax prediction for fill
            pred = (p1[:, :, :d_eff] >= 0.5).to(torch.uint8)  # [1,L,d_eff]

            # Choose which tokens to transmit
            if mask_strategy in ("none", "gm"):
                # transmit all active bits
                kept_channels = list(range(d_eff))
                bits_1d, p1_1d = _flatten_for_channel_mask(bits_gt, p1, kept_channels, d_eff)

                # reconstruct = GT for active bits
                rec_bits = pred.clone()
                rec_bits[:, :, :] = bits_gt[:, 0].reshape(1, L, d)[:, :, :d_eff]

            elif mask_strategy == "entropy_channel":
                kept_channels = _select_channels_by_entropy(
                    p1_Bl_d=p1,
                    d_eff=d_eff,
                    keep_ratio=mask_params.get("keep_ratio", 0.5),
                    entropy_thr=mask_params.get("entropy_thr", None),
                )
                bits_1d, p1_1d = _flatten_for_channel_mask(bits_gt, p1, kept_channels, d_eff)

                # reconstruct = argmax, but overwrite kept channels with GT
                rec_bits = pred.clone()
                gt_flat = bits_gt[:, 0].reshape(1, L, d)[:, :, :d_eff]
                if kept_channels:
                    kept = torch.tensor(kept_channels, device=gt_flat.device, dtype=torch.long)
                    rec_bits.index_copy_(2, kept, gt_flat.index_select(2, kept))

            elif mask_strategy == "entropy_scale":
                # derive keep_ratio from mean entropy
                Hb = _binary_entropy(p1[:, :, :d_eff])  # [1,L,d_eff]
                Hmean = float(Hb.mean().item())  # in bits (0..1)
                min_r = float(mask_params.get("min_keep_ratio", 0.2))
                max_r = float(mask_params.get("max_keep_ratio", 1.0))
                gamma = float(mask_params.get("gamma", 1.0))
                # normalize by max entropy=1
                r = min_r + (max_r - min_r) * (max(0.0, min(1.0, Hmean)) ** gamma)
                r = max(0.0, min(1.0, r))

                kept_channels = _select_channels_by_entropy(p1_Bl_d=p1, d_eff=d_eff, keep_ratio=r, entropy_thr=None)
                bits_1d, p1_1d = _flatten_for_channel_mask(bits_gt, p1, kept_channels, d_eff)

                rec_bits = pred.clone()
                gt_flat = bits_gt[:, 0].reshape(1, L, d)[:, :, :d_eff]
                if kept_channels:
                    kept = torch.tensor(kept_channels, device=gt_flat.device, dtype=torch.long)
                    rec_bits.index_copy_(2, kept, gt_flat.index_select(2, kept))

            elif mask_strategy == "entropy_spatial":
                kept_pos = _select_positions_by_entropy(
                    p1_Bl_d=p1,
                    H=H,
                    W=W,
                    d_eff=d_eff,
                    keep_ratio=mask_params.get("keep_ratio", 0.25),
                    entropy_thr=mask_params.get("entropy_thr", None),
                )
                bits_1d, p1_1d = _flatten_for_spatial_mask(bits_gt, p1, kept_pos, d_eff)

                rec_bits = pred.clone()
                gt_flat = bits_gt[:, 0].reshape(1, L, d)[:, :, :d_eff]
                # overwrite all bits at kept positions with GT
                rec_bits.index_copy_(1, kept_pos.to(rec_bits.device), gt_flat.index_select(1, kept_pos.to(gt_flat.device)))

            else:
                raise ValueError(f"Unknown mask_strategy: {mask_strategy}")

            # Arithmetic encode the selected stream
            _encode_stream_bits(enc, bits_1d, p1_1d, token_coding=token_coding)

            # Build reconstructed bits tensor [1,1,H,W,d]
            rec_full = torch.zeros((1, 1, H, W, d), device=self.device, dtype=torch.uint8)
            rec_full[:, 0].reshape(1, L, d)[:, :, :d_eff] = rec_bits
            # bits beyond d_eff are kept at 0
            reconstructed_bits.append(rec_full)

        payload = enc.finish()

        header = ARPCHeader(
            version=2,
            num_scales=int(K),
            bit_depth=int(d),
            k_transmit=int(k_transmit),
            seed=int(seed),
            prompt=prompt or "",
            active_bits=[int(x) for x in active_bits],
            scale_shapes=[(int(s[1]), int(s[2])) for s in scale_schedule],
            mask_strategy=str(mask_strategy),
            mask_params=mask_params,
        )
        header.validate()
        save_arpc_bitstream(out_path, header, payload)
        return header

    @torch.no_grad()
    def decompress(
        self,
        stream_path: str,
        out_path: str,
        prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Decompress ARPC bitstream -> reconstructed image tensor (uint8 B,H,W,3)."""
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        header, payload = load_arpc_bitstream(stream_path)
        header.validate()

        K = int(header.num_scales)
        d = int(header.bit_depth)
        k_transmit = int(header.k_transmit)
        active_bits = [int(x) for x in header.active_bits]
        scale_shapes = header.scale_shapes
        mask_strategy = (header.mask_strategy or "none").lower()
        mask_params = dict(header.mask_params or {})

        token_coding = str(mask_params.get("token_coding", "direct")).lower()
        if token_coding not in ("direct", "residual", "residual_sparse"):
            token_coding = "direct"

        prompt = header.prompt if prompt is None else prompt
        seed = int(header.seed if seed is None else seed)

        scale_schedule = [(1, int(h), int(w)) for (h, w) in scale_shapes]
        text_cond_tuple = self._prepare_text_cond_tuple(prompt)

        dec = RangeDecoder(payload)
        reconstructed_bits: List[torch.Tensor] = []

        for si in range(k_transmit):
            H, W = scale_shapes[si]
            L = int(H * W)
            d_eff = int(active_bits[si])

            logits = self._get_logits_for_scale(text_cond_tuple, scale_schedule, reconstructed_bits, si, seed)
            p1 = _logits_to_p1(logits, bit_depth=d)  # [1,L,d]

            pred = (p1[:, :, :d_eff] >= 0.5).to(torch.uint8)  # [1,L,d_eff]

            # decode kept stream and merge with argmax fill
            if mask_strategy in ("none", "gm"):
                kept_channels = list(range(d_eff))
                # decode all
                m = len(kept_channels)
                bits_stream = np.zeros((L * m,), dtype=np.uint8)
                p1_stream = p1[:, :, :d_eff].reshape(-1).detach().cpu().numpy().astype(np.float32)
                bits_stream = _decode_stream_bits(dec, p1_stream, token_coding=token_coding)
                rec_bits = torch.from_numpy(bits_stream.reshape(1, L, m)).to(self.device).to(torch.uint8)

            elif mask_strategy == "entropy_channel":
                kept_channels = _select_channels_by_entropy(
                    p1_Bl_d=p1,
                    d_eff=d_eff,
                    keep_ratio=mask_params.get("keep_ratio", 0.5),
                    entropy_thr=mask_params.get("entropy_thr", None),
                )
                m = len(kept_channels)
                bits_stream = np.zeros((L * m,), dtype=np.uint8)
                # probs in the same order as encoding: pos-major then kept_channels
                p1_stream = p1[:, :, :d_eff].index_select(2, torch.tensor(kept_channels, device=self.device)).reshape(-1)
                p1_stream = p1_stream.detach().cpu().numpy().astype(np.float32)
                bits_stream = _decode_stream_bits(dec, p1_stream, token_coding=token_coding)

                rec_bits = pred.clone()  # start from argmax
                rec_bits.index_copy_(2, torch.tensor(kept_channels, device=self.device), torch.from_numpy(bits_stream.reshape(1, L, m)).to(self.device))

            elif mask_strategy == "entropy_scale":
                Hb = _binary_entropy(p1[:, :, :d_eff])
                Hmean = float(Hb.mean().item())
                min_r = float(mask_params.get("min_keep_ratio", 0.2))
                max_r = float(mask_params.get("max_keep_ratio", 1.0))
                gamma = float(mask_params.get("gamma", 1.0))
                r = min_r + (max_r - min_r) * (max(0.0, min(1.0, Hmean)) ** gamma)
                r = max(0.0, min(1.0, r))

                kept_channels = _select_channels_by_entropy(p1_Bl_d=p1, d_eff=d_eff, keep_ratio=r, entropy_thr=None)
                m = len(kept_channels)
                bits_stream = np.zeros((L * m,), dtype=np.uint8)
                p1_stream = p1[:, :, :d_eff].index_select(2, torch.tensor(kept_channels, device=self.device)).reshape(-1)
                p1_stream = p1_stream.detach().cpu().numpy().astype(np.float32)
                bits_stream = _decode_stream_bits(dec, p1_stream, token_coding=token_coding)

                rec_bits = pred.clone()
                rec_bits.index_copy_(2, torch.tensor(kept_channels, device=self.device), torch.from_numpy(bits_stream.reshape(1, L, m)).to(self.device))

            elif mask_strategy == "entropy_spatial":
                kept_pos = _select_positions_by_entropy(
                    p1_Bl_d=p1,
                    H=H,
                    W=W,
                    d_eff=d_eff,
                    keep_ratio=mask_params.get("keep_ratio", 0.25),
                    entropy_thr=mask_params.get("entropy_thr", None),
                )
                Kpos = int(kept_pos.numel())
                bits_stream = np.zeros((Kpos * d_eff,), dtype=np.uint8)
                # probs order: kept positions (sorted), then bits 0..d_eff-1
                p1_stream = p1[:, :, :d_eff].index_select(1, kept_pos.to(self.device)).reshape(-1)
                p1_stream = p1_stream.detach().cpu().numpy().astype(np.float32)
                bits_stream = _decode_stream_bits(dec, p1_stream, token_coding=token_coding)

                rec_bits = pred.clone()
                decoded_bits = torch.from_numpy(bits_stream.reshape(1, Kpos, d_eff)).to(self.device).to(torch.uint8)
                rec_bits.index_copy_(1, kept_pos.to(self.device), decoded_bits)

            else:
                raise ValueError(f"Unknown mask_strategy in stream: {mask_strategy}")

            # Ensure shape [1,L,d_eff]
            if rec_bits.ndim == 3 and rec_bits.shape[2] == d_eff:
                pass
            else:
                # baseline decode path returns rec_bits of shape [1,L,d_eff]
                raise RuntimeError(f"bad rec_bits shape: {tuple(rec_bits.shape)}")

            rec_full = torch.zeros((1, 1, H, W, d), device=self.device, dtype=torch.uint8)
            rec_full[:, 0].reshape(1, L, d)[:, :, :d_eff] = rec_bits
            reconstructed_bits.append(rec_full)

        # Full decode with forced early scales
        forced_bits: List[Optional[torch.Tensor]] = [None] * K
        forced_mask: List[Optional[torch.Tensor]] = [None] * K
        for si in range(k_transmit):
            forced_bits[si] = reconstructed_bits[si]
            h, w = scale_shapes[si]
            forced_mask[si] = torch.ones((1, h, w), device=self.device, dtype=torch.bool)

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

        out_lower = out_path.lower()
        if out_lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
            Image.fromarray(img_u8[0].detach().cpu().numpy()).save(out_path)
        else:
            torch.save(img_u8.detach().cpu(), out_path)

        return img_u8
