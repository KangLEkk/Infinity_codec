"""Infinity + entropy-map progressive token transmission (B2-style).

This module is intended to be used *inside* the official Infinity repository.
It relies on Infinity's `autoregressive_infer_cfg`.

Key additions assumed:
- `autoregressive_infer_cfg(..., forced_bitidx_per_scale=..., forced_mask_per_scale=..., return_logits_per_scale=True)`
  (see `infinity_patched.py` in this folder for a reference patch).

This codec works with Infinity's bit-label tokenizer (BSQ) and packs per-position bits to uint64 for compact storage.

Usage sketch (encoder):
    # 1) Encode image -> gt_bitidx_per_scale using VAE
    # 2) For each scale s, run infer_cfg up to s to get logits and compute entropy map
    # 3) Select top-k positions by entropy and transmit their packed tokens

Usage sketch (decoder):
    # 1) Read sparse tokens -> build forced_bitidx/mask per scale
    # 2) Run infer_cfg once to sample missing tokens and overwrite forced positions
    # 3) Decode final codes via VAE

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

# Reuse the generic utilities from the earlier standalone project
from codec.bitstream import BitStreamReader, BitStreamWriter
from codec.entropy import entropy_map_from_logits
from codec.selection import select_topk_entropy


@torch.no_grad()
def pack_bits_to_uint(bits: torch.Tensor) -> torch.Tensor:
    """Pack last-dim bits (0/1) into uint64.

    bits: (..., d) int/bool
    returns: (...) uint64
    """
    if bits.dtype not in (torch.uint8, torch.int64, torch.int32, torch.bool, torch.uint64):
        bits = bits.to(torch.uint64)
    else:
        bits = bits.to(torch.uint64)    
    d = bits.shape[-1]
    if d > 64:
        raise ValueError(f"Bit depth {d} exceeds uint64 packing capacity.")
    # Use MSB-first packing for consistency
    # weights = (2 ** torch.arange(d - 1, -1, -1, device=bits.device, dtype=torch.int64))
    shifts = torch.arange(d - 1, -1, -1, device=bits.device, dtype=torch.uint64)
    weights = torch.bitwise_left_shift(torch.ones_like(shifts), shifts)
    return (bits * weights).sum(dim=-1).to(torch.uint64)


@torch.no_grad()
def unpack_uint_to_bits(vals: torch.Tensor, d: int) -> torch.Tensor:
    """Unpack uint64 into bits (0/1) along last dimension."""
    # vals64 = vals.to(torch.int64)
    # weights = (2 ** torch.arange(d - 1, -1, -1, device=vals.device, dtype=torch.int64))
    # bits = (vals64.unsqueeze(-1) // weights) % 2
    if d > 64:
        raise ValueError(f"Bit depth {d} exceeds uint64 packing capacity.")
    vals64 = vals.to(torch.uint64)
    shifts = torch.arange(d - 1, -1, -1, device=vals.device, dtype=torch.uint64)
    bits = torch.bitwise_and(torch.bitwise_right_shift(vals64.unsqueeze(-1), shifts), 1)
    return bits.to(torch.uint8)


@dataclass
class ProgressiveB2Config:
    """Transmission policy.

    keep_ratios_per_scale: list of ratios in (0,1], length = num_scales.
        - scale 0 is typically 1.0 (send all)
        - later scales: e.g. [1.0, 0.1, 0.05, 0.02, ...]

    topk_min_per_scale: minimal number of tokens to send at each scale (avoid empty scale)
    """

    keep_ratios_per_scale: List[float]
    topk_min_per_scale: int = 1


class InfinityEntropyProgressiveCodec:
    """Progressive token transmission using Infinity's entropy map."""

    def __init__(self, model, vae, scale_schedule: List[Tuple[int, int, int]]):
        self.model = model
        self.vae = vae
        self.scale_schedule = scale_schedule

    @torch.no_grad()
    def encode_image_to_bitidx(self, img_B3HW: torch.Tensor):
        """Encode an RGB image to bit-label indices for each scale.

        Returns:
          bitidx_per_scale: list of tensors [B, 1, h, w, d] (uint8, 0/1)
          d: bit depth
        """
        _, _, all_indices, all_bit_indices, _, _ = self.vae.encode(img_B3HW, scale_schedule=self.scale_schedule)

        bitidx_per_scale: List[torch.Tensor] = []
        for s, bits in enumerate(all_bit_indices):
            # bits: [B, T, h, w, d] for images T=1
            if bits.ndim == 5:
                bits = bits[:, 0:1]  # keep T=1
            elif bits.ndim == 4:
                bits = bits.unsqueeze(1)
            else:
                raise ValueError(f"Unexpected bits shape at scale {s}: {tuple(bits.shape)}")
            bitidx_per_scale.append(bits.to(torch.uint8))

        d = int(bitidx_per_scale[0].shape[-1])
        return bitidx_per_scale, d

    @torch.no_grad()
    def _entropy_map_for_scale(
        self,
        label_B_or_BLT,
        scale_idx: int,
        gt_bitidx_per_scale: List[torch.Tensor],
        cfg_list: Optional[List[float]] = None,
        tau_list: Optional[List[float]] = None,
        top_k: int = 0,
        top_p: float = 0.0,
        g_seed: Optional[int] = 123,
    ) -> torch.Tensor:
        """Compute entropy map for a given scale using Infinity logits.

        Returns entropy map [H, W] (float32) for B=1.
        """
        num_scales = len(self.scale_schedule)
        if cfg_list is None:
            cfg_list = [1.0] * num_scales
        if tau_list is None:
            tau_list = [1.0] * num_scales

        # Force earlier scales to ground-truth so the entropy is computed under correct context.
        gt_leak = scale_idx  # use gt for [0, scale_idx-1]

        out = self.model.autoregressive_infer_cfg(
            vae=self.vae,
            scale_schedule=self.scale_schedule,
            label_B_or_BLT=label_B_or_BLT,
            B=1,
            negative_label_B_or_BLT=None,
            g_seed=g_seed,
            cfg_list=cfg_list,
            tau_list=tau_list,
            top_k=top_k,
            top_p=top_p,
            returns_vemb=1,
            vae_type=1,
            ret_img=False,
            trunk_scale=scale_idx + 1,
            gt_leak=gt_leak,
            gt_ls_Bl=gt_bitidx_per_scale,
            inference_mode=True,
            sampling_per_bits=1,
            return_logits_per_scale=True,
        )
        # patched method returns 4-tuple
        if len(out) != 4:
            raise RuntimeError(
                "Infinity.autoregressive_infer_cfg did not return logits_per_scale. "
                "Please apply the patch from official_patch/infinity_patched.py"
            )
        _ret, _idxs, _imgs, logits_per_scale = out
        logits = logits_per_scale[-1]  # scale_idx

        # Compute entropy per position. For Infinity bit-label: logits last dim = 2*d.
        bits = gt_bitidx_per_scale[scale_idx]
        H = int(bits.shape[2])
        W = int(bits.shape[3])
        # entropy_map_from_logits expects [B, L, V]
        ent_BHW = entropy_map_from_logits(logits, H=H, W=W, bitlabel=True)
        return ent_BHW[0]

    @torch.no_grad()
    def compress(
        self,
        img_B3HW: torch.Tensor,
        label_B_or_BLT,
        out_path: str,
        cfg: ProgressiveB2Config,
        g_seed: Optional[int] = 123,
    ):
        """Compress image to a bitstream.

        Note: this writes *sparse* token maps for scales >=1.
        """
        bitidx_per_scale, d = self.encode_image_to_bitidx(img_B3HW)
        num_scales = len(bitidx_per_scale)
        assert len(cfg.keep_ratios_per_scale) == num_scales

        writer = BitStreamWriter(out_path)
        writer.write_header(num_scales=num_scales)

        for s in range(num_scales):
            bits = bitidx_per_scale[s]  # [1,1,H,W,d]
            if bits.shape[0] != 1 or bits.shape[1] != 1:
                raise ValueError(f"compress expects B=1,T=1, got {tuple(bits.shape[:2])} at scale {s}.")
            H, W = int(bits.shape[2]), int(bits.shape[3])

            tokens_uint = pack_bits_to_uint(bits[0, 0])  # [H,W] uint64

            if s == 0 or cfg.keep_ratios_per_scale[s] >= 0.9999:
                # Send full token map
                writer.write_token_map(scale_idx=s, token_map=tokens_uint.cpu().numpy())
                continue

            # Compute entropy map for scale s
            ent_hw = self._entropy_map_for_scale(
                label_B_or_BLT=label_B_or_BLT,
                scale_idx=s,
                gt_bitidx_per_scale=bitidx_per_scale,
                g_seed=g_seed,
            )

            topk = max(cfg.topk_min_per_scale, int(math.ceil(cfg.keep_ratios_per_scale[s] * H * W)))
            pos, vals = select_topk_entropy(ent_hw, tokens_uint, topk=topk)

            writer.write_sparse_tokens(
                scale_idx=s,
                positions=pos.cpu().numpy(),
                values=vals.cpu().numpy(),
                shape=(H, W),
            )

        writer.close()

    @torch.no_grad()
    def decompress(
        self,
        bitstream_path: str,
        label_B_or_BLT,
        g_seed: Optional[int] = 123,
        cfg_list: Optional[List[float]] = None,
        tau_list: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """Decompress bitstream to an image tensor (uint8 Bx3xHxW)."""

        reader = BitStreamReader(bitstream_path)
        header = reader.read_header()
        num_scales = header.num_scales

        forced_bits_list = []
        forced_mask_list = []

        for s in range(num_scales):
            payload = reader.read_next()
            H, W = payload.shape
            d_guess = None

            if payload.is_dense:
                tokens_uint = torch.from_numpy(payload.token_map).to(torch.uint64)
                mask = torch.ones((H, W), dtype=torch.bool)
            else:
                tokens_uint = torch.zeros((H, W), dtype=torch.uint64)
                mask = torch.zeros((H, W), dtype=torch.bool)
                pos = torch.from_numpy(payload.positions).to(torch.int64)
                vals = torch.from_numpy(payload.values).to(torch.uint64)
                tokens_uint.view(-1)[pos] = vals
                mask.view(-1)[pos] = True

            # Infer bit-depth d from scale_schedule and model settings.
            # Infinity uses a fixed d; we can retrieve it from model attributes if present.
            # Fallback: assume d=16.
            d = getattr(self.vae.quantizer.lfq, 'codebook_dim', None)
            if d is None:
                # last dimension in scale schedule doesn't contain d; fallback.
                d = 16

            bits_hw_d = unpack_uint_to_bits(tokens_uint, d=d)  # [H,W,d]
            forced_bits_list.append(bits_hw_d.unsqueeze(0).unsqueeze(0))  # [1,1,H,W,d]
            # forced_mask_list.append(mask.unsqueeze(0))  # [1,H,W]
            forced_mask_list.append(mask.unsqueeze(0).unsqueeze(0))  # [1,1,H,W]

        reader.close()

        if cfg_list is None:
            cfg_list = [1.0] * len(self.scale_schedule)
        if tau_list is None:
            tau_list = [1.0] * len(self.scale_schedule)

        if hasattr(self.model, "parameters"):
            device = next(self.model.parameters()).device
            forced_bits_list = [bits.to(device) for bits in forced_bits_list]
            forced_mask_list = [mask.to(device) for mask in forced_mask_list]

        out = self.model.autoregressive_infer_cfg(
            vae=self.vae,
            scale_schedule=self.scale_schedule,
            label_B_or_BLT=label_B_or_BLT,
            B=1,
            g_seed=g_seed,
            cfg_list=cfg_list,
            tau_list=tau_list,
            top_k=0,
            top_p=0.0,
            returns_vemb=1,
            vae_type=1,
            ret_img=True,
            trunk_scale=len(self.scale_schedule),
            gt_leak=0,
            gt_ls_Bl=None,
            inference_mode=True,
            sampling_per_bits=1,
            forced_bitidx_per_scale=forced_bits_list,
            forced_mask_per_scale=forced_mask_list,
            return_logits_per_scale=False,
        )
        # out: (ret, idx_list, img)
        ret, idxs, img = out[:3]
        # img returned as uint8 BHWC maybe; convert to tensor Bx3xHxW
        if isinstance(img, list):
            raise RuntimeError('Unexpected img list output')
        if isinstance(img, torch.Tensor):
            if img.ndim == 4 and img.shape[-1] == 3:
                img_t = img.permute(0, 3, 1, 2).contiguous()
            else:
                img_t = img
        else:
            import numpy as np
            if isinstance(img, np.ndarray) and img.ndim == 4 and img.shape[-1] == 3:
                img_t = torch.from_numpy(img).permute(0, 3, 1, 2).contiguous()
            else:
                raise RuntimeError(f'Unknown img type: {type(img)}')

        return img_t
