import os
os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import re
import zlib
import json
import time
import math
import argparse
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.append("/workspace/Infinity_codec")

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
import lpips
from pytorch_msssim import ms_ssim
from DISTS_pytorch import DISTS

from tools.run_infinity_refiner import *
from compression.util import *
from utils.arithmeticcoding import compress_to_bit_list, decompress_from_bit_list
from infinity.models.same_scale_refiner import SameScaleRefinementHead, compute_normalized_uncertainty, compute_neighborhood_context


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =========================
# Prompt bits helpers
# =========================
def _encode_prompt_ids_t5(tokenizer, prompt: str, max_len: int = 512):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len, add_special_tokens=True)
    return list(map(int, enc["input_ids"][0].tolist()))


def _arith_encode_ids_packet(ids):
    import torchac

    if ids is None or len(ids) == 0:
        return {"payload": b"", "cdf_1d": torch.tensor([0.0, 1.0], dtype=torch.float32), "unique_chars": [], "char_len": 0, "ids_len": 0, "bits": 0}

    text = ",".join(map(str, ids))
    char_freq = {}
    for ch in text:
        if ch.isdigit() or ch == ",":
            char_freq[ch] = char_freq.get(ch, 0) + 1

    unique_chars = sorted(char_freq.keys())
    total = sum(char_freq.values())
    prob = [char_freq[c] / total for c in unique_chars]

    cdf_1d = torch.zeros(len(unique_chars) + 1, dtype=torch.float32)
    cdf_1d[1:] = torch.cumsum(torch.tensor(prob, dtype=torch.float32), dim=0)
    cdf_1d[-1] = 1.0

    L = len(text)
    cdf = cdf_1d.view(1, 1, -1).expand(1, L, -1).contiguous()
    sym = torch.tensor([unique_chars.index(ch) for ch in text], dtype=torch.int16).view(1, -1)
    payload = torchac.encode_float_cdf(cdf, sym, check_input_bounds=True)
    return {
        "payload": payload,
        "cdf_1d": cdf_1d,
        "unique_chars": unique_chars,
        "char_len": L,
        "ids_len": len(ids),
        "bits": len(payload) * 8,
    }


def _arith_decode_ids_packet(packet):
    import torchac

    if packet["char_len"] == 0:
        return []
    cdf = packet["cdf_1d"].view(1, 1, -1).expand(1, packet["char_len"], -1).contiguous()
    decoded_sym = torchac.decode_float_cdf(cdf, packet["payload"]).view(-1).tolist()
    text = "".join(packet["unique_chars"][int(i)] for i in decoded_sym)
    return [int(x) for x in text.split(",") if x != ""]


def _decode_prompt_text_from_ids_t5(tokenizer, ids):
    if ids is None or len(ids) == 0:
        return ""
    try:
        return tokenizer.decode(ids, skip_special_tokens=True).strip()
    except TypeError:
        return tokenizer.decode(ids)


def _prompt_zlib_bytes(prompt: str) -> int:
    return len(zlib.compress(prompt.encode("utf-8", errors="ignore")))


# =========================
# Bit-logit helpers
# =========================
def pair_logits_to_single_bit_logits(logits_BLD2: torch.Tensor) -> torch.Tensor:
    return logits_BLD2[..., 1] - logits_BLD2[..., 0]


def single_bit_logits_to_pair_logits(bit_logits_BLD: torch.Tensor) -> torch.Tensor:
    return torch.stack((-0.5 * bit_logits_BLD, 0.5 * bit_logits_BLD), dim=-1)


def sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:
    B, l, V = logits_BlV.shape
    if top_k > 0:
        top_k = min(top_k, V)
        idx_to_remove = logits_BlV < logits_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        logits_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), -torch.inf)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(logits_BlV.softmax(dim=-1).view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)


# =========================
# Model/refiner loading
# =========================
def _collect_refiner_state(ckpt: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
    candidates = []
    root = ckpt.get("trainer", ckpt)
    candidates.extend([
        root.get("same_scale_refiner_wo_ddp"),
        root.get("same_scale_refiner_fsdp"),
        ckpt.get("same_scale_refiner_wo_ddp"),
        ckpt.get("same_scale_refiner_fsdp"),
    ])
    for cand in candidates:
        if isinstance(cand, dict) and len(cand):
            return cand
    return None


def _get_refiner_hparams_from_state(state: Dict[str, torch.Tensor], default_hidden_dim: int, default_depth: int) -> Tuple[int, int]:
    hidden_dim = default_hidden_dim
    depth = default_depth
    if "in_proj.0.weight" in state:
        hidden_dim = int(state["in_proj.0.weight"].shape[0])
    block_idxs = []
    pat = re.compile(r"blocks\.(\d+)\.")
    for k in state.keys():
        m = pat.match(k)
        if m:
            block_idxs.append(int(m.group(1)))
    if block_idxs:
        depth = max(block_idxs) + 1
    return hidden_dim, depth


def maybe_load_same_scale_refiner(args, infinity, model_path: str, device: str):
    if not getattr(args, "enable_same_scale_refiner", 1):
        return None

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    state = _collect_refiner_state(ckpt)
    if state is None:
        print(f"[same_scale_refiner] no refiner weights found in {model_path}; fallback to first-pass logits.")
        return None

    hidden_dim, depth = _get_refiner_hparams_from_state(
        state,
        int(getattr(args, "same_scale_refiner_hidden_dim", 64)),
        int(getattr(args, "same_scale_refiner_depth", 2)),
    )

    refiner = SameScaleRefinementHead(
        codebook_dim=int(getattr(infinity, "V") // 2),
        text_dim=int(getattr(args, "text_channels", 2048)),
        hidden_state_dim=int(getattr(infinity, "C", 0)),
        hidden_dim=hidden_dim,
        depth=depth,
        max_scales=max(32, int(getattr(args, "max_scales_hint", 20)) + 4),
        dropout=float(getattr(args, "same_scale_refiner_dropout", 0.0)),
        neighborhood_kernel=int(getattr(args, "same_scale_refiner_kernel", 3)),
        max_delta=float(getattr(args, "same_scale_calibrator_max_delta", 4.0)),
        gate_bias_init=float(getattr(args, "same_scale_gate_bias_init", -2.0)),
    ).to(device)
    own_state = refiner.state_dict()
    compatible_state = {k: v for k, v in state.items() if k in own_state and tuple(own_state[k].shape) == tuple(v.shape)}
    skipped = sorted(k for k in state.keys() if k not in compatible_state)
    missing, unexpected = refiner.load_state_dict(compatible_state, strict=False)
    print(f"[same_scale_refiner] loaded from {model_path}")
    if skipped:
        print(f"[same_scale_refiner] skipped incompatible keys: {skipped}")
    if missing:
        print(f"[same_scale_refiner] missing keys: {missing}")
    if unexpected:
        print(f"[same_scale_refiner] unexpected keys: {unexpected}")
    refiner.eval()
    for p in refiner.parameters():
        p.requires_grad_(False)
    return refiner


# =========================
# Forward helpers shared by sender/receiver
# =========================
def _get_attn_module(block):
    return block.sa if hasattr(block, "sa") else block.attn


def _toggle_kv_cache(infinity, enabled: bool):
    if getattr(infinity, "num_block_chunks", 1) == 1 and hasattr(infinity, "blocks"):
        for b in infinity.blocks:
            _get_attn_module(b).kv_caching(enabled)
    elif hasattr(infinity, "block_chunks"):
        for chunk in infinity.block_chunks:
            inner = chunk.module if hasattr(chunk, "module") else chunk
            if hasattr(inner, "module"):
                for b in inner.module:
                    _get_attn_module(b).kv_caching(enabled)
            else:
                for b in inner:
                    _get_attn_module(b).kv_caching(enabled)
    else:
        for b in infinity.unregistered_blocks:
            _get_attn_module(b).kv_caching(enabled)


def _iter_chunks(infinity):
    if getattr(infinity, "num_block_chunks", 1) == 1 and hasattr(infinity, "blocks"):
        return [(0, list(infinity.blocks))]
    out = []
    for i, chunk in enumerate(infinity.block_chunks):
        inner = chunk.module if hasattr(chunk, "module") else chunk
        blocks = list(inner.module) if hasattr(inner, "module") else list(inner)
        out.append((i, blocks))
    return out


def _prepare_prompt_cond(infinity, prompt, text_tokenizer, text_encoder, cfg_list):
    label_B_or_BLT = encode_prompt(text_tokenizer, text_encoder, prompt)
    kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
    B = 1
    if any(np.array(cfg_list) != 1):
        bs = 2 * B
        kv_compact_un = kv_compact.clone()
        total = 0
        for le in lens:
            kv_compact_un[total:total + le] = infinity.cfg_uncond[:le]
            total += le
        kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
        cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:] + cu_seqlens_k[-1]), dim=0)
    else:
        bs = B

    kv_compact = infinity.text_norm(kv_compact)
    sos = cond_BD = infinity.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k))
    kv_compact = infinity.text_proj_for_ca(kv_compact)
    ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
    last_stage = sos.unsqueeze(1).expand(bs, 1, -1) + infinity.pos_start.expand(bs, 1, -1)
    with torch.amp.autocast("cuda", enabled=False):
        cond_BD_or_gss = infinity.shared_ada_lin(cond_BD.float()).float().contiguous()
    return B, bs, ca_kv, cond_BD, cond_BD_or_gss, last_stage


def _forward_one_scale(infinity, last_stage, cond_BD_or_gss, ca_kv, scale_schedule, si):
    attn_fn = None
    if getattr(infinity, "use_flex_attn", False):
        attn_fn = infinity.attn_fn_compile_dict.get(tuple(scale_schedule[:(si + 1)]), None)

    for chunk_idx, blocks in _iter_chunks(infinity):
        if getattr(infinity, "add_lvl_embeding_only_first_block", 1) and chunk_idx == 0:
            last_stage = infinity.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=0)
        if not getattr(infinity, "add_lvl_embeding_only_first_block", 1):
            last_stage = infinity.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=0)
        for block in blocks:
            last_stage = block(
                x=last_stage,
                cond_BD=cond_BD_or_gss,
                ca_kv=ca_kv,
                attn_bias_or_two_vector=None,
                attn_fn=attn_fn,
                scale_schedule=scale_schedule,
                rope2d_freqs_grid=infinity.rope2d_freqs_grid,
                scale_ind=si,
            )
    return last_stage


def _get_cfg_logits(infinity, last_stage, cond_BD, cfg, tau, B):
    if cfg != 1:
        logits_all = infinity.get_logits(last_stage, cond_BD).mul(1 / tau)
        return cfg * logits_all[:B] + (1 - cfg) * logits_all[B:]
    return infinity.get_logits(last_stage[:B], cond_BD[:B]).mul(1 / tau)


def _parse_float_or_list(value, name: str):
    if isinstance(value, (list, tuple, np.ndarray)):
        vals = [float(x) for x in value]
        return vals[0] if len(vals) == 1 else vals
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    vals = [float(x.strip()) for x in text.replace(';', ',').split(',') if x.strip()]
    if not vals:
        raise ValueError(f"{name} must contain at least one numeric value")
    return vals[0] if len(vals) == 1 else vals


def _expand_scale_param(value, num_scales: int, name: str) -> List[float]:
    value = _parse_float_or_list(value, name)
    if isinstance(value, list):
        if len(value) == 1:
            return value * num_scales
        if len(value) != num_scales:
            raise ValueError(f"{name} length must be 1 or match num_scales={num_scales}, got {len(value)}")
        return value
    return [float(value)] * num_scales


def _get_tau_cfg_lists(args, vae_scale_schedule, tau_list=None, cfg_list=None) -> Tuple[List[float], List[float]]:
    num_scales = len(vae_scale_schedule)
    if tau_list is None:
        tau_list = getattr(args, "tau_list", 0.5)
    if cfg_list is None:
        cfg_list = getattr(args, "cfg_list", 3)
    return (
        _expand_scale_param(tau_list, num_scales, "tau_list"),
        _expand_scale_param(cfg_list, num_scales, "cfg_list"),
    )


def _apply_same_scale_refiner(refiner, infinity, vae, pair_logits_BLD2, pn, si, num_scales, summed_codes, vae_scale_schedule, cond_BD, apply_spatial_patchify=0, infer_progress: float = 1.0, infer_alpha: float = 1.0):
    if refiner is None:
        return pair_logits_BLD2
    B, seq_len, D, _ = pair_logits_BLD2.shape
    assert pn[0] == 1, "same-scale refiner inference currently assumes pt=1"

    bit_logits_BLD = pair_logits_to_single_bit_logits(pair_logits_BLD2)
    bit_logits_BDHW = bit_logits_BLD.reshape(B, pn[1], pn[2], D).permute(0, 3, 1, 2).contiguous()

    if si == 0:
        prefix_feat_BDHW = torch.zeros_like(bit_logits_BDHW)
    else:
        prefix_feat_BDHW = F.interpolate(summed_codes, size=vae_scale_schedule[si], mode=vae.quantizer.z_interplote_up).contiguous().squeeze(2).float()
        if apply_spatial_patchify:
            prefix_feat_BDHW = torch.nn.functional.pixel_unshuffle(prefix_feat_BDHW, 2)

    uncertainty_B1HW = compute_normalized_uncertainty(bit_logits_BDHW)
    neighbor_ctx = compute_neighborhood_context(torch.sigmoid(bit_logits_BDHW), kernel_size=refiner.neighborhood_kernel)
    delta_logits_BDHW = refiner(
        prefix_feat_BDHW,
        bit_logits_BDHW,
        uncertainty=uncertainty_B1HW,
        neighbor_context=neighbor_ctx,
        text_summary=None,
        scale_hidden=cond_BD[:B],
        scale_id=si,
    )
    refined_bit_logits_BDHW = bit_logits_BDHW + infer_alpha * delta_logits_BDHW
    refined_bit_logits_BLD = refined_bit_logits_BDHW.permute(0, 2, 3, 1).reshape(B, seq_len, D)
    return single_bit_logits_to_pair_logits(refined_bit_logits_BLD)


def _codes_to_next_last_stage(infinity, vae, codes, summed_codes, vae_scale_schedule, si, num_stages_minus_1, apply_spatial_patchify=0):
    if si != num_stages_minus_1:
        summed_codes = summed_codes + F.interpolate(codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up)
        last_stage = F.interpolate(summed_codes, size=vae_scale_schedule[si + 1], mode=vae.quantizer.z_interplote_up)
        last_stage = last_stage.squeeze(-3)
        if apply_spatial_patchify:
            last_stage = torch.nn.functional.pixel_unshuffle(last_stage, 2)
        last_stage = last_stage.reshape(*last_stage.shape[:2], -1)
        last_stage = torch.permute(last_stage, [0, 2, 1])
        last_stage = infinity.word_embed(infinity.norm0_ve(last_stage))
        return summed_codes, last_stage
    summed_codes = summed_codes + codes
    return summed_codes, None


# =========================
# Sender: refined probabilities + arithmetic coding
# =========================
def get_prob_refined(args, infinity, vae, vae_scale_schedule, prompt, text_tokenizer, text_encoder, gt_ls_Bl, refiner=None, tau_list=None, cfg_list=None):
    tau_list, cfg_list = _get_tau_cfg_lists(args, vae_scale_schedule, tau_list=tau_list, cfg_list=cfg_list)

    B, bs, ca_kv, cond_BD, cond_BD_or_gss, last_stage = _prepare_prompt_cond(infinity, prompt, text_tokenizer, text_encoder, cfg_list)
    num_stages_minus_1 = len(vae_scale_schedule) - 1
    summed_codes = 0
    prob_list = []

    _toggle_kv_cache(infinity, True)
    try:
        for si, pn in enumerate(vae_scale_schedule):
            last_stage = _forward_one_scale(infinity, last_stage, cond_BD_or_gss, ca_kv, vae_scale_schedule, si)
            logits_BLV = _get_cfg_logits(infinity, last_stage, cond_BD, cfg_list[si], tau_list[si], B)
            pair_logits_BLD2 = logits_BLV.reshape(B, logits_BLV.shape[1], -1, 2)
            pair_logits_BLD2 = _apply_same_scale_refiner(
                refiner,
                infinity,
                vae,
                pair_logits_BLD2,
                pn,
                si,
                len(vae_scale_schedule),
                summed_codes,
                vae_scale_schedule,
                cond_BD,
                apply_spatial_patchify=getattr(args, "apply_spatial_patchify", 0),
                infer_progress=float(getattr(args, "same_scale_infer_progress", 1.0)),
                infer_alpha=float(getattr(args, "same_scale_infer_alpha", 1.0)),
            )
            prob_list.append(pair_logits_BLD2.reshape(B, -1, 2).softmax(dim=-1).view(-1, 2))

            idx_Bld = gt_ls_Bl[si].reshape(B, pn[1], pn[2], -1)
            idx_Bld = idx_Bld.unsqueeze(1)
            codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label')
            summed_codes, next_stage = _codes_to_next_last_stage(
                infinity, vae, codes, summed_codes, vae_scale_schedule, si, num_stages_minus_1,
                apply_spatial_patchify=getattr(args, "apply_spatial_patchify", 0),
            )
            if next_stage is not None:
                last_stage = next_stage.repeat(bs // B, 1, 1)
    finally:
        _toggle_kv_cache(infinity, False)

    return prob_list


def encoding_refined(args, infinity, vae, scale_schedule, text, text_tokenizer, text_encoder, gt_ms_idx_Bl, refiner=None):
    prob_list = get_prob_refined(args, infinity, vae, scale_schedule, text, text_tokenizer, text_encoder, gt_ms_idx_Bl, refiner=refiner)
    trans_list, help_list, bpp_list = [], [], []
    sum_len = 0
    for i in range(len(gt_ms_idx_Bl)):
        gt_idx = gt_ms_idx_Bl[i].view(-1).cpu().tolist()
        prob = prob_list[i][:, 0].cpu().tolist()
        arithmetic_bits, hbits = [], []
        for j in range(int(len(prob) / args.vae_type)):
            gt_token = gt_idx[j * args.vae_type:(j + 1) * args.vae_type]
            p_token = prob[j * args.vae_type:(j + 1) * args.vae_type]
            bits = compress_to_bit_list(gt_token, p_token)
            if len(bits) < args.vae_type:
                arithmetic_bits.append(bits)
                hbits.append(1)   # arithmetic-coded
                sum_len += len(bits)
            else:
                arithmetic_bits.append(gt_token)
                hbits.append(0)   # raw bits
                sum_len += args.vae_type
        trans_list.append(arithmetic_bits)
        help_list.append(hbits)
        sum_len += len(hbits)
        bpp_list.append(sum_len / 1024 / 1024)
    return trans_list, help_list, bpp_list


# =========================
# Receiver: refined decode + refined sampling
# =========================
def decoding_refined(args, infinity, vae, vae_scale_schedule, prompt, text_tokenizer, text_encoder, trans_list, help_list, refiner=None, tau_list=None, cfg_list=None):
    tau_list, cfg_list = _get_tau_cfg_lists(args, vae_scale_schedule, tau_list=tau_list, cfg_list=cfg_list)

    B, bs, ca_kv, cond_BD, cond_BD_or_gss, last_stage = _prepare_prompt_cond(infinity, prompt, text_tokenizer, text_encoder, cfg_list)
    num_stages_minus_1 = len(vae_scale_schedule) - 1
    summed_codes = 0
    decode_idx = []

    _toggle_kv_cache(infinity, True)
    try:
        for si, pn in enumerate(vae_scale_schedule):
            last_stage = _forward_one_scale(infinity, last_stage, cond_BD_or_gss, ca_kv, vae_scale_schedule, si)
            logits_BLV = _get_cfg_logits(infinity, last_stage, cond_BD, cfg_list[si], tau_list[si], B)
            pair_logits_BLD2 = logits_BLV.reshape(B, logits_BLV.shape[1], -1, 2)
            pair_logits_BLD2 = _apply_same_scale_refiner(
                refiner,
                infinity,
                vae,
                pair_logits_BLD2,
                pn,
                si,
                len(vae_scale_schedule),
                summed_codes,
                vae_scale_schedule,
                cond_BD,
                apply_spatial_patchify=getattr(args, "apply_spatial_patchify", 0),
                infer_progress=float(getattr(args, "same_scale_infer_progress", 1.0)),
                infer_alpha=float(getattr(args, "same_scale_infer_alpha", 1.0)),
            )
            prob = pair_logits_BLD2.reshape(B, -1, 2).softmax(dim=-1).view(-1, 2)[:, 0].cpu().tolist()

            decompressed_string = []
            bit_string = trans_list[si]
            h_string = help_list[si]
            for j, flag in enumerate(h_string):
                p_token = prob[j * args.vae_type:(j + 1) * args.vae_type]
                if flag == 0:
                    decompressed_string.extend(bit_string[j])
                elif flag == 1:
                    decompressed_string.extend(decompress_from_bit_list(bit_string[j], args.vae_type, p_token))
                else:
                    raise ValueError(f"Unknown flag={flag} at scale={si}, token={j}")

            dec_idx = torch.tensor(decompressed_string, dtype=torch.int32, device=pair_logits_BLD2.device).reshape(B, pn[1] * pn[2], -1)
            decode_idx.append(dec_idx)
            idx_Bld = dec_idx.reshape(B, pn[1], pn[2], -1).unsqueeze(1)
            codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label')
            summed_codes, next_stage = _codes_to_next_last_stage(
                infinity, vae, codes, summed_codes, vae_scale_schedule, si, num_stages_minus_1,
                apply_spatial_patchify=getattr(args, "apply_spatial_patchify", 0),
            )
            if next_stage is not None:
                last_stage = next_stage.repeat(bs // B, 1, 1)
    finally:
        _toggle_kv_cache(infinity, False)

    return decode_idx


def decompress_cfg_refined(args, infinity, vae, vae_scale_schedule, prompt, text_tokenizer, text_encoder, gt_leak, decoded_idx_list, refiner=None, tau_list=None, cfg_list=None):
    rng = infinity.rng
    tau_list, cfg_list = _get_tau_cfg_lists(args, vae_scale_schedule, tau_list=tau_list, cfg_list=cfg_list)

    B, bs, ca_kv, cond_BD, cond_BD_or_gss, last_stage = _prepare_prompt_cond(infinity, prompt, text_tokenizer, text_encoder, cfg_list)
    num_stages_minus_1 = len(vae_scale_schedule) - 1
    summed_codes = 0

    _toggle_kv_cache(infinity, True)
    try:
        for si, pn in enumerate(vae_scale_schedule):
            last_stage = _forward_one_scale(infinity, last_stage, cond_BD_or_gss, ca_kv, vae_scale_schedule, si)
            logits_BLV = _get_cfg_logits(infinity, last_stage, cond_BD, cfg_list[si], tau_list[si], B)
            pair_logits_BLD2 = logits_BLV.reshape(B, logits_BLV.shape[1], -1, 2)
            pair_logits_BLD2 = _apply_same_scale_refiner(
                refiner,
                infinity,
                vae,
                pair_logits_BLD2,
                pn,
                si,
                len(vae_scale_schedule),
                summed_codes,
                vae_scale_schedule,
                cond_BD,
                apply_spatial_patchify=getattr(args, "apply_spatial_patchify", 0),
                infer_progress=float(getattr(args, "same_scale_infer_progress", 1.0)),
                infer_alpha=float(getattr(args, "same_scale_infer_alpha", 1.0)),
            )

            if si <= gt_leak:
                idx_Bld = decoded_idx_list[si]
            else:
                flat_logits = pair_logits_BLD2.reshape(B, -1, 2)
                idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(flat_logits, rng=rng, top_k=0, top_p=0.0, num_samples=1)[:, :, 0]
                idx_Bld = idx_Bld.reshape(B, pair_logits_BLD2.shape[1], -1)

            idx_Bld = idx_Bld.reshape(B, pn[1], pn[2], -1).unsqueeze(1)
            codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label')
            summed_codes, next_stage = _codes_to_next_last_stage(
                infinity, vae, codes, summed_codes, vae_scale_schedule, si, num_stages_minus_1,
                apply_spatial_patchify=getattr(args, "apply_spatial_patchify", 0),
            )
            if next_stage is not None:
                last_stage = next_stage.repeat(bs // B, 1, 1)
    finally:
        _toggle_kv_cache(infinity, False)

    img = vae.decode(summed_codes.squeeze(-3))
    return (img + 1) / 2


# =========================
# Eval
# =========================
def build_scale_schedule_and_q(args, img_path: str):
    inp = load_img(img_path, args)
    _, _, h, w = inp.shape
    h_div_w = h / w
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
    scale_schedule = [(1, h_, w_) for (_, h_, w_) in scale_schedule]
    scale_q = [(scale_schedule[i][0], scale_schedule[i][1], scale_schedule[i][2], int((i + 1) // ((len(scale_schedule) // 3) + 1) + 2)) for i in range(len(scale_schedule))]
    return inp, h, w, scale_schedule, scale_q


def compress_image(args, vae, scale_schedule, scale_q, infinity, text_tokenizer, text_encoder, img_path, text, refiner=None, device='cuda'):
    inp_B3HW = load_img(img_path, args)
    raw_features, _, _ = vae.encode_for_raw_features(inp_B3HW.to(device), scale_schedule=scale_schedule)
    _, gt_ms_idx_Bl = mask_quant(vae, scale_q, raw_features, device)
    trans_list, help_list, bpp_list = encoding_refined(args, infinity, vae, scale_schedule, text, text_tokenizer, text_encoder, gt_ms_idx_Bl, refiner=refiner)
    return trans_list, help_list, bpp_list, gt_ms_idx_Bl


def _count_bits_until_scale(trans_list, help_list, gt_leak: int) -> int:
    total_bits = 0
    for i in range(gt_leak + 1):
        total_bits += len(help_list[i])
        for payload in trans_list[i]:
            total_bits += len(payload)
    return total_bits


def _compute_image_metrics(std_tensor, rec_tensor, lpips_fn, dists_fn, device: str):
    mse = torch.mean((std_tensor - rec_tensor) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse)).item() if mse.item() > 0 else float('inf')
    msssim_val = ms_ssim(std_tensor, rec_tensor, data_range=1.0, size_average=True).item()
    lpips_val = lpips_fn(std_tensor.to(device) * 2 - 1, rec_tensor.to(device) * 2 - 1).item()
    dists_val = dists_fn(std_tensor.to(device), rec_tensor.to(device)).item()
    return {"psnr": psnr, "msssim": msssim_val, "lpips": lpips_val, "dists": dists_val}


def evaluate_one_image(args, infinity, vae, refiner, text_tokenizer, text_encoder, img_path, text, rec_base_path, lpips_fn, dists_fn, device: str):
    inp, h, w, scale_schedule, scale_q = build_scale_schedule_and_q(args, img_path)

    prompt_bits = 0
    text_for_decode = text
    if args.add_prompt_bits:
        if args.prompt_bits_mode == "arith":
            prompt_ids = _encode_prompt_ids_t5(text_tokenizer, text, max_len=args.tlen)
            packet = _arith_encode_ids_packet(prompt_ids)
            prompt_bits = packet["bits"]
            text_for_decode = _decode_prompt_text_from_ids_t5(text_tokenizer, _arith_decode_ids_packet(packet))
        elif args.prompt_bits_mode == "zlib":
            prompt_bits = _prompt_zlib_bytes(text) * 8
    prompt_bpp = prompt_bits / (h * w) if args.add_prompt_bits else 0.0

    autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16) if device.startswith("cuda") and bool(getattr(args, "bf16", 0)) else nullcontext()
    with torch.no_grad(), autocast_ctx:
        trans_list, help_list, _, gt_ms_idx_Bl = compress_image(args, vae, scale_schedule, scale_q, infinity, text_tokenizer, text_encoder, img_path, text, refiner=refiner, device=device)
        decoded_idx = decoding_refined(args, infinity, vae, scale_schedule, text_for_decode, text_tokenizer, text_encoder, trans_list, help_list, refiner=refiner)

    std_img_cv = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    std_tensor = torch.tensor(std_img_cv).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    scale_results = {}
    img_name = os.path.basename(img_path)
    for gt_leak in range(len(gt_ms_idx_Bl)):
        with torch.no_grad(), autocast_ctx:
            img = decompress_cfg_refined(args, infinity, vae, scale_schedule, text_for_decode, text_tokenizer, text_encoder, gt_leak, decoded_idx, refiner=refiner)
        scale_folder = os.path.join(rec_base_path, f"scale_{gt_leak}")
        os.makedirs(scale_folder, exist_ok=True)
        rec_img_path = os.path.join(scale_folder, img_name)
        torchvision.utils.save_image(img.cpu(), rec_img_path)

        rec_img_cv = cv2.cvtColor(cv2.imread(rec_img_path), cv2.COLOR_BGR2RGB)
        rec_tensor = torch.tensor(rec_img_cv).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        metrics = _compute_image_metrics(std_tensor, rec_tensor, lpips_fn, dists_fn, device)
        img_bpp = _count_bits_until_scale(trans_list, help_list, gt_leak) / (h * w)
        scale_results[str(gt_leak)] = {
            "bpp": img_bpp + prompt_bpp,
            "psnr": metrics["psnr"],
            "msssim": metrics["msssim"],
            "lpips": metrics["lpips"],
            "dists": metrics["dists"],
        }

    return {
        "image_name": img_name,
        "text": text,
        "prompt_bpp": prompt_bpp,
        "scales_data": scale_results,
    }


def summarize_dataset(dataset_metrics_data: List[Dict[str, Any]]):
    scale_aggregates: Dict[str, Dict[str, List[float]]] = {}
    for item in dataset_metrics_data:
        for scale_idx, metrics in item["scales_data"].items():
            if scale_idx not in scale_aggregates:
                scale_aggregates[scale_idx] = {"bpp": [], "psnr": [], "msssim": [], "lpips": [], "dists": []}
            for k in scale_aggregates[scale_idx]:
                scale_aggregates[scale_idx][k].append(float(metrics[k]))

    summary = {}
    for scale_idx, vals in scale_aggregates.items():
        summary[scale_idx] = {
            "avg_bpp": float(np.mean(vals["bpp"])),
            "avg_psnr": float(np.mean(vals["psnr"])),
            "avg_msssim": float(np.mean(vals["msssim"])),
            "avg_lpips": float(np.mean(vals["lpips"])),
            "avg_dists": float(np.mean(vals["dists"])),
            "image_count": len(vals["bpp"]),
        }
    return summary


def default_args():
    initial_model_path = '/workspace/Infinity_codec/local_output/debug_stage2_student_1024_125M_16vae_refine_orimodel_nodetach/ar-ckpt-giter045K-ep0-iter45000-last.pth'
    vae_path = '/workspace/CKPT/Infinity/infinity_vae_d16.pth'
    text_encoder_ckpt = '/workspace/CKPT/flan-t5-xl'
    return argparse.Namespace(
        pn='1M',
        model_path=initial_model_path,
        dataset_json='/workspace/ARPC/data/DIV2K.json',
        output_root='/workspace/Infinity_codec/results/refiner_eval2_40k_cfg1_nodetach',
        cfg_insertion_layer=0,
        vae_type=16,
        vae_path=vae_path,
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        model_type='infinity_layer12',
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        use_scale_schedule_embedding=0,
        sampling_per_bits=1,
        text_encoder_ckpt=text_encoder_ckpt,
        text_channels=2048,
        apply_spatial_patchify=0,
        h_div_w_template=1.000,
        use_flex_attn=0,
        cache_dir='/workspace/Infinity_codec/local_output/refiner_eval_cache',
        enable_model_cache=1,
        checkpoint_type='torch',
        seed=0,
        bf16=0,
        rec_path='',
        add_prompt_bits=1,
        prompt_bits_mode='arith',
        tlen=512,
        tau_list='0.5',
        cfg_list='1',
        keep_gpu_busy=0,
        sweep_all_iters=1,
        start_iter=None,
        limit_images=0,
        enable_same_scale_refiner=1,
        same_scale_refiner_hidden_dim=64,
        same_scale_refiner_depth=2,
        same_scale_refiner_dropout=0.0,
        same_scale_refiner_kernel=3,
        same_scale_calibrator_max_delta=4.0,
        same_scale_gate_bias_init=-2.0,
        same_scale_infer_progress=1.0,
        same_scale_infer_alpha=1.0,
        max_scales_hint=20,
    )


def _parse_cli(base_args):
    parser = argparse.ArgumentParser(description="Evaluate Infinity codec with same-scale refinement head.")
    for k, v in vars(base_args).items():
        arg = f"--{k}"
        if isinstance(v, bool):
            parser.add_argument(arg, type=int, default=int(v))
        elif isinstance(v, int):
            parser.add_argument(arg, type=int, default=v)
        elif isinstance(v, float):
            parser.add_argument(arg, type=float, default=v)
        else:
            parser.add_argument(arg, type=str, default=v)
    parsed = parser.parse_args()
    parsed.cfg_insertion_layer = [int(x) for x in str(parsed.cfg_insertion_layer).replace(';', ',').split(',') if str(x).strip()]
    parsed.tau_list = _parse_float_or_list(parsed.tau_list, "tau_list")
    parsed.cfg_list = _parse_float_or_list(parsed.cfg_list, "cfg_list")
    return parsed


def _list_models_to_test(model_path: str, sweep_all_iters: int, start_iter: Optional[int]):
    if not sweep_all_iters:
        return [model_path]
    model_dir = os.path.dirname(model_path)
    base_filename = os.path.basename(model_path)
    if start_iter is None:
        match_init = re.search(r'-iter(\d+)-', base_filename)
        if not match_init:
            return [model_path]
        start_iter = int(match_init.group(1))
    available_models = []
    for f in os.listdir(model_dir):
        if f.endswith('.pth') and 'iter' in f:
            m = re.search(r'-iter(\d+)-', f)
            if m:
                it = int(m.group(1))
                if it >= int(start_iter):
                    available_models.append((it, os.path.join(model_dir, f)))
    available_models.sort(key=lambda x: x[0])
    return [p for _, p in available_models] or [model_path]


def main():
    args = _parse_cli(default_args())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    lpips_evaluator = lpips.LPIPS(net='alex').to(device)
    dists_evaluator = DISTS().to(device)

    with open(args.dataset_json, 'rt', encoding='utf-8') as f:
        json_data = [json.loads(line) for line in f]
    if int(args.limit_images) > 0:
        json_data = json_data[:int(args.limit_images)]

    models_to_test = _list_models_to_test(args.model_path, int(args.sweep_all_iters), getattr(args, 'start_iter', None))
    print(f"[*] Found {len(models_to_test)} model(s) to test.")

    for current_model_path in models_to_test:
        args.model_path = current_model_path
        m = re.search(r'-iter(\d+)-', os.path.basename(current_model_path))
        current_iter = int(m.group(1)) if m else -1
        args.rec_path = os.path.join(args.output_root, f"iter_{current_iter}" if current_iter >= 0 else os.path.splitext(os.path.basename(current_model_path))[0])
        os.makedirs(args.rec_path, exist_ok=True)

        print("=" * 80)
        print(f"[*] Evaluating: {current_model_path}")
        print("=" * 80)

        infinity = load_transformer(vae, args)
        refiner = maybe_load_same_scale_refiner(args, infinity, current_model_path, device)

        dataset_metrics_data = []
        for idx, data in enumerate(json_data):
            img_path, text = data['img_path'], data['txt']
            print(f"[{idx + 1}/{len(json_data)}] {os.path.basename(img_path)}")
            result = evaluate_one_image(args, infinity, vae, refiner, text_tokenizer, text_encoder, img_path, text, args.rec_path, lpips_evaluator, dists_evaluator, device)
            dataset_metrics_data.append(result)
            bpps = [f"{v['bpp']:.6f}" for _, v in sorted(result['scales_data'].items(), key=lambda kv: int(kv[0]))]
            print(f"    scale total bpp: {bpps}")

        average_metrics_summary = summarize_dataset(dataset_metrics_data)
        print(f"\n[*] Iter {current_iter} summary:")
        for scale_idx, item in sorted(average_metrics_summary.items(), key=lambda kv: int(kv[0])):
            print(
                f"    scale {scale_idx} | BPP: {item['avg_bpp']:.6f} | PSNR: {item['avg_psnr']:.4f} | "
                f"MS-SSIM: {item['avg_msssim']:.4f} | LPIPS: {item['avg_lpips']:.4f} | DISTS: {item['avg_dists']:.4f}"
            )

        final_json_output = {
            "model_path": current_model_path,
            "model_iter": current_iter,
            "same_scale_refiner_enabled": bool(refiner is not None),
            "tau_list": args.tau_list,
            "cfg_list": args.cfg_list,
            "summary": average_metrics_summary,
            "details": dataset_metrics_data,
        }
        json_save_path = os.path.join(args.rec_path, f"metrics_iter_{current_iter}.json" if current_iter >= 0 else "metrics.json")
        with open(json_save_path, 'w', encoding='utf-8') as f:
            json.dump(final_json_output, f, indent=2, ensure_ascii=False)

        df_summary = pd.DataFrame(average_metrics_summary).T
        df_summary.index.name = 'scale'
        csv_save_path = os.path.join(args.rec_path, f"avg_metrics_iter_{current_iter}.csv" if current_iter >= 0 else "avg_metrics.csv")
        df_summary.to_csv(csv_save_path)
        print(f"[*] Saved JSON to {json_save_path}")
        print(f"[*] Saved CSV  to {csv_save_path}\n")

    print("[*] All models finished.")


if __name__ == '__main__':
    main()
