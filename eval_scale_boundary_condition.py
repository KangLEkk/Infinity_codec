import argparse
import importlib.util
import json
import math
import os
import re
import sys
import zlib
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.append("/workspace/Infinity_codec")

import cv2
import lpips
import numpy as np
import pandas as pd
import torch
if not hasattr(torch, "_dynamo"):
    torch._dynamo = SimpleNamespace(config=SimpleNamespace(cache_size_limit=None))
import torch.nn.functional as F
import torchvision
from DISTS_pytorch import DISTS
import PIL.Image as PImage
from pytorch_msssim import ms_ssim
from torchvision.transforms.functional import to_tensor

from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
from utils.arithmeticcoding import compress_to_bit_list, decompress_from_bit_list


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

_CODEC_SPEC = importlib.util.spec_from_file_location(
    "boundary_condition_codec",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "infinity", "models", "condition_codec.py"),
)
_CODEC_MOD = importlib.util.module_from_spec(_CODEC_SPEC)
_CODEC_SPEC.loader.exec_module(_CODEC_MOD)
BoundaryConditionCodec = _CODEC_MOD.BoundaryConditionCodec
BoundarySpatialAdapter = _CODEC_MOD.BoundarySpatialAdapter
TransformersDepthExtractor = _CODEC_MOD.TransformersDepthExtractor
TransformersSAMSegmentationExtractor = _CODEC_MOD.TransformersSAMSegmentationExtractor
image_to_luma_boundary = _CODEC_MOD.image_to_luma_boundary
image_to_spatial_condition = _CODEC_MOD.image_to_spatial_condition
var_token_condition_from_map = _CODEC_MOD.var_token_condition_from_map


def transform(pil_img, tgt_h, tgt_w):
    width, height = pil_img.size
    if width / height <= tgt_w / tgt_h:
        resized_width = tgt_w
        resized_height = int(tgt_w / (width / height))
    else:
        resized_height = tgt_h
        resized_width = int((width / height) * tgt_h)
    pil_img = pil_img.resize((resized_width, resized_height), resample=PImage.LANCZOS)
    arr = np.array(pil_img)
    crop_y = (arr.shape[0] - tgt_h) // 2
    crop_x = (arr.shape[1] - tgt_w) // 2
    im = to_tensor(arr[crop_y: crop_y + tgt_h, crop_x: crop_x + tgt_w])
    return im.add(im).add_(-1)


def load_img(img_path, args):
    with open(img_path, "rb") as f:
        img = PImage.open(f).convert("RGB")
        w, h = img.size
        h_div_w = h / w
        h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w - h_div_w_templates))]
        tgt_h, tgt_w = dynamic_resolution_h_w[h_div_w_template][args.pn]["pixel"]
        img_B3HW = transform(img, tgt_h, tgt_w)
    return img_B3HW.unsqueeze(0)


def mask_quant(vae, vae_scale_schedule, raw_features, device):
    with torch.amp.autocast("cuda", enabled=False):
        B = raw_features.shape[0]
        codes_out = raw_features.unsqueeze(2) if raw_features.dim() == 4 else raw_features
        cum_var_input = 0
        gt_all_bit_indices = []
        x_BLC_wo_prefix = []
        for si, (pt, ph, pw, pm) in enumerate(vae_scale_schedule):
            residual = codes_out - cum_var_input
            if si != len(vae_scale_schedule) - 1:
                residual = F.interpolate(residual, size=vae_scale_schedule[si][:3], mode=vae.quantizer.z_interplote_down).contiguous()
            quantized, _, bit_indices, _ = vae.quantizer.lfq(residual)
            gt_all_bit_indices.append(bit_indices)
            cum_var_input = cum_var_input + F.interpolate(quantized, size=vae_scale_schedule[-1][:3], mode=vae.quantizer.z_interplote_up).contiguous()
            if si < len(vae_scale_schedule) - 1:
                this_scale_input = F.interpolate(cum_var_input, size=vae_scale_schedule[si + 1][:3], mode=vae.quantizer.z_interplote_up).contiguous()
                x_BLC_wo_prefix.append(this_scale_input.reshape(*this_scale_input.shape[:2], -1).permute(0, 2, 1))
        gt_ms_idx_Bl = [item.reshape(B, -1, vae.codebook_dim) for item in gt_all_bit_indices]
        x_BLC_wo_prefix = torch.cat(x_BLC_wo_prefix, 1)
    return x_BLC_wo_prefix, gt_ms_idx_Bl


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
    return {"payload": payload, "cdf_1d": cdf_1d, "unique_chars": unique_chars, "char_len": L, "ids_len": len(ids), "bits": len(payload) * 8}


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


def _parse_float_or_list(value, name: str):
    if isinstance(value, (list, tuple, np.ndarray)):
        vals = [float(x) for x in value]
        return vals[0] if len(vals) == 1 else vals
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    vals = [float(x.strip()) for x in text.replace(";", ",").split(",") if x.strip()]
    if not vals:
        raise ValueError(f"{name} must contain at least one numeric value")
    return vals[0] if len(vals) == 1 else vals


def _parse_int_list(value) -> List[int]:
    text = str(value or "").strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.replace(";", ",").split(",") if x.strip()]


def _expand_scale_param(value, num_scales: int, name: str) -> List[float]:
    value = _parse_float_or_list(value, name)
    if isinstance(value, list):
        if len(value) == 1:
            return value * num_scales
        if len(value) != num_scales:
            raise ValueError(f"{name} length must be 1 or match num_scales={num_scales}, got {len(value)}")
        return value
    return [float(value)] * num_scales


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
    captions = [prompt]
    tokens = text_tokenizer(text=captions, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    input_ids = tokens.input_ids.cuda(non_blocking=True)
    mask = tokens.attention_mask.cuda(non_blocking=True)
    text_features = text_encoder(input_ids=input_ids, attention_mask=mask)["last_hidden_state"].float()
    lens = mask.sum(dim=-1).tolist()
    cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
    max_seqlen_k = max(lens)
    kv_compact = torch.cat([feat_i[:len_i] for len_i, feat_i in zip(lens, text_features.unbind(0))], dim=0)

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


def _get_cfg_logits(infinity, last_stage, cond_BD, cfg, tau, B):
    if cfg != 1:
        logits_all = infinity.get_logits(last_stage, cond_BD).mul(1 / tau)
        return cfg * logits_all[:B] + (1 - cfg) * logits_all[B:]
    return infinity.get_logits(last_stage[:B], cond_BD[:B]).mul(1 / tau)


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


def _count_bits_until_scale(trans_list, help_list, gt_leak: int) -> int:
    total_bits = 0
    for i in range(gt_leak + 1):
        total_bits += len(help_list[i])
        for payload in trans_list[i]:
            total_bits += len(payload)
    return total_bits


def _get_tau_cfg_lists(args, vae_scale_schedule, tau_list=None, cfg_list=None) -> Tuple[List[float], List[float]]:
    num_scales = len(vae_scale_schedule)
    if tau_list is None:
        tau_list = getattr(args, "tau_list", 0.5)
    if cfg_list is None:
        cfg_list = getattr(args, "cfg_list", 1.0)
    return (
        _expand_scale_param(tau_list, num_scales, "tau_list"),
        _expand_scale_param(cfg_list, num_scales, "cfg_list"),
    )


def _extract_module_state(state: Dict[str, torch.Tensor], module_name: str) -> Dict[str, torch.Tensor]:
    prefix = module_name + "."
    out = {}
    for k, v in state.items():
        if prefix in k:
            out[k.split(prefix, 1)[1]] = v
    return out


def _collect_gpt_state(ckpt: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
    root = ckpt.get("trainer", ckpt)
    for key in ("gpt_fsdp", "gpt_wo_ddp", "gpt_ema_fsdp", "gpt_ema_for_vis"):
        cand = root.get(key) if isinstance(root, dict) else None
        if isinstance(cand, dict) and any(("condition_codec." in k or "condition_adapter." in k) for k in cand.keys()):
            return cand
    if isinstance(root, dict) and any(("condition_codec." in k or "condition_adapter." in k) for k in root.keys()):
        return root
    if isinstance(ckpt, dict) and any(("condition_codec." in k or "condition_adapter." in k) for k in ckpt.keys()):
        return ckpt
    return None


def maybe_load_boundary_conditioning(args, infinity, model_path: str, device: str):
    if not int(getattr(args, "enable_boundary_condition", 1)):
        print("[boundary_condition] disabled by args.")
        return None

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    state = _collect_gpt_state(ckpt)
    if state is None:
        raise KeyError(
            f"No condition_adapter weights found in {model_path}. "
            "Use a checkpoint trained with --enable_boundary_condition 1."
        )

    codec_state = _extract_module_state(state, "condition_codec")
    adapter_state = _extract_module_state(state, "condition_adapter")
    if not adapter_state:
        raise KeyError("Checkpoint has no condition_adapter state.")

    codec_type = str(getattr(args, "condition_codec_type", "binary") or "binary").lower()
    hidden_dim = int(codec_state.get("encoder.0.weight").shape[0]) if "encoder.0.weight" in codec_state else int(args.boundary_cond_hidden_dim)
    latent_dim = int(codec_state.get("prior_logits").shape[1]) if "prior_logits" in codec_state else int(args.boundary_cond_latent_dim)
    if "decoder.6.weight" in codec_state:
        feature_dim = int(codec_state.get("decoder.6.weight").shape[0])
    elif "proj.0.weight" in adapter_state:
        feature_dim = int(adapter_state["proj.0.weight"].shape[1])
    else:
        feature_dim = int(args.boundary_cond_feature_dim)

    if codec_type == "vae_token":
        infinity.condition_codec = None
    else:
        if not codec_state:
            raise KeyError("Checkpoint has no condition_codec state for binary condition_codec_type.")
        infinity.condition_codec = BoundaryConditionCodec(
            input_size=int(args.boundary_cond_size),
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            feature_dim=feature_dim,
        ).to(device)
    infinity.condition_adapter = BoundarySpatialAdapter(
        feature_dim=feature_dim,
        embed_dim=int(getattr(infinity, "C")),
        depth=int(getattr(infinity, "depth")),
    ).to(device)

    if infinity.condition_codec is not None:
        missing, unexpected = infinity.condition_codec.load_state_dict(codec_state, strict=False)
        if missing:
            print(f"[boundary_condition] codec missing keys: {missing}")
        if unexpected:
            print(f"[boundary_condition] codec unexpected keys: {unexpected}")
    missing, unexpected = infinity.condition_adapter.load_state_dict(adapter_state, strict=False)
    if missing:
        print(f"[boundary_condition] adapter missing keys: {missing}")
    if unexpected:
        print(f"[boundary_condition] adapter unexpected keys: {unexpected}")

    if infinity.condition_codec is not None:
        infinity.condition_codec.eval()
    infinity.condition_adapter.eval()
    if infinity.condition_codec is not None:
        for p in infinity.condition_codec.parameters():
            p.requires_grad_(False)
    for p in infinity.condition_adapter.parameters():
        p.requires_grad_(False)

    print(
        "[spatial_condition] loaded "
        f"type={getattr(args, 'spatial_cond_type', 'boundary')}, size={args.boundary_cond_size}, "
        f"hidden={hidden_dim}, latent={latent_dim}, feature={feature_dim}"
    )
    return {
        "type": getattr(args, "spatial_cond_type", "boundary"),
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "feature_dim": feature_dim,
    }


def _boundary_side_bits_from_aux(args, aux: Dict[str, torch.Tensor], h: int, w: int) -> Dict[str, Any]:
    z_bits = aux["z_hard"].reshape(-1).to(dtype=torch.int64).cpu().tolist()
    prior_logits = aux["prior_logits"].reshape(-1).float()
    p0 = torch.sigmoid(-prior_logits).clamp(1e-5, 1.0 - 1e-5).cpu().tolist()
    mode = str(getattr(args, "boundary_side_bits_mode", "arith")).lower()

    if mode == "none":
        payload_bits = []
        side_bits = 0
    elif mode == "raw":
        payload_bits = z_bits
        side_bits = len(z_bits)
    elif mode == "entropy":
        payload_bits = []
        side_bits = int(round(float(aux["hard_side_bpp"].mean().item()) * h * w))
    elif mode == "arith":
        payload_bits = compress_to_bit_list(z_bits, p0)
        if int(getattr(args, "boundary_check_decode", 0)):
            decoded = decompress_from_bit_list(payload_bits, len(z_bits), p0)
            if decoded != z_bits:
                raise RuntimeError("Boundary side arithmetic decode mismatch.")
        side_bits = len(payload_bits)
    else:
        raise ValueError(f"Unknown boundary_side_bits_mode={mode!r}")

    expected_bits = float(aux["side_bpp"].mean().item()) * h * w
    hard_expected_bits = float(aux["hard_side_bpp"].mean().item()) * h * w
    return {
        "side_bits": int(side_bits),
        "side_bpp": float(side_bits / max(1, h * w)),
        "expected_side_bits": expected_bits,
        "expected_side_bpp": float(expected_bits / max(1, h * w)),
        "hard_expected_side_bits": hard_expected_bits,
        "hard_expected_side_bpp": float(hard_expected_bits / max(1, h * w)),
        "latent_shape": tuple(aux["z_hard"].shape[-3:]),
        "payload_bits_len": len(payload_bits),
    }


_EVAL_DEPTH_EXTRACTOR = None
_EVAL_SEG_EXTRACTOR = None


def _get_eval_depth_extractor(args, device: str):
    global _EVAL_DEPTH_EXTRACTOR
    if _EVAL_DEPTH_EXTRACTOR is None:
        depth_device = str(getattr(args, "depth_model_device", "") or device)
        print(f"[spatial_condition] loading online depth model {args.depth_model_name} on {depth_device}")
        _EVAL_DEPTH_EXTRACTOR = TransformersDepthExtractor(
            model_name=getattr(args, "depth_model_name", "depth-anything/Depth-Anything-V2-Small-hf"),
            device=depth_device,
            dtype=getattr(args, "depth_model_dtype", "fp16"),
            cache_dir=getattr(args, "depth_model_cache_dir", ""),
        )
    return _EVAL_DEPTH_EXTRACTOR


def _get_eval_seg_extractor(args, device: str):
    global _EVAL_SEG_EXTRACTOR
    if _EVAL_SEG_EXTRACTOR is None:
        seg_device = str(getattr(args, "seg_model_device", "") or device)
        print(f"[spatial_condition] loading online SAM/seg model {args.seg_model_name} on {seg_device}")
        _EVAL_SEG_EXTRACTOR = TransformersSAMSegmentationExtractor(
            model_name=getattr(args, "seg_model_name", "facebook/sam-vit-base"),
            device=seg_device,
            dtype=getattr(args, "seg_model_dtype", "fp16"),
            cache_dir=getattr(args, "seg_model_cache_dir", ""),
            output_mode=getattr(args, "seg_output_mode", "region_boundary"),
            max_masks=int(getattr(args, "seg_max_masks", 16)),
            points_per_batch=int(getattr(args, "seg_points_per_batch", 32)),
        )
    return _EVAL_SEG_EXTRACTOR


def prepare_boundary_condition(args, infinity, vae, scale_schedule, inp_B3HW: torch.Tensor, h: int, w: int, device: str):
    if not int(getattr(args, "enable_boundary_condition", 1)):
        return None, {
            "side_bits": 0,
            "side_bpp": 0.0,
            "expected_side_bpp": 0.0,
            "hard_expected_side_bpp": 0.0,
            "latent_shape": None,
        }
    if not hasattr(infinity, "condition_codec") or not hasattr(infinity, "condition_adapter"):
        raise AttributeError("Infinity model does not have boundary condition modules attached.")

    codec_type = str(getattr(args, "condition_codec_type", "binary") or "binary").lower()
    out_size = 0 if codec_type == "vae_token" else int(args.boundary_cond_size)
    cond_type = str(getattr(args, "spatial_cond_type", "boundary") or "boundary").lower()
    depth_source = str(getattr(args, "depth_condition_source", "proxy") or "proxy").lower()
    if cond_type in {"depth", "depth_model", "depth_anything"} and depth_source in {"transformers", "hf", "depth_anything", "depth_anything_v2"}:
        condition = _get_eval_depth_extractor(args, device)(inp_B3HW.to(device), out_size=out_size)
    elif cond_type in {"sam", "seg", "segmentation", "segment"} and str(getattr(args, "seg_condition_source", "transformers") or "transformers").lower() in {"transformers", "hf", "sam"}:
        condition = _get_eval_seg_extractor(args, device)(inp_B3HW.to(device), out_size=out_size)
    else:
        condition = image_to_spatial_condition(
            inp_B3HW.to(device),
            cond_type=cond_type,
            out_size=out_size,
        )
    if codec_type == "vae_token":
        aux = var_token_condition_from_map(
            vae,
            condition,
            scale_schedule,
            num_scales=int(getattr(args, "condition_token_scales", 2)),
            image_hw=(h, w),
        )
        side_bits = int(aux["side_bits_per_image"][0].item())
        side_bpp = side_bits / max(1, h * w)
        side_info = {
            "side_bits": side_bits,
            "side_bpp": float(side_bpp),
            "expected_side_bits": side_bits,
            "expected_side_bpp": float(side_bpp),
            "hard_expected_side_bits": side_bits,
            "hard_expected_side_bpp": float(side_bpp),
            "latent_shape": aux["latent_shape"],
            "num_scales": int(aux["num_scales"]),
            "payload_bits_len": side_bits,
        }
        return aux["features"].detach(), side_info

    aux = infinity.condition_codec(condition, image_hw=(h, w))
    side_info = _boundary_side_bits_from_aux(args, aux, h, w)
    return aux["features"].detach(), side_info


def _forward_one_scale_conditioned(infinity, last_stage, cond_BD_or_gss, ca_kv, scale_schedule, si, condition_features=None):
    attn_fn = None
    if getattr(infinity, "use_flex_attn", False):
        attn_fn = infinity.attn_fn_compile_dict.get(tuple(scale_schedule[:(si + 1)]), None)

    condition_scale_tokens = None
    condition_adapter = getattr(infinity, "condition_adapter", None)
    if condition_features is not None and condition_adapter is not None:
        condition_scale_tokens = condition_adapter.make_scale_tokens(condition_features, scale_schedule[si], need_to_pad=0)

    layer_idx = 0
    for chunk_idx, blocks in _iter_chunks(infinity):
        if getattr(infinity, "add_lvl_embeding_only_first_block", 1) and chunk_idx == 0:
            last_stage = infinity.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=0)
        if not getattr(infinity, "add_lvl_embeding_only_first_block", 1):
            last_stage = infinity.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=0)
        for block in blocks:
            if condition_scale_tokens is not None:
                last_stage = last_stage + condition_adapter(condition_scale_tokens, layer_idx, dtype=last_stage.dtype)
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
            layer_idx += 1
    return last_stage


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


def get_prob_conditioned(args, infinity, vae, vae_scale_schedule, prompt, text_tokenizer, text_encoder, gt_ls_Bl, condition_features=None):
    tau_list, cfg_list = _get_tau_cfg_lists(args, vae_scale_schedule)
    B, bs, ca_kv, cond_BD, cond_BD_or_gss, last_stage = _prepare_prompt_cond(infinity, prompt, text_tokenizer, text_encoder, cfg_list)
    if condition_features is not None and bs != B:
        condition_features = condition_features.repeat(bs // B, 1, 1, 1)

    num_stages_minus_1 = len(vae_scale_schedule) - 1
    summed_codes = 0
    prob_list = []

    _toggle_kv_cache(infinity, True)
    try:
        for si, pn in enumerate(vae_scale_schedule):
            last_stage = _forward_one_scale_conditioned(
                infinity, last_stage, cond_BD_or_gss, ca_kv, vae_scale_schedule, si, condition_features=condition_features
            )
            logits_BLV = _get_cfg_logits(infinity, last_stage, cond_BD, cfg_list[si], tau_list[si], B)
            pair_logits_BLD2 = logits_BLV.reshape(B, logits_BLV.shape[1], -1, 2)
            prob_list.append(pair_logits_BLD2.reshape(B, -1, 2).softmax(dim=-1).view(-1, 2))

            idx_Bld = gt_ls_Bl[si].reshape(B, pn[1], pn[2], -1).unsqueeze(1)
            codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type="bit_label")
            summed_codes, next_stage = _codes_to_next_last_stage(
                infinity, vae, codes, summed_codes, vae_scale_schedule, si, num_stages_minus_1,
                apply_spatial_patchify=getattr(args, "apply_spatial_patchify", 0),
            )
            if next_stage is not None:
                last_stage = next_stage.repeat(bs // B, 1, 1)
    finally:
        _toggle_kv_cache(infinity, False)
    return prob_list


def encoding_conditioned(args, infinity, vae, scale_schedule, text, text_tokenizer, text_encoder, gt_ms_idx_Bl, condition_features=None):
    prob_list = get_prob_conditioned(args, infinity, vae, scale_schedule, text, text_tokenizer, text_encoder, gt_ms_idx_Bl, condition_features)
    trans_list, help_list = [], []
    sum_len = 0
    bpp_list = []
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
                hbits.append(1)
                sum_len += len(bits)
            else:
                arithmetic_bits.append(gt_token)
                hbits.append(0)
                sum_len += args.vae_type
        trans_list.append(arithmetic_bits)
        help_list.append(hbits)
        sum_len += len(hbits)
        bpp_list.append(sum_len)
    return trans_list, help_list, bpp_list


def decoding_conditioned(args, infinity, vae, vae_scale_schedule, prompt, text_tokenizer, text_encoder, trans_list, help_list, condition_features=None):
    tau_list, cfg_list = _get_tau_cfg_lists(args, vae_scale_schedule)
    B, bs, ca_kv, cond_BD, cond_BD_or_gss, last_stage = _prepare_prompt_cond(infinity, prompt, text_tokenizer, text_encoder, cfg_list)
    if condition_features is not None and bs != B:
        condition_features = condition_features.repeat(bs // B, 1, 1, 1)

    num_stages_minus_1 = len(vae_scale_schedule) - 1
    summed_codes = 0
    decode_idx = []

    _toggle_kv_cache(infinity, True)
    try:
        for si, pn in enumerate(vae_scale_schedule):
            last_stage = _forward_one_scale_conditioned(
                infinity, last_stage, cond_BD_or_gss, ca_kv, vae_scale_schedule, si, condition_features=condition_features
            )
            logits_BLV = _get_cfg_logits(infinity, last_stage, cond_BD, cfg_list[si], tau_list[si], B)
            pair_logits_BLD2 = logits_BLV.reshape(B, logits_BLV.shape[1], -1, 2)
            prob = pair_logits_BLD2.reshape(B, -1, 2).softmax(dim=-1).view(-1, 2)[:, 0].cpu().tolist()

            decompressed_string = []
            for j, flag in enumerate(help_list[si]):
                p_token = prob[j * args.vae_type:(j + 1) * args.vae_type]
                if flag == 0:
                    decompressed_string.extend(trans_list[si][j])
                elif flag == 1:
                    decompressed_string.extend(decompress_from_bit_list(trans_list[si][j], args.vae_type, p_token))
                else:
                    raise ValueError(f"Unknown flag={flag} at scale={si}, token={j}")

            dec_idx = torch.tensor(decompressed_string, dtype=torch.int32, device=pair_logits_BLD2.device).reshape(B, pn[1] * pn[2], -1)
            decode_idx.append(dec_idx)
            idx_Bld = dec_idx.reshape(B, pn[1], pn[2], -1).unsqueeze(1)
            codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type="bit_label")
            summed_codes, next_stage = _codes_to_next_last_stage(
                infinity, vae, codes, summed_codes, vae_scale_schedule, si, num_stages_minus_1,
                apply_spatial_patchify=getattr(args, "apply_spatial_patchify", 0),
            )
            if next_stage is not None:
                last_stage = next_stage.repeat(bs // B, 1, 1)
    finally:
        _toggle_kv_cache(infinity, False)
    return decode_idx


def decompress_cfg_conditioned(args, infinity, vae, vae_scale_schedule, prompt, text_tokenizer, text_encoder, gt_leak, decoded_idx_list, condition_features=None):
    rng = infinity.rng
    tau_list, cfg_list = _get_tau_cfg_lists(args, vae_scale_schedule)
    B, bs, ca_kv, cond_BD, cond_BD_or_gss, last_stage = _prepare_prompt_cond(infinity, prompt, text_tokenizer, text_encoder, cfg_list)
    if condition_features is not None and bs != B:
        condition_features = condition_features.repeat(bs // B, 1, 1, 1)

    num_stages_minus_1 = len(vae_scale_schedule) - 1
    summed_codes = 0

    _toggle_kv_cache(infinity, True)
    try:
        for si, pn in enumerate(vae_scale_schedule):
            last_stage = _forward_one_scale_conditioned(
                infinity, last_stage, cond_BD_or_gss, ca_kv, vae_scale_schedule, si, condition_features=condition_features
            )
            logits_BLV = _get_cfg_logits(infinity, last_stage, cond_BD, cfg_list[si], tau_list[si], B)
            pair_logits_BLD2 = logits_BLV.reshape(B, logits_BLV.shape[1], -1, 2)

            if si <= gt_leak:
                idx_Bld = decoded_idx_list[si]
            else:
                flat_logits = pair_logits_BLD2.reshape(B, -1, 2)
                idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(
                    flat_logits, rng=rng, top_k=0, top_p=0.0, num_samples=1
                )[:, :, 0]
                idx_Bld = idx_Bld.reshape(B, pair_logits_BLD2.shape[1], -1)

            idx_Bld = idx_Bld.reshape(B, pn[1], pn[2], -1).unsqueeze(1)
            codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type="bit_label")
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


def build_scale_schedule_and_q(args, img_path: str):
    inp = load_img(img_path, args)
    _, _, h, w = inp.shape
    h_div_w = h / w
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]["scales"]
    scale_schedule = [(1, h_, w_) for (_, h_, w_) in scale_schedule]
    scale_q = [
        (scale_schedule[i][0], scale_schedule[i][1], scale_schedule[i][2], int((i + 1) // ((len(scale_schedule) // 3) + 1) + 2))
        for i in range(len(scale_schedule))
    ]
    return inp, h, w, scale_schedule, scale_q


def compress_image_conditioned(args, vae, scale_schedule, scale_q, infinity, text_tokenizer, text_encoder, img_path, text, device="cuda"):
    inp_B3HW = load_img(img_path, args)
    _, _, h, w = inp_B3HW.shape
    condition_features, condition_info = prepare_boundary_condition(args, infinity, vae, scale_schedule, inp_B3HW, h, w, device)
    raw_features, _, _ = vae.encode_for_raw_features(inp_B3HW.to(device), scale_schedule=scale_schedule)
    _, gt_ms_idx_Bl = mask_quant(vae, scale_q, raw_features, device)
    trans_list, help_list, latent_bits_per_scale = encoding_conditioned(
        args, infinity, vae, scale_schedule, text, text_tokenizer, text_encoder, gt_ms_idx_Bl, condition_features=condition_features
    )
    return trans_list, help_list, latent_bits_per_scale, gt_ms_idx_Bl, condition_features, condition_info


def _compute_image_metrics(std_tensor, rec_tensor, lpips_fn, dists_fn, device: str):
    mse = torch.mean((std_tensor - rec_tensor) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse)).item() if mse.item() > 0 else float("inf")
    msssim_val = ms_ssim(std_tensor, rec_tensor, data_range=1.0, size_average=True).item()
    lpips_val = lpips_fn(std_tensor.to(device) * 2 - 1, rec_tensor.to(device) * 2 - 1).item()
    dists_val = dists_fn(std_tensor.to(device), rec_tensor.to(device)).item()
    return {"psnr": psnr, "msssim": msssim_val, "lpips": lpips_val, "dists": dists_val}


def evaluate_one_image(args, infinity, vae, text_tokenizer, text_encoder, img_path, text, rec_base_path, lpips_fn, dists_fn, device: str):
    inp, h, w, scale_schedule, scale_q = build_scale_schedule_and_q(args, img_path)

    prompt_bits = 0
    text_for_decode = text
    if int(args.add_prompt_bits):
        if args.prompt_bits_mode == "arith":
            prompt_ids = _encode_prompt_ids_t5(text_tokenizer, text, max_len=args.tlen)
            packet = _arith_encode_ids_packet(prompt_ids)
            prompt_bits = packet["bits"]
            text_for_decode = _decode_prompt_text_from_ids_t5(text_tokenizer, _arith_decode_ids_packet(packet))
        elif args.prompt_bits_mode == "zlib":
            prompt_bits = _prompt_zlib_bytes(text) * 8
    prompt_bpp = prompt_bits / (h * w) if int(args.add_prompt_bits) else 0.0

    autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16) if device.startswith("cuda") and bool(getattr(args, "bf16", 0)) else nullcontext()
    with torch.no_grad(), autocast_ctx:
        trans_list, help_list, _, gt_ms_idx_Bl, condition_features, condition_info = compress_image_conditioned(
            args, vae, scale_schedule, scale_q, infinity, text_tokenizer, text_encoder, img_path, text, device=device
        )
        decoded_idx = decoding_conditioned(
            args, infinity, vae, scale_schedule, text_for_decode, text_tokenizer, text_encoder,
            trans_list, help_list, condition_features=condition_features,
        )

    std_img_cv = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    std_tensor = torch.tensor(std_img_cv).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    scale_results = {}
    img_name = os.path.basename(img_path)
    for gt_leak in range(len(gt_ms_idx_Bl)):
        with torch.no_grad(), autocast_ctx:
            img = decompress_cfg_conditioned(
                args, infinity, vae, scale_schedule, text_for_decode, text_tokenizer, text_encoder,
                gt_leak, decoded_idx, condition_features=condition_features,
            )
        scale_folder = os.path.join(rec_base_path, f"scale_{gt_leak}")
        os.makedirs(scale_folder, exist_ok=True)
        rec_img_path = os.path.join(scale_folder, img_name)
        torchvision.utils.save_image(img.cpu(), rec_img_path)

        rec_img_cv = cv2.cvtColor(cv2.imread(rec_img_path), cv2.COLOR_BGR2RGB)
        rec_tensor = torch.tensor(rec_img_cv).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        metrics = _compute_image_metrics(std_tensor, rec_tensor, lpips_fn, dists_fn, device)
        latent_bits = _count_bits_until_scale(trans_list, help_list, gt_leak)
        latent_bpp = latent_bits / (h * w)
        condition_bpp = float(condition_info["side_bpp"])
        total_bpp = latent_bpp + prompt_bpp + condition_bpp
        scale_results[str(gt_leak)] = {
            "bpp": total_bpp,
            "latent_bpp": latent_bpp,
            "prompt_bpp": prompt_bpp,
            "condition_bpp": condition_bpp,
            "condition_expected_bpp": float(condition_info["expected_side_bpp"]),
            "condition_hard_expected_bpp": float(condition_info["hard_expected_side_bpp"]),
            "psnr": metrics["psnr"],
            "msssim": metrics["msssim"],
            "lpips": metrics["lpips"],
            "dists": metrics["dists"],
        }

    return {
        "image_name": img_name,
        "text": text,
        "prompt_bpp": prompt_bpp,
        "condition": condition_info,
        "scales_data": scale_results,
    }


def summarize_dataset(dataset_metrics_data: List[Dict[str, Any]]):
    keys = ["bpp", "latent_bpp", "prompt_bpp", "condition_bpp", "condition_expected_bpp", "condition_hard_expected_bpp", "psnr", "msssim", "lpips", "dists"]
    scale_aggregates: Dict[str, Dict[str, List[float]]] = {}
    for item in dataset_metrics_data:
        for scale_idx, metrics in item["scales_data"].items():
            scale_aggregates.setdefault(scale_idx, {k: [] for k in keys})
            for k in keys:
                scale_aggregates[scale_idx][k].append(float(metrics[k]))

    summary = {}
    for scale_idx, vals in scale_aggregates.items():
        summary[scale_idx] = {
            "avg_bpp": float(np.mean(vals["bpp"])),
            "avg_latent_bpp": float(np.mean(vals["latent_bpp"])),
            "avg_prompt_bpp": float(np.mean(vals["prompt_bpp"])),
            "avg_condition_bpp": float(np.mean(vals["condition_bpp"])),
            "avg_condition_expected_bpp": float(np.mean(vals["condition_expected_bpp"])),
            "avg_condition_hard_expected_bpp": float(np.mean(vals["condition_hard_expected_bpp"])),
            "avg_psnr": float(np.mean(vals["psnr"])),
            "avg_msssim": float(np.mean(vals["msssim"])),
            "avg_lpips": float(np.mean(vals["lpips"])),
            "avg_dists": float(np.mean(vals["dists"])),
            "image_count": len(vals["bpp"]),
        }
    return summary


def default_args():
    return argparse.Namespace(
        pn="1M",
        model_path="/workspace/Infinity_codec/local_output/stage2_1024_125M_16vae_depth_8/ar-ckpt-giter020K-ep0-iter20000-last.pth",
        dataset_json="/workspace/ARPC/data/DIV2K.json",
        output_root="/workspace/Infinity_codec/results/depth_condition_eval",
        cfg_insertion_layer=0,
        vae_type=16,
        vae_path="/workspace/CKPT/Infinity/infinity_vae_d16.pth",
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        model_type="infinity_layer12",
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        use_scale_schedule_embedding=0,
        sampling_per_bits=1,
        text_encoder_ckpt="/workspace/CKPT/flan-t5-xl",
        text_channels=2048,
        apply_spatial_patchify=0,
        h_div_w_template=1.000,
        use_flex_attn=0,
        cache_dir="/workspace/Infinity_codec/local_output/boundary_eval_cache",
        enable_model_cache=1,
        checkpoint_type="torch",
        seed=0,
        bf16=0,
        rec_path="",
        add_prompt_bits=1,
        prompt_bits_mode="arith",
        tlen=512,
        tau_list="0.5",
        cfg_list="1",
        keep_gpu_busy=0,
        sweep_all_iters=0,
        start_iter=None,
        limit_images=0,
        enable_boundary_condition=1,
        spatial_cond_type="depth",
        condition_codec_type="vae_token",
        condition_token_scales=2,
        condition_token_scales_list="1,2,3,4,5",
        depth_condition_source="transformers",
        depth_model_name="depth-anything/Depth-Anything-V2-Small-hf",
        depth_model_dtype="fp32",
        depth_model_device="",
        depth_model_cache_dir="",
        seg_condition_source="transformers",
        seg_model_name="facebook/sam-vit-base",
        seg_model_dtype="fp32",
        seg_model_device="",
        seg_model_cache_dir="",
        seg_max_masks=16,
        seg_points_per_batch=32,
        seg_output_mode="region_boundary",
        boundary_cond_size=128,
        boundary_cond_hidden_dim=48,
        boundary_cond_latent_dim=8,
        boundary_cond_feature_dim=32,
        boundary_side_bits_mode="arith",
        boundary_check_decode=0,
    )


def _parse_cli(base_args):
    parser = argparse.ArgumentParser(description="Evaluate Infinity codec with spatial side-condition adapters.")
    for k, v in vars(base_args).items():
        arg = f"--{k}"
        if isinstance(v, bool):
            parser.add_argument(arg, type=int, default=int(v))
        elif isinstance(v, int):
            parser.add_argument(arg, type=int, default=v)
        elif isinstance(v, float):
            parser.add_argument(arg, type=float, default=v)
        elif v is None:
            parser.add_argument(arg, type=str, default="")
        else:
            parser.add_argument(arg, type=str, default=v)
    parsed = parser.parse_args()
    parsed.cfg_insertion_layer = [int(x) for x in str(parsed.cfg_insertion_layer).replace(";", ",").split(",") if str(x).strip()]
    parsed.tau_list = _parse_float_or_list(parsed.tau_list, "tau_list")
    parsed.cfg_list = _parse_float_or_list(parsed.cfg_list, "cfg_list")
    parsed.start_iter = None if str(parsed.start_iter).strip() == "" else int(parsed.start_iter)
    parsed.condition_token_scales_list = _parse_int_list(getattr(parsed, "condition_token_scales_list", ""))
    return parsed


def _list_models_to_test(model_path: str, sweep_all_iters: int, start_iter: Optional[int]):
    if not sweep_all_iters:
        return [model_path]
    model_dir = os.path.dirname(model_path)
    base_filename = os.path.basename(model_path)
    if start_iter is None:
        match_init = re.search(r"-iter(\d+)-", base_filename)
        if not match_init:
            return [model_path]
        start_iter = int(match_init.group(1))
    available_models = []
    for f in os.listdir(model_dir):
        if f.endswith(".pth") and "iter" in f:
            m = re.search(r"-iter(\d+)-", f)
            if m:
                it = int(m.group(1))
                if it >= int(start_iter):
                    available_models.append((it, os.path.join(model_dir, f)))
    available_models.sort(key=lambda x: x[0])
    return [p for _, p in available_models] or [model_path]


def _condition_token_sweep_values(args) -> List[int]:
    if str(getattr(args, "condition_codec_type", "binary") or "binary").lower() != "vae_token":
        return [int(getattr(args, "condition_token_scales", 2))]
    vals = list(getattr(args, "condition_token_scales_list", []) or [])
    if not vals:
        vals = [int(getattr(args, "condition_token_scales", 2))]
    return sorted({max(1, int(v)) for v in vals})


def main():
    args = _parse_cli(default_args())
    from tools.run_infinity_refiner import load_tokenizer, load_transformer, load_visual_tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_device(0)

    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    lpips_evaluator = lpips.LPIPS(net="alex").to(device)
    dists_evaluator = DISTS().to(device)

    with open(args.dataset_json, "rt", encoding="utf-8") as f:
        json_data = [json.loads(line) for line in f]
    if int(args.limit_images) > 0:
        json_data = json_data[:int(args.limit_images)]

    models_to_test = _list_models_to_test(args.model_path, int(args.sweep_all_iters), args.start_iter)
    print(f"[*] Found {len(models_to_test)} model(s) to test.")
    condition_token_sweep_values = _condition_token_sweep_values(args)
    if len(condition_token_sweep_values) > 1:
        print(f"[*] Sweeping condition_token_scales={condition_token_sweep_values}")

    for current_model_path in models_to_test:
        args.model_path = current_model_path
        m = re.search(r"-iter(\d+)-", os.path.basename(current_model_path))
        current_iter = int(m.group(1)) if m else -1
        base_rec_path = os.path.join(args.output_root, f"iter_{current_iter}" if current_iter >= 0 else os.path.splitext(os.path.basename(current_model_path))[0])

        print("=" * 80)
        print(f"[*] Evaluating: {current_model_path}")
        print("=" * 80)

        infinity = load_transformer(vae, args)
        boundary_meta = maybe_load_boundary_conditioning(args, infinity, current_model_path, device)

        for cond_token_scales in condition_token_sweep_values:
            if str(getattr(args, "condition_codec_type", "binary") or "binary").lower() == "vae_token":
                args.condition_token_scales = int(cond_token_scales)
            args.rec_path = base_rec_path if len(condition_token_sweep_values) == 1 else os.path.join(base_rec_path, f"condtok_{int(cond_token_scales)}")
            os.makedirs(args.rec_path, exist_ok=True)
            if len(condition_token_sweep_values) > 1:
                print(f"[*] condition_token_scales={args.condition_token_scales} -> {args.rec_path}")

            dataset_metrics_data = []
            for idx, data in enumerate(json_data):
                img_path = data.get("img_path", data.get("path"))
                text = data.get("txt", data.get("text", data.get("caption", "")))
                if not img_path:
                    raise KeyError(f"Dataset row {idx} has no img_path/path field.")
                print(f"[{idx + 1}/{len(json_data)}] {os.path.basename(img_path)}")
                result = evaluate_one_image(
                    args, infinity, vae, text_tokenizer, text_encoder, img_path, text,
                    args.rec_path, lpips_evaluator, dists_evaluator, device,
                )
                dataset_metrics_data.append(result)
                bpps = [f"{v['bpp']:.6f}" for _, v in sorted(result["scales_data"].items(), key=lambda kv: int(kv[0]))]
                cond_bpp = result["condition"]["side_bpp"]
                print(f"    condition bpp: {cond_bpp:.6f} | scale total bpp: {bpps}")

            average_metrics_summary = summarize_dataset(dataset_metrics_data)
            print(f"\n[*] Iter {current_iter} summary:")
            for scale_idx, item in sorted(average_metrics_summary.items(), key=lambda kv: int(kv[0])):
                print(
                    f"    scale {scale_idx} | BPP: {item['avg_bpp']:.6f} "
                    f"(latent {item['avg_latent_bpp']:.6f} + cond {item['avg_condition_bpp']:.6f}) | "
                    f"PSNR: {item['avg_psnr']:.4f} | MS-SSIM: {item['avg_msssim']:.4f} | "
                    f"LPIPS: {item['avg_lpips']:.4f} | DISTS: {item['avg_dists']:.4f}"
                )

            final_json_output = {
                "model_path": current_model_path,
                "model_iter": current_iter,
                "boundary_condition_enabled": bool(int(args.enable_boundary_condition)),
                "spatial_cond_type": getattr(args, "spatial_cond_type", "boundary"),
                "condition_codec_type": getattr(args, "condition_codec_type", "binary"),
                "condition_token_scales": int(getattr(args, "condition_token_scales", cond_token_scales)),
                "depth_condition_source": getattr(args, "depth_condition_source", "proxy"),
                "seg_condition_source": getattr(args, "seg_condition_source", "transformers"),
                "seg_output_mode": getattr(args, "seg_output_mode", "region_boundary"),
                "boundary_condition_meta": boundary_meta,
                "boundary_side_bits_mode": args.boundary_side_bits_mode,
                "tau_list": args.tau_list,
                "cfg_list": args.cfg_list,
                "summary": average_metrics_summary,
                "details": dataset_metrics_data,
            }
            json_save_path = os.path.join(args.rec_path, f"metrics_iter_{current_iter}.json" if current_iter >= 0 else "metrics.json")
            with open(json_save_path, "w", encoding="utf-8") as f:
                json.dump(final_json_output, f, indent=2, ensure_ascii=False)

            df_summary = pd.DataFrame(average_metrics_summary).T
            df_summary.index.name = "scale"
            csv_save_path = os.path.join(args.rec_path, f"avg_metrics_iter_{current_iter}.csv" if current_iter >= 0 else "avg_metrics.csv")
            df_summary.to_csv(csv_save_path)
            print(f"[*] Saved JSON to {json_save_path}")
            print(f"[*] Saved CSV  to {csv_save_path}\n")

    print("[*] All models finished.")


if __name__ == "__main__":
    main()
