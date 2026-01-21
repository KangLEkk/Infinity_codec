"""Evaluate ARPC entropy-masking variants.

This script runs:
  image -> compress(.arpc) -> decompress(recon)
then reports bitrate (bpp) and distortion metrics:
  - PSNR / SSIM / MS-SSIM
  - LPIPS (optional)
  - DISTS (optional)

It is designed for **quick ablations** of the following stage-1 policies:
- none (baseline GM-BMSRQ / transmit all active bits)
- entropy_channel
- entropy_scale
- entropy_spatial

Example
-------
  python -m scripts.arpc_eval_entropy_mask \
    --data_dir /path/to/kodak \
    --vae_ckpt /path/to/vae.ckpt \
    --model_ckpt /path/to/infinity_ar.ckpt \
    --strategies none,entropy_channel,entropy_spatial \
    --keep_ratios 0.25,0.5,0.75 \
    --k_list 1,3,5 \
    --pn 1M \
    --out_csv results.csv \
    --save_recon_dir recon

Notes
-----
- B=1 only.
- We resize+center-crop to Infinity's dynamic resolution template (pn + ratio).
- Recon images are saved into subfolders by (pn/template/strategy/k/keep_ratio) to avoid overwriting.

Optional dependencies
---------------------
To enable perceptual metrics, install one of the following:
  - piq:        pip install piq
  - lpips:      pip install lpips
  - DISTS:      pip install DISTS-pytorch

If not installed, LPIPS/DISTS columns will be NaN.
"""

from __future__ import annotations

import json
import zlib
import argparse
import csv
import math
import os
import time
from glob import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Reuse the exact build & preprocessing utilities from the CLI
from scripts.arpc_cli import build_codec, _get_target_hw, _resize_center_crop, _to_tensor_minus1_1


# -------------------------
# Optional perceptual metrics
# -------------------------


def _try_build_lpips(device: torch.device, net: str = "alex"):
    """Return a callable lpips(x_01, y_01)->float or None."""
    # Preference order:
    #   1) official lpips (expects [-1,1])
    #   2) piq.LPIPS (expects [0,1])
    try:
        import lpips  # type: ignore

        m = lpips.LPIPS(net=net).to(device).eval()

        @torch.no_grad()
        def _lp(x01: torch.Tensor, y01: torch.Tensor) -> float:
            x = (x01 * 2.0 - 1.0).clamp(-1, 1)
            y = (y01 * 2.0 - 1.0).clamp(-1, 1)
            v = m(x, y)
            return float(v.mean().item())

        return _lp
    except Exception:
        pass

    try:
        import piq  # type: ignore

        m = piq.LPIPS(replace_pooling=True, reduction="none").to(device).eval()

        @torch.no_grad()
        def _lp(x01: torch.Tensor, y01: torch.Tensor) -> float:
            v = m(x01, y01)
            return float(v.mean().item())

        return _lp
    except Exception:
        return None


def _try_build_dists(device: torch.device):
    """Return a callable dists(x_01, y_01)->float or None."""
    # Preference order:
    #   1) piq.DISTS (expects [0,1])
    #   2) DISTS-pytorch (expects [0,1])
    try:
        import piq  # type: ignore

        m = piq.DISTS(reduction="none").to(device).eval()

        @torch.no_grad()
        def _d(x01: torch.Tensor, y01: torch.Tensor) -> float:
            v = m(x01, y01)
            return float(v.mean().item())

        return _d
    except Exception:
        pass

    try:
        from DISTS_pytorch import DISTS  # type: ignore

        m = DISTS().to(device).eval()

        @torch.no_grad()
        def _d(x01: torch.Tensor, y01: torch.Tensor) -> float:
            v = m(x01, y01)
            return float(v.mean().item())

        return _d
    except Exception:
        return None


# -------------------------
# Metrics (pure torch)
# -------------------------


def _psnr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    """PSNR on [0,1] tensors, shape [1,3,H,W]."""
    mse = torch.mean((x - y) ** 2).clamp_min(eps)
    return float(10.0 * torch.log10(1.0 / mse).item())


def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5, device: Optional[torch.device] = None) -> torch.Tensor:
    coords = torch.arange(window_size, device=device).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    k2d = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)  # [1,1,ws,ws]
    return k2d


def _ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> float:
    """SSIM on [0,1], shape [1,3,H,W]."""
    device = x.device
    k = _gaussian_kernel(window_size, sigma, device=device)
    k = k.repeat(3, 1, 1, 1)  # [3,1,ws,ws]

    mu_x = torch.nn.functional.conv2d(x, k, padding=window_size // 2, groups=3)
    mu_y = torch.nn.functional.conv2d(y, k, padding=window_size // 2, groups=3)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = torch.nn.functional.conv2d(x * x, k, padding=window_size // 2, groups=3) - mu_x2
    sigma_y2 = torch.nn.functional.conv2d(y * y, k, padding=window_size // 2, groups=3) - mu_y2
    sigma_xy = torch.nn.functional.conv2d(x * y, k, padding=window_size // 2, groups=3) - mu_xy

    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / den.clamp_min(1e-12)
    return float(ssim_map.mean().item())


def _ssim_cs(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (ssim, cs) per-image for MS-SSIM."""
    device = x.device
    k = _gaussian_kernel(window_size, sigma, device=device)
    k = k.repeat(3, 1, 1, 1)

    mu_x = torch.nn.functional.conv2d(x, k, padding=window_size // 2, groups=3)
    mu_y = torch.nn.functional.conv2d(y, k, padding=window_size // 2, groups=3)

    sigma_x2 = torch.nn.functional.conv2d(x * x, k, padding=window_size // 2, groups=3) - mu_x * mu_x
    sigma_y2 = torch.nn.functional.conv2d(y * y, k, padding=window_size // 2, groups=3) - mu_y * mu_y
    sigma_xy = torch.nn.functional.conv2d(x * y, k, padding=window_size // 2, groups=3) - mu_x * mu_y

    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x2 + sigma_y2 + C2)).clamp_min(1e-12)
    cs_map = (2 * sigma_xy + C2) / (sigma_x2 + sigma_y2 + C2).clamp_min(1e-12)

    ssim_val = ssim_map.mean(dim=(1, 2, 3))
    cs_val = cs_map.mean(dim=(1, 2, 3))
    return ssim_val, cs_val


def _ms_ssim(x: torch.Tensor, y: torch.Tensor, levels: int = 5) -> float:
    """MS-SSIM on [0,1], shape [1,3,H,W]."""
    weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=x.device)
    weights = weights[:levels]

    mssim = []
    mcs = []

    xx = x
    yy = y
    for _ in range(levels):
        ssim_val, cs_val = _ssim_cs(xx, yy)
        mssim.append(ssim_val)
        mcs.append(cs_val)
        # downsample
        xx = torch.nn.functional.avg_pool2d(xx, kernel_size=2, stride=2, padding=0)
        yy = torch.nn.functional.avg_pool2d(yy, kernel_size=2, stride=2, padding=0)

    mssim = torch.stack(mssim)  # [L, B]
    mcs = torch.stack(mcs)

    # MS-SSIM = prod(cs_i^w_i) * ssim_L^w_L
    pow1 = (mcs[:-1] ** weights[:-1, None]).prod(dim=0)
    pow2 = (mssim[-1] ** weights[-1]).view(-1)
    return float((pow1 * pow2).mean().item())


# -------------------------
# Helpers
# -------------------------


def _parse_floats(spec: str) -> List[float]:
    return [float(x.strip()) for x in (spec or "").split(",") if x.strip()]


def _parse_ints(spec: str) -> List[int]:
    return [int(x.strip()) for x in (spec or "").split(",") if x.strip()]


def list_images(data_dir: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
    files: List[str] = []
    for e in exts:
        files.extend(glob(os.path.join(data_dir, e)))
    files = sorted(files)
    return files


def load_and_preprocess(
    path: str,
    pn: str,
    ratio: float,
    keep_original: bool = False,
    pad_to: int = 16,
) -> Tuple[torch.Tensor, int, int]:
    pil = Image.open(path).convert("RGB")

    # Default behavior: follow Infinity template (resize+center-crop)
    tgt_h, tgt_w = _get_target_hw(pn, ratio)
    pil = _resize_center_crop(pil, tgt_h, tgt_w)

    ten = _to_tensor_minus1_1(pil)  # [1,3,H,W] in [-1,1]
    return ten, tgt_h, tgt_w


def to_01(t: torch.Tensor) -> torch.Tensor:
    return (t.clamp(-1, 1) + 1.0) / 2.0


def _tag(x) -> str:
    """Filesystem-safe tag."""
    s = str(x)
    return (
        s.replace(" ", "")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(".", "p")
        .replace(":", "_")
        .replace("|", "_")
    )


def _ratio_tag(r) -> str:
    if r is None:
        return "auto"
    return f"{float(r):.3f}".rstrip("0").rstrip(".").replace(".", "p")


# -------------------------
# Main
# -------------------------


def _safe_tag(x) -> str:
    """Make a filesystem-safe tag."""
    s = str(x)
    s = s.replace("/", "_").replace(" ", "")
    s = s.replace(":", "_").replace(",", "_")
    s = s.replace("(", "").replace(")", "")
    s = s.replace("+", "p").replace("-", "m")
    s = s.replace(".", "p")
    return s

def _norm_path(p: str) -> str:
    return os.path.realpath(os.path.abspath(p))

def _safe_float(x):
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")

def _mean_ignore_nan(vals):
    vals = [v for v in vals if v == v]  # NaN check
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)

def _print_summary_means(rows, group_keys=("strategy", "keep_ratio", "k_transmit")):
    """Print mean metrics for each group and overall."""
    if not rows:
        print("[summary] no rows to summarize.")
        return

    # numeric fields we care about (only average if present)
    num_fields = [
        "bpp_img", "prompt_bpp", "bpp", "psnr", "ssim", "ms_ssim", "lpips", "dists",
        "prompt_bits"
    ]

    # overall
    print("\n" + "=" * 80)
    print("[summary] Overall means across all images/settings")
    overall = {}
    for f in num_fields:
        overall[f] = _mean_ignore_nan([_safe_float(r.get(f, float("nan"))) for r in rows])
    overall_line = " | ".join([f"{k}={overall[k]:.6g}" for k in num_fields if overall[k] == overall[k]])
    print("  " + overall_line)
    print("=" * 80)

    # grouped
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        key = tuple(r.get(k, None) for k in group_keys)
        groups[key].append(r)

    # sort keys for stable output
    def _key_sort(k):
        # strategy (str), keep_ratio (float), k (int)
        kk = []
        for x in k:
            if isinstance(x, (int, float)):
                kk.append(x)
            else:
                kk.append(str(x))
        return tuple(kk)

    print("[summary] Group means by " + ",".join(group_keys))
    for key in sorted(groups.keys(), key=_key_sort):
        rs = groups[key]
        means = {f: _mean_ignore_nan([_safe_float(r.get(f, float("nan"))) for r in rs]) for f in num_fields}
        # build label
        label = ", ".join([f"{k}={v}" for k, v in zip(group_keys, key)])
        line = " | ".join([f"{f}={means[f]:.6g}" for f in ["bpp","psnr","ssim","ms_ssim","lpips","dists"] if means[f]==means[f]])
        print(f"  [{label}] N={len(rs)} -> {line}")
    print("=" * 80 + "\n")



def _load_caption_db(caption_json: str):
    """Load caption json: {image_path: [caption,...], ...}. Returns (by_path, by_basename)."""
    with open(caption_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    by_path = {}
    by_base = {}
    for k, v in data.items():
        if isinstance(v, list) and len(v) > 0:
            cap = v[0]
        elif isinstance(v, str):
            cap = v
        else:
            continue
        if not isinstance(cap, str) or not cap.strip():
            continue
        nk = _norm_path(k)
        cap = cap.strip()
        by_path[nk] = cap
        by_base.setdefault(os.path.basename(nk), cap)
    return by_path, by_base

def _get_caption(img_path: str, by_path: dict, by_base: dict, fallback: str) -> str:
    if not by_path:
        return fallback
    key = _norm_path(img_path)
    if key in by_path:
        return by_path[key]
    base = os.path.basename(key)
    if base in by_base:
        return by_base[base]
    return fallback

def _try_build_hf_tokenizer(text_encoder_ckpt: str = ""):
    """Try to build a tokenizer. Prefer open_clip tokenizer if available (nn_indices style)."""
    try:
        import prompt_inversion.open_clip as open_clip  # type: ignore
        tok = open_clip.tokenizer._tokenizer
        return tok, "open_clip"
    except Exception:
        pass
    if text_encoder_ckpt:
        try:
            from transformers import AutoTokenizer  # type: ignore
            tok = AutoTokenizer.from_pretrained(text_encoder_ckpt)
            return tok, "hf"
        except Exception:
            pass
    return None, "none"

def _encode_prompt_ids(tokenizer, prompt: str, max_len: int = 77):
    """Encode prompt -> token ids. Works for open_clip tokenizer and HF tokenizer."""
    if tokenizer is None:
        return []
    try:
        ids = tokenizer.encode(prompt)
    except Exception:
        try:
            ids = tokenizer.encode(prompt, add_special_tokens=True)
        except Exception:
            return []
    if max_len is not None and max_len > 0:
        ids = ids[:max_len]
    return list(map(int, ids))

def _prompt_zlib_bytes(prompt: str) -> int:
    b = prompt.encode("utf-8", errors="ignore")
    return len(zlib.compress(b))

def _arith_bits_from_ids(ids):
    """Arithmetic-code token ids like nn_indices.py (char-level) and return payload bits."""
    if ids is None or len(ids) == 0:
        return 0
    text = ",".join(map(str, ids))
    # char freqs over digits and comma
    char_freq = {}
    for ch in text:
        if ch.isdigit() or ch == ",":
            char_freq[ch] = char_freq.get(ch, 0) + 1
    if not char_freq:
        return 0
    try:
        import torch
        import torchac  # type: ignore

        total = sum(char_freq.values())
        unique_chars = sorted(char_freq.keys())
        prob = [char_freq.get(c, 0) / total for c in unique_chars]

        cdf = torch.zeros(len(unique_chars) + 1, dtype=torch.float32)
        cdf[1:] = torch.cumsum(torch.tensor(prob, dtype=torch.float32), dim=0)
        cdf[-1] = 1.0

        L = len(text)
        cdf = cdf.view(1, 1, -1).expand(1, L, -1).contiguous()
        sym = torch.tensor([unique_chars.index(ch) for ch in text], dtype=torch.int16).view(1, -1)

        encoded = torchac.encode_float_cdf(cdf, sym, check_input_bounds=True)
        return len(encoded) * 8
    except Exception:
        # fallback: zlib as approximation
        return _prompt_zlib_bytes(text) * 8


def main():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--out_csv", type=str, required=True)

    p.add_argument("--print_summary", type=int, default=1, help="Print mean metrics summary at the end (0/1).")
    p.add_argument("--work_dir", type=str, default="./_arpc_eval_work")
    p.add_argument("--save_recon_dir", type=str, default=None)


# captions / prompt coding
    p.add_argument("--caption_json", type=str, default="/data/home/keanle/datasets/caption/DIV2K_captions_768.json", help="JSON mapping image path -> [caption].")
    p.add_argument("--caption_fallback", type=str, default="", help="Fallback caption if not found; default to --prompt.")
    p.add_argument("--print_prompts", type=int, default=1, help="Print per-image prompt for debugging (0/1).")
    # p.add_argument("--text_encoder_ckpt", type=str, default="/data/home/keanle/pretrain_model/flan-t5-xl", help="HF tokenizer ckpt for encoding prompt ids (optional).")
    # p.add_argument("--tlen", type=int, default=512, help="Max token length when encoding prompts.")
    p.add_argument("--prompt_bits_mode", type=str, default="arith", choices=["arith", "zlib", "none"], help="How to estimate prompt bits.")
    p.add_argument("--add_prompt_bits", type=int, default=1, help="Add prompt bits to bpp (0/1).")


    # sweep
    p.add_argument("--strategies", type=str, default="none,entropy_channel,entropy_spatial")
    p.add_argument("--keep_ratios", type=str, default="0.25,0.5")
    p.add_argument("--k_list", type=str, default="1,3,5")

    # schedule/model
    p.add_argument("--device", type=str, default="cuda", choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", type=str, default="a clear photo")

    p.add_argument("--pn", type=str, default="1M", choices=["0.06M", "0.25M", "0.60M", "1M"])
    p.add_argument("--h_div_w_template", type=float, default=1.0)
    p.add_argument("--keep_original", type=int, default=0, choices=[0, 1])
    p.add_argument("--active_bits", type=str, default="default")

    p.add_argument("--vae_type", type=int, default=32)
    p.add_argument("--apply_spatial_patchify", type=int, default=0, choices=[0, 1])

    p.add_argument("--vae_ckpt", type=str, required=True)
    p.add_argument("--model_ckpt", type=str, required=True)
    p.add_argument("--text_encoder_ckpt", type=str, default="/data/home/keanle/pretrain_model/flan-t5-xl")

    p.add_argument("--tlen", type=int, default=512)
    p.add_argument("--text_channels", type=int, default=2048)
    p.add_argument("--model_type", type=str, default="infinity_2b")

    p.add_argument("--use_bit_label", type=int, default=1, choices=[0, 1])
    p.add_argument("--use_flex_attn", type=int, default=0, choices=[0, 1])
    p.add_argument("--bf16", type=int, default=0, choices=[0, 1])

    p.add_argument("--add_lvl_embeding_only_first_block", type=int, default=0, choices=[0, 1])
    p.add_argument("--rope2d_each_sa_layer", type=int, default=1, choices=[0, 1])
    p.add_argument("--rope2d_normalized_by_hw", type=int, default=2, choices=[0, 1, 2])

    # perceptual metrics
    p.add_argument("--lpips_net", type=str, default="alex", choices=["alex", "vgg", "squeeze"], help="LPIPS backbone (if lpips is installed)")

    args = p.parse_args()
    args.bf16 = bool(int(args.bf16))
    args.keep_original = bool(int(args.keep_original))
    args.use_flex_attn = bool(int(args.use_flex_attn))

    os.makedirs(args.work_dir, exist_ok=True)
    if args.save_recon_dir:
        os.makedirs(args.save_recon_dir, exist_ok=True)

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "cpu" else "cuda")
    )

    # ---- captions / prompts ----
    caption_by_path, caption_by_base = None, None
    if args.caption_json:
        try:
            caption_by_path, caption_by_base = _load_caption_db(args.caption_json)
            print(f"[caption] loaded {len(caption_by_path)} captions from {args.caption_json}")
        except Exception as e:
            print(f"[caption] failed to load {args.caption_json}: {e}")
            caption_by_path, caption_by_base = None, None

    tok, tok_kind = _try_build_hf_tokenizer(args.text_encoder_ckpt)
    if tok is None:
        print("[prompt] tokenizer not available; prompt bits will fallback to zlib/none")
    else:
        print(f"[prompt] tokenizer={tok_kind}")

    # build codec once
        codec = build_codec(args, device)

        # optional metrics
        lpips_fn = _try_build_lpips(device, net=args.lpips_net)
        dists_fn = _try_build_dists(device)
        if lpips_fn is None:
            print("[WARN] LPIPS not available (install: pip install lpips OR piq). Will write NaN.")
        if dists_fn is None:
            print("[WARN] DISTS not available (install: pip install piq OR DISTS-pytorch). Will write NaN.")

        strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
        keep_ratios = _parse_floats(args.keep_ratios)
        k_list = _parse_ints(args.k_list)

        images = list_images(args.data_dir)
        if not images:
            raise RuntimeError(f"No images found in {args.data_dir}")

        fieldnames = [
            "image",
            "pn",
            "H",
            "W",
            "strategy",
            "keep_ratio",
            "k_transmit",
            "bytes",
            "bpp_img",
            "prompt_bits",
            "prompt_bpp",
            "bpp",
            "psnr",
            "ssim",
            "ms_ssim",
            "lpips",
            "dists",
            "enc_s",
            "dec_s",
        ]

        pn_tag = _tag(args.pn)
        tmpl_tag = _tag(args.h_div_w_template)

        summary_rows = []  # for mean stats

        with open(args.out_csv, "w", newline="") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
            writer.writeheader()

            for img_path in images:
                name = os.path.basename(img_path)
                stem = os.path.splitext(name)[0]

                x_m11, H, W = load_and_preprocess(img_path, args.pn, args.h_div_w_template, args.keep_original, 16)
                # choose per-image prompt from caption_json (if provided)
                fallback_prompt = args.caption_fallback if args.caption_fallback else args.prompt
                prompt_text = _get_caption(img_path, caption_by_path, caption_by_base, fallback=fallback_prompt)
                if args.print_prompts:
                    print(f"[prompt] {os.path.basename(img_path)} :: {prompt_text}")

                prompt_ids = _encode_prompt_ids(tok, prompt_text, max_len=int(args.tlen))
                if args.prompt_bits_mode == "arith":
                    prompt_bits = _arith_bits_from_ids(prompt_ids)
                elif args.prompt_bits_mode == "zlib":
                    prompt_bits = _prompt_zlib_bytes(prompt_text) * 8
                else:
                    prompt_bits = 0

                x_m11 = x_m11.to(device)

                for k_transmit in k_list:
                    for strategy in strategies:
                        if strategy == "none":
                            ratios = [1.0]
                        elif strategy == "entropy_scale":
                            ratios = [None]
                        else:
                            ratios = keep_ratios

                        for r in ratios:
                            r_tag = _ratio_tag(r)

                            mask_params: Dict = {}
                            if strategy in ("entropy_channel", "entropy_spatial"):
                                mask_params["keep_ratio"] = float(r)
                            elif strategy == "entropy_scale":
                                # default parameters; keep_ratio derived from entropy
                                mask_params = {"min_keep_ratio": 0.2, "max_keep_ratio": 1.0, "gamma": 1.0}

                            # Make per-setting unique filenames to avoid overwriting
                            stream_name = f"{stem}.pn{pn_tag}.tmpl{tmpl_tag}.k{k_transmit}.{strategy}.r{r_tag}.arpc"
                            recon_name = f"{stem}.png"  # cleaner inside folder

                            stream_path = os.path.join(args.work_dir, stream_name)

                            if args.save_recon_dir:
                                recon_dir = os.path.join(
                                    args.save_recon_dir,
                                    f"pn_{pn_tag}",
                                    f"tmpl_{tmpl_tag}",
                                    strategy,
                                    f"k{k_transmit}",
                                    f"r{r_tag}",
                                )
                                os.makedirs(recon_dir, exist_ok=True)
                                recon_path = os.path.join(recon_dir, recon_name)
                            else:
                                recon_path = os.path.join(args.work_dir, f"{stem}.k{k_transmit}.{strategy}.r{r_tag}.png")

                            # encode
                            t0 = time.time()
                            codec.compress(
                                img_B3HW=x_m11,
                                prompt=prompt_text,
                                out_path=stream_path,
                                k_transmit=int(k_transmit),
                                active_bits_spec=str(args.active_bits),
                                mask_strategy=strategy,
                                mask_params=mask_params,
                                seed=int(args.seed),
                            )
                            enc_s = time.time() - t0

                            nbytes = os.path.getsize(stream_path)
                            bpp_img = (nbytes * 8.0) / (H * W)
                            prompt_bpp = (prompt_bits / (H * W)) if args.add_prompt_bits else 0.0
                            bpp = bpp_img + prompt_bpp

                            # decode
                            t1 = time.time()
                            y_u8 = codec.decompress(stream_path=stream_path, out_path=recon_path)
                            dec_s = time.time() - t1

                            # metrics
                            y_01 = (y_u8.permute(0, 3, 1, 2).float().to(device) / 255.0).clamp(0, 1)
                            x_01 = to_01(x_m11)

                            psnr = _psnr(x_01, y_01)
                            ssim = _ssim(x_01, y_01)
                            ms_ssim = _ms_ssim(x_01, y_01)

                            lpips_v = float("nan")
                            dists_v = float("nan")
                            if lpips_fn is not None:
                                lpips_v = float(lpips_fn(x_01, y_01))
                            if dists_fn is not None:
                                dists_v = float(dists_fn(x_01, y_01))

                            row = {
                                "image": name,
                                "pn": args.pn,
                                "H": H,
                                "W": W,
                                "strategy": strategy,
                                "keep_ratio": ("" if r is None else float(r)),
                                "k_transmit": int(k_transmit),
                                "bytes": int(nbytes),
                                "bpp_img": float(bpp_img),
                                "prompt_bits": int(prompt_bits),
                                "prompt_bpp": float(prompt_bpp),
                                "bpp": float(bpp),
                                "psnr": float(psnr),
                                "ssim": float(ssim),
                                "ms_ssim": float(ms_ssim),
                                "lpips": float(lpips_v),
                                "dists": float(dists_v),
                                "enc_s": float(enc_s),
                                "dec_s": float(dec_s),
                            }
                            writer.writerow(row)
                            summary_rows.append(row)
                            fcsv.flush()

                            msg = (
                                f"[{name}] k={k_transmit} {strategy} r={r} -> {bpp:.6f} bpp (img {bpp_img:.6f} + prompt {prompt_bpp:.6f}) | "
                                f"PSNR {psnr:.2f} | SSIM {ssim:.4f} | MS-SSIM {ms_ssim:.4f}"
                            )
                            if lpips_fn is not None:
                                msg += f" | LPIPS {lpips_v:.4f}"
                            if dists_fn is not None:
                                msg += f" | DISTS {dists_v:.4f}"
                            print(msg)





        if getattr(args, "print_summary", 1):
            _print_summary_means(summary_rows, group_keys=("strategy", "keep_ratio", "k_transmit"))

if __name__ == "__main__":
    main()
