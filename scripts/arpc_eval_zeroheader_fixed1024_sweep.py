"""Evaluate ARPC ZERO-header (NONE strategy) on a dataset.

This variant is specialized for your setup:
- NONE masking
- ZERO header stream (payload-only)
- fixed resolution (default: 1024x1024)
- k_transmit is NOT stored in the stream (must be consistent for enc/dec)

Metrics:
  - PSNR / SSIM / MS-SSIM
  - LPIPS (optional)
  - DISTS (optional)
"""

from __future__ import annotations

import json
import zlib
import argparse
import csv
import os
import time
from glob import glob
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# Reuse build & preprocessing utilities from the CLI
from scripts.arpc_cli_zeroheader_fixed1024_sweep import build_codec, _resize_center_crop, _to_tensor_minus1_1


# -------------------------
# Optional perceptual metrics
# -------------------------

def _try_build_lpips(device: torch.device, net: str = "alex"):
    """Return a callable lpips(x_01, y_01)->float or None."""
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
    mse = torch.mean((x - y) ** 2).clamp_min(eps)
    return float(10.0 * torch.log10(1.0 / mse).item())


def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5, device: Optional[torch.device] = None) -> torch.Tensor:
    coords = torch.arange(window_size, device=device).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    k2d = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
    return k2d


def _ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> float:
    device = x.device
    k = _gaussian_kernel(window_size, sigma, device=device).repeat(3, 1, 1, 1)

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
    return float((num / den.clamp_min(1e-12)).mean().item())


def _ssim_cs(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5):
    device = x.device
    k = _gaussian_kernel(window_size, sigma, device=device).repeat(3, 1, 1, 1)

    mu_x = torch.nn.functional.conv2d(x, k, padding=window_size // 2, groups=3)
    mu_y = torch.nn.functional.conv2d(y, k, padding=window_size // 2, groups=3)

    sigma_x2 = torch.nn.functional.conv2d(x * x, k, padding=window_size // 2, groups=3) - mu_x * mu_x
    sigma_y2 = torch.nn.functional.conv2d(y * y, k, padding=window_size // 2, groups=3) - mu_y * mu_y
    sigma_xy = torch.nn.functional.conv2d(x * y, k, padding=window_size // 2, groups=3) - mu_x * mu_y

    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x2 + sigma_y2 + C2)).clamp_min(1e-12)
    cs_map = (2 * sigma_xy + C2) / (sigma_x2 + sigma_y2 + C2).clamp_min(1e-12)

    return ssim_map.mean(dim=(1,2,3)), cs_map.mean(dim=(1,2,3))


def _ms_ssim(x: torch.Tensor, y: torch.Tensor, levels: int = 5) -> float:
    weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=x.device)[:levels]
    mssim = []
    mcs = []
    xx, yy = x, y
    for _ in range(levels):
        ssim_val, cs_val = _ssim_cs(xx, yy)
        mssim.append(ssim_val)
        mcs.append(cs_val)
        xx = torch.nn.functional.avg_pool2d(xx, 2, 2)
        yy = torch.nn.functional.avg_pool2d(yy, 2, 2)

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)
    return float(((mcs[:-1] ** weights[:-1, None]).prod(dim=0) * (mssim[-1] ** weights[-1])).mean().item())


# -------------------------
# Helpers
# -------------------------

def list_images(data_dir: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
    files: List[str] = []
    for e in exts:
        files.extend(glob(os.path.join(data_dir, e)))
    return sorted(files)


def load_and_preprocess(path: str, fixed_hw: int) -> Tuple[torch.Tensor, int, int]:
    pil = Image.open(path).convert("RGB")
    pil = _resize_center_crop(pil, fixed_hw, fixed_hw)
    ten = _to_tensor_minus1_1(pil)
    return ten, fixed_hw, fixed_hw


def to_01(t: torch.Tensor) -> torch.Tensor:
    return (t.clamp(-1, 1) + 1.0) / 2.0


def _load_caption_db(caption_json: str):
    with open(caption_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    by_path, by_base = {}, {}
    for k, v in data.items():
        cap = v[0] if isinstance(v, list) and len(v) else (v if isinstance(v, str) else "")
        cap = (cap or "").strip()
        if not cap:
            continue
        key = os.path.realpath(os.path.abspath(k))
        by_path[key] = cap
        by_base.setdefault(os.path.basename(key), cap)
    return by_path, by_base


def _get_caption(img_path: str, by_path: dict, by_base: dict, fallback: str) -> str:
    if not by_path:
        return fallback
    key = os.path.realpath(os.path.abspath(img_path))
    if key in by_path:
        return by_path[key]
    base = os.path.basename(key)
    if base in by_base:
        return by_base[base]
    return fallback


def _prompt_zlib_bits(prompt: str) -> int:
    b = prompt.encode("utf-8", errors="ignore")
    return len(zlib.compress(b)) * 8


def main():
    p = argparse.ArgumentParser(description="Evaluate ARPC ZERO-header codec with strategy/keep_ratio sweeps.")

    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument("--work_dir", type=str, default="./_arpc_eval_work")
    p.add_argument("--save_recon_dir", type=str, default=None)

    # shared ZERO-header assumptions
    p.add_argument("--fixed_hw", type=int, default=1024)
    p.add_argument("--k_list", type=str, default="5")

    # sweep controls
    p.add_argument("--mask_strategies", type=str, default="none",
                   help="Comma-separated list. e.g. 'none,entropy_spatial,entropy_channel,entropy_scale'")
    p.add_argument("--keep_ratios", type=str, default="",
                   help="Comma-separated list of keep_ratio values for strategies that use it (entropy_channel/spatial). "
                        "e.g. '0.02,0.05,0.1'. Empty -> use only one run per strategy.")
    p.add_argument("--mask_params", type=str, default="{}",
                   help="Base JSON dict for mask params. keep_ratio will be overridden per run if --keep_ratios is set.")
    p.add_argument("--active_bits", type=str, default="all",
                   help="Shared active bits spec (0-header). e.g. 'all' or '16'.")

    # prompt / captions
    p.add_argument("--caption_json", type=str, default="")
    p.add_argument("--prompt", type=str, default="a clear photo")
    p.add_argument("--print_prompts", type=int, default=0, choices=[0, 1])
    p.add_argument("--add_prompt_bits", type=int, default=1, choices=[0, 1])
    p.add_argument("--prompt_bits_mode", type=str, default="zlib", choices=["zlib", "none"])

    # model / vae
    p.add_argument("--device", type=str, default="cuda", choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vae_type", type=int, default=32)
    p.add_argument("--apply_spatial_patchify", type=int, default=0, choices=[0, 1])

    p.add_argument("--vae_ckpt", type=str, required=True)
    p.add_argument("--model_ckpt", type=str, required=True)
    p.add_argument("--text_encoder_ckpt", type=str, default="google/flan-t5-xl")

    p.add_argument("--tlen", type=int, default=512)
    p.add_argument("--text_channels", type=int, default=2048)
    p.add_argument("--model_type", type=str, default="infinity_2b")

    p.add_argument("--use_bit_label", type=int, default=1, choices=[0, 1])
    p.add_argument("--use_flex_attn", type=int, default=0, choices=[0, 1])
    p.add_argument("--bf16", type=int, default=0, choices=[0, 1])

    p.add_argument("--add_lvl_embeding_only_first_block", type=int, default=0, choices=[0, 1])
    p.add_argument("--rope2d_each_sa_layer", type=int, default=1, choices=[0, 1])
    p.add_argument("--rope2d_normalized_by_hw", type=int, default=2, choices=[0, 1, 2])

    # metrics
    p.add_argument("--lpips_net", type=str, default="alex", choices=["alex", "vgg", "squeeze"])

    args = p.parse_args()
    args.bf16 = bool(int(args.bf16))
    args.use_flex_attn = bool(int(args.use_flex_attn))

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else (
        "cpu" if args.device == "cpu" else "cuda"
    ))

    os.makedirs(args.work_dir, exist_ok=True)
    if args.save_recon_dir:
        os.makedirs(args.save_recon_dir, exist_ok=True)

    # captions (optional)
    caption_by_path, caption_by_base = None, None
    if args.caption_json:
        try:
            caption_by_path, caption_by_base = _load_caption_db(args.caption_json)
        except Exception as e:
            print(f"[caption] failed to load {args.caption_json}: {e}")
            caption_by_path, caption_by_base = None, None

    # build codec once; override k_transmit/mask each run
    args.k_transmit = 1
    codec = build_codec(args, device)

    # parse sweeps
    strategies = [s.strip() for s in str(args.mask_strategies).split(",") if s.strip()]
    if not strategies:
        strategies = ["none"]

    keep_ratios = []
    if str(args.keep_ratios).strip():
        for x in str(args.keep_ratios).split(","):
            x = x.strip()
            if not x:
                continue
            keep_ratios.append(float(x))
    # Base mask_params dict
    try:
        base_params = json.loads(str(args.mask_params or "{}"))
        if not isinstance(base_params, dict):
            base_params = {}
    except Exception:
        base_params = {}

    lpips_fn = _try_build_lpips(device, net=args.lpips_net)
    dists_fn = _try_build_dists(device)

    k_list = [int(x.strip()) for x in str(args.k_list).split(",") if x.strip()]
    images = list_images(args.data_dir)
    if not images:
        raise RuntimeError(f"No images found in {args.data_dir}")

    fields = [
        "image","H","W",
        "mask_strategy","keep_ratio","active_bits",
        "k_transmit","bytes","bpp_img","prompt_bits","prompt_bpp","bpp",
        "psnr","ssim","ms_ssim","lpips","dists",
        "enc_s","dec_s"
    ]

    with open(args.out_csv, "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=fields)
        w.writeheader()

        for img_path in images:
            name = os.path.basename(img_path)
            stem = os.path.splitext(name)[0]

            pil = Image.open(img_path).convert("RGB")
            pil = _resize_center_crop(pil, int(args.fixed_hw), int(args.fixed_hw))
            x = _to_tensor_minus1_1(pil).to(device)
            H, W = int(x.shape[2]), int(x.shape[3])
            x_01 = to_01(x)

            for k in k_list:
                for ms in strategies:
                    ms = ms.strip()

                    # decide ratio sweep list for this strategy
                    if ms in ("entropy_channel", "entropy_spatial") and keep_ratios:
                        ratio_list = keep_ratios
                    else:
                        ratio_list = [None]

                    for kr in ratio_list:
                        # set codec run-time config (ZERO-header; must match encoder/decoder)
                        if hasattr(codec, "k_transmit"):
                            codec.k_transmit = int(k)
                        if hasattr(codec, "mask_strategy"):
                            codec.mask_strategy = str(ms)
                        if hasattr(codec, "active_bits_spec"):
                            codec.active_bits_spec = str(args.active_bits or "all")

                        run_params = dict(base_params)
                        if kr is not None and ms in ("entropy_channel", "entropy_spatial"):
                            run_params["keep_ratio"] = float(kr)
                        if hasattr(codec, "mask_params"):
                            codec.mask_params = run_params

                        # prompt / caption
                        prompt_text = str(args.prompt)
                        if caption_by_path is not None:
                            prompt_text = caption_by_path.get(img_path, prompt_text)
                        if caption_by_base is not None:
                            prompt_text = caption_by_base.get(name, prompt_text)

                        if int(args.print_prompts):
                            print(f"[prompt] {name} :: {prompt_text}")

                        prompt_bits = 0
                        if args.prompt_bits_mode == "zlib" and int(args.add_prompt_bits):
                            prompt_bits = _prompt_zlib_bits(prompt_text)
                        prompt_bpp = (prompt_bits / (H * W)) if int(args.add_prompt_bits) else 0.0

                        tag_ms = ms.replace("/", "_")
                        tag_kr = "auto" if kr is None else f"{kr:.6f}".rstrip("0").rstrip(".")
                        # Organized filenames to avoid overwriting and make metric evaluation easier
                        stream_name = f"{stem}.k{k}.ms_{tag_ms}.kr_{tag_kr}.arpc"
                        tag_ab = str(args.active_bits or "all").replace("/", "_")

                        # Save streams into a structured folder
                        stream_dir = os.path.join(args.work_dir, "streams", f"ab_{tag_ab}", f"ms_{tag_ms}", f"k{k}", f"r{tag_kr}")
                        os.makedirs(stream_dir, exist_ok=True)
                        stream_path = os.path.join(stream_dir, stream_name)

                        # Save recon images into a structured folder (like arpc_eval_entropy_mask_lpips_dists.py)
                        if args.save_recon_dir:
                            recon_dir = os.path.join(args.save_recon_dir, f"ab_{tag_ab}", f"ms_{tag_ms}", f"k{k}", f"r{tag_kr}")
                        else:
                            recon_dir = os.path.join(args.work_dir, "recon", f"ab_{tag_ab}", f"ms_{tag_ms}", f"k{k}", f"r{tag_kr}")
                        os.makedirs(recon_dir, exist_ok=True)
                        recon_path = os.path.join(recon_dir, f"{stem}.png")
                        # compress
                        t0 = time.time()
                        codec.compress(img_B3HW=x, out_stream_path=stream_path, prompt=prompt_text)
                        enc_s = time.time() - t0

                        nbytes = os.path.getsize(stream_path)
                        bpp_img = (nbytes * 8.0) / (H * W)
                        bpp = bpp_img + prompt_bpp

                        # decompress
                        t1 = time.time()
                        y_u8 = codec.decompress(stream_path=stream_path, out_path=recon_path, prompt=prompt_text)
                        dec_s = time.time() - t1

                        y_01 = (y_u8.permute(0,3,1,2).float().to(device) / 255.0).clamp(0,1)

                        psnr = _psnr(x_01, y_01)
                        ssim = _ssim(x_01, y_01)
                        ms_ssim = _ms_ssim(x_01, y_01)
                        lp = float("nan") if lpips_fn is None else float(lpips_fn(x_01, y_01))
                        di = float("nan") if dists_fn is None else float(dists_fn(x_01, y_01))

                        row = dict(
                            image=name, H=H, W=W,
                            mask_strategy=ms, keep_ratio=("" if kr is None else float(kr)),
                            active_bits=str(args.active_bits or "all"),
                            k_transmit=int(k), bytes=int(nbytes),
                            bpp_img=float(bpp_img),
                            prompt_bits=int(prompt_bits),
                            prompt_bpp=float(prompt_bpp),
                            bpp=float(bpp),
                            psnr=float(psnr),
                            ssim=float(ssim),
                            ms_ssim=float(ms_ssim),
                            lpips=float(lp),
                            dists=float(di),
                            enc_s=float(enc_s),
                            dec_s=float(dec_s),
                        )
                        w.writerow(row)
                        fcsv.flush()

                        msg = f"[{name}] ms={ms} kr={tag_kr} k={k} -> {bpp:.8f} bpp | PSNR {psnr:.2f} | SSIM {ssim:.4f} | MS-SSIM {ms_ssim:.4f}"
                        if lpips_fn is not None:
                            msg += f" | LPIPS {lp:.4f}"
                        if dists_fn is not None:
                            msg += f" | DISTS {di:.4f}"
                        print(msg)

if __name__ == "__main__":
    main()
