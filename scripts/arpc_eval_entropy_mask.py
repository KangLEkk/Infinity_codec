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


def main():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument("--work_dir", type=str, default="./_arpc_eval_work")
    p.add_argument("--save_recon_dir", type=str, default=None)

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

    with open(args.out_csv, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        for img_path in images:
            name = os.path.basename(img_path)
            stem = os.path.splitext(name)[0]

            x_m11, H, W = load_and_preprocess(img_path, args.pn, args.h_div_w_template, args.keep_original, 16)
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
                            prompt=args.prompt,
                            out_path=stream_path,
                            k_transmit=int(k_transmit),
                            active_bits_spec=str(args.active_bits),
                            mask_strategy=strategy,
                            mask_params=mask_params,
                            seed=int(args.seed),
                        )
                        enc_s = time.time() - t0

                        nbytes = os.path.getsize(stream_path)
                        bpp = (nbytes * 8.0) / (H * W)

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
                        fcsv.flush()

                        msg = (
                            f"[{name}] k={k_transmit} {strategy} r={r} -> {bpp:.6f} bpp | "
                            f"PSNR {psnr:.2f} | SSIM {ssim:.4f} | MS-SSIM {ms_ssim:.4f}"
                        )
                        if lpips_fn is not None:
                            msg += f" | LPIPS {lpips_v:.4f}"
                        if dists_fn is not None:
                            msg += f" | DISTS {dists_v:.4f}"
                        print(msg)


if __name__ == "__main__":
    main()
