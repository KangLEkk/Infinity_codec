import sys
import json
import math
import argparse
import tqdm
import torch
import pyiqa

from pathlib import Path
from PIL import Image
from torchvision.transforms import ToTensor
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from neuralcompression.metrics import update_patch_fid


IMG_GLOB = "*.[jpJP][pnPN]*[gG]"  # jpg/jpeg/png (case-insensitive)

DEFAULT_PALETTE = [
    "#6baed6", "#9ecae1", "#c6dbef", "#3182bd", "#08519c",
    "#74c476", "#31a354", "#006d2c", "#fd8d3c", "#e6550d",
]


def _list_images(folder: Path):
    return sorted([x for x in folder.glob(IMG_GLOB)])


def _build_stem_map(paths):
    m = {}
    for p in paths:
        m[p.stem] = p
    return m


@torch.no_grad()
def evaluate_one(recon_dir: Path, gt_dir: Path, ntest=None, device="cuda", fid_min_pairs=51):
    """
    Return dict with metrics averaged on matched pairs:
      psnr, dists, ms_ssim, lpips, (fid, kid_mean, kid_std if pairs >= fid_min_pairs)
    """
    device = torch.device(device)
    totensor = ToTensor()

    recon_dir = Path(recon_dir)
    gt_dir = Path(gt_dir)

    if (not recon_dir.is_dir()) or (not gt_dir.is_dir()):
        raise FileNotFoundError(f"Missing dir: recon={recon_dir} gt={gt_dir}")

    recon_paths = _list_images(recon_dir)
    gt_paths = _list_images(gt_dir)

    if len(recon_paths) == 0:
        raise RuntimeError(f"No images in recon_dir: {recon_dir}")
    if len(gt_paths) == 0:
        raise RuntimeError(f"No images in gt_dir: {gt_dir}")

    recon_map = _build_stem_map(recon_paths)
    gt_map = _build_stem_map(gt_paths)

    common = sorted(set(recon_map.keys()) & set(gt_map.keys()))
    if ntest is not None:
        common = common[:ntest]
    if len(common) == 0:
        raise RuntimeError(
            f"No matched pairs between {recon_dir} and {gt_dir}. "
            f"(Matched by filename stem.)"
        )

    # paired metrics
    metric_paired = {
        "psnr": pyiqa.create_metric("psnr").to(device),
        "dists": pyiqa.create_metric("dists").to(device),
        "ms_ssim": pyiqa.create_metric("ms_ssim").to(device),
        "lpips": LearnedPerceptualImagePatchSimilarity(normalize=True).to(device),
    }

    fid_metric = FrechetInceptionDistance().to(device)
    kid_metric = KernelInceptionDistance().to(device)

    result_sum = {}
    n = len(common)

    for stem in tqdm.tqdm(common, desc=f"Eval {recon_dir.parent.name}/{recon_dir.name}", dynamic_ncols=True):
        recon_path = recon_map[stem]
        gt_path = gt_map[stem]

        with open(recon_path, "rb") as f:
            image_recon = Image.open(f).convert("RGB")
        with open(gt_path, "rb") as f:
            image_gt = Image.open(f).convert("RGB")

        recon_tensor = totensor(image_recon).unsqueeze(0).to(device)
        gt_tensor = totensor(image_gt).unsqueeze(0).to(device)

        # Update patch FID/KID
        update_patch_fid(gt_tensor, recon_tensor, fid_metric=fid_metric, kid_metric=kid_metric)

        # Paired metrics
        for k, metric in metric_paired.items():
            v = float(metric(recon_tensor, gt_tensor).item())
            result_sum[k] = result_sum.get(k, 0.0) + v

    out = {k: (v / n) for k, v in result_sum.items()}

    # FID/KID only when enough pairs
    if n >= fid_min_pairs:
        out["fid"] = float(fid_metric.compute())
        kid_mean, kid_std = kid_metric.compute()
        out["kid_mean"] = float(kid_mean)
        out["kid_std"] = float(kid_std)

    # 防止显存碎片
    torch.cuda.empty_cache()
    return out


def _discover_scale_folders(root: Path, scale_names=None):
    if scale_names is not None and len(scale_names) > 0:
        scales = [root / s for s in scale_names]
    else:
        # 自动扫描 scale_0 ... scale_12，并按数字大小排序
        def get_scale_idx(p):
            try:
                return int(p.name.split('_')[1])
            except (IndexError, ValueError):
                return -1
                
        scales = sorted(
            [p for p in root.glob("scale_*") if p.is_dir()],
            key=get_scale_idx
        )
    if len(scales) == 0:
        raise RuntimeError(f"No scale folders found under: {root}")
    return scales


def _save_output(obj: dict, out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".json":
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# Auto-generated results\n")
            f.write("results = ")
            f.write(repr(obj))
            f.write("\n")


def parse_args(argv):
    p = argparse.ArgumentParser("Evaluate scale_0 to scale_12 results.")
    p.add_argument("--root", type=str, default="/workspace/Infinity_codec/results/DIV2K_dynamic_18000",
                   help="Root folder containing scale_X directories.")
    p.add_argument("--gt_dir", type=str, default="/workspace/data/DIV2K_1024",
                   help="Ground-truth folder")
    p.add_argument("--scales", nargs="*", default=[],
                   help="Optional: specify scale folders explicitly, e.g. scale_0 scale_1 ...")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--ntest", type=int, default=None,
                   help="Optional: evaluate first N matched pairs only")
    p.add_argument("--fid_min_pairs", type=int, default=51,
                   help="Compute FID/KID only if matched pairs >= this value (default: 51)")
    p.add_argument("--codec_name", type=str, default="ARPC_scales",
                   help="Prefix for output keys, default: ARPC_scales")
    p.add_argument("--out", type=str, default="arpc_scale_results_18000.py",
                   help="Output path (.py or .json). Default: arpc_scale_results_dynamic.py")
    p.add_argument("--x_axis_mode", type=str, default="index", choices=["name", "index"],
                   help='How to fill the list: name -> ["scale_0",...], index -> [0,1,...]')
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    root = Path(args.root)
    gt_dir = Path(args.gt_dir)
    out_path = Path(args.out)

    scale_dirs = _discover_scale_folders(root, args.scales)
    scale_names = [s.name for s in scale_dirs]

    grouped = {}
    key = f"{args.codec_name}"
    style = {"color": DEFAULT_PALETTE[3], "linestyle": "-", "marker": "o", "smooth": False}

    if args.x_axis_mode == "index":
        x_list = [int(s.split('_')[1]) for s in scale_names]
    else:
        x_list = scale_names.copy()

    entry = {
        "bpp": x_list,  # 保留 bpp 键位以兼容你可能有的旧画图脚本
        "scales": x_list,
        "lpips": [],
        "dists": [],
        "fid": [],
        "psnr": [],
        "ms_ssim": [],
        **style,
    }

    print(f"\n==================== Group: {key} ====================")
    for scale_dir in scale_dirs:
        if not scale_dir.is_dir():
            print(f"[Warn] missing scale_dir: {scale_dir} -> fill None")
            entry["lpips"].append(None)
            entry["dists"].append(None)
            entry["fid"].append(None)
            entry["psnr"].append(None)
            entry["ms_ssim"].append(None)
            continue

        try:
            res = evaluate_one(
                recon_dir=scale_dir,
                gt_dir=gt_dir,
                ntest=args.ntest,
                device=args.device,
                fid_min_pairs=args.fid_min_pairs,
            )
        except Exception as e:
            print(f"[Error] eval failed for {scale_dir}: {e} -> fill None")
            entry["lpips"].append(None)
            entry["dists"].append(None)
            entry["fid"].append(None)
            entry["psnr"].append(None)
            entry["ms_ssim"].append(None)
            continue

        entry["lpips"].append(res.get("lpips", None))
        entry["dists"].append(res.get("dists", None))
        entry["fid"].append(res.get("fid", None))
        entry["psnr"].append(res.get("psnr", None))
        entry["ms_ssim"].append(res.get("ms_ssim", None))

        print(f"[{scale_dir.name}]  "
              f"lpips={res.get('lpips', None)}  "
              f"dists={res.get('dists', None)}  "
              f"fid={res.get('fid', None)}  "
              f"psnr={res.get('psnr', None)}  "
              f"ms_ssim={res.get('ms_ssim', None)}")

    grouped[key] = entry

    _save_output(grouped, out_path)
    print(f"\nSaved to: {out_path.resolve()}")
    print("\n========== OUTPUT DICT (preview) ==========")
    print(repr(grouped))


if __name__ == "__main__":
    main(sys.argv[1:])