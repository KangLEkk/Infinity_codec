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
# from torchmetrics.image import (
#     FrechetInceptionDistance,
#     KernelInceptionDistance,
#     LearnedPerceptualImagePatchSimilarity,
# )
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


def _parse_float_after_prefix(s: str, prefix: str):
    # "r0.1" -> 0.1, "k8" -> 8.0
    if not s.startswith(prefix):
        return math.nan
    try:
        return float(s[len(prefix):])
    except Exception:
        return math.nan


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
        #（你也可以把 autocast 打开，但部分指标可能数值略有差异；这里保守起见不开）
        for k, metric in metric_paired.items():
            v = float(metric(recon_tensor, gt_tensor).item())
            result_sum[k] = result_sum.get(k, 0.0) + v

    out = {k: (v / n) for k, v in result_sum.items()}

    # FID/KID only when enough pairs (默认 >50，和你原脚本一致)
    if n >= fid_min_pairs:
        out["fid"] = float(fid_metric.compute())
        kid_mean, kid_std = kid_metric.compute()
        out["kid_mean"] = float(kid_mean)
        out["kid_std"] = float(kid_std)

    # 防止显存碎片（多次循环时有用）
    torch.cuda.empty_cache()
    return out


def _discover_k_folders(root: Path, k_names=None):
    if k_names is not None and len(k_names) > 0:
        ks = [root / k for k in k_names]
    else:
        ks = sorted([p for p in root.glob("k*") if p.is_dir()],
                    key=lambda p: _parse_float_after_prefix(p.name, "k"))
    if len(ks) == 0:
        raise RuntimeError(f"No k-folders found under: {root}")
    return ks


def _discover_r_folders(k_folder: Path, r_names=None):
    if r_names is not None and len(r_names) > 0:
        rs = [k_folder / r for r in r_names]
    else:
        rs = sorted([p for p in k_folder.glob("r*") if p.is_dir()],
                    key=lambda p: _parse_float_after_prefix(p.name, "r"))
    if len(rs) == 0:
        raise RuntimeError(f"No r-folders found under: {k_folder}")
    return rs


def _make_style_for_r(idx: int):
    color = DEFAULT_PALETTE[idx % len(DEFAULT_PALETTE)]
    return {"color": color, "linestyle": "-", "marker": "o", "smooth": False}


def _save_output(obj: dict, out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".json":
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    else:
        # Save as a python file with a dict variable
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# Auto-generated results\n")
            f.write("results = ")
            f.write(repr(obj))
            f.write("\n")


def parse_args(argv):
    p = argparse.ArgumentParser("Evaluate all k1..k8 and group results by rate folder rX.")
    p.add_argument("--root", type=str, default="/workspace/Infinity_codec/results/3/ab_all/ms_none",
                   help="Root folder containing k1..k8, e.g. .../ms_entropy_spatial")
    p.add_argument("--gt_dir", type=str, default="/workspace/data/DIV2K_1024",
                   help="Ground-truth folder")
    p.add_argument("--k", nargs="*", default=["k2","k3","k4","k5","k6","k7","k8"],
                   help="Optional: specify k folders explicitly, e.g. k1 k2 ... k8")
    p.add_argument("--r", nargs="*", default=["rauto"],
                   help="Optional: specify r folders explicitly, e.g. r0.1 r0.02 ...")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--ntest", type=int, default=None,
                   help="Optional: evaluate first N matched pairs only")
    p.add_argument("--fid_min_pairs", type=int, default=51,
                   help="Compute FID/KID only if matched pairs >= this value (default: 51)")
    p.add_argument("--codec_name", type=str, default="VAR_codec",
                   help="Prefix for output keys, default: VAR_codec")
    p.add_argument("--out", type=str, default="ms_rdproxy_spatial_global3.py",
                   help="Output path (.py or .json). Default: eval_groupby_r.py")
    p.add_argument("--bpp_mode", type=str, default="kindex", choices=["kname", "kindex"],
                   help='How to fill "bpp" list: kname -> ["k1","k2"...], kindex -> [1,2,...]')
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    root = Path(args.root)
    gt_dir = Path(args.gt_dir)
    out_path = Path(args.out)

    ks = _discover_k_folders(root, args.k)
    # 用第一个 k 来发现 r（也可手动传 --r）
    rs0 = _discover_r_folders(ks[0], args.r)

    k_names = [k.name for k in ks]
    r_names = [r.name for r in rs0]

    # 你要的最终结构：按 r 聚合（每个 r 下有 8 个 k 的指标）
    grouped = {}

    for ridx, rname in enumerate(r_names):
        key = f"{args.codec_name}_{rname}"
        style = _make_style_for_r(ridx)

        if args.bpp_mode == "kindex":
            bpp_list = [int(_parse_float_after_prefix(k, "k")) for k in k_names]
        else:
            # 按你示例：把 k1..k8 当成 bpp list 的占位符/标签
            bpp_list = k_names.copy()

        entry = {
            "bpp": bpp_list,
            "lpips": [],
            "dists": [],
            "fid": [],
            # 你也可以顺手存更多（不影响你后续使用）
            "psnr": [],
            "ms_ssim": [],
            **style,
        }

        print(f"\n==================== Group: {key} ====================")
        for k in ks:
            recon_dir = k / rname
            if not recon_dir.is_dir():
                print(f"[Warn] missing recon_dir: {recon_dir} -> fill None")
                entry["lpips"].append(None)
                entry["dists"].append(None)
                entry["fid"].append(None)
                entry["psnr"].append(None)
                entry["ms_ssim"].append(None)
                continue

            try:
                res = evaluate_one(
                    recon_dir=recon_dir,
                    gt_dir=gt_dir,
                    ntest=args.ntest,
                    device=args.device,
                    fid_min_pairs=args.fid_min_pairs,
                )
            except Exception as e:
                print(f"[Error] eval failed for {recon_dir}: {e} -> fill None")
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

            print(f"[{k.name}/{rname}]  "
                  f"lpips={res.get('lpips', None)}  "
                  f"dists={res.get('dists', None)}  "
                  f"fid={res.get('fid', None)}  "
                  f"psnr={res.get('psnr', None)}  "
                  f"ms_ssim={res.get('ms_ssim', None)}")

        grouped[key] = entry

    _save_output(grouped, out_path)
    print(f"\nSaved to: {out_path.resolve()}")
    # 同时打印一个可直接复制的 dict（可选）
    print("\n========== OUTPUT DICT (preview) ==========")
    print(repr(grouped))


if __name__ == "__main__":
    main(sys.argv[1:])
