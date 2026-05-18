import argparse
import json
import os
import sys
import zlib

os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.append("/workspace/Infinity_codec")

import cv2
import lpips
import numpy as np
import pandas as pd
import torch
import torchvision
from DISTS_pytorch import DISTS
from pytorch_msssim import ms_ssim

from compression.reciever import decoding, decompress_cfg
from compression.sender import encoding
from compression.util import h_div_w_templates, load_img, mask_quant
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w
from tools.run_infinity import load_tokenizer, load_transformer, load_visual_tokenizer


def _parse_ratio_list(text: str):
    out = []
    for item in str(text or "").split(","):
        item = item.strip()
        if not item:
            continue
        out.append(float(item))
    return out if out else [1.0]


def _format_setting_tag(strategy: str, keep_ratio: float) -> str:
    return f"{strategy}_keep_{keep_ratio:.2f}".replace(".", "p")


def _encode_prompt_ids_t5(tokenizer, prompt: str, max_len: int = 512):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len, add_special_tokens=True)
    return list(map(int, enc["input_ids"][0].tolist()))


def _arith_encode_ids_packet(ids):
    import torchac

    if ids is None or len(ids) == 0:
        return {
            "payload": b"",
            "cdf_1d": torch.tensor([0.0, 1.0], dtype=torch.float32),
            "unique_chars": [],
            "char_len": 0,
            "ids_len": 0,
            "bits": 0,
        }

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


def build_gt_tokens(args, vae, scale_schedule, scale_q, img_path, device):
    inp_B3HW = load_img(img_path, args)
    raw_features, _, _ = vae.encode_for_raw_features(inp_B3HW.to(device), scale_schedule=scale_schedule)
    _, gt_ms_idx_Bl = mask_quant(vae, scale_q, raw_features, device)
    return inp_B3HW, gt_ms_idx_Bl


def progressive_evaluate(
    args,
    infinity,
    vae,
    scale_schedule,
    text,
    text_tokenizer,
    text_encoder,
    decoded_packet,
    packet_meta,
    rec_base_path,
    setting_tag,
    img_path,
    img_name,
    device,
    h,
    w,
    prompt_bpp,
    lpips_fn,
    dists_fn,
):
    std_img_cv = cv2.imread(img_path)
    std_img_cv = cv2.cvtColor(std_img_cv, cv2.COLOR_BGR2RGB)
    std_tensor = torch.tensor(std_img_cv).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    cumulative_bits = packet_meta.get("cumulative_total_bits", [])
    max_scale = min(len(scale_schedule), int(packet_meta.get("k_transmit", len(scale_schedule))))
    scale_results = {}

    for gt_leak in range(max_scale):
        current_img_bits = float(cumulative_bits[gt_leak]) if gt_leak < len(cumulative_bits) else 0.0
        current_total_bpp = current_img_bits / float(h * w) + float(prompt_bpp)

        scale_folder = os.path.join(rec_base_path, setting_tag, f"scale_{gt_leak}")
        os.makedirs(scale_folder, exist_ok=True)
        img = decompress_cfg(
            infinity,
            vae,
            scale_schedule,
            text,
            text_tokenizer,
            text_encoder,
            gt_leak,
            decoded_packet,
            cfg_list=getattr(args, "progressive_cfg", 3.0),
            tau_list=getattr(args, "progressive_tau", 0.5),
            cfg_insertion_layer=[getattr(args, "cfg_insertion_layer", 0)],
            decode_mode=getattr(args, "codec_progressive_fill_mode", "map"),
        )
        rec_img_path = os.path.join(scale_folder, img_name)
        torchvision.utils.save_image(img.cpu(), rec_img_path)

        rec_img_cv = cv2.imread(rec_img_path)
        rec_img_cv = cv2.cvtColor(rec_img_cv, cv2.COLOR_BGR2RGB)
        rec_tensor = torch.tensor(rec_img_cv).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        mse = torch.mean((std_tensor - rec_tensor) ** 2)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse)).item() if mse.item() > 0 else float("inf")
        msssim_val = ms_ssim(std_tensor, rec_tensor, data_range=1.0, size_average=True).item()
        lpips_val = lpips_fn(std_tensor.to(device) * 2 - 1, rec_tensor.to(device) * 2 - 1).item()
        dists_val = dists_fn(std_tensor.to(device), rec_tensor.to(device)).item()

        scale_results[str(gt_leak)] = {
            "bpp": current_total_bpp,
            "psnr": psnr,
            "msssim": msssim_val,
            "lpips": lpips_val,
            "dists": dists_val,
        }

    return scale_results


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default='/workspace/Infinity_codec/local_output/debug_stage2_student_1024_3/slim-ckpt-giter004K-ep0-iter4000-last.pth')
    p.add_argument("--vae_path", type=str, default='/workspace/Infinity_codec/outputs/bitvae_tok_stage1_dino0.1_8bs_dynamic/checkpoints/model_step_249999.ckpt')
    p.add_argument("--text_encoder_ckpt", type=str, default='/workspace/CKPT/flan-t5-xl')
    p.add_argument("--data_json", type=str, default='/workspace/ARPC/data/DIV2K.json')
    p.add_argument("--rec_path", type=str, default='/workspace/Infinity_codec/results/dynamic_3_new/DIV2K-sample_rdproxy_spatial')

    p.add_argument("--pn", type=str, default="1M")
    p.add_argument("--vae_type", type=int, default=32)
    p.add_argument("--model_type", type=str, default="infinity_2b")
    p.add_argument("--cfg_insertion_layer", type=int, default=0)
    p.add_argument("--add_lvl_embeding_only_first_block", type=int, default=1)
    p.add_argument("--use_bit_label", type=int, default=1)
    p.add_argument("--rope2d_each_sa_layer", type=int, default=1)
    p.add_argument("--rope2d_normalized_by_hw", type=int, default=2)
    p.add_argument("--use_scale_schedule_embedding", type=int, default=0)
    p.add_argument("--sampling_per_bits", type=int, default=1)
    p.add_argument("--text_channels", type=int, default=2048)
    p.add_argument("--apply_spatial_patchify", type=int, default=0)
    p.add_argument("--h_div_w_template", type=float, default=1.0)
    p.add_argument("--use_flex_attn", type=int, default=0)
    p.add_argument("--cache_dir", type=str, default="/workspace/Infinity_codec/local_output/debug_stage2_student_1024——2")
    p.add_argument("--enable_model_cache", type=int, default=0)
    p.add_argument("--checkpoint_type", type=str, default="torch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bf16", type=int, default=0)
    p.add_argument("--tlen", type=int, default=512)

    p.add_argument("--add_prompt_bits", type=int, default=1)
    p.add_argument("--prompt_bits_mode", type=str, default="arith", choices=["arith", "zlib"])

    p.add_argument("--codec_mask_strategy", type=str, default="rdproxy_spatial")
    p.add_argument("--codec_keep_ratios", type=str, default="0.10,0.25,0.50")
    p.add_argument("--codec_k_transmit", type=int, default=0)
    p.add_argument("--codec_active_bits", type=str, default="all")
    p.add_argument("--codec_flag_bits", type=int, default=1)
    p.add_argument("--codec_entropy_thr", type=float, default=None)
    p.add_argument("--codec_score_thr", type=float, default=None)
    p.add_argument("--codec_score_type", type=str, default="rd_ratio")
    p.add_argument("--codec_lambda_rd", type=float, default=0.0)
    p.add_argument("--codec_chan_weight", type=str, default="uniform")
    p.add_argument("--codec_scale_weight", type=str, default="area")
    p.add_argument("--codec_min_keep_per_scale", type=int, default=1)
    p.add_argument("--codec_min_keep_ratio", type=float, default=0.2)
    p.add_argument("--codec_max_keep_ratio", type=float, default=1.0)
    p.add_argument("--codec_gamma", type=float, default=1.0)
    p.add_argument("--codec_fill_mode", type=str, default="auto", choices=["auto", "map", "prev_nearest"])
    p.add_argument("--codec_texture_weight", type=float, default=0.5)
    p.add_argument("--codec_structure_weight", type=float, default=0.5)
    p.add_argument("--codec_pca_rank", type=int, default=4)
    p.add_argument("--codec_texture_kernel", type=str, default="laplacian")
    p.add_argument("--codec_progressive_fill_mode", type=str, default="sample", choices=["map", "sample", "adaptive"])

    p.add_argument("--progressive_cfg", type=float, default=3.0)
    p.add_argument("--progressive_tau", type=float, default=0.5)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_device(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(args.rec_path, exist_ok=True)

    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    infinity = load_transformer(vae, args)

    lpips_evaluator = lpips.LPIPS(net="alex").to(device)
    dists_evaluator = DISTS().to(device)
    keep_ratios = _parse_ratio_list(args.codec_keep_ratios)
    if str(args.codec_mask_strategy).lower() == "none":
        keep_ratios = [1.0]

    json_data = []
    with open(args.data_json, "rt") as f:
        for line in f:
            json_data.append(json.loads(line))

    results_by_setting = {}

    for data in json_data:
        img_path, text = data["img_path"], data["txt"]
        img_name = os.path.basename(img_path)

        inp_B3HW = load_img(img_path, args)
        _, _, h, w = inp_B3HW.shape
        h_div_w = h / w
        h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
        scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]["scales"]
        scale_schedule = [(1, hh, ww) for (_, hh, ww) in scale_schedule]
        scale_q = [
            (scale_schedule[i][0], scale_schedule[i][1], scale_schedule[i][2], int((i + 1) // ((len(scale_schedule) // 3) + 1) + 2))
            for i in range(len(scale_schedule))
        ]

        prompt_bits = 0
        text_for_decode = text
        if args.add_prompt_bits:
            if args.prompt_bits_mode == "arith":
                prompt_ids = _encode_prompt_ids_t5(text_tokenizer, text, max_len=args.tlen)
                prompt_packet = _arith_encode_ids_packet(prompt_ids)
                prompt_bits = prompt_packet["bits"]
                recovered_ids = _arith_decode_ids_packet(prompt_packet)
                assert recovered_ids == prompt_ids, "Prompt arithmetic decode mismatch"
                text_for_decode = _decode_prompt_text_from_ids_t5(text_tokenizer, recovered_ids)
            else:
                prompt_bits = _prompt_zlib_bytes(text) * 8
        prompt_bpp = prompt_bits / float(h * w) if args.add_prompt_bits else 0.0

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=(device == "cuda")):
            _, gt_ms_idx_Bl = build_gt_tokens(args, vae, scale_schedule, scale_q, img_path, device)

            for keep_ratio in keep_ratios:
                args.codec_keep_ratio = keep_ratio
                setting_tag = _format_setting_tag(args.codec_mask_strategy, keep_ratio)

                trans_list, help_list, packet_meta = encoding(
                    args,
                    infinity,
                    vae,
                    scale_schedule,
                    text,
                    text_tokenizer,
                    text_encoder,
                    gt_ms_idx_Bl,
                )
                decoded_packet = decoding(
                    args,
                    infinity,
                    vae,
                    scale_schedule,
                    text_for_decode,
                    text_tokenizer,
                    text_encoder,
                    gt_ms_idx_Bl,
                    trans_list,
                    help_list,
                    packet_meta=packet_meta,
                    return_meta=True,
                )
                scale_metrics = progressive_evaluate(
                    args,
                    infinity,
                    vae,
                    scale_schedule,
                    text_for_decode,
                    text_tokenizer,
                    text_encoder,
                    decoded_packet,
                    packet_meta,
                    args.rec_path,
                    setting_tag,
                    img_path,
                    img_name,
                    device,
                    h,
                    w,
                    prompt_bpp,
                    lpips_evaluator,
                    dists_evaluator,
                )

                results_by_setting.setdefault(setting_tag, []).append(
                    {
                        "image_name": img_name,
                        "text": text,
                        "prompt_bpp": prompt_bpp,
                        "packet_meta": packet_meta,
                        "decode_meta": decoded_packet["packet_meta"],
                        "scales_data": scale_metrics,
                    }
                )
                last_scale_key = str(max(int(k) for k in scale_metrics.keys()))
                print(
                    f"[{setting_tag}] {img_name} | prompt_bpp={prompt_bpp:.6f} | "
                    f"last_bpp={scale_metrics[last_scale_key]['bpp']:.6f} | "
                    f"psnr={scale_metrics[last_scale_key]['psnr']:.4f}"
                )

    final_output = {
        "model_path": args.model_path,
        "codec_mask_strategy": args.codec_mask_strategy,
        "keep_ratios": keep_ratios,
        "codec_fill_mode": args.codec_fill_mode,
        "codec_texture_weight": args.codec_texture_weight,
        "codec_structure_weight": args.codec_structure_weight,
        "codec_pca_rank": args.codec_pca_rank,
        "codec_texture_kernel": args.codec_texture_kernel,
        "results_by_setting": {},
    }

    for setting_tag, dataset_metrics in results_by_setting.items():
        scale_aggregates = {}
        for item in dataset_metrics:
            for scale_idx, metrics in item["scales_data"].items():
                scale_aggregates.setdefault(scale_idx, {"bpp": [], "psnr": [], "msssim": [], "lpips": [], "dists": []})
                for key in scale_aggregates[scale_idx]:
                    scale_aggregates[scale_idx][key].append(metrics[key])

        average_metrics_summary = {}
        for scale_idx, metric_lists in scale_aggregates.items():
            average_metrics_summary[scale_idx] = {
                "avg_bpp": float(np.mean(metric_lists["bpp"])),
                "avg_psnr": float(np.mean(metric_lists["psnr"])),
                "avg_msssim": float(np.mean(metric_lists["msssim"])),
                "avg_lpips": float(np.mean(metric_lists["lpips"])),
                "avg_dists": float(np.mean(metric_lists["dists"])),
                "image_count": len(metric_lists["bpp"]),
            }

        df_summary = pd.DataFrame(average_metrics_summary).T
        df_summary.index.name = "scale"
        csv_path = os.path.join(args.rec_path, f"{setting_tag}_summary.csv")
        df_summary.to_csv(csv_path)
        final_output["results_by_setting"][setting_tag] = {
            "summary": average_metrics_summary,
            "details": dataset_metrics,
        }
        print(f"[*] Saved summary CSV: {csv_path}")

    json_path = os.path.join(args.rec_path, "progressive_codec_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    print(f"[*] Saved metrics JSON: {json_path}")
