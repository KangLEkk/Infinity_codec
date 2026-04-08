import os
os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
sys.path.append("/workspace/Infinity_codec")

import torch, torchvision
import zlib
import time
import re
import json
import argparse
import cv2
import pandas as pd
import numpy as np

import lpips
from pytorch_msssim import ms_ssim
from DISTS_pytorch import DISTS

torch.cuda.set_device(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from tools.run_infinity import *
from compression.util import *
from compression.sender_drop import encoding, get_prob
from compression.reciever_drop import decompress_cfg, decoding


def _format_ratio_tag(r: float) -> str:
    return f"drop_{r:.2f}".replace('.', 'p')


def _encode_prompt_ids_t5(tokenizer, prompt: str, max_len: int = 512):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len, add_special_tokens=True)
    ids = enc["input_ids"][0].tolist()
    return list(map(int, ids))


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
    packet = {
        "payload": payload,
        "cdf_1d": cdf_1d,
        "unique_chars": unique_chars,
        "char_len": L,
        "ids_len": len(ids),
        "bits": len(payload) * 8,
    }
    return packet


def _arith_decode_ids_packet(packet):
    import torchac
    if packet["char_len"] == 0:
        return []
    cdf = packet["cdf_1d"].view(1, 1, -1).expand(1, packet["char_len"], -1).contiguous()
    decoded_sym = torchac.decode_float_cdf(cdf, packet["payload"])
    decoded_sym = decoded_sym.view(-1).tolist()
    text = "".join(packet["unique_chars"][int(i)] for i in decoded_sym)
    ids = [int(x) for x in text.split(",") if x != ""]
    return ids


def _decode_prompt_text_from_ids_t5(tokenizer, ids):
    if ids is None or len(ids) == 0:
        return ""
    try:
        return tokenizer.decode(ids, skip_special_tokens=True).strip()
    except TypeError:
        return tokenizer.decode(ids)


def _prompt_zlib_bytes(prompt: str) -> int:
    b = prompt.encode("utf-8", errors="ignore")
    return len(zlib.compress(b))


def build_gt_tokens(args, vae, scale_schedule, scale_q, img_path, device):
    inp_B3HW = load_img(img_path, args)
    raw_features, _, _ = vae.encode_for_raw_features(inp_B3HW.to(device), scale_schedule=scale_schedule)
    _, gt_ms_idx_Bl = mask_quant(vae, scale_q, raw_features, device)
    return inp_B3HW, gt_ms_idx_Bl


def decompress_modified(
    args,
    infinity,
    vae,
    scale_schedule,
    text,
    text_tokenizer,
    text_encoder,
    gt_ms_idx_Bl,
    rec_base_path,
    img_path,
    img_name,
    trans_list,
    help_list,
    packet_meta,
    device,
    h,
    w,
    prompt_bpp,
    lpips_fn,
    dists_fn,
    ratio_tag,
):
    dec_idx = decoding(args, infinity, vae, scale_schedule, text, text_tokenizer, text_encoder, gt_ms_idx_Bl, trans_list, help_list)
    scale_results = {}

    std_img_cv = cv2.imread(img_path)
    std_img_cv = cv2.cvtColor(std_img_cv, cv2.COLOR_BGR2RGB)
    std_tensor = torch.tensor(std_img_cv).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    cumulative_bits = packet_meta.get("cumulative_total_bits", None)

    for gt_leak in range(0, len(gt_ms_idx_Bl)):
        if cumulative_bits is not None:
            current_img_bpp = float(cumulative_bits[gt_leak]) / float(h * w)
        else:
            sum_bit = 0
            side_bits_per_token = int(getattr(args, 'rd_side_bits', 2))
            for i in range(gt_leak + 1):
                sum_bit += len(help_list[i]) * side_bits_per_token
                for payload in trans_list[i]:
                    sum_bit += len(payload)
            current_img_bpp = sum_bit / (h * w)
        current_total_bpp = current_img_bpp + prompt_bpp

        scale_folder = os.path.join(rec_base_path, ratio_tag, f"scale_{gt_leak}")
        os.makedirs(scale_folder, exist_ok=True)
        img = decompress_cfg(infinity, vae, scale_schedule, text, text_tokenizer, text_encoder, gt_leak, dec_idx)
        rec_img_path = os.path.join(scale_folder, img_name)
        torchvision.utils.save_image(img.cpu(), rec_img_path)

        rec_img_cv = cv2.imread(rec_img_path)
        rec_img_cv = cv2.cvtColor(rec_img_cv, cv2.COLOR_BGR2RGB)
        rec_tensor = torch.tensor(rec_img_cv).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        mse = torch.mean((std_tensor - rec_tensor) ** 2)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse)).item() if mse.item() > 0 else float('inf')
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


if __name__ == "__main__":
    initial_model_path = '/workspace/Infinity_codec/local_output/debug_stage2_student_1024——2/slim-ckpt-giter021K-ep0-iter21000-last.pth'
    vae_path = '/workspace/Infinity_codec/outputs/bitvae_tok_stage1_dino0.1_2/checkpoints/model_step_249999.ckpt'
    text_encoder_ckpt = '/workspace/CKPT/flan-t5-xl'

    args = argparse.Namespace(
        pn='1M',
        model_path=initial_model_path,
        cfg_insertion_layer=0,
        vae_type=32,
        vae_path=vae_path,
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        model_type='infinity_2b',
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        use_scale_schedule_embedding=0,
        sampling_per_bits=1,
        text_encoder_ckpt=text_encoder_ckpt,
        text_channels=2048,
        apply_spatial_patchify=0,
        h_div_w_template=1.000,
        use_flex_attn=0,
        cache_dir='/workspace/Infinity_codec/local_output/debug_stage2_student_1024——2',
        enable_model_cache=1,
        checkpoint_type='torch',
        seed=0,
        bf16=0,
        rec_path='',
        add_prompt_bits=1,
        prompt_bits_mode='arith',
        tlen=512,
        keep_gpu_busy=1,
        rd_drop_ratios=[0.10, 0.25, 0.50, 0.75],
        rd_drop_start_scale=0,
        rd_side_bits=2,
        rd_scale_beta=1.0,
        rd_lambda_mismatch=2.0,
        rd_lambda_entropy=0.25,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    lpips_evaluator = lpips.LPIPS(net='alex').to(device)
    dists_evaluator = DISTS().to(device)

    json_data = []
    with open('/workspace/ARPC/data/DIV2K.json', 'rt') as f:
        for line in f:
            json_data.append(json.loads(line))

    model_dir = os.path.dirname(initial_model_path)
    base_filename = os.path.basename(initial_model_path)
    match_init = re.search(r'-iter(\d+)-', base_filename)
    if not match_init:
        raise ValueError("无法在给定的 initial_model_path 中找到有效的 '-iterXXXX-' 结构。")
    start_iter = int(match_init.group(1))

    available_models = []
    for f in os.listdir(model_dir):
        if f.endswith('.pth') and 'iter' in f and 'slim' in f:
            m = re.search(r'-iter(\d+)-', f)
            if m:
                available_models.append((int(m.group(1)), f))
    available_models.sort(key=lambda x: x[0])
    models_to_test = [f for it, f in available_models if it >= start_iter]
    print(f"[*] 成功扫描到 {len(models_to_test)} 个待测试模型。")

    for current_filename in models_to_test:
        current_model_path = os.path.join(model_dir, current_filename)
        current_iter = int(re.search(r'-iter(\d+)-', current_filename).group(1))
        print(f"\n{'='*60}")
        print(f"[*] 开始测试模型: {current_filename} (Iter: {current_iter})")
        print(f"{'='*60}")

        args.model_path = current_model_path
        args.rec_path = f'/workspace/Infinity_codec/results/dynamic/DIV2K_dynamic_rdproxy_{current_iter}'
        os.makedirs(args.rec_path, exist_ok=True)
        infinity = load_transformer(vae, args)

        dataset_metrics_by_ratio = {_format_ratio_tag(r): [] for r in args.rd_drop_ratios}

        for data in json_data:
            img_path, text = data['img_path'], data['txt']
            img_name = img_path.split('/')[-1]

            inp_B3HW = load_img(img_path, args)
            _, _, h, w = inp_B3HW.shape
            h_div_w = h / w
            h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
            scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
            scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
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
                    assert recovered_ids == prompt_ids, "Prompt arithmetic decode mismatch!"
                    text_for_decode = _decode_prompt_text_from_ids_t5(text_tokenizer, recovered_ids)
                elif args.prompt_bits_mode == "zlib":
                    prompt_bits = _prompt_zlib_bytes(text) * 8
                    text_for_decode = text
            prompt_bpp = prompt_bits / (h * w) if args.add_prompt_bits else 0.0

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                _, gt_ms_idx_Bl = build_gt_tokens(args, vae, scale_schedule, scale_q, img_path, device)
                prob_list = get_prob(infinity, vae, scale_schedule, text, text_tokenizer, text_encoder, gt_ms_idx_Bl)

                print(f"Processing {img_name}... [Prompt Bits: {prompt_bits} | Prompt BPP: {prompt_bpp:.6f}]")
                for rd_ratio in args.rd_drop_ratios:
                    ratio_tag = _format_ratio_tag(rd_ratio)
                    trans_list, help_list, packet_meta = encoding(
                        args,
                        infinity,
                        vae,
                        scale_schedule,
                        text,
                        text_tokenizer,
                        text_encoder,
                        gt_ms_idx_Bl,
                        rd_drop_ratio=rd_ratio,
                        precomputed_prob_list=prob_list,
                    )
                    scale_metrics = decompress_modified(
                        args,
                        infinity,
                        vae,
                        scale_schedule,
                        text_for_decode,
                        text_tokenizer,
                        text_encoder,
                        gt_ms_idx_Bl,
                        args.rec_path,
                        img_path,
                        img_name,
                        trans_list,
                        help_list,
                        packet_meta,
                        device,
                        h,
                        w,
                        prompt_bpp,
                        lpips_evaluator,
                        dists_evaluator,
                        ratio_tag,
                    )
                    dataset_metrics_by_ratio[ratio_tag].append({
                        "image_name": img_name,
                        "text": text,
                        "prompt_bpp": prompt_bpp,
                        "rd_drop_ratio": rd_ratio,
                        "packet_meta": packet_meta,
                        "scales_data": scale_metrics,
                    })
                    last_scale_key = str(len(scale_schedule) - 1)
                    print(
                        f"  [{ratio_tag}] dropped={packet_meta['num_dropped_tokens']}/{packet_meta['num_candidate_tokens']} | "
                        f"last-scale bpp={scale_metrics[last_scale_key]['bpp']:.6f} | "
                        f"psnr={scale_metrics[last_scale_key]['psnr']:.4f}"
                    )

        results_by_ratio = {}
        for ratio_tag, dataset_metrics_data in dataset_metrics_by_ratio.items():
            scale_aggregates = {}
            for item in dataset_metrics_data:
                for scale_idx, metrics in item["scales_data"].items():
                    if scale_idx not in scale_aggregates:
                        scale_aggregates[scale_idx] = {"bpp": [], "psnr": [], "msssim": [], "lpips": [], "dists": []}
                    for k in scale_aggregates[scale_idx].keys():
                        scale_aggregates[scale_idx][k].append(metrics[k])

            average_metrics_summary = {}
            print(f"\n[*] 模型 Iter: {current_iter} | {ratio_tag} 测试集平均指标汇总:")
            for scale_idx, metric_lists in scale_aggregates.items():
                avg_bpp = float(np.mean(metric_lists["bpp"]))
                avg_psnr = float(np.mean(metric_lists["psnr"]))
                avg_msssim = float(np.mean(metric_lists["msssim"]))
                avg_lpips = float(np.mean(metric_lists["lpips"]))
                avg_dists = float(np.mean(metric_lists["dists"]))
                average_metrics_summary[scale_idx] = {
                    "avg_bpp": avg_bpp,
                    "avg_psnr": avg_psnr,
                    "avg_msssim": avg_msssim,
                    "avg_lpips": avg_lpips,
                    "avg_dists": avg_dists,
                    "image_count": len(metric_lists["bpp"]),
                }
                print(
                    f"    Scale {scale_idx} | BPP: {avg_bpp:.4f} | PSNR: {avg_psnr:.4f} | "
                    f"MS-SSIM: {avg_msssim:.4f} | LPIPS: {avg_lpips:.4f} | DISTS: {avg_dists:.4f}"
                )

            results_by_ratio[ratio_tag] = {
                "summary": average_metrics_summary,
                "details": dataset_metrics_data,
            }

            df_summary = pd.DataFrame(average_metrics_summary).T
            df_summary.index.name = "scale"
            csv_save_path = os.path.join(args.rec_path, f"avg_metrics_iter_{current_iter}_{ratio_tag}.csv")
            df_summary.to_csv(csv_save_path)
            print(f"[*] {ratio_tag} 平均值 CSV 已保存至: {csv_save_path}")

        final_json_output = {
            "model_iter": current_iter,
            "rd_proxy_config": {
                "rd_drop_ratios": list(args.rd_drop_ratios),
                "rd_drop_start_scale": args.rd_drop_start_scale,
                "rd_side_bits": args.rd_side_bits,
                "rd_scale_beta": args.rd_scale_beta,
                "rd_lambda_mismatch": args.rd_lambda_mismatch,
                "rd_lambda_entropy": args.rd_lambda_entropy,
            },
            "results_by_ratio": results_by_ratio,
        }

        json_save_path = os.path.join(args.rec_path, f"metrics_iter_{current_iter}_rdproxy.json")
        with open(json_save_path, 'w', encoding='utf-8') as f:
            json.dump(final_json_output, f, indent=4, ensure_ascii=False)
        print(f"[*] 当前模型全部 ratio 的详细指标已保存至: {json_save_path}\n")

    print("\n[*] 所有可用模型已测试完毕。退出测试循环。")
    if getattr(args, 'keep_gpu_busy', 0) == 1 and len(json_data) > 0:
        print("[*] 检测到 keep_gpu_busy=1，进入无限循环以保持 GPU 活跃...")
        dummy_data = json_data[0]
        dummy_img_path, dummy_text = dummy_data['img_path'], dummy_data['txt']
        dummy_inp = load_img(dummy_img_path, args)
        _, _, dummy_h, dummy_w = dummy_inp.shape
        dummy_h_div_w = dummy_h / dummy_w
        dummy_h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - dummy_h_div_w))]
        dummy_scale_schedule = dynamic_resolution_h_w[dummy_h_div_w_template_][args.pn]['scales']
        dummy_scale_schedule = [(1, dummy_h, dummy_w) for (_, dummy_h, dummy_w) in dummy_scale_schedule]
        dummy_scale_q = [
            (dummy_scale_schedule[i][0], dummy_scale_schedule[i][1], dummy_scale_schedule[i][2], int((i + 1) // ((len(dummy_scale_schedule) // 3) + 1) + 2))
            for i in range(len(dummy_scale_schedule))
        ]
        while True:
            try:
                _, gt_ms_idx_Bl = build_gt_tokens(args, vae, dummy_scale_schedule, dummy_scale_q, dummy_img_path, device)
                trans_list, help_list, _ = encoding(
                    args, infinity, vae, dummy_scale_schedule, dummy_text, text_tokenizer, text_encoder, gt_ms_idx_Bl, rd_drop_ratio=0.0
                )
                dec_idx = decoding(args, infinity, vae, dummy_scale_schedule, dummy_text, text_tokenizer, text_encoder, gt_ms_idx_Bl, trans_list, help_list)
                for gt_leak in range(0, len(gt_ms_idx_Bl)):
                    _ = decompress_cfg(infinity, vae, dummy_scale_schedule, dummy_text, text_tokenizer, text_encoder, gt_leak, dec_idx)
            except Exception:
                time.sleep(1)
                pass
