import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["CC"]  = "gcc"
os.environ["CXX"] = "g++"
import sys
sys.path.append("/workspace/Infinity_codec")

import torch, torchvision
import zlib  # Added for zlib prompt compression
import time  # Added for the infinite keep-alive loop
torch.cuda.set_device(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import json
import argparse
from tools.run_infinity import *
import lpips
from compression.util import *
from compression.sender import encoding
from compression.reciever import decompress_cfg, decoding
import pandas as pd
import numpy as np

# --- Prompt Compression Helpers ---
def _encode_prompt_ids(tokenizer, prompt: str, max_len: int = 77):
    """Encode prompt -> token ids."""
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

def _arith_bits_from_ids(ids):
    """Arithmetic-code token ids and return payload bits."""
    if ids is None or len(ids) == 0:
        return 0
    text = ",".join(map(str, ids))
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

def _prompt_zlib_bytes(prompt: str) -> int:
    b = prompt.encode("utf-8", errors="ignore")
    return len(zlib.compress(b))
# ----------------------------------

def calculate_lpips(img1, img2, device=0):
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    image1_tensor = torch.tensor(np.array(img1)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image2_tensor = torch.tensor(np.array(img2)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    distance = lpips_fn(image1_tensor.to(device), image2_tensor.to(device))
    return distance.item()

def calculate_psnr(img1, img2, max_val=255.):
    """
    Based on `tf.image.psnr`
    https://www.tensorflow.org/api_docs/python/tf/image/psnr
    """
    float_type = 'float64' 

    img1 = img1.astype(float_type)
    img2 = img2.astype(float_type)
    mse = np.mean(np.square(img1 - img2), axis=(0, 1, 2))
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    return psnr

def compress(args, vae, scale_schedule, scale_q, infinity, text_tokenizer, text_encoder, img_path, text):
    inp_B3HW = load_img(img_path, args)
    raw_features, _, _ = vae.encode_for_raw_features(inp_B3HW.to(device), scale_schedule=scale_schedule)
    x_BLC_wo_prefix, gt_ms_idx_Bl = mask_quant(vae, scale_q, raw_features, device)
    trans_list, help_list, bpp = encoding(args, infinity, vae, scale_schedule, text, text_tokenizer, text_encoder, gt_ms_idx_Bl)

    return trans_list, help_list, bpp, gt_ms_idx_Bl

def decompress(args, infinity, vae, scale_schedule, text, text_tokenizer, text_encoder, gt_ms_idx_Bl, rec_path, trans_list, help_list, device):
    dec_idx = decoding(args, infinity, vae, scale_schedule, text, text_tokenizer, text_encoder, gt_ms_idx_Bl, trans_list, help_list)
    results_data = []
    for gt_leak in range(0, len(gt_ms_idx_Bl)):
        sum_bit = 0
        for i in range(gt_leak+1):
            sum_bit += len(help_list[i])
            for j in range(len(trans_list[i])):
                sum_bit += len(trans_list[i][j])
        bpp = sum_bit/1024/1024
        img = decompress_cfg(infinity, vae, scale_schedule, text, text_tokenizer, text_encoder, gt_leak, dec_idx)
        torchvision.utils.save_image(img.cpu(), os.path.join(rec_path, str(gt_leak)+".png"))

        psnr, lpips = cal_score(gt_leak, rec_path, device)

        results_data.append(
            {
                "gt_leak": gt_leak,
                "bpp": bpp,
                "psnr": psnr,
                "lpips": lpips
            }
        )

    results_df = pd.DataFrame(data=results_data)
    results_df.to_csv(os.path.join(rec_path, "results.csv"))

def decompress_modified(args, infinity, vae, scale_schedule, text, text_tokenizer, text_encoder, 
                        gt_ms_idx_Bl, rec_base_path, img_name, trans_list, help_list, device, 
                        h, w, prompt_bpp):
    dec_idx = decoding(args, infinity, vae, scale_schedule, text, text_tokenizer, text_encoder, gt_ms_idx_Bl, trans_list, help_list)
    
    img_bpps = [] 
    
    for gt_leak in range(0, len(gt_ms_idx_Bl)):
        # 1. 计算当前尺度的累计 bpp
        sum_bit = 0
        for i in range(gt_leak + 1):
            sum_bit += len(help_list[i])
            for j in range(len(trans_list[i])):
                sum_bit += len(trans_list[i][j])
        
        current_img_bpp = sum_bit / (h * w)
        current_total_bpp = current_img_bpp + prompt_bpp
        img_bpps.append(current_total_bpp)

        # 2. 保存图像到尺度文件夹
        scale_folder = os.path.join(rec_base_path, f"scale_{gt_leak}")
        os.makedirs(scale_folder, exist_ok=True)
        img = decompress_cfg(infinity, vae, scale_schedule, text, text_tokenizer, text_encoder, gt_leak, dec_idx)
        torchvision.utils.save_image(img.cpu(), os.path.join(scale_folder, img_name))

    return img_bpps

def cal_score(gt_leak, rec_path, device):
    std_img_path = os.path.join(rec_path, "ori.png")
    rec_img_path = os.path.join(rec_path, str(gt_leak)+".png")
    std_img = cv2.imread(std_img_path)
    rec_img = cv2.imread(rec_img_path)
    std_img = cv2.cvtColor(std_img, cv2.COLOR_BGR2RGB)
    rec_img = cv2.cvtColor(rec_img, cv2.COLOR_BGR2RGB)
    psnr = calculate_psnr(std_img, rec_img)
    lpips = calculate_lpips(std_img, rec_img, device)

    return psnr, lpips

if __name__ == "__main__":

    model_path='/workspace/Infinity_codec/local_output/debug_stage2_student_1024——2/ar-ckpt-giter013K-ep0-iter13000-last.pth'
    vae_path='/workspace/Infinity_codec/outputs/bitvae_tok_stage1_dino0.1_2/checkpoints/model_step_249999.ckpt'
    text_encoder_ckpt = '/workspace/CKPT/flan-t5-xl'
    args=argparse.Namespace(
        pn='1M',
        model_path=model_path,
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
        rec_path='/workspace/Infinity_codec/results/DIV2K_3',
        
        # Args for prompt compression
        add_prompt_bits=1,
        prompt_bits_mode='arith', # Options: 'arith', 'zlib', 'none'
        tlen=512,
        
        # GPU Keep-alive switch
        keep_gpu_busy=1 # Set to 0 to disable
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    infinity = load_transformer(vae, args)

    json_data = []
    with open('/workspace/ARPC/data/DIV2K.json', 'rt') as f:
        for line in f:
            json_data.append(json.loads(line))

    base_rec_path = args.rec_path
    if not os.path.exists(base_rec_path):
        os.makedirs(base_rec_path)

    all_scales_stats = {}

    for data in json_data:
        img_path, text = data['img_path'], data['txt']
        img_name = img_path.split('/')[-1] 
        
        inp_B3HW = load_img(img_path, args)
        b, c, h, w = inp_B3HW.shape
        h_div_w = h / w
        h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
        scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
        scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
        scale_q = [(scale_schedule[i][0], scale_schedule[i][1], scale_schedule[i][2], 
                    int((i+1)//((len(scale_schedule)//3)+1)+2)) for i in range(len(scale_schedule))]
        
        # Prompt bits
        prompt_bits = 0
        if args.add_prompt_bits:
            prompt_ids = _encode_prompt_ids(text_tokenizer, text, max_len=args.tlen)
            if args.prompt_bits_mode == "arith":
                prompt_bits = _arith_bits_from_ids(prompt_ids)
            elif args.prompt_bits_mode == "zlib":
                prompt_bits = _prompt_zlib_bytes(text) * 8
                
        prompt_bpp = prompt_bits / (h * w) if args.add_prompt_bits else 0.0
        
        trans_list, help_list, bpp, gt_ms_idx_Bl = compress(
            args, vae, scale_schedule, scale_q, infinity, 
            text_tokenizer, text_encoder, img_path, text
        )
        
        print(f"Processing {img_name}... [Prompt Bits: {prompt_bits} | Prompt BPP: {prompt_bpp:.6f}]")
        img_bpps = decompress_modified(
            args, infinity, vae, scale_schedule, text, 
            text_tokenizer, text_encoder, gt_ms_idx_Bl, 
            base_rec_path, img_name, trans_list, help_list, device,
            h, w, prompt_bpp
        )
        for idx, bpp_val in enumerate(img_bpps):
            if idx not in all_scales_stats:
                all_scales_stats[idx] = []
            all_scales_stats[idx].append(bpp_val)
            
        print(f"Finished {img_name}, current scales total bpp: {[f'{b:.8f}' for b in img_bpps]}")

    final_summary = []
    for idx, bpps in all_scales_stats.items():
        avg_bpp = np.mean(bpps)
        final_summary.append({
            "scale": idx,
            "avg_bpp": avg_bpp,
            "img_count": len(bpps)
        })
        print(f"Scale {idx}: Average BPP = {avg_bpp:.4f}")

    df_summary = pd.DataFrame(final_summary)
    df_summary.to_csv(os.path.join(args.rec_path, "overall_bpp_stats.csv"), index=False)

    # --- GPU Keep-Alive Logic ---
    if getattr(args, 'keep_gpu_busy', 0) == 1:
        print("\n[*] 评估已完成。检测到 keep_gpu_busy=1，进入无限循环以保持 GPU 活跃...")
        
        # 取第一张图作为循环测试的数据，避免 I/O 开销
        if len(json_data) > 0:
            dummy_data = json_data[0]
            dummy_img_path, dummy_text = dummy_data['img_path'], dummy_data['txt']
            
            # 预计算尺度信息
            dummy_inp = load_img(dummy_img_path, args)
            _, _, dummy_h, dummy_w = dummy_inp.shape
            dummy_h_div_w = dummy_h / dummy_w
            dummy_h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - dummy_h_div_w))]
            dummy_scale_schedule = dynamic_resolution_h_w[dummy_h_div_w_template_][args.pn]['scales']
            dummy_scale_schedule = [(1, dummy_h, dummy_w) for (_, dummy_h, dummy_w) in dummy_scale_schedule]
            dummy_scale_q = [(dummy_scale_schedule[i][0], dummy_scale_schedule[i][1], dummy_scale_schedule[i][2], 
                        int((i+1)//((len(dummy_scale_schedule)//3)+1)+2)) for i in range(len(dummy_scale_schedule))]
            
            while True:
                try:
                    # 仅执行模型推理，不保存图片或计算指标
                    trans_list, help_list, _, gt_ms_idx_Bl = compress(
                        args, vae, dummy_scale_schedule, dummy_scale_q, infinity, 
                        text_tokenizer, text_encoder, dummy_img_path, dummy_text
                    )
                    
                    dec_idx = decoding(args, infinity, vae, dummy_scale_schedule, dummy_text, text_tokenizer, text_encoder, gt_ms_idx_Bl, trans_list, help_list)
                    for gt_leak in range(0, len(gt_ms_idx_Bl)):
                        _ = decompress_cfg(infinity, vae, dummy_scale_schedule, dummy_text, text_tokenizer, text_encoder, gt_leak, dec_idx)
                        
                except Exception as e:
                    # 忽略偶发异常，确保循环不中断退出
                    time.sleep(1)
                    pass