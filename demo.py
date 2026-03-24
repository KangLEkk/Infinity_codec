import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["CC"]  = "gcc"
os.environ["CXX"] = "g++"
import sys
sys.path.append("/workspace/Infinity_codec")

import torch, torchvision
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
    # print(mse)
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
                        gt_ms_idx_Bl, rec_base_path, img_name, trans_list, help_list, device):
    dec_idx = decoding(args, infinity, vae, scale_schedule, text, text_tokenizer, text_encoder, gt_ms_idx_Bl, trans_list, help_list)
    
    img_bpps = [] # 记录当前图片在各个尺度下的累计 bpp
    
    for gt_leak in range(0, len(gt_ms_idx_Bl)):
        # 1. 计算当前尺度的累计 bpp (单位: bits per pixel)
        # 根据你的逻辑：bpp = 总比特数 / (宽 * 高)
        # 注意：原代码中除以 1024/1024 得到的是 MB，bpp 通常指 bits per pixel
        sum_bit = 0
        for i in range(gt_leak + 1):
            sum_bit += len(help_list[i])
            for j in range(len(trans_list[i])):
                sum_bit += len(trans_list[i][j])
        
        # 获取原始图像分辨率计算真正的 bpp
        # _, _, h, w = scale_schedule[-1] # 取最大尺度的分辨率
        current_bpp = sum_bit / (1024 * 1024)
        img_bpps.append(current_bpp)

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

    model_path='/workspace/CKPT/Infinity/infinity_2b_reg.pth'
    vae_path='/workspace/CKPT/Infinity/infinity_vae_d32reg.pth'
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
    cache_dir='/dev/shm',
    enable_model_cache=0,
    checkpoint_type='torch',
    seed=0,
    bf16=0,
    rec_path='/workspace/Infinity_codec/results/DIV2K_orimodel'
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load text encoder
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    # load vae
    vae = load_visual_tokenizer(args)
    # load infinity
    infinity = load_transformer(vae, args)

    json_data = []
    with open('/workspace/ARPC/data/DIV2K.json', 'rt') as f:
        for line in f:
            json_data.append(json.loads(line))

    base_rec_path = args.rec_path # './rec_img/DIV2K'
    if not os.path.exists(base_rec_path):
        os.makedirs(base_rec_path)

    all_scales_stats = {}

    for data in json_data:
        img_path, text = data['img_path'], data['txt']
        img_name = img_path.split('/')[-1] # 获取 0801.png
        
        # 加载图像并获取尺度信息
        inp_B3HW = load_img(img_path, args)
        b, c, h, w = inp_B3HW.shape
        h_div_w = h / w
        h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
        scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
        scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
        scale_q = [(scale_schedule[i][0], scale_schedule[i][1], scale_schedule[i][2], 
                    int((i+1)//((len(scale_schedule)//3)+1)+2)) for i in range(len(scale_schedule))]
        
        # 压缩
        trans_list, help_list, bpp, gt_ms_idx_Bl = compress(
            args, vae, scale_schedule, scale_q, infinity, 
            text_tokenizer, text_encoder, img_path, text
        )
        
        # 解压并按尺度保存
        print(f"Processing {img_name}...")
        img_bpps =decompress_modified(
            args, infinity, vae, scale_schedule, text, 
            text_tokenizer, text_encoder, gt_ms_idx_Bl, 
            base_rec_path, img_name, trans_list, help_list, device
        )
        # 累加统计数据
        for idx, bpp_val in enumerate(img_bpps):
            if idx not in all_scales_stats:
                all_scales_stats[idx] = []
            all_scales_stats[idx].append(bpp_val)
            
        print(f"Finished {img_name}, current scales bpp: {[f'{b:.8f}' for b in img_bpps]}")

    # --- 计算并保存平均值 ---
    final_summary = []
    for idx, bpps in all_scales_stats.items():
        avg_bpp = np.mean(bpps)
        final_summary.append({
            "scale": idx,
            "avg_bpp": avg_bpp,
            "img_count": len(bpps)
        })
        print(f"Scale {idx}: Average BPP = {avg_bpp:.8f}")

    # 保存总统计表
    df_summary = pd.DataFrame(final_summary)
    df_summary.to_csv(os.path.join(args.rec_path, "overall_bpp_stats.csv"), index=False)