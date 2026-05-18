from compression.util import *
from utils.arithmeticcoding import decompress_from_bit_list, compress_to_bit_list
import math
from typing import Dict, Any, Tuple, Optional

import torch


import math
from typing import Dict, Any, Tuple, Optional

import torch


def _relative_boundary_factor(
    rel_stage: int,
    boundary_temp_boost: float = 0.25,
    boundary_decay: float = 0.55,
    tail_cool_rate: float = 0.03,
    tail_cool_start: int = 4,
    tail_cool_min: float = 0.85,
) -> float:
    """
    rel_stage:
        当前尺度距离“最后一个已传输尺度”的相对距离。
        - rel_stage = 1: 第一个缺失尺度（最关键）
        - rel_stage = 2: 第二个缺失尺度
        - ...

    返回一个温度乘子：
        - 第一个缺失尺度给更高温度
        - 后续逐渐回落
        - 再往后略微降温，避免高尺度过于随机
    """
    rel_stage = max(1, int(rel_stage))

    # 第一个缺失尺度温度更高，后面指数衰减
    boost = boundary_temp_boost * math.exp(-boundary_decay * float(rel_stage - 1))

    # 到更后面的缺失尺度，略微偏保守
    if rel_stage >= tail_cool_start:
        cool = max(tail_cool_min, 1.0 - tail_cool_rate * float(rel_stage - tail_cool_start + 1))
    else:
        cool = 1.0

    return (1.0 + boost) * cool


def apply_entropy_adaptive_temperature(
    raw_logits: torch.Tensor,
    si: int,
    last_observed_scale_idx: int,
    selective_ratio: float = 0.20,
    t0: float = 1.60,
    alpha: float = 0.45,
    theta: float = 0.55,
    base_temperature: float = 0.85,
    min_temperature: float = 0.10,
    max_temperature: float = 2.50,
    boundary_temp_boost: float = 0.25,
    boundary_decay: float = 0.55,
    tail_cool_rate: float = 0.03,
    tail_cool_start: int = 4,
    tail_cool_min: float = 0.85,
    eps: float = 1e-10,
    return_debug: bool = False,
) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
    """
    输入:
        raw_logits: [B, seq_len, 64]
            默认按 Infinity bitwise 格式解释为 [B, seq_len, 32, 2]
        si:
            当前正在生成的尺度 index
        last_observed_scale_idx:
            最后一个“已传输/已知”的尺度 index
            例如传输了前 3 个 token map，则 last_observed_scale_idx = 2
            若一个都没传，设为 -1
        selective_ratio:
            只对当前尺度中熵最高的前 selective_ratio 比例 token 做动态温度
        t0, alpha, theta:
            动态温度公式参数
            T = t0 * exp(-H_norm / alpha) + theta
            注意这里 H_norm 已经归一化到 [0, 1]
        base_temperature:
            对“不进行动态采样”的 token 使用的固定温度
        min_temperature, max_temperature:
            温度安全上下界

    输出:
        logits_scaled: [B, seq_len, 64]
        debug_info: 可选调试信息
    """
    if raw_logits.ndim != 3:
        raise ValueError(f"raw_logits should be [B, seq_len, 64], got shape={tuple(raw_logits.shape)}")
    if raw_logits.size(-1) % 2 != 0:
        raise ValueError(f"Last dim of raw_logits must be even, got {raw_logits.size(-1)}")

    B, L, V = raw_logits.shape
    num_bits = V // 2

    # [B, L, 64] -> [B, L, 32, 2]
    logits_bits = raw_logits.reshape(B, L, num_bits, 2)

    # bit 概率
    probs_bits = torch.softmax(logits_bits, dim=-1)

    # 每个 bit 的熵，单位 nat，最大值 ln(2)
    entropy_bits = -(probs_bits * torch.log(probs_bits.clamp_min(eps))).sum(dim=-1)  # [B, L, num_bits]

    # 归一化到 [0, 1]
    entropy_bits_norm = (entropy_bits / math.log(2.0)).clamp_(0.0, 1.0)  # [B, L, num_bits]

    # token 级不确定性：对 32 个 bit 的归一化熵取均值
    token_uncertainty = entropy_bits_norm.mean(dim=-1)  # [B, L]

    # 相对缺失边界：第一个缺失尺度最重要
    rel_stage = max(1, int(si - last_observed_scale_idx))
    rel_factor = _relative_boundary_factor(
        rel_stage=rel_stage,
        boundary_temp_boost=boundary_temp_boost,
        boundary_decay=boundary_decay,
        tail_cool_rate=tail_cool_rate,
        tail_cool_start=tail_cool_start,
        tail_cool_min=tail_cool_min,
    )

    # bit-wise 动态温度
    # 低熵 -> 温度更高一些
    # 高熵 -> 温度更低一些
    dynamic_temp_bits = t0 * torch.exp(-entropy_bits_norm / max(alpha, 1e-6)) + theta  # [B, L, num_bits]
    dynamic_temp_bits = dynamic_temp_bits * rel_factor

    # selective sampling: 只对最不确定的一部分 token 用动态温度
    selective_ratio = float(max(0.0, min(1.0, selective_ratio)))
    if selective_ratio <= 0.0:
        use_dynamic_mask = torch.zeros_like(token_uncertainty, dtype=torch.bool)
    elif selective_ratio >= 1.0:
        use_dynamic_mask = torch.ones_like(token_uncertainty, dtype=torch.bool)
    else:
        # 每个 batch 样本内部单独算分位数阈值
        threshold = torch.quantile(token_uncertainty, q=1.0 - selective_ratio, dim=1, keepdim=True)
        use_dynamic_mask = token_uncertainty >= threshold  # [B, L]

    # 未选中的 token 用固定低温
    base_temp_bits = torch.full_like(dynamic_temp_bits, fill_value=base_temperature)

    final_temp_bits = torch.where(
        use_dynamic_mask.unsqueeze(-1),
        dynamic_temp_bits,
        base_temp_bits,
    )

    final_temp_bits = final_temp_bits.clamp_(min=min_temperature, max=max_temperature)

    # 应用到 bit logits
    logits_bits_scaled = logits_bits / final_temp_bits.unsqueeze(-1)  # [B, L, num_bits, 2]
    logits_scaled = logits_bits_scaled.reshape_as(raw_logits)

    if not return_debug:
        return logits_scaled, None

    debug_info = {
        "rel_stage": rel_stage,
        "rel_factor": rel_factor,
        "token_uncertainty_mean": token_uncertainty.mean().item(),
        "token_uncertainty_max": token_uncertainty.max().item(),
        "dynamic_ratio": use_dynamic_mask.float().mean().item(),
        "temp_mean": final_temp_bits.mean().item(),
        "temp_min": final_temp_bits.min().item(),
        "temp_max": final_temp_bits.max().item(),
    }
    return logits_scaled, debug_info


def calc_token_entropy(p_list):
    ent = 0.0
    for p in p_list:
        if 0.0 < p < 1.0:
            ent += -p * math.log2(p) - (1 - p) * math.log2(1 - p)
    return ent
def sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:  # return idx, shaped (B, l)
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
    # sample (have to squeeze cuz multinomial can only be used on 2D tensor)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(logits_BlV.softmax(dim=-1).view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)

def decompress_cfg(infinity, vae, vae_scale_schedule, prompt, text_tokenizer, text_encoder, gt_leak, gt_ls_Bl, cfg_list=3, tau_list=0.5, cfg_insertion_layer=[0]):
    # infinity.rng.manual_seed(9306)
    rng = infinity.rng
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(vae_scale_schedule)
        cfg_list = [cfg_list] * len(vae_scale_schedule)
    label_B_or_BLT = encode_prompt(text_tokenizer, text_encoder, prompt)
    kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
    B = 1
    if any(np.array(cfg_list) != 1):
        bs = 2*B
        kv_compact_un = kv_compact.clone()
        total = 0
        for le in lens:
            kv_compact_un[total:total+le] = (infinity.cfg_uncond)[:le]
            total += le
        kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
        cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:]+cu_seqlens_k[-1]), dim=0)
    else:
        bs = B
    
    kv_compact = infinity.text_norm(kv_compact)
    sos = cond_BD = infinity.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k)) # sos shape: [2, 4096]
    kv_compact = infinity.text_proj_for_ca(kv_compact) # kv_compact shape: [304, 4096]
    ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
    last_stage = sos.unsqueeze(1).expand(bs, 1, -1) + infinity.pos_start.expand(bs, 1, -1)
    with torch.amp.autocast('cuda', enabled=False):
        cond_BD_or_gss = infinity.shared_ada_lin(cond_BD.float()).float().contiguous()
    for b in infinity.unregistered_blocks: (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(True)

    abs_cfg_insertion_layers = []
    add_cfg_on_logits, add_cfg_on_probs = False, False
    leng = len(infinity.unregistered_blocks)
    for item in cfg_insertion_layer:
        if item == 0: # add cfg on logits
            add_cfg_on_logits = True
        elif item == 1: # add cfg on probs
            add_cfg_on_probs = True # todo in the future, we may want to add cfg on logits and probs
        elif item < 0: # determine to add cfg at item-th layer's output
            assert leng+item > 0, f'cfg_insertion_layer: {item} is not valid since len(unregistered_blocks)={infinity.num_block_chunks}'
            abs_cfg_insertion_layers.append(leng+item)
        else:
            raise ValueError(f'cfg_insertion_layer: {item} is not valid')
        
    accu_BChw, cur_L, ret = None, 0, []  # current length, list of reconstructed images
    idx_Bl_list, idx_Bld_list = [], []
    num_stages_minus_1 = len(vae_scale_schedule)-1
    summed_codes = 0
    cur_L = 0
    for si, pn in enumerate(vae_scale_schedule):
        cfg = cfg_list[si]
        cur_L += np.array(pn).prod()
        need_to_pad = 0
        attn_fn = None
        if infinity.use_flex_attn:
            attn_fn = infinity.attn_fn_compile_dict.get(tuple(vae_scale_schedule[:(si+1)]), None)
        layer_idx = 0
        for block_idx, b in enumerate(infinity.block_chunks):
            # last_stage shape: [4, 1, 2048], cond_BD_or_gss.shape: [4, 1, 6, 2048], ca_kv[0].shape: [64, 2048], ca_kv[1].shape [5], ca_kv[2]: int
            if infinity.add_lvl_embeding_only_first_block and block_idx == 0:
                last_stage = infinity.add_lvl_embeding(last_stage, si, vae_scale_schedule, need_to_pad=need_to_pad)
            if not infinity.add_lvl_embeding_only_first_block: 
                last_stage = infinity.add_lvl_embeding(last_stage, si, vae_scale_schedule, need_to_pad=need_to_pad)
            
            for m in b.module:
                last_stage = m(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=None, attn_fn=attn_fn, scale_schedule=vae_scale_schedule, rope2d_freqs_grid=infinity.rope2d_freqs_grid, scale_ind=si)
                layer_idx += 1
        
        if (cfg != 1) and add_cfg_on_logits:
                # print(f'add cfg on add_cfg_on_logits')
                logits_BlV = infinity.get_logits(last_stage, cond_BD).mul(1/tau_list[si])
                logits_BlV = cfg * logits_BlV[:B] + (1-cfg) * logits_BlV[B:]
        else:
            logits_BlV = infinity.get_logits(last_stage[:B], cond_BD[:B]).mul(1/tau_list[si])

        # # ---------- 替换为基于信息熵的动态温度采样 ----------
        # # 1. 获取纯净的原始 Logits (暂时不乘以静态的 1/tau)
        # if (cfg != 1) and add_cfg_on_logits:
        #     logits_all = infinity.get_logits(last_stage, cond_BD)
        #     logits_cond = logits_all[:B]
        #     logits_uncond = logits_all[B:]
        #     raw_logits = cfg * logits_cond + (1 - cfg) * logits_uncond
        # else:
        #     raw_logits = infinity.get_logits(last_stage[:B], cond_BD[:B])

        # # 2. 计算比特级信息熵 (Bitwise Entropy)
        # # Infinity 的 raw_logits 形状通常为 [B, seq_len, 64] (32个bit，每个bit有2个状态)
        # tmp_bs, tmp_seq_len = raw_logits.shape[:2]
        
        # # 将其 Reshape 为 [B, seq_len, 32, 2] 以便计算每个 Bit 的 Softmax 概率
        # logits_bits = raw_logits.view(tmp_bs, tmp_seq_len, 32, 2)
        # probs_bits = torch.softmax(logits_bits, dim=-1)
        
        # # 计算每个 Bit 的熵，并求和得到当前 Token 的总信息熵 epsilon
        # # 论文公式: \epsilon = -\sum p_k * log(p_k)
        # entropy_bits = -torch.sum(probs_bits * torch.log(probs_bits + 1e-10), dim=-1) # [B, seq_len, 32]
        # epsilon = torch.sum(entropy_bits, dim=-1) # [B, seq_len]

        # # 3. 基于熵的动态温度映射
        # # 论文核心公式: T = T_0 * exp(-epsilon / alpha) + theta
        # T_0 = 2.0     # 低熵区域的最大额外温度增益
        # alpha = 8.0   # 衰减系数 (Infinity 32-bit 最大熵约22，alpha调大以匹配尺度)
        # theta = 0.5   # 基础保障温度 (高熵区域的严格采样阈值)
        
        # T_dynamic = T_0 * torch.exp(-epsilon / alpha) + theta # [B, seq_len]
        
        # # 4. 结合论文中专门针对 Scale-wise 模型的尺度退火 (Scale-wise Decay)
        # # 论文公式: T_s = T * [1 - beta * (s - S/2)]
        # S_total = len(vae_scale_schedule)
        # beta = 0.05 # 论文默认0.3，但因尺度多达13-15层，降低 beta 防止后期温度变为负数
        # scale_decay = 1.0 - beta * (si - (S_total // 2))
        
        # # 防止超高尺度时温度跌破安全下限
        # T_final = torch.clamp(T_dynamic * scale_decay, min=0.1) 
        
        # # 5. 将动态温度应用到 Logits 上进行重塑
        # T_final = T_final.unsqueeze(-1) # [B, seq_len, 1] 广播到全部 64 维
        # logits_BlV = raw_logits / T_final
        # # ----------------------------------------------------
        
        # ---------- Relative-boundary + bitwise entropy adaptive temperature ----------
        # 1. 获取原始 logits
        # if (cfg != 1) and add_cfg_on_logits:
        #     logits_all = infinity.get_logits(last_stage, cond_BD)
        #     logits_cond = logits_all[:B]
        #     logits_uncond = logits_all[B:]
        #     raw_logits = cfg * logits_cond + (1 - cfg) * logits_uncond
        # else:
        #     raw_logits = infinity.get_logits(last_stage[:B], cond_BD[:B])

        # # 2. 设置“最后一个已传输尺度”
        # #    你需要把 transmitted_token_maps 改成你代码里真实的变量名
        # #    例如如果传输了前 k 个 token map，那么 last_observed_scale_idx = k - 1
        # #    如果一个都没传，就设成 -1
        # last_observed_scale_idx = gt_leak - 1

        # # 3. 当前尺度用多大比例 token 做动态采样
        # #    第一个缺失尺度更激进，后面更保守
        # rel_stage = max(1, int(si - last_observed_scale_idx))
        # if rel_stage == 1:
        #     selective_ratio = 0.50
        # elif rel_stage == 2:
        #     selective_ratio = 0.30
        # else:
        #     selective_ratio = 0.15

        # # 4. 应用新的动态温度
        # logits_BlV, temp_debug = apply_entropy_adaptive_temperature(
        #     raw_logits=raw_logits,
        #     si=si,
        #     last_observed_scale_idx=last_observed_scale_idx,
        #     selective_ratio=selective_ratio,

        #     # 下面这组参数是“归一化熵”版本，不再是你原来 alpha=8.0 那套量纲
        #     t0=1.60,
        #     alpha=0.45,
        #     theta=0.55,

        #     # 非高不确定 token 的固定低温
        #     base_temperature=0.5,

        #     # 安全边界
        #     min_temperature=0.10,
        #     max_temperature=2.20,

        #     # relative-boundary 调节
        #     boundary_temp_boost=0.25,   # 第一个缺失尺度额外升温
        #     boundary_decay=0.55,        # 升温衰减速度
        #     tail_cool_rate=0.03,        # 后续尺度逐渐保守
        #     tail_cool_start=4,          # 从第几个缺失尺度开始轻微降温
        #     tail_cool_min=0.85,

        #     return_debug=False,         # 调参数时可改成 True
        # )
        # # ----------------------------------------------------------------------------


        if infinity.use_bit_label:
            tmp_bs, tmp_seq_len = logits_BlV.shape[:2] #[bs, seq_len, 64]
            logits_BlV = logits_BlV.reshape(tmp_bs, -1, 2) #[bs, seq_len*32, 2]
            prob = logits_BlV.softmax(dim=-1).view(-1, 2)
            idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=0, top_p=0.0, num_samples=1)[:, :, 0]
            idx_Bld = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1)
        else:
            idx_Bl = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=0, top_p=0.0, num_samples=1)[:, :, 0]
        
        if si <= gt_leak:
            idx_Bld = gt_ls_Bl[si]
        idx_Bld = idx_Bld.reshape(B, pn[1], pn[2], -1)
        idx_Bld = idx_Bld.unsqueeze(1) # [B, 1, h, w, d] or [B, 1, 2h, 2w, d]
        codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label') # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
        if si != num_stages_minus_1:
            summed_codes += F.interpolate(codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up)
            last_stage = F.interpolate(summed_codes, size=vae_scale_schedule[si+1], mode=vae.quantizer.z_interplote_up) # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
            last_stage = last_stage.squeeze(-3) # [B, d, h, w] or [B, d, 2h, 2w]
            last_stage = last_stage.reshape(*last_stage.shape[:2], -1) # [B, d, h*w] or [B, 4d, h*w]
            last_stage = torch.permute(last_stage, [0,2,1]) # [B, h*w, d] or [B, h*w, 4d]
        else:
            summed_codes += codes
        if si != num_stages_minus_1:
            last_stage = infinity.word_embed(infinity.norm0_ve(last_stage))
            last_stage = last_stage.repeat(bs//B, 1, 1)
    for b in infinity.unregistered_blocks: (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(False)
    img = vae.decode(summed_codes.squeeze(-3))
    img = (img + 1) / 2
    # img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8)
    return img

def decoding(args, infinity, vae, scale_schedule, text, text_tokenizer, text_encoder, trans_list, help_list):
    pass

# def decoding(args, infinity, vae, vae_scale_schedule, prompt, text_tokenizer, text_encoder, gt_ls_Bl, trans_list, help_list, tau_list=0.5, cfg_insertion_layer=[0]):
#     if not isinstance(tau_list, list):
#         tau_list = [tau_list] * len(vae_scale_schedule)
#     label_B_or_BLT = encode_prompt(text_tokenizer, text_encoder, prompt)
#     kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
#     bs = B = 1
#     kv_compact = infinity.text_norm(kv_compact)
#     sos = cond_BD = infinity.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k)) 
#     kv_compact = infinity.text_proj_for_ca(kv_compact) # kv_compact shape: [304, 4096]
#     ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
#     last_stage = sos.unsqueeze(1).expand(bs, 1, -1) + infinity.pos_start.expand(bs, 1, -1)
#     with torch.amp.autocast('cuda', enabled=False):
#         cond_BD_or_gss = infinity.shared_ada_lin(cond_BD.float()).float().contiguous()
#     for b in infinity.unregistered_blocks: (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(True)
#     abs_cfg_insertion_layers = []
#     add_cfg_on_logits, add_cfg_on_probs = False, False
#     leng = len(infinity.unregistered_blocks)
#     for item in cfg_insertion_layer:
#         if item == 0: # add cfg on logits
#             add_cfg_on_logits = True
#         elif item == 1: # add cfg on probs
#             add_cfg_on_probs = True # todo in the future, we may want to add cfg on logits and probs
#         elif item < 0: # determine to add cfg at item-th layer's output
#             assert leng+item > 0, f'cfg_insertion_layer: {item} is not valid since len(unregistered_blocks)={infinity.num_block_chunks}'
#             abs_cfg_insertion_layers.append(leng+item)
#         else:
#             raise ValueError(f'cfg_insertion_layer: {item} is not valid')
        
#     num_stages_minus_1 = len(vae_scale_schedule)-1
#     summed_codes = 0
#     cur_L = 0
#     decode_idx = []
#     for si, pn in enumerate(vae_scale_schedule):
#         cur_L += np.array(pn).prod()
#         need_to_pad = 0
#         attn_fn = None
#         if infinity.use_flex_attn:
#             attn_fn = infinity.attn_fn_compile_dict.get(tuple(vae_scale_schedule[:(si+1)]), None)
#         layer_idx = 0
#         for block_idx, b in enumerate(infinity.block_chunks):
#             # last_stage shape: [4, 1, 2048], cond_BD_or_gss.shape: [4, 1, 6, 2048], ca_kv[0].shape: [64, 2048], ca_kv[1].shape [5], ca_kv[2]: int
#             if infinity.add_lvl_embeding_only_first_block and block_idx == 0:
#                 last_stage = infinity.add_lvl_embeding(last_stage, si, vae_scale_schedule, need_to_pad=need_to_pad)
#             if not infinity.add_lvl_embeding_only_first_block: 
#                 last_stage = infinity.add_lvl_embeding(last_stage, si, vae_scale_schedule, need_to_pad=need_to_pad)
            
#             for m in b.module:
#                 last_stage = m(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=None, attn_fn=attn_fn, scale_schedule=vae_scale_schedule, rope2d_freqs_grid=infinity.rope2d_freqs_grid, scale_ind=si)
#                 layer_idx += 1
#         logits_BlV = infinity.get_logits(last_stage[:B], cond_BD[:B]).mul(1/tau_list[si])
    
#         tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
#         logits_BlV = logits_BlV.reshape(tmp_bs, -1, 2) #[bs, vocabulary_size*seq_len, 2]
#         B, l, V = logits_BlV.shape
#         prob = logits_BlV.softmax(dim=-1).view(-1, V)
#         prob = prob[:, 0].cpu().tolist()
#         bit_string = trans_list[si]
#         h_string = help_list[si]
#         # decompressed_string = []
#         # for j in range(len(h_string)):
#         #     if h_string[j] == 0:
#         #         decompressed_string.extend(bit_string[j])
#         #     else:
#         #         dec_str = decompress_from_bit_list(bit_string[j], args.vae_type, prob[j*args.vae_type:(j+1)*args.vae_type])
#                 # decompressed_string.extend(dec_str)
        
#         # 2. 在 decoding 函数的内部，找到遍历 j (Token) 的循环，用以下逻辑完全替换：
#         # ================= 新增：自适应 Mask 核心配置 =================
#         START_SCALE = 3         # 必须与 Sender 保持绝对一致
#         ENTROPY_THRESHOLD = 12.0 # 必须与 Sender 保持绝对一致
#         # =========================================================

#         # 假设当前外层循环尺度索引变量名叫 si （或者是 i，根据你的代码上下文调整）
#         decompressed_string = []
#         trans_idx = 0  # 独立流指针
        
#         for j in range(int(len(prob)/args.vae_type)):
#             p_token = prob[j*args.vae_type:(j+1)*args.vae_type]
            
#             # 直接从 help_list 读取发送端传来的指令 (j 代表当前第几个 Token)
#             flag = help_list[si][j] 
            
#             if flag == 2:
#                 # 状态 2：发送端指示安全，直接取概率最大值 (Argmax) 脑补
#                 dec_str = [0 if p > 0.5 else 1 for p in p_token]
#             elif flag == 0:
#                 # 状态 0：算术解码
#                 dec_str = decompress_from_bit_list(trans_list[si][trans_idx], args.vae_type, p_token)
#                 trans_idx += 1 # 只有真正读了流，指针才走
#             elif flag == 1:
#                 # 状态 1：原文透传
#                 dec_str = trans_list[si][trans_idx]
#                 trans_idx += 1
#             else:
#                 raise ValueError(f"Unknown flag: {flag}")
                
#             decompressed_string.extend(dec_str)
        
#         dec_idx = torch.tensor(decompressed_string).to(dtype=torch.int32).to(device=logits_BlV.device)
#         dec_idx = dec_idx.reshape(B, pn[1]*pn[2], -1)
#         decode_idx.append(dec_idx)
#         idx_Bld = dec_idx.reshape(B, pn[1], pn[2], -1)
#         idx_Bld = idx_Bld.unsqueeze(1) # [B, 1, h, w, d] or [B, 1, 2h, 2w, d]
#         codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label') # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
#         if si != num_stages_minus_1:
#             summed_codes += F.interpolate(codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up)
#             last_stage = F.interpolate(summed_codes, size=vae_scale_schedule[si+1], mode=vae.quantizer.z_interplote_up) # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
#             last_stage = last_stage.squeeze(-3) # [B, d, h, w] or [B, d, 2h, 2w]
#             last_stage = last_stage.reshape(*last_stage.shape[:2], -1) # [B, d, h*w] or [B, 4d, h*w]
#             last_stage = torch.permute(last_stage, [0,2,1]) # [B, h*w, d] or [B, h*w, 4d]
#         else:
#             summed_codes += codes
#         if si != num_stages_minus_1:
#             last_stage = infinity.word_embed(infinity.norm0_ve(last_stage))
#             last_stage = last_stage.repeat(bs//B, 1, 1)
#     for b in infinity.unregistered_blocks: (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(False)

#     return decode_idx


def decoding(args, infinity, vae, vae_scale_schedule, prompt, text_tokenizer, text_encoder, gt_ls_Bl, trans_list, help_list, tau_list=0.5, cfg_insertion_layer=[0]):
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(vae_scale_schedule)
    label_B_or_BLT = encode_prompt(text_tokenizer, text_encoder, prompt)
    kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
    bs = B = 1
    kv_compact = infinity.text_norm(kv_compact)
    sos = cond_BD = infinity.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k)) 
    kv_compact = infinity.text_proj_for_ca(kv_compact) # kv_compact shape: [304, 4096]
    ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
    last_stage = sos.unsqueeze(1).expand(bs, 1, -1) + infinity.pos_start.expand(bs, 1, -1)
    with torch.amp.autocast('cuda', enabled=False):
        cond_BD_or_gss = infinity.shared_ada_lin(cond_BD.float()).float().contiguous()
    for b in infinity.unregistered_blocks: (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(True)
    abs_cfg_insertion_layers = []
    add_cfg_on_logits, add_cfg_on_probs = False, False
    leng = len(infinity.unregistered_blocks)
    for item in cfg_insertion_layer:
        if item == 0: # add cfg on logits
            add_cfg_on_logits = True
        elif item == 1: # add cfg on probs
            add_cfg_on_probs = True # todo in the future, we may want to add cfg on logits and probs
        elif item < 0: # determine to add cfg at item-th layer's output
            assert leng+item > 0, f'cfg_insertion_layer: {item} is not valid since len(unregistered_blocks)={infinity.num_block_chunks}'
            abs_cfg_insertion_layers.append(leng+item)
        else:
            raise ValueError(f'cfg_insertion_layer: {item} is not valid')
        
    num_stages_minus_1 = len(vae_scale_schedule)-1
    summed_codes = 0
    cur_L = 0
    decode_idx = []
    for si, pn in enumerate(vae_scale_schedule):
        cur_L += np.array(pn).prod()
        need_to_pad = 0
        attn_fn = None
        if infinity.use_flex_attn:
            attn_fn = infinity.attn_fn_compile_dict.get(tuple(vae_scale_schedule[:(si+1)]), None)
        layer_idx = 0
        for block_idx, b in enumerate(infinity.block_chunks):
            # last_stage shape: [4, 1, 2048], cond_BD_or_gss.shape: [4, 1, 6, 2048], ca_kv[0].shape: [64, 2048], ca_kv[1].shape [5], ca_kv[2]: int
            if infinity.add_lvl_embeding_only_first_block and block_idx == 0:
                last_stage = infinity.add_lvl_embeding(last_stage, si, vae_scale_schedule, need_to_pad=need_to_pad)
            if not infinity.add_lvl_embeding_only_first_block: 
                last_stage = infinity.add_lvl_embeding(last_stage, si, vae_scale_schedule, need_to_pad=need_to_pad)
            
            for m in b.module:
                last_stage = m(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=None, attn_fn=attn_fn, scale_schedule=vae_scale_schedule, rope2d_freqs_grid=infinity.rope2d_freqs_grid, scale_ind=si)
                layer_idx += 1
        logits_BlV = infinity.get_logits(last_stage[:B], cond_BD[:B]).mul(1/tau_list[si])
    
        tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
        logits_BlV = logits_BlV.reshape(tmp_bs, -1, 2) #[bs, vocabulary_size*seq_len, 2]
        B, l, V = logits_BlV.shape
        prob = logits_BlV.softmax(dim=-1).view(-1, V)
        prob = prob[:, 0].cpu().tolist()
        bit_string = trans_list[si]
        h_string = help_list[si]
        decompressed_string = []
        for j in range(len(h_string)):
            if h_string[j] == 0:
                decompressed_string.extend(bit_string[j])
            else:
                dec_str = decompress_from_bit_list(bit_string[j], args.vae_type, prob[j*args.vae_type:(j+1)*args.vae_type])
                decompressed_string.extend(dec_str)
        
        dec_idx = torch.tensor(decompressed_string).to(dtype=torch.int32).to(device=logits_BlV.device)
        dec_idx = dec_idx.reshape(B, pn[1]*pn[2], -1)
        decode_idx.append(dec_idx)
        idx_Bld = dec_idx.reshape(B, pn[1], pn[2], -1)
        idx_Bld = idx_Bld.unsqueeze(1) # [B, 1, h, w, d] or [B, 1, 2h, 2w, d]
        codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label') # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
        if si != num_stages_minus_1:
            summed_codes += F.interpolate(codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up)
            last_stage = F.interpolate(summed_codes, size=vae_scale_schedule[si+1], mode=vae.quantizer.z_interplote_up) # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
            last_stage = last_stage.squeeze(-3) # [B, d, h, w] or [B, d, 2h, 2w]
            last_stage = last_stage.reshape(*last_stage.shape[:2], -1) # [B, d, h*w] or [B, 4d, h*w]
            last_stage = torch.permute(last_stage, [0,2,1]) # [B, h*w, d] or [B, h*w, 4d]
        else:
            summed_codes += codes
        if si != num_stages_minus_1:
            last_stage = infinity.word_embed(infinity.norm0_ve(last_stage))
            last_stage = last_stage.repeat(bs//B, 1, 1)
    for b in infinity.unregistered_blocks: (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(False)

    return decode_idx