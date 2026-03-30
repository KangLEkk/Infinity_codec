from compression.util import *
from utils.arithmeticcoding import decompress_from_bit_list, compress_to_bit_list
import math

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

def decompress_cfg(infinity, vae, vae_scale_schedule, prompt, text_tokenizer, text_encoder, gt_leak, gt_ls_Bl, cfg_list=3, tau_list=1, cfg_insertion_layer=[0]):
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
        
        # if (cfg != 1) and add_cfg_on_logits:
        #         # print(f'add cfg on add_cfg_on_logits')
        #         logits_BlV = infinity.get_logits(last_stage, cond_BD).mul(1/tau_list[si])
        #         logits_BlV = cfg * logits_BlV[:B] + (1-cfg) * logits_BlV[B:]
        # else:
        #     logits_BlV = infinity.get_logits(last_stage[:B], cond_BD[:B]).mul(1/tau_list[si])

        # ---------- 替换为基于信息熵的动态温度采样 ----------
        # 1. 获取纯净的原始 Logits (暂时不乘以静态的 1/tau)
        if (cfg != 1) and add_cfg_on_logits:
            logits_all = infinity.get_logits(last_stage, cond_BD)
            logits_cond = logits_all[:B]
            logits_uncond = logits_all[B:]
            raw_logits = cfg * logits_cond + (1 - cfg) * logits_uncond
        else:
            raw_logits = infinity.get_logits(last_stage[:B], cond_BD[:B])

        # 2. 计算比特级信息熵 (Bitwise Entropy)
        # Infinity 的 raw_logits 形状通常为 [B, seq_len, 64] (32个bit，每个bit有2个状态)
        tmp_bs, tmp_seq_len = raw_logits.shape[:2]
        
        # 将其 Reshape 为 [B, seq_len, 32, 2] 以便计算每个 Bit 的 Softmax 概率
        logits_bits = raw_logits.view(tmp_bs, tmp_seq_len, 32, 2)
        probs_bits = torch.softmax(logits_bits, dim=-1)
        
        # 计算每个 Bit 的熵，并求和得到当前 Token 的总信息熵 epsilon
        # 论文公式: \epsilon = -\sum p_k * log(p_k)
        entropy_bits = -torch.sum(probs_bits * torch.log(probs_bits + 1e-10), dim=-1) # [B, seq_len, 32]
        epsilon = torch.sum(entropy_bits, dim=-1) # [B, seq_len]

        # 3. 基于熵的动态温度映射
        # 论文核心公式: T = T_0 * exp(-epsilon / alpha) + theta
        T_0 = 2.0     # 低熵区域的最大额外温度增益
        alpha = 8.0   # 衰减系数 (Infinity 32-bit 最大熵约22，alpha调大以匹配尺度)
        theta = 0.5   # 基础保障温度 (高熵区域的严格采样阈值)
        
        T_dynamic = T_0 * torch.exp(-epsilon / alpha) + theta # [B, seq_len]
        
        # 4. 结合论文中专门针对 Scale-wise 模型的尺度退火 (Scale-wise Decay)
        # 论文公式: T_s = T * [1 - beta * (s - S/2)]
        S_total = len(vae_scale_schedule)
        beta = 0.05 # 论文默认0.3，但因尺度多达13-15层，降低 beta 防止后期温度变为负数
        scale_decay = 1.0 - beta * (si - (S_total // 2))
        
        # 防止超高尺度时温度跌破安全下限
        T_final = torch.clamp(T_dynamic * scale_decay, min=0.1) 
        
        # 5. 将动态温度应用到 Logits 上进行重塑
        T_final = T_final.unsqueeze(-1) # [B, seq_len, 1] 广播到全部 64 维
        logits_BlV = raw_logits / T_final
        # ----------------------------------------------------
        

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