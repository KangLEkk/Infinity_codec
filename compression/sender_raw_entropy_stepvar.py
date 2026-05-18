from compression.util import *
from utils.arithmeticcoding import compress_to_bit_list
import math
import numpy as np
import torch
import torch.nn.functional as F


def calc_token_entropy(p_list):
    ent = 0.0
    for p in p_list:
        if 0.0 < p < 1.0:
            ent += -p * math.log2(p) - (1 - p) * math.log2(1 - p)
    return ent


def _logical_token_entropy_from_logits(logits_bit_BLD2: torch.Tensor, bits_per_token: int) -> torch.Tensor:
    """
    logits_bit_BLD2: [B, L*bits_per_token, 2]
    return: [B, L]
    """
    with torch.amp.autocast('cuda', enabled=False):
        logits_f = logits_bit_BLD2.float()
        log_probs = F.log_softmax(logits_f, dim=-1)
        probs = log_probs.exp()
        bit_ent = -(probs * (log_probs / math.log(2.0))).sum(dim=-1)
    B, Lbits = bit_ent.shape
    if Lbits % bits_per_token != 0:
        raise ValueError(f"Lbits={Lbits} is not divisible by bits_per_token={bits_per_token}")
    return bit_ent.view(B, Lbits // bits_per_token, bits_per_token).sum(dim=-1)



def _apply_entropy_temperature(
    logits_bit_BLD2: torch.Tensor,
    scale_idx: int,
    num_scales: int,
    bits_per_token: int,
    args,
):
    """
    Entropy-aware dynamic temperature for binary-bit logits.
    This is optional for arithmetic-coding probabilities; keep it OFF by default,
    because it changes the entropy model and sender/receiver must be perfectly synced.
    """
    if not getattr(args, "entropy_on_codec", 0):
        return logits_bit_BLD2

    with torch.amp.autocast('cuda', enabled=False):
        logits_f = logits_bit_BLD2.float()
        token_entropy = _logical_token_entropy_from_logits(logits_f, bits_per_token)

        T0 = float(getattr(args, "entropy_T0", 2.0))
        alpha = float(getattr(args, "entropy_alpha", 2.5))
        theta = float(getattr(args, "entropy_theta", 0.6))
        beta = float(getattr(args, "entropy_beta", 0.0))
        tmin = float(getattr(args, "entropy_min_t", 0.45))
        tmax = float(getattr(args, "entropy_max_t", 2.50))

        T = T0 * torch.exp(-token_entropy / max(alpha, 1e-6)) + theta

        if beta > 0 and num_scales > 1:
            # The paper gives a scale-wise decay term for scale-wise AR models.
            # Here we use a normalized variant to avoid negative temperatures.
            mid = 0.5 * (num_scales - 1)
            offset = (scale_idx - mid) / max(mid, 1.0)
            scale_factor = 1.0 - beta * offset
            T = T * scale_factor

        T = T.clamp(min=tmin, max=tmax)
        B, Lt = token_entropy.shape
        T_bit = T.unsqueeze(-1).expand(B, Lt, bits_per_token).reshape(B, Lt * bits_per_token, 1)
        return logits_f / T_bit



def get_prob(infinity, vae, vae_scale_schedule, prompt, text_tokenizer, text_encoder, gt_ls_Bl, args=None, tau_list=0.5, cfg_insertion_layer=[0]):
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(vae_scale_schedule)
    label_B_or_BLT = encode_prompt(text_tokenizer, text_encoder, prompt)
    kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
    bs = B = 1
    kv_compact = infinity.text_norm(kv_compact)
    sos = cond_BD = infinity.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k))
    kv_compact = infinity.text_proj_for_ca(kv_compact)
    ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
    last_stage = sos.unsqueeze(1).expand(bs, 1, -1) + infinity.pos_start.expand(bs, 1, -1)
    with torch.amp.autocast('cuda', enabled=False):
        cond_BD_or_gss = infinity.shared_ada_lin(cond_BD.float()).float().contiguous()

    for b in infinity.unregistered_blocks:
        (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(True)

    add_cfg_on_logits, add_cfg_on_probs = False, False
    leng = len(infinity.unregistered_blocks)
    abs_cfg_insertion_layers = []
    for item in cfg_insertion_layer:
        if item == 0:
            add_cfg_on_logits = True
        elif item == 1:
            add_cfg_on_probs = True
        elif item < 0:
            assert leng + item > 0, f'cfg_insertion_layer: {item} is not valid since len(unregistered_blocks)={infinity.num_block_chunks}'
            abs_cfg_insertion_layers.append(leng + item)
        else:
            raise ValueError(f'cfg_insertion_layer: {item} is not valid')

    num_stages_minus_1 = len(vae_scale_schedule) - 1
    summed_codes = 0
    prob_list = []

    for si, pn in enumerate(vae_scale_schedule):
        need_to_pad = 0
        attn_fn = None
        if infinity.use_flex_attn:
            attn_fn = infinity.attn_fn_compile_dict.get(tuple(vae_scale_schedule[:(si + 1)]), None)

        for block_idx, b in enumerate(infinity.block_chunks):
            if infinity.add_lvl_embeding_only_first_block and block_idx == 0:
                last_stage = infinity.add_lvl_embeding(last_stage, si, vae_scale_schedule, need_to_pad=need_to_pad)
            if not infinity.add_lvl_embeding_only_first_block:
                last_stage = infinity.add_lvl_embeding(last_stage, si, vae_scale_schedule, need_to_pad=need_to_pad)
            for m in b.module:
                last_stage = m(
                    x=last_stage,
                    cond_BD=cond_BD_or_gss,
                    ca_kv=ca_kv,
                    attn_bias_or_two_vector=None,
                    attn_fn=attn_fn,
                    scale_schedule=vae_scale_schedule,
                    rope2d_freqs_grid=infinity.rope2d_freqs_grid,
                    scale_ind=si,
                )

        logits_BlV = infinity.get_logits(last_stage[:B], cond_BD[:B]).mul(1 / tau_list[si])
        tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
        logits_bit = logits_BlV.reshape(tmp_bs, -1, 2)
        if args is not None:
            logits_bit = _apply_entropy_temperature(logits_bit, si, len(vae_scale_schedule), getattr(args, "vae_type", 32), args)
        prob = logits_bit.softmax(dim=-1).view(-1, 2)
        prob_list.append(prob)

        idx_Bld = gt_ls_Bl[si]
        idx_Bld = idx_Bld.reshape(B, pn[1], pn[2], -1).unsqueeze(1)
        codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label')
        if si != num_stages_minus_1:
            summed_codes += F.interpolate(codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up)
            last_stage = F.interpolate(summed_codes, size=vae_scale_schedule[si + 1], mode=vae.quantizer.z_interplote_up)
            last_stage = last_stage.squeeze(-3)
            last_stage = last_stage.reshape(*last_stage.shape[:2], -1)
            last_stage = torch.permute(last_stage, [0, 2, 1])
            last_stage = infinity.word_embed(infinity.norm0_ve(last_stage))
            last_stage = last_stage.repeat(bs // B, 1, 1)
        else:
            summed_codes += codes

    for b in infinity.unregistered_blocks:
        (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(False)
    return prob_list



def encoding(args, infinity, vae, scale_schedule, text, text_tokenizer, text_encoder, gt_ms_idx_Bl):
    prob_list = get_prob(infinity, vae, scale_schedule, text, text_tokenizer, text_encoder, gt_ms_idx_Bl, args=args)
    trans_list = []
    help_list = []
    bpp_list = []

    sum_len = 0
    for i in range(len(gt_ms_idx_Bl)):
        gt_idx = gt_ms_idx_Bl[i]
        prob = prob_list[i][:, 0].cpu().tolist()
        gt_idx = gt_idx.view(-1, 1).squeeze().cpu().tolist()
        arithmetic_bits = []
        hbits = []
        for j in range(int(len(prob) / args.vae_type)):
            gt_token = gt_idx[j * args.vae_type:(j + 1) * args.vae_type]
            p_token = prob[j * args.vae_type:(j + 1) * args.vae_type]
            bits = compress_to_bit_list(gt_token, p_token)
            if len(bits) < args.vae_type:
                arithmetic_bits.append(bits)
                hbits.append(1)
                sum_len += len(bits)
            else:
                arithmetic_bits.append(gt_token)
                hbits.append(0)
                sum_len += args.vae_type
        trans_list.append(arithmetic_bits)
        help_list.append(hbits)
        sum_len += len(hbits)
        bpp_list.append(sum_len / 1024 / 1024)

    return trans_list, help_list, bpp_list
