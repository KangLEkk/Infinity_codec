from compression.util import *
from utils.arithmeticcoding import decompress_from_bit_list
import math
from typing import Dict, Any, Tuple, Optional

import torch

FLAG_RAW = 0
FLAG_ARITH = 1
FLAG_DROP = 2


def _relative_boundary_factor(
    rel_stage: int,
    boundary_temp_boost: float = 0.25,
    boundary_decay: float = 0.55,
    tail_cool_rate: float = 0.03,
    tail_cool_start: int = 4,
    tail_cool_min: float = 0.85,
) -> float:
    rel_stage = max(1, int(rel_stage))
    boost = boundary_temp_boost * math.exp(-boundary_decay * float(rel_stage - 1))
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
    if raw_logits.ndim != 3:
        raise ValueError(f"raw_logits should be [B, seq_len, 64], got shape={tuple(raw_logits.shape)}")
    if raw_logits.size(-1) % 2 != 0:
        raise ValueError(f"Last dim of raw_logits must be even, got {raw_logits.size(-1)}")

    B, L, V = raw_logits.shape
    num_bits = V // 2
    logits_bits = raw_logits.reshape(B, L, num_bits, 2)
    probs_bits = torch.softmax(logits_bits, dim=-1)
    entropy_bits = -(probs_bits * torch.log(probs_bits.clamp_min(eps))).sum(dim=-1)
    entropy_bits_norm = (entropy_bits / math.log(2.0)).clamp_(0.0, 1.0)
    token_uncertainty = entropy_bits_norm.mean(dim=-1)

    rel_stage = max(1, int(si - last_observed_scale_idx))
    rel_factor = _relative_boundary_factor(
        rel_stage=rel_stage,
        boundary_temp_boost=boundary_temp_boost,
        boundary_decay=boundary_decay,
        tail_cool_rate=tail_cool_rate,
        tail_cool_start=tail_cool_start,
        tail_cool_min=tail_cool_min,
    )

    dynamic_temp_bits = t0 * torch.exp(-entropy_bits_norm / max(alpha, 1e-6)) + theta
    dynamic_temp_bits = dynamic_temp_bits * rel_factor

    selective_ratio = float(max(0.0, min(1.0, selective_ratio)))
    if selective_ratio <= 0.0:
        use_dynamic_mask = torch.zeros_like(token_uncertainty, dtype=torch.bool)
    elif selective_ratio >= 1.0:
        use_dynamic_mask = torch.ones_like(token_uncertainty, dtype=torch.bool)
    else:
        threshold = torch.quantile(token_uncertainty, q=1.0 - selective_ratio, dim=1, keepdim=True)
        use_dynamic_mask = token_uncertainty >= threshold

    base_temp_bits = torch.full_like(dynamic_temp_bits, fill_value=base_temperature)
    final_temp_bits = torch.where(use_dynamic_mask.unsqueeze(-1), dynamic_temp_bits, base_temp_bits)
    final_temp_bits = final_temp_bits.clamp_(min=min_temperature, max=max_temperature)
    logits_bits_scaled = logits_bits / final_temp_bits.unsqueeze(-1)
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


def sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:
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
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(logits_BlV.softmax(dim=-1).view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)


def _forward_probs_one_scale(infinity, cond_BD, cond_BD_or_gss, ca_kv, last_stage, si, vae_scale_schedule):
    B = 1
    need_to_pad = 0
    attn_fn = None
    if infinity.use_flex_attn:
        attn_fn = infinity.attn_fn_compile_dict.get(tuple(vae_scale_schedule[: (si + 1)]), None)
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
    logits_BlV = infinity.get_logits(last_stage[:B], cond_BD[:B])
    return logits_BlV, last_stage


def decompress_cfg(infinity, vae, vae_scale_schedule, prompt, text_tokenizer, text_encoder, gt_leak, gt_ls_Bl, cfg_list=3, tau_list=0.5, cfg_insertion_layer=[0]):
    rng = infinity.rng
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(vae_scale_schedule)
        cfg_list = [cfg_list] * len(vae_scale_schedule)
    label_B_or_BLT = encode_prompt(text_tokenizer, text_encoder, prompt)
    kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
    B = 1
    if any(np.array(cfg_list) != 1):
        bs = 2 * B
        kv_compact_un = kv_compact.clone()
        total = 0
        for le in lens:
            kv_compact_un[total : total + le] = (infinity.cfg_uncond)[:le]
            total += le
        kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
        cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:] + cu_seqlens_k[-1]), dim=0)
    else:
        bs = B

    kv_compact = infinity.text_norm(kv_compact)
    sos = cond_BD = infinity.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k))
    kv_compact = infinity.text_proj_for_ca(kv_compact)
    ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
    last_stage = sos.unsqueeze(1).expand(bs, 1, -1) + infinity.pos_start.expand(bs, 1, -1)
    with torch.amp.autocast('cuda', enabled=False):
        cond_BD_or_gss = infinity.shared_ada_lin(cond_BD.float()).float().contiguous()
    for b in infinity.unregistered_blocks:
        (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(True)

    accu_BChw, cur_L, ret = None, 0, []
    idx_Bl_list, idx_Bld_list = [], []
    num_stages_minus_1 = len(vae_scale_schedule) - 1
    summed_codes = 0
    cur_L = 0
    for si, pn in enumerate(vae_scale_schedule):
        cfg = cfg_list[si]
        cur_L += np.array(pn).prod()
        need_to_pad = 0
        attn_fn = None
        if infinity.use_flex_attn:
            attn_fn = infinity.attn_fn_compile_dict.get(tuple(vae_scale_schedule[: (si + 1)]), None)
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

        if (cfg != 1) and False:
            logits_all = infinity.get_logits(last_stage, cond_BD)
            logits_cond = logits_all[:B]
            logits_uncond = logits_all[B:]
            raw_logits = cfg * logits_cond + (1 - cfg) * logits_uncond
        else:
            raw_logits = infinity.get_logits(last_stage[:B], cond_BD[:B])

        last_observed_scale_idx = gt_leak - 1
        rel_stage = max(1, int(si - last_observed_scale_idx))
        if rel_stage == 1:
            selective_ratio = 0.50
        elif rel_stage == 2:
            selective_ratio = 0.30
        else:
            selective_ratio = 0.15

        logits_BlV, _ = apply_entropy_adaptive_temperature(
            raw_logits=raw_logits,
            si=si,
            last_observed_scale_idx=last_observed_scale_idx,
            selective_ratio=selective_ratio,
            t0=1.60,
            alpha=0.45,
            theta=0.55,
            base_temperature=0.5,
            min_temperature=0.10,
            max_temperature=2.20,
            boundary_temp_boost=0.25,
            boundary_decay=0.55,
            tail_cool_rate=0.03,
            tail_cool_start=4,
            tail_cool_min=0.85,
            return_debug=False,
        )

        if infinity.use_bit_label:
            tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
            logits_BlV = logits_BlV.reshape(tmp_bs, -1, 2)
            idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=0, top_p=0.0, num_samples=1)[:, :, 0]
            idx_Bld = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1)
        else:
            idx_Bl = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=0, top_p=0.0, num_samples=1)[:, :, 0]

        if si <= gt_leak:
            idx_Bld = gt_ls_Bl[si]
        idx_Bld = idx_Bld.reshape(B, pn[1], pn[2], -1)
        idx_Bld = idx_Bld.unsqueeze(1)
        codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label')
        if si != num_stages_minus_1:
            summed_codes += F.interpolate(codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up)
            last_stage = F.interpolate(summed_codes, size=vae_scale_schedule[si + 1], mode=vae.quantizer.z_interplote_up)
            last_stage = last_stage.squeeze(-3)
            last_stage = last_stage.reshape(*last_stage.shape[:2], -1)
            last_stage = torch.permute(last_stage, [0, 2, 1])
        else:
            summed_codes += codes
        if si != num_stages_minus_1:
            last_stage = infinity.word_embed(infinity.norm0_ve(last_stage))
            last_stage = last_stage.repeat(bs // B, 1, 1)
    for b in infinity.unregistered_blocks:
        (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(False)
    img = vae.decode(summed_codes.squeeze(-3))
    img = (img + 1) / 2
    return img


def decoding(args, infinity, vae, vae_scale_schedule, prompt, text_tokenizer, text_encoder, gt_ls_Bl, trans_list, help_list, tau_list=0.5, cfg_insertion_layer=[0]):
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

    num_stages_minus_1 = len(vae_scale_schedule) - 1
    summed_codes = 0
    decode_idx = []
    for si, pn in enumerate(vae_scale_schedule):
        need_to_pad = 0
        attn_fn = None
        if infinity.use_flex_attn:
            attn_fn = infinity.attn_fn_compile_dict.get(tuple(vae_scale_schedule[: (si + 1)]), None)
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
        logits_BlV = logits_BlV.reshape(tmp_bs, -1, 2)
        _, _, V = logits_BlV.shape
        prob = logits_BlV.softmax(dim=-1).view(-1, V)
        prob = prob[:, 0].detach().cpu().tolist()

        scale_payloads = trans_list[si]
        scale_flags = help_list[si]
        decompressed_string = []
        trans_idx = 0
        num_tokens = len(scale_flags)
        for j in range(num_tokens):
            l = j * args.vae_type
            r = (j + 1) * args.vae_type
            p_token = prob[l:r]
            flag = scale_flags[j]
            if flag == FLAG_DROP:
                dec_str = [0 if p >= 0.5 else 1 for p in p_token]
            elif flag == FLAG_RAW:
                dec_str = scale_payloads[trans_idx]
                trans_idx += 1
            elif flag == FLAG_ARITH:
                dec_str = decompress_from_bit_list(scale_payloads[trans_idx], args.vae_type, p_token)
                trans_idx += 1
            else:
                raise ValueError(f"Unknown help flag: {flag}")
            decompressed_string.extend(dec_str)

        dec_idx = torch.tensor(decompressed_string, dtype=torch.int32, device=logits_BlV.device)
        dec_idx = dec_idx.reshape(B, pn[1] * pn[2], -1)
        decode_idx.append(dec_idx)
        idx_Bld = dec_idx.reshape(B, pn[1], pn[2], -1)
        idx_Bld = idx_Bld.unsqueeze(1)
        codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label')
        if si != num_stages_minus_1:
            summed_codes += F.interpolate(codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up)
            last_stage = F.interpolate(summed_codes, size=vae_scale_schedule[si + 1], mode=vae.quantizer.z_interplote_up)
            last_stage = last_stage.squeeze(-3)
            last_stage = last_stage.reshape(*last_stage.shape[:2], -1)
            last_stage = torch.permute(last_stage, [0, 2, 1])
        else:
            summed_codes += codes
        if si != num_stages_minus_1:
            last_stage = infinity.word_embed(infinity.norm0_ve(last_stage))
            last_stage = last_stage.repeat(bs // B, 1, 1)

    for b in infinity.unregistered_blocks:
        (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(False)
    return decode_idx
