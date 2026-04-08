from compression.util import *
from utils.arithmeticcoding import compress_to_bit_list
import math
import torch

FLAG_RAW = 0
FLAG_ARITH = 1
FLAG_DROP = 2


def calc_token_entropy(p_list):
    ent = 0.0
    for p in p_list:
        if 0.0 < p < 1.0:
            ent += -p * math.log2(p) - (1 - p) * math.log2(1 - p)
    return ent


def _bit_gt_prob(p0: float, gt_bit: int, eps: float = 1e-8) -> float:
    p = p0 if int(gt_bit) == 0 else (1.0 - p0)
    return min(max(float(p), eps), 1.0 - eps)


def _token_distortion_proxy(
    p_token,
    gt_token,
    lambda_mismatch: float = 2.0,
    lambda_entropy: float = 0.25,
    eps: float = 1e-8,
):
    gt_probs = [_bit_gt_prob(p, g, eps=eps) for p, g in zip(p_token, gt_token)]
    bit_nll = -sum(math.log2(x) for x in gt_probs) / max(1, len(gt_probs))
    argmax_bits = [0 if p >= 0.5 else 1 for p in p_token]
    bit_mismatch = sum(int(a != int(g)) for a, g in zip(argmax_bits, gt_token)) / max(1, len(gt_token))
    bit_entropy = calc_token_entropy(p_token) / max(1, len(gt_token))
    dist_proxy = bit_nll + lambda_mismatch * bit_mismatch + lambda_entropy * bit_entropy
    return {
        "bit_nll": float(bit_nll),
        "bit_mismatch": float(bit_mismatch),
        "bit_entropy": float(bit_entropy),
        "dist_proxy": float(dist_proxy),
    }


def _token_rd_drop_score(
    keep_bits: int,
    scale_idx: int,
    num_scales: int,
    p_token,
    gt_token,
    rd_scale_beta: float = 1.0,
    lambda_mismatch: float = 2.0,
    lambda_entropy: float = 0.25,
    eps: float = 1e-8,
):
    dist_stats = _token_distortion_proxy(
        p_token,
        gt_token,
        lambda_mismatch=lambda_mismatch,
        lambda_entropy=lambda_entropy,
        eps=eps,
    )
    denom = max(1, num_scales - 1)
    coarse_importance = 1.0 + float(rd_scale_beta) * (1.0 - float(scale_idx) / float(denom))
    effective_dist = dist_stats["dist_proxy"] * coarse_importance
    score = float(keep_bits) / max(effective_dist, eps)
    return score, dist_stats, coarse_importance


def get_prob(infinity, vae, vae_scale_schedule, prompt, text_tokenizer, text_encoder, gt_ls_Bl, tau_list=0.5, cfg_insertion_layer=[0]):
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

    leng = len(infinity.unregistered_blocks)
    add_cfg_on_logits, add_cfg_on_probs = False, False
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
    cur_L = 0
    prob_list = []
    for si, pn in enumerate(vae_scale_schedule):
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
        logits_BlV = infinity.get_logits(last_stage[:B], cond_BD[:B]).mul(1 / tau_list[si])
        tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
        logits_BlV = logits_BlV.reshape(tmp_bs, -1, 2)
        B, l, V = logits_BlV.shape
        prob = logits_BlV.softmax(dim=-1).view(-1, V)
        prob_list.append(prob)

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
    return prob_list


def encoding(
    args,
    infinity,
    vae,
    scale_schedule,
    text,
    text_tokenizer,
    text_encoder,
    gt_ms_idx_Bl,
    rd_drop_ratio: float = 0.0,
    precomputed_prob_list=None,
):
    prob_list = precomputed_prob_list
    if prob_list is None:
        prob_list = get_prob(infinity, vae, scale_schedule, text, text_tokenizer, text_encoder, gt_ms_idx_Bl)

    rd_drop_ratio = float(max(0.0, min(1.0, rd_drop_ratio)))
    rd_drop_start_scale = int(getattr(args, 'rd_drop_start_scale', 0))
    rd_side_bits = int(getattr(args, 'rd_side_bits', 2))
    rd_scale_beta = float(getattr(args, 'rd_scale_beta', 1.0))
    rd_lambda_mismatch = float(getattr(args, 'rd_lambda_mismatch', 2.0))
    rd_lambda_entropy = float(getattr(args, 'rd_lambda_entropy', 0.25))

    num_scales = len(gt_ms_idx_Bl)
    token_infos_by_scale = []
    candidates = []

    for i in range(num_scales):
        gt_idx = gt_ms_idx_Bl[i]
        prob = prob_list[i][:, 0].detach().cpu().tolist()
        gt_idx = gt_idx.view(-1, 1).squeeze().detach().cpu().tolist()

        num_tokens = int(len(prob) / args.vae_type)
        scale_infos = []
        for j in range(num_tokens):
            l = j * args.vae_type
            r = (j + 1) * args.vae_type
            p_token = prob[l:r]
            gt_token = gt_idx[l:r]

            arith_bits = compress_to_bit_list(gt_token, p_token)
            arith_len = len(arith_bits)
            raw_bits = list(gt_token)
            raw_len = len(raw_bits)

            if arith_len < raw_len:
                keep_payload = arith_bits
                keep_flag = FLAG_ARITH
                keep_bits = arith_len
            else:
                keep_payload = raw_bits
                keep_flag = FLAG_RAW
                keep_bits = raw_len

            score, dist_stats, coarse_importance = _token_rd_drop_score(
                keep_bits=keep_bits,
                scale_idx=i,
                num_scales=num_scales,
                p_token=p_token,
                gt_token=gt_token,
                rd_scale_beta=rd_scale_beta,
                lambda_mismatch=rd_lambda_mismatch,
                lambda_entropy=rd_lambda_entropy,
            )

            info = {
                'scale_idx': i,
                'token_idx': j,
                'p_token': p_token,
                'gt_token': gt_token,
                'keep_payload': keep_payload,
                'keep_flag': keep_flag,
                'keep_bits': int(keep_bits),
                'arith_len': int(arith_len),
                'raw_len': int(raw_len),
                'rd_score': float(score),
                'coarse_importance': float(coarse_importance),
                **dist_stats,
            }
            scale_infos.append(info)
            if i >= rd_drop_start_scale:
                candidates.append(info)
        token_infos_by_scale.append(scale_infos)

    num_candidates = len(candidates)
    num_drop = int(num_candidates * rd_drop_ratio)
    drop_keys = set()
    if num_drop > 0:
        ranked = sorted(candidates, key=lambda x: x['rd_score'], reverse=True)
        drop_keys = {(x['scale_idx'], x['token_idx']) for x in ranked[:num_drop]}

    trans_list = []
    help_list = []
    bpp_list = []
    scale_stats = []
    cumulative_total_bits = []
    cumulative_bits = 0

    for i, scale_infos in enumerate(token_infos_by_scale):
        payloads = []
        flags = []
        dropped = 0
        payload_bits_this_scale = 0

        for info in scale_infos:
            key = (info['scale_idx'], info['token_idx'])
            if key in drop_keys:
                flags.append(FLAG_DROP)
                dropped += 1
            else:
                flags.append(info['keep_flag'])
                payloads.append(info['keep_payload'])
                payload_bits_this_scale += len(info['keep_payload'])

        side_bits_this_scale = len(scale_infos) * rd_side_bits
        total_bits_this_scale = payload_bits_this_scale + side_bits_this_scale
        cumulative_bits += total_bits_this_scale
        cumulative_total_bits.append(int(cumulative_bits))
        bpp_list.append(int(cumulative_bits))

        trans_list.append(payloads)
        help_list.append(flags)
        scale_stats.append({
            'scale_idx': i,
            'num_tokens': len(scale_infos),
            'num_dropped_tokens': dropped,
            'drop_ratio_in_scale': float(dropped / max(1, len(scale_infos))),
            'payload_bits': int(payload_bits_this_scale),
            'side_bits': int(side_bits_this_scale),
            'total_bits': int(total_bits_this_scale),
            'cumulative_total_bits': int(cumulative_bits),
        })

    packet_meta = {
        'rd_drop_ratio': float(rd_drop_ratio),
        'rd_drop_start_scale': int(rd_drop_start_scale),
        'rd_side_bits': int(rd_side_bits),
        'rd_scale_beta': float(rd_scale_beta),
        'rd_lambda_mismatch': float(rd_lambda_mismatch),
        'rd_lambda_entropy': float(rd_lambda_entropy),
        'num_total_tokens': int(sum(len(x) for x in token_infos_by_scale)),
        'num_candidate_tokens': int(num_candidates),
        'num_dropped_tokens': int(len(drop_keys)),
        'num_kept_tokens': int(sum(len(x) for x in token_infos_by_scale) - len(drop_keys)),
        'scale_stats': scale_stats,
        'cumulative_total_bits': cumulative_total_bits,
    }

    return trans_list, help_list, packet_meta
