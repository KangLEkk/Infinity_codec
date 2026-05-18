from compression.util import *
from utils.arithmeticcoding import decompress_from_bit_list
import math
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

try:
    from infinity.models.basic import CrossAttnBlock
except Exception:
    class CrossAttnBlock:  # fallback sentinel for isinstance checks
        pass


# ==============================
# Entropy-aware sampling helpers
# ==============================

def calc_token_entropy(p_list):
    ent = 0.0
    for p in p_list:
        if 0.0 < p < 1.0:
            ent += -p * math.log2(p) - (1 - p) * math.log2(1 - p)
    return ent



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
    return torch.multinomial(
        logits_BlV.softmax(dim=-1).view(-1, V),
        num_samples=num_samples,
        replacement=replacement,
        generator=rng,
    ).view(B, l, num_samples)



def _logical_token_entropy_from_logits(logits_bit_BLD2: torch.Tensor, bits_per_token: int) -> torch.Tensor:
    probs0 = logits_bit_BLD2.softmax(dim=-1)[..., 0].clamp_(1e-8, 1 - 1e-8)
    bit_ent = -(probs0 * torch.log2(probs0) + (1.0 - probs0) * torch.log2(1.0 - probs0))
    B, Lbits = bit_ent.shape
    return bit_ent.view(B, Lbits // bits_per_token, bits_per_token).sum(dim=-1)



def entropy_scale_temperature(
    logits_bit_BLD2: torch.Tensor,
    scale_idx: int,
    num_scales: int,
    bits_per_token: int,
    args,
):
    token_entropy = _logical_token_entropy_from_logits(logits_bit_BLD2, bits_per_token)

    T0 = float(getattr(args, "entropy_T0", 2.0))
    alpha = float(getattr(args, "entropy_alpha", 2.5))
    theta = float(getattr(args, "entropy_theta", 0.6))
    beta = float(getattr(args, "entropy_beta", 0.0))
    tmin = float(getattr(args, "entropy_min_t", 0.45))
    tmax = float(getattr(args, "entropy_max_t", 2.50))

    T = T0 * torch.exp(-token_entropy / max(alpha, 1e-6)) + theta

    if beta > 0 and num_scales > 1:
        mid = 0.5 * (num_scales - 1)
        offset = (scale_idx - mid) / max(mid, 1.0)
        scale_factor = 1.0 - beta * offset
        T = T * scale_factor

    T = T.clamp(min=tmin, max=tmax)
    B, Lt = token_entropy.shape
    T_bit = T.unsqueeze(-1).expand(B, Lt, bits_per_token).reshape(B, Lt * bits_per_token, 1)
    return logits_bit_BLD2 / T_bit, token_entropy, T


# ==============================
# StepVAR helpers / wrappers
# ==============================

def _minmax_norm(x: torch.Tensor) -> torch.Tensor:
    xmin = x.amin(dim=1, keepdim=True)
    xmax = x.amax(dim=1, keepdim=True)
    return (x - xmin) / (xmax - xmin + 1e-6)



def stepvar_total_score(x_BLC: torch.Tensor, H: int, W: int, pca_iters: int = 3, pca_weight: float = 0.5) -> torch.Tensor:
    B, L, C = x_BLC.shape
    assert L == H * W, f"L={L}, but H*W={H*W}"

    x_centered = x_BLC - x_BLC.mean(dim=1, keepdim=True)
    v = torch.randn(B, C, 1, device=x_BLC.device, dtype=x_BLC.dtype)
    v = F.normalize(v, dim=1)
    for _ in range(max(1, pca_iters)):
        v = torch.bmm(x_centered.transpose(1, 2), torch.bmm(x_centered, v))
        v = F.normalize(v, dim=1)

    s_str = torch.abs(torch.bmm(x_centered, v).squeeze(-1))

    x_spatial = x_BLC.transpose(1, 2).reshape(B, C, H, W)
    x_low = F.avg_pool2d(x_spatial, kernel_size=3, stride=1, padding=1)
    x_high = x_spatial - x_low
    s_txt = x_high.pow(2).sum(dim=1).reshape(B, L)

    s_total = pca_weight * _minmax_norm(s_str) + _minmax_norm(s_txt)
    return s_total



def upsample_bit_labels_nearest(idx_prev_Bld: torch.Tensor, prev_hw, curr_hw) -> torch.Tensor:
    prev_h, prev_w = prev_hw
    curr_h, curr_w = curr_hw
    B, _, d = idx_prev_Bld.shape
    x = idx_prev_Bld.reshape(B, prev_h, prev_w, d).permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(curr_h, curr_w), mode='nearest')
    x = x.round().clamp_(0, 1)
    x = x.permute(0, 2, 3, 1).reshape(B, curr_h * curr_w, d)
    return x.to(dtype=idx_prev_Bld.dtype)



def _flat_idx_to_yx(idx: torch.Tensor, W: int):
    y = idx // W
    x = idx % W
    return torch.stack([y, x], dim=-1)



def nearest_neighbor_token_fill(sampled_kept_Bkd: torch.Tensor, keep_idx_Bk: torch.Tensor, H: int, W: int) -> torch.Tensor:
    B, k, d = sampled_kept_Bkd.shape
    L = H * W
    full_idx = torch.arange(L, device=sampled_kept_Bkd.device)
    full_coords = _flat_idx_to_yx(full_idx, W).float()
    dense_list = []
    for b in range(B):
        keep_coords = _flat_idx_to_yx(keep_idx_Bk[b], W).float()
        dists = torch.cdist(full_coords.unsqueeze(0), keep_coords.unsqueeze(0)).squeeze(0)
        nn_idx = dists.argmin(dim=1)
        dense = sampled_kept_Bkd[b][nn_idx]
        dense_list.append(dense)
    return torch.stack(dense_list, dim=0)



def _get_block_chunks(infinity):
    if hasattr(infinity, "block_chunks"):
        return list(infinity.block_chunks)
    if hasattr(infinity, "blocks"):
        return [SimpleNamespace(module=list(infinity.blocks))]
    raise AttributeError("Infinity model has neither block_chunks nor blocks")



def _call_add_lvl_embedding(infinity, last_stage, si, scale_schedule, need_to_pad=0, token_indices=None):
    try:
        return infinity.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad, token_indices=token_indices)
    except TypeError:
        return infinity.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad)



def _call_block_module(module, x, cond_BD, ca_kv, attn_fn, scale_schedule, rope2d_freqs_grid, si, token_indices=None):
    try:
        return module(
            x=x,
            cond_BD=cond_BD,
            ca_kv=ca_kv,
            attn_bias_or_two_vector=None,
            attn_fn=attn_fn,
            scale_schedule=scale_schedule,
            rope2d_freqs_grid=rope2d_freqs_grid,
            scale_ind=si,
            token_indices=token_indices,
        )
    except TypeError:
        return module(
            x=x,
            cond_BD=cond_BD,
            ca_kv=ca_kv,
            attn_bias_or_two_vector=None,
            attn_fn=attn_fn,
            scale_schedule=scale_schedule,
            rope2d_freqs_grid=rope2d_freqs_grid,
            scale_ind=si,
        )



def _resolve_stepvar_ratio(args, infinity, si: int, num_scales: int) -> float:
    ratios = getattr(args, "stepvar_prune_ratios", None)
    if hasattr(infinity, "_stepvar_resolve_prune_ratio"):
        ratio = float(infinity._stepvar_resolve_prune_ratio(ratios, si, num_scales))
    else:
        if ratios is None:
            ratio = 0.0
        elif isinstance(ratios, dict):
            ratio = float(ratios.get(si, ratios.get(str(si), 0.0)))
        elif isinstance(ratios, (list, tuple, np.ndarray)):
            if len(ratios) == 0:
                ratio = 0.0
            elif len(ratios) == 1:
                ratio = float(ratios[0])
            elif len(ratios) == num_scales:
                ratio = float(ratios[si])
            else:
                tail_offset = num_scales - len(ratios)
                ratio = float(ratios[si - tail_offset]) if si >= tail_offset else 0.0
        else:
            ratio = float(ratios)

    skip_last_n = int(getattr(args, "stepvar_skip_last_n", 0))
    if skip_last_n > 0 and si >= (num_scales - skip_last_n):
        ratio = max(ratio, 1.0)
    return float(max(0.0, min(1.0, ratio)))



def _idx3d_to_5d(idx_Bld: torch.Tensor, pn, apply_spatial_patchify: bool = False) -> torch.Tensor:
    B = idx_Bld.shape[0]
    pt, ph, pw = int(pn[0]), int(pn[1]), int(pn[2])
    x = idx_Bld.reshape(B, ph, pw, -1)
    if apply_spatial_patchify:
        dtype = x.dtype
        x = x.permute(0, 3, 1, 2).contiguous().float()
        x = torch.nn.functional.pixel_shuffle(x, 2)
        x = x.permute(0, 2, 3, 1).contiguous()
        if dtype.is_floating_point:
            x = x.to(dtype=dtype)
        else:
            x = x.round().to(dtype=dtype)
    return x.unsqueeze(1)



def _codes_to_next_last_stage(infinity, vae, summed_codes, si, vae_scale_schedule, bs, B):
    last_stage = F.interpolate(summed_codes, size=vae_scale_schedule[si + 1], mode=vae.quantizer.z_interplote_up)
    last_stage = last_stage.squeeze(-3)
    if getattr(infinity, "apply_spatial_patchify", 0):
        last_stage = torch.nn.functional.pixel_unshuffle(last_stage, 2)
    last_stage = last_stage.reshape(*last_stage.shape[:2], -1)
    last_stage = torch.permute(last_stage, [0, 2, 1])
    last_stage = infinity.word_embed(infinity.norm0_ve(last_stage))
    last_stage = last_stage.repeat(bs // B, 1, 1)
    return last_stage


# ==============================
# Main decoding / reconstruction
# ==============================

def decompress_cfg(args, infinity, vae, vae_scale_schedule, prompt, text_tokenizer, text_encoder, gt_leak, gt_ls_Bl, cfg_list=3, tau_list=0.5, cfg_insertion_layer=[0]):
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
            kv_compact_un[total:total + le] = (infinity.cfg_uncond)[:le]
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

    add_cfg_on_logits = False
    leng = len(infinity.unregistered_blocks)
    abs_cfg_insertion_layers = []
    for item in cfg_insertion_layer:
        if item == 0:
            add_cfg_on_logits = True
        elif item == 1:
            pass
        elif item < 0:
            assert leng + item > 0, f'cfg_insertion_layer: {item} is not valid since len(unregistered_blocks)={leng}'
            abs_cfg_insertion_layers.append(leng + item)
        else:
            raise ValueError(f'cfg_insertion_layer: {item} is not valid')

    num_stages_minus_1 = len(vae_scale_schedule) - 1
    summed_codes = 0
    block_chunks = _get_block_chunks(infinity)
    apply_spatial_patchify = bool(getattr(infinity, "apply_spatial_patchify", 0))

    for si, pn in enumerate(vae_scale_schedule):
        curr_h, curr_w = int(pn[1]), int(pn[2])

        # For transmitted / already-decoded scales, no need to rerun transformer blocks.
        if si <= gt_leak:
            idx_Bld = gt_ls_Bl[si]
            idx_Bld_5d = _idx3d_to_5d(idx_Bld, pn, apply_spatial_patchify=apply_spatial_patchify)
            codes = vae.quantizer.lfq.indices_to_codes(idx_Bld_5d, label_type='bit_label')
        else:
            cfg = cfg_list[si]
            need_to_pad = 0
            attn_fn = None
            if getattr(infinity, "use_flex_attn", False):
                attn_fn = infinity.attn_fn_compile_dict.get(tuple(vae_scale_schedule[:(si + 1)]), None)

            stepvar_ratio = 0.0
            if getattr(args, "stepvar_enabled", False) and si >= int(getattr(args, "stepvar_min_scale", 0)):
                stepvar_ratio = _resolve_stepvar_ratio(args, infinity, si, len(vae_scale_schedule))

            token_indices = None
            recover_index = None
            logits_hidden = None
            skip_transformer_this_scale = False

            if stepvar_ratio > 0 and hasattr(infinity, "_stepvar_select_keep_indices"):
                keep_idx = infinity._stepvar_select_keep_indices(
                    last_stage[:B],
                    pn=pn,
                    prune_ratio=stepvar_ratio,
                    w_str=float(getattr(args, "stepvar_w_str", 0.5)),
                    power_iter=int(getattr(args, "stepvar_power_iter", 3)),
                )
                if keep_idx is not None:
                    if keep_idx.numel() == 0:
                        skip_transformer_this_scale = True
                        logits_hidden = last_stage
                    elif keep_idx.numel() < last_stage.shape[1]:
                        token_indices = keep_idx
                        if hasattr(infinity, "_stepvar_build_recover_index"):
                            recover_index = infinity._stepvar_build_recover_index(
                                keep_idx,
                                pn=pn,
                                chunk_size=int(getattr(args, "stepvar_chunk_size", 4096)),
                            )
                        last_stage = last_stage.index_select(dim=1, index=keep_idx)

            layer_idx = 0
            if not skip_transformer_this_scale:
                for block_idx, b in enumerate(block_chunks):
                    if getattr(infinity, "add_lvl_embeding_only_first_block", 1) and block_idx == 0:
                        last_stage = _call_add_lvl_embedding(infinity, last_stage, si, vae_scale_schedule, need_to_pad=need_to_pad, token_indices=token_indices)
                    if not getattr(infinity, "add_lvl_embeding_only_first_block", 1):
                        last_stage = _call_add_lvl_embedding(infinity, last_stage, si, vae_scale_schedule, need_to_pad=need_to_pad, token_indices=token_indices)
                    for m in b.module:
                        last_stage = _call_block_module(
                            m,
                            x=last_stage,
                            cond_BD=cond_BD_or_gss,
                            ca_kv=ca_kv,
                            attn_fn=attn_fn,
                            scale_schedule=vae_scale_schedule,
                            rope2d_freqs_grid=infinity.rope2d_freqs_grid,
                            si=si,
                            token_indices=token_indices,
                        )
                        if (cfg != 1) and (layer_idx in abs_cfg_insertion_layers):
                            last_stage = cfg * last_stage[:B] + (1 - cfg) * last_stage[B:]
                            last_stage = torch.cat((last_stage, last_stage), 0)
                        layer_idx += 1

                if recover_index is not None and hasattr(infinity, "_stepvar_recover_dense"):
                    logits_hidden = infinity._stepvar_recover_dense(last_stage, recover_index)
                else:
                    logits_hidden = last_stage
            else:
                logits_hidden = last_stage if logits_hidden is None else logits_hidden

            if (cfg != 1) and add_cfg_on_logits:
                logits_BlV = infinity.get_logits(logits_hidden, cond_BD).mul(1 / tau_list[si])
                logits_BlV = cfg * logits_BlV[:B] + (1 - cfg) * logits_BlV[B:]
            else:
                logits_BlV = infinity.get_logits(logits_hidden[:B], cond_BD[:B]).mul(1 / tau_list[si])

            if infinity.use_bit_label:
                tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
                logits_bit = logits_BlV.reshape(tmp_bs, -1, 2)

                if getattr(args, "entropy_sampling", 1):
                    logits_bit, _, _ = entropy_scale_temperature(
                        logits_bit,
                        scale_idx=si,
                        num_scales=len(vae_scale_schedule),
                        bits_per_token=getattr(args, "vae_type", 32),
                        args=args,
                    )

                idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(
                    logits_bit,
                    rng=rng,
                    top_k=int(getattr(args, "sample_top_k", 0)),
                    top_p=float(getattr(args, "sample_top_p", 0.0)),
                    num_samples=1,
                )[:, :, 0]
                idx_Bld = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1)
            else:
                idx_Bl = sample_with_top_k_top_p_also_inplace_modifying_logits_(
                    logits_BlV,
                    rng=rng,
                    top_k=int(getattr(args, "sample_top_k", 0)),
                    top_p=float(getattr(args, "sample_top_p", 0.0)),
                    num_samples=1,
                )[:, :, 0]
                idx_Bld = idx_Bl

            idx_Bld_5d = _idx3d_to_5d(idx_Bld, pn, apply_spatial_patchify=apply_spatial_patchify)
            codes = vae.quantizer.lfq.indices_to_codes(idx_Bld_5d, label_type='bit_label')

        if si != num_stages_minus_1:
            summed_codes += F.interpolate(codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up)
            last_stage = _codes_to_next_last_stage(infinity, vae, summed_codes, si, vae_scale_schedule, bs, B)
        else:
            summed_codes += codes

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

    leng = len(infinity.unregistered_blocks)
    abs_cfg_insertion_layers = []
    for item in cfg_insertion_layer:
        if item == 0 or item == 1:
            continue
        elif item < 0:
            assert leng + item > 0, f'cfg_insertion_layer: {item} is not valid since len(unregistered_blocks)={leng}'
            abs_cfg_insertion_layers.append(leng + item)
        else:
            raise ValueError(f'cfg_insertion_layer: {item} is not valid')

    num_stages_minus_1 = len(vae_scale_schedule) - 1
    summed_codes = 0
    decode_idx = []
    block_chunks = _get_block_chunks(infinity)
    apply_spatial_patchify = bool(getattr(infinity, "apply_spatial_patchify", 0))

    for si, pn in enumerate(vae_scale_schedule):
        need_to_pad = 0
        attn_fn = None
        if getattr(infinity, "use_flex_attn", False):
            attn_fn = infinity.attn_fn_compile_dict.get(tuple(vae_scale_schedule[:(si + 1)]), None)
        for block_idx, b in enumerate(block_chunks):
            if getattr(infinity, "add_lvl_embeding_only_first_block", 1) and block_idx == 0:
                last_stage = _call_add_lvl_embedding(infinity, last_stage, si, vae_scale_schedule, need_to_pad=need_to_pad, token_indices=None)
            if not getattr(infinity, "add_lvl_embeding_only_first_block", 1):
                last_stage = _call_add_lvl_embedding(infinity, last_stage, si, vae_scale_schedule, need_to_pad=need_to_pad, token_indices=None)
            for m in b.module:
                last_stage = _call_block_module(
                    m,
                    x=last_stage,
                    cond_BD=cond_BD_or_gss,
                    ca_kv=ca_kv,
                    attn_fn=attn_fn,
                    scale_schedule=vae_scale_schedule,
                    rope2d_freqs_grid=infinity.rope2d_freqs_grid,
                    si=si,
                    token_indices=None,
                )
        logits_BlV = infinity.get_logits(last_stage[:B], cond_BD[:B]).mul(1 / tau_list[si])
        logits_bit = logits_BlV.reshape(B, -1, 2)

        if getattr(args, "entropy_on_codec", 0):
            logits_bit, _, _ = entropy_scale_temperature(
                logits_bit,
                scale_idx=si,
                num_scales=len(vae_scale_schedule),
                bits_per_token=getattr(args, "vae_type", 32),
                args=args,
            )

        prob = logits_bit.softmax(dim=-1).view(-1, 2)
        prob = prob[:, 0].cpu().tolist()
        bit_string = trans_list[si]
        h_string = help_list[si]

        decompressed_string = []
        for j in range(len(h_string)):
            if h_string[j] == 0:
                decompressed_string.extend(bit_string[j])
            else:
                dec_str = decompress_from_bit_list(
                    bit_string[j],
                    args.vae_type,
                    prob[j * args.vae_type:(j + 1) * args.vae_type],
                )
                decompressed_string.extend(dec_str)

        dec_idx = torch.tensor(decompressed_string, dtype=torch.int32, device=logits_BlV.device)
        dec_idx = dec_idx.reshape(B, int(pn[1] * pn[2]), -1)
        decode_idx.append(dec_idx)

        idx_Bld = _idx3d_to_5d(dec_idx, pn, apply_spatial_patchify=apply_spatial_patchify)
        codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label')
        if si != num_stages_minus_1:
            summed_codes += F.interpolate(codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up)
            last_stage = _codes_to_next_last_stage(infinity, vae, summed_codes, si, vae_scale_schedule, bs, B)
        else:
            summed_codes += codes

    for b in infinity.unregistered_blocks:
        (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(False)

    return decode_idx
