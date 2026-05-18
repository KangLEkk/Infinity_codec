import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from compression.util import encode_prompt


_EPS = 1e-12

FLAG_RAW = 0
FLAG_ARITH = 1


def calc_token_entropy(p_list):
    ent = 0.0
    for p in p_list:
        if 0.0 < p < 1.0:
            ent += -p * math.log2(p) - (1 - p) * math.log2(1 - p)
    return ent


def logits_to_prob0(logits_BlV: torch.Tensor) -> torch.Tensor:
    if logits_BlV.ndim != 3:
        raise ValueError(f"logits_BlV must be [B, L, 2*d], got {tuple(logits_BlV.shape)}")
    B, L, V = logits_BlV.shape
    if V % 2 != 0:
        raise ValueError(f"Last dim must be even, got {V}")
    d = V // 2
    probs = torch.softmax(logits_BlV.view(B, L, d, 2).float(), dim=-1)
    return probs[..., 0].clamp(_EPS, 1.0 - _EPS)


def map_bits_from_prob0(prob0_BLd: torch.Tensor) -> torch.Tensor:
    return (prob0_BLd < 0.5).to(torch.uint8)


def binary_entropy(prob0: torch.Tensor) -> torch.Tensor:
    prob0 = prob0.clamp(_EPS, 1.0 - _EPS)
    prob1 = 1.0 - prob0
    return -(prob0 * torch.log2(prob0) + prob1 * torch.log2(prob1))


def _topk_indices(scores_1d: torch.Tensor, k: int) -> torch.Tensor:
    L = int(scores_1d.numel())
    k = int(max(0, min(int(k), L)))
    if k == 0:
        return torch.zeros((0,), device=scores_1d.device, dtype=torch.long)
    idx = torch.arange(L, device=scores_1d.device, dtype=scores_1d.dtype)
    adj = scores_1d + (-idx) * 1e-7
    _, top_idx = torch.topk(adj, k=k, largest=True, sorted=True)
    return torch.sort(top_idx.to(torch.long))[0]


def _normalize_score_1d(scores_1d: torch.Tensor) -> torch.Tensor:
    if scores_1d.numel() == 0:
        return scores_1d.float()
    scores = scores_1d.float()
    lo = scores.min()
    hi = scores.max()
    span = hi - lo
    if float(span.abs().item()) <= _EPS:
        return torch.zeros_like(scores)
    return (scores - lo) / span.clamp_min(_EPS)


def select_channels_by_entropy(
    prob0_BLd: torch.Tensor,
    d_eff: int,
    keep_ratio: Optional[float] = None,
    entropy_thr: Optional[float] = None,
) -> List[int]:
    p = prob0_BLd[:, :, :d_eff]
    H = binary_entropy(p)
    Hc = H.mean(dim=1)[0]
    if entropy_thr is not None:
        keep = torch.nonzero(Hc >= float(entropy_thr), as_tuple=False).view(-1)
        if keep.numel() == 0:
            keep = _topk_indices(Hc, 1)
        return keep.to(torch.long).tolist()
    r = 1.0 if keep_ratio is None else float(keep_ratio)
    r = max(0.0, min(1.0, r))
    m = int(math.ceil(d_eff * r))
    m = max(1, min(m, d_eff))
    return _topk_indices(Hc, m).to(torch.long).tolist()


def select_positions_by_entropy(
    prob0_BLd: torch.Tensor,
    Hs: int,
    Ws: int,
    d_eff: int,
    keep_ratio: Optional[float] = None,
    entropy_thr: Optional[float] = None,
) -> torch.Tensor:
    del Hs, Ws
    p = prob0_BLd[:, :, :d_eff]
    Hpos = binary_entropy(p).mean(dim=2)[0]
    if entropy_thr is not None:
        keep = torch.nonzero(Hpos >= float(entropy_thr), as_tuple=False).view(-1)
        if keep.numel() == 0:
            keep = _topk_indices(Hpos, 1)
        return keep.to(torch.long)
    r = 1.0 if keep_ratio is None else float(keep_ratio)
    r = max(0.0, min(1.0, r))
    k = int(round(Hpos.numel() * r))
    k = max(1, min(k, int(Hpos.numel())))
    return _topk_indices(Hpos, k)


def get_latent_channel_weights(vae, d: int, device: str, mode: str = "uniform") -> torch.Tensor:
    mode = str(mode or "uniform").lower()
    if mode in ("uniform", "none", "1", "unity"):
        return torch.ones((d,), device=device, dtype=torch.float32)

    w = None
    if mode in ("decoder_conv_in_l2", "decoder", "conv_in", "convin"):
        try:
            conv = getattr(getattr(vae, "decoder", None), "conv_in", None)
            W = getattr(conv, "weight", None)
            if W is None and hasattr(conv, "conv"):
                W = getattr(conv.conv, "weight", None)
            if W is None:
                raise AttributeError("conv_in.weight not found")
            W = W.detach().float()
            if W.ndim == 4:
                w = (W ** 2).sum(dim=(0, 2, 3))
            elif W.ndim == 5:
                w = (W ** 2).sum(dim=(0, 2, 3, 4))
            else:
                raise ValueError(f"unexpected conv weight ndim={W.ndim}")
            w = w[:d].contiguous()
        except Exception:
            w = None
    if w is None:
        w = torch.ones((d,), device=device, dtype=torch.float32)
    else:
        w = w.to(device=device, dtype=torch.float32)
        w = w / (w.mean() + _EPS)
    return w


def select_positions_by_rdproxy(
    prob0_BLd: torch.Tensor,
    Hs: int,
    Ws: int,
    d_eff: int,
    d_total: int,
    fixed_hw: Tuple[int, int],
    vae,
    device: str,
    mask_params: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    keep_ratio = float(mask_params.get("keep_ratio", 0.25))
    score_thr = mask_params.get("score_thr", None)
    score_type = str(mask_params.get("score_type", "rd_ratio") or "rd_ratio").lower()
    if score_type in ("rdproxy", "rd", "rdproxy_ratio"):
        score_type = "rd_ratio"
    lambda_rd = float(mask_params.get("lambda_rd", 0.0))
    chan_weight_mode = str(mask_params.get("chan_weight", "uniform") or "uniform")
    scale_weight_mode = str(mask_params.get("scale_weight", "area") or "area").lower()
    min_keep = int(mask_params.get("min_keep_per_scale", 1))

    p = prob0_BLd[:, :, :d_eff]
    Hb = binary_entropy(p)
    Hpos = Hb.mean(dim=2)[0].contiguous()
    Rpos = Hb.sum(dim=2)[0].contiguous()
    q = torch.minimum(p, 1.0 - p)

    w_chan = get_latent_channel_weights(vae, d_total, device=device, mode=chan_weight_mode)[:d_eff]
    qpos_w = (q * w_chan.view(1, 1, d_eff)).sum(dim=2)[0].contiguous()

    Hf, Wf = int(fixed_hw[0]), int(fixed_hw[1])
    if scale_weight_mode in ("none", "1", "unity"):
        w = 1.0
    else:
        w_area = float(Hf * Wf) / max(1.0, float(Hs * Ws))
        if scale_weight_mode in ("sqrt", "sqrt_area", "sqrtarea"):
            w = math.sqrt(w_area)
        else:
            w = w_area
    bit_delta2 = 4.0 / max(1.0, float(d_total))
    Dpos = (qpos_w * float(w) * float(bit_delta2)).contiguous()

    if score_type == "entropy":
        Spos = Hpos
    elif score_type == "rd_lagrange":
        Spos = Dpos - float(lambda_rd) * Rpos
    else:
        Spos = Dpos / (Rpos + _EPS)

    if score_thr is not None:
        kept = torch.nonzero(Spos >= float(score_thr), as_tuple=False).view(-1)
        if kept.numel() < max(1, min_keep):
            kept = _topk_indices(Spos, max(1, min_keep))
        kept_pos = kept.to(torch.long)
    else:
        r = max(0.0, min(1.0, float(keep_ratio)))
        k = int(round(int(Hs * Ws) * r))
        k = max(max(1, min_keep), min(k, int(Hs * Ws)))
        kept_pos = _topk_indices(Spos, k).to(torch.long)

    diag = {
        "score_type": score_type,
        "lambda_rd": float(lambda_rd),
        "chan_weight": chan_weight_mode,
        "scale_weight": scale_weight_mode,
        "scale_w": float(w),
        "keep_ratio": float(keep_ratio),
        "score_thr": "" if score_thr is None else float(score_thr),
        "min_keep_per_scale": int(min_keep),
        "L": int(Hs * Ws),
        "kept": int(kept_pos.numel()),
        "score_keep_mean": float(Spos.index_select(0, kept_pos).mean().item()) if kept_pos.numel() > 0 else float("nan"),
        "score_all_mean": float(Spos.mean().item()),
    }
    return kept_pos, diag


def _compute_texture_scores_stepvar(
    prob0_BLd: torch.Tensor,
    Hs: int,
    Ws: int,
    d_eff: int,
    mask_params: Dict[str, Any],
) -> torch.Tensor:
    p = prob0_BLd[:, :, :d_eff]
    entropy_map = binary_entropy(p).mean(dim=2).reshape(1, 1, Hs, Ws)

    kernel_name = str(mask_params.get("texture_kernel", "laplacian") or "laplacian").lower()
    if kernel_name in ("lap4", "laplacian4", "cross"):
        kernel = torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
            device=entropy_map.device,
            dtype=entropy_map.dtype,
        )
    else:
        kernel = torch.tensor(
            [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]],
            device=entropy_map.device,
            dtype=entropy_map.dtype,
        )
    response = F.conv2d(entropy_map, kernel.view(1, 1, 3, 3), padding=1)
    return response.abs().reshape(-1)


def _compute_structure_scores_stepvar(
    prob0_BLd: torch.Tensor,
    d_eff: int,
    mask_params: Dict[str, Any],
) -> torch.Tensor:
    X = prob0_BLd[0, :, :d_eff].float().contiguous()
    L = int(X.shape[0])
    if L == 0:
        return X.new_zeros((0,))
    if L == 1 or d_eff <= 1:
        return X.new_ones((L,))

    Xc = X - X.mean(dim=0, keepdim=True)
    cov = Xc.transpose(0, 1) @ Xc
    cov = cov / max(1, L - 1)

    _, eigvecs = torch.linalg.eigh(cov)
    pca_rank = int(mask_params.get("pca_rank", 4))
    pca_rank = max(1, min(pca_rank, int(eigvecs.shape[1])))
    basis = eigvecs[:, -pca_rank:]
    proj = Xc @ basis
    return proj.pow(2).sum(dim=1).sqrt()


def select_positions_by_stepvar(
    prob0_BLd: torch.Tensor,
    Hs: int,
    Ws: int,
    d_eff: int,
    mask_params: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    # StepVAR-style approximation for this codec:
    # combine a high-pass texture response with PCA-based structural energy,
    # then keep the top-scoring spatial positions.
    keep_ratio = float(mask_params.get("keep_ratio", 0.25))
    score_thr = mask_params.get("score_thr", None)
    min_keep = int(mask_params.get("min_keep_per_scale", 1))
    texture_weight = float(mask_params.get("texture_weight", 0.5))
    structure_weight = float(mask_params.get("structure_weight", 0.5))

    texture_scores = _compute_texture_scores_stepvar(prob0_BLd, Hs, Ws, d_eff=d_eff, mask_params=mask_params)
    structure_scores = _compute_structure_scores_stepvar(prob0_BLd, d_eff=d_eff, mask_params=mask_params)

    texture_norm = _normalize_score_1d(texture_scores)
    structure_norm = _normalize_score_1d(structure_scores)

    weight_sum = max(float(texture_weight + structure_weight), _EPS)
    combined_scores = (texture_weight * texture_norm + structure_weight * structure_norm) / weight_sum

    if score_thr is not None:
        kept = torch.nonzero(combined_scores >= float(score_thr), as_tuple=False).view(-1)
        if kept.numel() < max(1, min_keep):
            kept = _topk_indices(combined_scores, max(1, min_keep))
        kept_pos = kept.to(torch.long)
    else:
        keep_ratio = max(0.0, min(1.0, float(keep_ratio)))
        k = int(round(int(Hs * Ws) * keep_ratio))
        k = max(max(1, min_keep), min(k, int(Hs * Ws)))
        kept_pos = _topk_indices(combined_scores, k).to(torch.long)

    diag = {
        "score_type": "stepvar_structure_texture",
        "texture_weight": float(texture_weight),
        "structure_weight": float(structure_weight),
        "texture_kernel": str(mask_params.get("texture_kernel", "laplacian")),
        "pca_rank": int(mask_params.get("pca_rank", 4)),
        "keep_ratio": float(keep_ratio),
        "score_thr": "" if score_thr is None else float(score_thr),
        "min_keep_per_scale": int(min_keep),
        "L": int(Hs * Ws),
        "kept": int(kept_pos.numel()),
        "score_keep_mean": float(combined_scores.index_select(0, kept_pos).mean().item()) if kept_pos.numel() > 0 else float("nan"),
        "score_all_mean": float(combined_scores.mean().item()) if combined_scores.numel() > 0 else float("nan"),
        "texture_all_mean": float(texture_scores.mean().item()) if texture_scores.numel() > 0 else float("nan"),
        "structure_all_mean": float(structure_scores.mean().item()) if structure_scores.numel() > 0 else float("nan"),
    }
    return kept_pos, diag


def resolve_active_bits(active_bits_spec: Any, d_total: int, num_scales: int) -> List[int]:
    spec = str(active_bits_spec if active_bits_spec is not None else "all").strip().lower()
    if spec in ("all", "full", "d", "default"):
        return [int(d_total)] * int(num_scales)
    try:
        v = int(spec)
        v = max(1, min(v, int(d_total)))
        return [int(v)] * int(num_scales)
    except Exception:
        return [int(d_total)] * int(num_scales)


def build_codec_config(args, scale_schedule, d_total: int, num_scales: int, packet_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    packet_meta = packet_meta or {}
    mask_strategy = str(packet_meta.get("mask_strategy", getattr(args, "codec_mask_strategy", "none")) or "none").lower()
    mask_params = dict(packet_meta.get("mask_params", {}))
    defaults = {
        "keep_ratio": float(getattr(args, "codec_keep_ratio", 1.0)),
        "entropy_thr": getattr(args, "codec_entropy_thr", None),
        "score_thr": getattr(args, "codec_score_thr", None),
        "score_type": str(getattr(args, "codec_score_type", "rd_ratio")),
        "lambda_rd": float(getattr(args, "codec_lambda_rd", 0.0)),
        "chan_weight": str(getattr(args, "codec_chan_weight", "uniform")),
        "scale_weight": str(getattr(args, "codec_scale_weight", "area")),
        "min_keep_per_scale": int(getattr(args, "codec_min_keep_per_scale", 1)),
        "min_keep_ratio": float(getattr(args, "codec_min_keep_ratio", 0.2)),
        "max_keep_ratio": float(getattr(args, "codec_max_keep_ratio", 1.0)),
        "gamma": float(getattr(args, "codec_gamma", 1.0)),
        "texture_weight": float(getattr(args, "codec_texture_weight", 0.5)),
        "structure_weight": float(getattr(args, "codec_structure_weight", 0.5)),
        "pca_rank": int(getattr(args, "codec_pca_rank", 4)),
        "texture_kernel": str(getattr(args, "codec_texture_kernel", "laplacian")),
    }
    for key, value in defaults.items():
        mask_params.setdefault(key, value)

    k_transmit = int(packet_meta.get("k_transmit", getattr(args, "codec_k_transmit", num_scales)))
    if k_transmit <= 0:
        k_transmit = int(num_scales)
    k_transmit = max(1, min(int(num_scales), int(k_transmit)))

    active_bits_spec = packet_meta.get("active_bits_spec", getattr(args, "codec_active_bits", "all"))
    active_bits = packet_meta.get("active_bits", resolve_active_bits(active_bits_spec, d_total, num_scales))
    flag_bits = int(packet_meta.get("flag_bits", getattr(args, "codec_flag_bits", 1)))
    fixed_hw = tuple(packet_meta.get("fixed_hw", (int(scale_schedule[-1][1]), int(scale_schedule[-1][2]))))
    fill_mode = str(packet_meta.get("fill_mode", getattr(args, "codec_fill_mode", "auto")) or "auto").lower()
    if fill_mode == "auto":
        fill_mode = "prev_nearest" if mask_strategy == "stepvar_spatial" else "map"

    return {
        "mask_strategy": mask_strategy,
        "mask_params": mask_params,
        "k_transmit": k_transmit,
        "active_bits_spec": active_bits_spec,
        "active_bits": [int(x) for x in active_bits],
        "flag_bits": int(flag_bits),
        "fixed_hw": fixed_hw,
        "fill_mode": fill_mode,
    }


def setup_progressive_context(
    infinity,
    vae_scale_schedule,
    prompt,
    text_tokenizer,
    text_encoder,
    cfg_list=1.0,
    cfg_insertion_layer=None,
):
    if cfg_insertion_layer is None:
        cfg_insertion_layer = [0]
    if not isinstance(cfg_list, list):
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
    with torch.amp.autocast("cuda", enabled=False):
        cond_BD_or_gss = infinity.shared_ada_lin(cond_BD.float()).float().contiguous()
    for block in infinity.unregistered_blocks:
        attn_mod = block.sa if hasattr(block, "sa") else block.attn
        attn_mod.kv_caching(True)

    add_cfg_on_logits = False
    for item in cfg_insertion_layer:
        if item == 0:
            add_cfg_on_logits = True

    return {
        "infinity": infinity,
        "scale_schedule": vae_scale_schedule,
        "cfg_list": cfg_list,
        "B": B,
        "bs": bs,
        "cond_BD": cond_BD,
        "cond_BD_or_gss": cond_BD_or_gss,
        "ca_kv": ca_kv,
        "last_stage": last_stage,
        "summed_codes": 0,
        "num_stages_minus_1": len(vae_scale_schedule) - 1,
        "add_cfg_on_logits": add_cfg_on_logits,
    }


def cleanup_progressive_context(ctx: Dict[str, Any]) -> None:
    infinity = ctx["infinity"]
    for block in infinity.unregistered_blocks:
        attn_mod = block.sa if hasattr(block, "sa") else block.attn
        attn_mod.kv_caching(False)


def forward_scale_raw_logits(ctx: Dict[str, Any], si: int) -> torch.Tensor:
    infinity = ctx["infinity"]
    last_stage = ctx["last_stage"]
    cond_BD = ctx["cond_BD"]
    cond_BD_or_gss = ctx["cond_BD_or_gss"]
    ca_kv = ctx["ca_kv"]
    vae_scale_schedule = ctx["scale_schedule"]
    B = ctx["B"]
    cfg = ctx["cfg_list"][si]

    need_to_pad = 0
    attn_fn = None
    if infinity.use_flex_attn:
        attn_fn = infinity.attn_fn_compile_dict.get(tuple(vae_scale_schedule[: (si + 1)]), None)

    for block_idx, block in enumerate(infinity.block_chunks):
        if infinity.add_lvl_embeding_only_first_block and block_idx == 0:
            last_stage = infinity.add_lvl_embeding(last_stage, si, vae_scale_schedule, need_to_pad=need_to_pad)
        if not infinity.add_lvl_embeding_only_first_block:
            last_stage = infinity.add_lvl_embeding(last_stage, si, vae_scale_schedule, need_to_pad=need_to_pad)
        for module in block.module:
            last_stage = module(
                x=last_stage,
                cond_BD=cond_BD_or_gss,
                ca_kv=ca_kv,
                attn_bias_or_two_vector=None,
                attn_fn=attn_fn,
                scale_schedule=vae_scale_schedule,
                rope2d_freqs_grid=infinity.rope2d_freqs_grid,
                scale_ind=si,
            )

    if (cfg != 1) and ctx["add_cfg_on_logits"]:
        logits_all = infinity.get_logits(last_stage, cond_BD)
        logits_cond = logits_all[:B]
        logits_uncond = logits_all[B:]
        raw_logits = cfg * logits_cond + (1 - cfg) * logits_uncond
    else:
        raw_logits = infinity.get_logits(last_stage[:B], cond_BD[:B])
    return raw_logits


def advance_progressive_context(ctx: Dict[str, Any], vae, si: int, idx_Bld: torch.Tensor) -> None:
    scale_schedule = ctx["scale_schedule"]
    pn = scale_schedule[si]
    B = ctx["B"]

    if idx_Bld.ndim == 3:
        idx_Bld = idx_Bld.reshape(B, pn[1], pn[2], -1)
    elif idx_Bld.ndim == 5:
        idx_Bld = idx_Bld[:, 0]
    bits_B1HWd = idx_Bld.unsqueeze(1)
    codes = vae.quantizer.lfq.indices_to_codes(bits_B1HWd, label_type="bit_label")

    if si != ctx["num_stages_minus_1"]:
        ctx["summed_codes"] = ctx["summed_codes"] + F.interpolate(
            codes,
            size=scale_schedule[-1],
            mode=vae.quantizer.z_interplote_up,
        )
        last_stage = F.interpolate(
            ctx["summed_codes"],
            size=scale_schedule[si + 1],
            mode=vae.quantizer.z_interplote_up,
        )
        last_stage = last_stage.squeeze(-3)
        last_stage = last_stage.reshape(*last_stage.shape[:2], -1)
        last_stage = torch.permute(last_stage, [0, 2, 1])
        last_stage = ctx["infinity"].word_embed(ctx["infinity"].norm0_ve(last_stage))
        ctx["last_stage"] = last_stage.repeat(ctx["bs"] // B, 1, 1)
    else:
        ctx["summed_codes"] = ctx["summed_codes"] + codes


def upsample_previous_bits_nearest(
    prev_bits: Optional[torch.Tensor],
    prev_hw: Optional[Tuple[int, int]],
    out_hw: Tuple[int, int],
) -> Optional[torch.Tensor]:
    if prev_bits is None or prev_hw is None:
        return None
    Hp, Wp = int(prev_hw[0]), int(prev_hw[1])
    Hs, Ws = int(out_hw[0]), int(out_hw[1])
    if int(prev_bits.shape[1]) != Hp * Wp:
        return None

    prev_grid = prev_bits.reshape(prev_bits.shape[0], Hp, Wp, prev_bits.shape[-1]).permute(0, 3, 1, 2).float()
    up_grid = F.interpolate(prev_grid, size=(Hs, Ws), mode="nearest")
    return up_grid.permute(0, 2, 3, 1).reshape(prev_bits.shape[0], Hs * Ws, prev_bits.shape[-1]).to(dtype=prev_bits.dtype)


def fill_pruned_positions(
    base_bits: torch.Tensor,
    keep_mask: Optional[torch.Tensor],
    fill_mode: str = "map",
    prev_bits: Optional[torch.Tensor] = None,
    prev_hw: Optional[Tuple[int, int]] = None,
    out_hw: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    if keep_mask is None:
        return base_bits
    fill_mode = str(fill_mode or "map").lower()
    if fill_mode in ("map", "argmax", "none"):
        return base_bits
    if bool(keep_mask.all().item()):
        return base_bits

    if fill_mode in ("nearest", "prev_nearest", "stepvar_nn"):
        propagated = upsample_previous_bits_nearest(prev_bits=prev_bits, prev_hw=prev_hw, out_hw=out_hw or prev_hw or (0, 0))
        if propagated is None:
            return base_bits
        missing_mask = (~keep_mask.reshape(1, -1, 1)).expand_as(base_bits)
        return torch.where(missing_mask, propagated.to(device=base_bits.device, dtype=base_bits.dtype), base_bits)

    return base_bits


def build_scale_keep_plan(
    codec_cfg: Dict[str, Any],
    prob0_BLd: torch.Tensor,
    si: int,
    Hs: int,
    Ws: int,
    d_total: int,
    vae,
    device: str,
    global_kept_pos: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    d_eff = int(codec_cfg["active_bits"][si])
    L = int(Hs * Ws)
    if si >= int(codec_cfg["k_transmit"]):
        keep_mask = torch.zeros((1, Hs, Ws), device=prob0_BLd.device, dtype=torch.bool)
        return {
            "unit_mode": "spatial",
            "kept_pos": torch.zeros((0,), device=prob0_BLd.device, dtype=torch.long),
            "keep_mask": keep_mask,
            "diag": {"kept": 0, "L": L},
        }

    mask_strategy = codec_cfg["mask_strategy"]
    mask_params = codec_cfg["mask_params"]
    if mask_strategy == "none":
        kept_pos = torch.arange(L, device=prob0_BLd.device, dtype=torch.long)
        keep_mask = torch.ones((1, Hs, Ws), device=prob0_BLd.device, dtype=torch.bool)
        return {"unit_mode": "spatial", "kept_pos": kept_pos, "keep_mask": keep_mask, "diag": None}
    if mask_strategy == "entropy_channel":
        kept_channels = select_channels_by_entropy(
            prob0_BLd,
            d_eff=d_eff,
            keep_ratio=mask_params.get("keep_ratio", 0.5),
            entropy_thr=mask_params.get("entropy_thr", None),
        )
        keep_mask = torch.ones((1, Hs, Ws), device=prob0_BLd.device, dtype=torch.bool)
        return {"unit_mode": "channel", "kept_channels": kept_channels, "keep_mask": keep_mask, "diag": None}
    if mask_strategy == "entropy_scale":
        Hb = binary_entropy(prob0_BLd[:, :, :d_eff])
        Hmean = float(Hb.mean().item())
        min_r = float(mask_params.get("min_keep_ratio", 0.2))
        max_r = float(mask_params.get("max_keep_ratio", 1.0))
        gamma = float(mask_params.get("gamma", 1.0))
        keep_ratio = min_r + (max_r - min_r) * (max(0.0, min(1.0, Hmean)) ** gamma)
        kept_channels = select_channels_by_entropy(
            prob0_BLd,
            d_eff=d_eff,
            keep_ratio=keep_ratio,
            entropy_thr=None,
        )
        keep_mask = torch.ones((1, Hs, Ws), device=prob0_BLd.device, dtype=torch.bool)
        return {
            "unit_mode": "channel",
            "kept_channels": kept_channels,
            "keep_mask": keep_mask,
            "diag": {"scale_entropy_mean": Hmean, "keep_ratio": float(keep_ratio)},
        }
    if mask_strategy == "entropy_spatial":
        kept_pos = select_positions_by_entropy(
            prob0_BLd,
            Hs=Hs,
            Ws=Ws,
            d_eff=d_eff,
            keep_ratio=mask_params.get("keep_ratio", 0.25),
            entropy_thr=mask_params.get("entropy_thr", None),
        )
        keep_mask = torch.zeros((1, Hs, Ws), device=prob0_BLd.device, dtype=torch.bool)
        keep_mask.view(-1)[kept_pos] = True
        return {"unit_mode": "spatial", "kept_pos": kept_pos, "keep_mask": keep_mask, "diag": None}
    if mask_strategy == "rdproxy_spatial":
        kept_pos, diag = select_positions_by_rdproxy(
            prob0_BLd,
            Hs=Hs,
            Ws=Ws,
            d_eff=d_eff,
            d_total=d_total,
            fixed_hw=codec_cfg["fixed_hw"],
            vae=vae,
            device=device,
            mask_params=mask_params,
        )
        keep_mask = torch.zeros((1, Hs, Ws), device=prob0_BLd.device, dtype=torch.bool)
        if kept_pos.numel() > 0:
            keep_mask.view(-1)[kept_pos] = True
        return {"unit_mode": "spatial", "kept_pos": kept_pos, "keep_mask": keep_mask, "diag": diag}
    if mask_strategy == "stepvar_spatial":
        kept_pos, diag = select_positions_by_stepvar(
            prob0_BLd,
            Hs=Hs,
            Ws=Ws,
            d_eff=d_eff,
            mask_params=mask_params,
        )
        keep_mask = torch.zeros((1, Hs, Ws), device=prob0_BLd.device, dtype=torch.bool)
        if kept_pos.numel() > 0:
            keep_mask.view(-1)[kept_pos] = True
        return {"unit_mode": "spatial", "kept_pos": kept_pos, "keep_mask": keep_mask, "diag": diag}
    if mask_strategy in ("entropy_spatial_global", "rdproxy_spatial_global"):
        if global_kept_pos is None:
            raise ValueError(f"global_kept_pos is required for strategy={mask_strategy}")
        kept_pos = global_kept_pos.to(prob0_BLd.device).to(torch.long)
        keep_mask = torch.zeros((1, Hs, Ws), device=prob0_BLd.device, dtype=torch.bool)
        if kept_pos.numel() > 0:
            keep_mask.view(-1)[kept_pos] = True
        return {"unit_mode": "spatial", "kept_pos": kept_pos, "keep_mask": keep_mask, "diag": None}
    raise ValueError(f"Unknown codec mask_strategy: {mask_strategy}")


@torch.no_grad()
def precompute_global_spatial_masks(
    codec_cfg: Dict[str, Any],
    infinity,
    vae,
    scale_schedule,
    prompt,
    text_tokenizer,
    text_encoder,
    cfg_list=1.0,
    cfg_insertion_layer=None,
) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
    mask_strategy = codec_cfg["mask_strategy"]
    if mask_strategy not in ("entropy_spatial_global", "rdproxy_spatial_global"):
        raise ValueError(f"Strategy {mask_strategy} does not use global spatial masks")

    if cfg_insertion_layer is None:
        cfg_insertion_layer = [0]
    ctx = setup_progressive_context(
        infinity,
        scale_schedule,
        prompt,
        text_tokenizer,
        text_encoder,
        cfg_list=cfg_list,
        cfg_insertion_layer=cfg_insertion_layer,
    )

    mask_params = codec_cfg["mask_params"]
    score_type = str(mask_params.get("score_type", "entropy" if mask_strategy == "entropy_spatial_global" else "rd_ratio") or "entropy").lower()
    if score_type in ("rdproxy", "rd", "rdproxy_ratio"):
        score_type = "rd_ratio"
    lambda_rd = float(mask_params.get("lambda_rd", 0.0))
    scale_weight_mode = str(mask_params.get("scale_weight", "area") or "area").lower()
    min_keep_per_scale = int(mask_params.get("min_keep_per_scale", 1))
    entropy_thr = mask_params.get("entropy_thr", None)
    score_thr = mask_params.get("score_thr", None)
    keep_ratio = float(mask_params.get("keep_ratio", 0.25))

    Hf, Wf = int(codec_cfg["fixed_hw"][0]), int(codec_cfg["fixed_hw"][1])
    score_list = []
    L_list = []
    HW_list = []
    w_list = []

    try:
        for si in range(int(codec_cfg["k_transmit"])):
            _, Hs, Ws = scale_schedule[si]
            Hs, Ws = int(Hs), int(Ws)
            d_eff = int(codec_cfg["active_bits"][si])
            raw_logits = forward_scale_raw_logits(ctx, si)
            prob0 = logits_to_prob0(raw_logits)
            p_eff = prob0[:, :, :d_eff]
            Hb = binary_entropy(p_eff)
            Hpos = Hb.mean(dim=2)[0].contiguous()
            Rpos = Hb.sum(dim=2)[0].contiguous()

            q = torch.minimum(p_eff, 1.0 - p_eff)
            chan_weight_mode = str(mask_params.get("chan_weight", "uniform") or "uniform")
            w_chan = get_latent_channel_weights(vae, int(prob0.shape[-1]), device=prob0.device, mode=chan_weight_mode)[:d_eff]
            qpos_w = (q * w_chan.view(1, 1, d_eff)).sum(dim=2)[0].contiguous()

            if scale_weight_mode in ("none", "1", "unity"):
                w = 1.0
            else:
                w_area = float(Hf * Wf) / max(1.0, float(Hs * Ws))
                if scale_weight_mode in ("sqrt", "sqrt_area", "sqrtarea"):
                    w = math.sqrt(w_area)
                else:
                    w = w_area

            bit_delta2 = 4.0 / max(1.0, float(prob0.shape[-1]))
            Dpos = (qpos_w * float(w) * float(bit_delta2)).contiguous()
            if score_type == "entropy":
                Spos = Hpos
            elif score_type == "rd_lagrange":
                Spos = Dpos - float(lambda_rd) * Rpos
            else:
                Spos = Dpos / (Rpos + _EPS)

            score_list.append(Spos)
            L_list.append(int(Hs * Ws))
            HW_list.append((Hs, Ws))
            w_list.append(float(w))

            rec_bits = torch.zeros((1, int(Hs * Ws), int(prob0.shape[-1])), device=prob0.device, dtype=torch.uint8)
            rec_bits[:, :, :d_eff] = map_bits_from_prob0(p_eff)
            advance_progressive_context(ctx, vae, si, rec_bits)
    finally:
        cleanup_progressive_context(ctx)

    global_scores = torch.cat(score_list, dim=0)
    kept_global = None
    if score_type == "entropy" and entropy_thr is not None:
        kept_global = torch.nonzero(global_scores >= float(entropy_thr), as_tuple=False).view(-1).to(torch.long)
    elif score_type != "entropy" and score_thr is not None:
        kept_global = torch.nonzero(global_scores >= float(score_thr), as_tuple=False).view(-1).to(torch.long)
    if kept_global is None or kept_global.numel() == 0:
        k_keep = int(round(int(global_scores.numel()) * max(0.0, min(1.0, keep_ratio))))
        k_keep = max(1, min(k_keep, int(global_scores.numel())))
        kept_global = _topk_indices(global_scores, k_keep)

    kept_pos_list = []
    per_total = []
    per_keep = []
    offset = 0
    for si, L in enumerate(L_list):
        sel = kept_global[(kept_global >= offset) & (kept_global < offset + L)] - offset
        sel = sel.to(torch.long)
        if min_keep_per_scale > 0 and int(sel.numel()) < min_keep_per_scale:
            extra = _topk_indices(score_list[si], min_keep_per_scale)
            sel = torch.unique(torch.cat([sel, extra.to(torch.long)], dim=0))
            sel = torch.sort(sel)[0]
        kept_pos_list.append(sel)
        per_total.append(int(L))
        per_keep.append(int(sel.numel()))
        offset += L

    stats = {
        "strategy": mask_strategy,
        "score_type": score_type,
        "lambda_rd": float(lambda_rd),
        "scale_weight": scale_weight_mode,
        "per_scale_weight": w_list,
        "keep_ratio": float(keep_ratio),
        "entropy_thr": "" if entropy_thr is None else float(entropy_thr),
        "score_thr": "" if score_thr is None else float(score_thr),
        "min_keep_per_scale": int(min_keep_per_scale),
        "k_transmit": int(codec_cfg["k_transmit"]),
        "per_scale_hw": HW_list,
        "per_scale_total_pos": per_total,
        "per_scale_kept_pos": per_keep,
        "per_scale_drop_pos": [int(t - k) for t, k in zip(per_total, per_keep)],
        "total_pos": int(sum(per_total)),
        "total_kept": int(sum(per_keep)),
        "total_drop": int(sum(per_total) - sum(per_keep)),
        "keep_ratio_effective": float(sum(per_keep) / max(1, sum(per_total))),
    }
    return kept_pos_list, stats
