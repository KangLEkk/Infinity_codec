import math
from typing import Any, Dict, List, Optional, Tuple

import torch

from compression.progressive_masked_codec import (
    FLAG_ARITH,
    FLAG_RAW,
    build_codec_config,
    build_scale_keep_plan,
    cleanup_progressive_context,
    forward_scale_raw_logits,
    fill_pruned_positions,
    logits_to_prob0,
    map_bits_from_prob0,
    precompute_global_spatial_masks,
    setup_progressive_context,
    advance_progressive_context,
)
from utils.arithmeticcoding import decompress_from_bit_list


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
        raise ValueError(f"raw_logits should be [B, seq_len, 2*d], got shape={tuple(raw_logits.shape)}")
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


def sample_with_top_k_top_p_also_inplace_modifying_logits_(
    logits_BlV: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    rng=None,
    num_samples=1,
) -> torch.Tensor:
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


def _sample_bits_from_logits(raw_logits: torch.Tensor, rng=None) -> torch.Tensor:
    B, L, V = raw_logits.shape
    sampled = sample_with_top_k_top_p_also_inplace_modifying_logits_(
        raw_logits.reshape(B, -1, 2),
        rng=rng,
        top_k=0,
        top_p=0.0,
        num_samples=1,
    )[:, :, 0]
    return sampled.reshape(B, L, V // 2).to(torch.uint8)


def _generate_bits_from_logits(
    raw_logits: torch.Tensor,
    si: int,
    gt_leak: int,
    decode_mode: str,
    rng=None,
) -> torch.Tensor:
    decode_mode = str(decode_mode or "adaptive").lower()
    if decode_mode == "map":
        return raw_logits.reshape(raw_logits.shape[0], raw_logits.shape[1], -1, 2).argmax(dim=-1).to(torch.uint8)
    if decode_mode == "sample":
        return _sample_bits_from_logits(raw_logits, rng=rng)

    last_observed_scale_idx = int(gt_leak - 1)
    rel_stage = max(1, int(si - last_observed_scale_idx))
    if rel_stage == 1:
        selective_ratio = 0.50
    elif rel_stage == 2:
        selective_ratio = 0.30
    else:
        selective_ratio = 0.15
    logits_scaled, _ = apply_entropy_adaptive_temperature(
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
    return _sample_bits_from_logits(logits_scaled, rng=rng)


def _unpack_decoded_packet(decoded_packet):
    if isinstance(decoded_packet, dict):
        return (
            decoded_packet.get("decode_idx", []),
            decoded_packet.get("keep_masks"),
            decoded_packet.get("packet_meta"),
        )
    return decoded_packet, None, None


@torch.no_grad()
def decoding(
    args,
    infinity,
    vae,
    vae_scale_schedule,
    prompt,
    text_tokenizer,
    text_encoder,
    gt_ls_Bl,
    trans_list,
    help_list,
    packet_meta: Optional[Dict[str, Any]] = None,
    tau_list=1.0,
    cfg_insertion_layer=None,
    return_meta: bool = True,
):
    del tau_list
    if cfg_insertion_layer is None:
        cfg_insertion_layer = [0]

    num_scales = len(vae_scale_schedule)
    d_total = int(gt_ls_Bl[0].shape[-1]) if gt_ls_Bl is not None and len(gt_ls_Bl) else int(getattr(args, "vae_type", 32))
    codec_cfg = build_codec_config(args, vae_scale_schedule, d_total, num_scales, packet_meta=packet_meta)

    global_kept_pos = None
    global_stats = None
    if codec_cfg["mask_strategy"] in ("entropy_spatial_global", "rdproxy_spatial_global"):
        global_kept_pos, global_stats = precompute_global_spatial_masks(
            codec_cfg,
            infinity,
            vae,
            vae_scale_schedule,
            prompt,
            text_tokenizer,
            text_encoder,
            cfg_list=1.0,
            cfg_insertion_layer=cfg_insertion_layer,
        )

    ctx = setup_progressive_context(
        infinity,
        vae_scale_schedule,
        prompt,
        text_tokenizer,
        text_encoder,
        cfg_list=1.0,
        cfg_insertion_layer=cfg_insertion_layer,
    )

    decode_idx = []
    keep_masks = []
    scale_debug = []
    prev_bits = None
    prev_hw = None

    try:
        for si, pn in enumerate(vae_scale_schedule):
            _, Hs, Ws = pn
            Hs, Ws = int(Hs), int(Ws)
            L = int(Hs * Ws)
            raw_logits = forward_scale_raw_logits(ctx, si)
            prob0 = logits_to_prob0(raw_logits)
            d_eff = int(codec_cfg["active_bits"][si])

            plan = build_scale_keep_plan(
                codec_cfg=codec_cfg,
                prob0_BLd=prob0,
                si=si,
                Hs=Hs,
                Ws=Ws,
                d_total=d_total,
                vae=vae,
                device=str(prob0.device),
                global_kept_pos=(None if global_kept_pos is None or si >= len(global_kept_pos) else global_kept_pos[si]),
            )

            rec_bits = torch.zeros((1, L, d_total), device=prob0.device, dtype=torch.uint8)
            rec_bits[:, :, :d_eff] = map_bits_from_prob0(prob0[:, :, :d_eff])

            payloads = trans_list[si] if si < len(trans_list) else []
            flags = help_list[si] if si < len(help_list) else []
            trans_idx = 0

            if plan["unit_mode"] == "channel":
                kept_channels = [int(x) for x in plan.get("kept_channels", [])]
                kept_tensor = torch.tensor(kept_channels, device=prob0.device, dtype=torch.long)
                if kept_channels:
                    for pos in range(L):
                        flag = flags[trans_idx]
                        prob0_token = prob0[0, pos, kept_tensor].detach().cpu().tolist()
                        if flag == FLAG_RAW:
                            decoded_bits = payloads[trans_idx]
                        elif flag == FLAG_ARITH:
                            decoded_bits = decompress_from_bit_list(payloads[trans_idx], len(kept_channels), prob0_token)
                        else:
                            raise ValueError(f"Unknown help flag: {flag}")
                        rec_bits[0, pos, kept_tensor] = torch.tensor(decoded_bits, device=prob0.device, dtype=torch.uint8)
                        trans_idx += 1
            else:
                kept_pos = plan.get("kept_pos", torch.zeros((0,), device=prob0.device, dtype=torch.long))
                for pos in kept_pos.tolist():
                    flag = flags[trans_idx]
                    prob0_token = prob0[0, pos, :d_eff].detach().cpu().tolist()
                    if flag == FLAG_RAW:
                        decoded_bits = payloads[trans_idx]
                    elif flag == FLAG_ARITH:
                        decoded_bits = decompress_from_bit_list(payloads[trans_idx], d_eff, prob0_token)
                    else:
                        raise ValueError(f"Unknown help flag: {flag}")
                    rec_bits[0, pos, :d_eff] = torch.tensor(decoded_bits, device=prob0.device, dtype=torch.uint8)
                    trans_idx += 1

            rec_bits = fill_pruned_positions(
                rec_bits,
                keep_mask=plan["keep_mask"],
                fill_mode=codec_cfg["fill_mode"],
                prev_bits=prev_bits,
                prev_hw=prev_hw,
                out_hw=(Hs, Ws),
            )
            advance_progressive_context(ctx, vae, si, rec_bits)
            prev_bits = rec_bits.detach()
            prev_hw = (Hs, Ws)
            decode_idx.append(rec_bits.to(dtype=torch.int32))
            keep_masks.append(plan["keep_mask"].detach().cpu())
            scale_debug.append(
                {
                    "scale_idx": int(si),
                    "hw": [Hs, Ws],
                    "d_eff": int(d_eff),
                    "unit_mode": str(plan["unit_mode"]),
                    "keep_ratio": float(plan["keep_mask"].float().mean().item()),
                    "fill_mode": str(codec_cfg["fill_mode"]),
                    "diag": plan.get("diag"),
                }
            )
    finally:
        cleanup_progressive_context(ctx)

    merged_packet_meta = dict(packet_meta or {})
    merged_packet_meta.setdefault("mask_strategy", codec_cfg["mask_strategy"])
    merged_packet_meta.setdefault("mask_params", codec_cfg["mask_params"])
    merged_packet_meta.setdefault("active_bits", codec_cfg["active_bits"])
    merged_packet_meta.setdefault("k_transmit", codec_cfg["k_transmit"])
    merged_packet_meta.setdefault("fill_mode", codec_cfg["fill_mode"])
    if global_stats is not None:
        merged_packet_meta["global_stats"] = global_stats

    result = {
        "decode_idx": decode_idx,
        "keep_masks": keep_masks,
        "packet_meta": merged_packet_meta,
        "scale_debug": scale_debug,
    }
    return result if return_meta else decode_idx


@torch.no_grad()
def decompress_cfg(
    infinity,
    vae,
    vae_scale_schedule,
    prompt,
    text_tokenizer,
    text_encoder,
    gt_leak,
    gt_ls_Bl,
    cfg_list=3,
    tau_list=0.5,
    cfg_insertion_layer=None,
    decode_mode: str = "adaptive",
):
    if cfg_insertion_layer is None:
        cfg_insertion_layer = [0]
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(vae_scale_schedule)
    if not isinstance(cfg_list, list):
        cfg_list = [cfg_list] * len(vae_scale_schedule)

    decoded_idx, keep_masks, packet_meta = _unpack_decoded_packet(gt_ls_Bl)
    packet_fill_mode = str((packet_meta or {}).get("fill_mode", "map"))
    ctx = setup_progressive_context(
        infinity,
        vae_scale_schedule,
        prompt,
        text_tokenizer,
        text_encoder,
        cfg_list=cfg_list,
        cfg_insertion_layer=cfg_insertion_layer,
    )
    rng = infinity.rng
    prev_bits = None
    prev_hw = None

    try:
        for si, pn in enumerate(vae_scale_schedule):
            _, Hs, Ws = pn
            Hs, Ws = int(Hs), int(Ws)
            L = int(Hs * Ws)
            raw_logits = forward_scale_raw_logits(ctx, si)
            d_total = raw_logits.shape[-1] // 2

            if si <= int(gt_leak) and si < len(decoded_idx):
                forced_bits = decoded_idx[si].to(device=raw_logits.device, dtype=torch.uint8).reshape(1, L, d_total)
                keep_mask = None
                if keep_masks is not None and si < len(keep_masks):
                    keep_mask = keep_masks[si].to(device=raw_logits.device, dtype=torch.bool)
                if keep_mask is None or bool(keep_mask.all().item()):
                    idx_flat = forced_bits
                else:
                    base_bits = _generate_bits_from_logits(
                        raw_logits,
                        si=si,
                        gt_leak=gt_leak,
                        decode_mode="map",
                        rng=rng,
                    )
                    mask_flat = keep_mask.reshape(1, L, 1).expand_as(base_bits)
                    idx_flat = torch.where(mask_flat, forced_bits, base_bits)
                    idx_flat = fill_pruned_positions(
                        idx_flat,
                        keep_mask=keep_mask,
                        fill_mode=packet_fill_mode,
                        prev_bits=prev_bits,
                        prev_hw=prev_hw,
                        out_hw=(Hs, Ws),
                    )
            else:
                idx_flat = _generate_bits_from_logits(
                    raw_logits,
                    si=si,
                    gt_leak=gt_leak,
                    decode_mode=decode_mode,
                    rng=rng,
                )

            advance_progressive_context(ctx, vae, si, idx_flat)
            prev_bits = idx_flat.detach()
            prev_hw = (Hs, Ws)
    finally:
        cleanup_progressive_context(ctx)

    img = vae.decode(ctx["summed_codes"].squeeze(-3))
    img = (img + 1) / 2
    return img
