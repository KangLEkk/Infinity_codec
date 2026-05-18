from typing import Any, Dict, List, Optional, Tuple

import torch

from compression.progressive_masked_codec import (
    FLAG_ARITH,
    FLAG_RAW,
    fill_pruned_positions,
    build_codec_config,
    build_scale_keep_plan,
    cleanup_progressive_context,
    forward_scale_raw_logits,
    logits_to_prob0,
    map_bits_from_prob0,
    precompute_global_spatial_masks,
    setup_progressive_context,
    advance_progressive_context,
)
from utils.arithmeticcoding import compress_to_bit_list


@torch.no_grad()
def get_prob(
    infinity,
    vae,
    vae_scale_schedule,
    prompt,
    text_tokenizer,
    text_encoder,
    gt_ls_Bl,
    tau_list=0.5,
    cfg_insertion_layer=None,
):
    del tau_list
    if cfg_insertion_layer is None:
        cfg_insertion_layer = [0]

    ctx = setup_progressive_context(
        infinity,
        vae_scale_schedule,
        prompt,
        text_tokenizer,
        text_encoder,
        cfg_list=1.0,
        cfg_insertion_layer=cfg_insertion_layer,
    )
    prob_list = []
    try:
        for si in range(len(vae_scale_schedule)):
            raw_logits = forward_scale_raw_logits(ctx, si)
            prob0 = logits_to_prob0(raw_logits)
            prob = torch.stack((prob0, 1.0 - prob0), dim=-1).reshape(-1, 2)
            prob_list.append(prob)
            advance_progressive_context(ctx, vae, si, gt_ls_Bl[si].to(device=raw_logits.device, dtype=torch.uint8))
    finally:
        cleanup_progressive_context(ctx)
    return prob_list


def _encode_token(gt_token: List[int], prob0_token: List[float]) -> Tuple[List[int], int, int]:
    arith_bits = compress_to_bit_list(gt_token, prob0_token)
    raw_bits = list(gt_token)
    if len(arith_bits) < len(raw_bits):
        return arith_bits, FLAG_ARITH, len(arith_bits)
    return raw_bits, FLAG_RAW, len(raw_bits)


@torch.no_grad()
def encoding(
    args,
    infinity,
    vae,
    scale_schedule,
    text,
    text_tokenizer,
    text_encoder,
    gt_ms_idx_Bl,
    packet_meta: Optional[Dict[str, Any]] = None,
    precomputed_prob_list=None,
):
    del precomputed_prob_list

    num_scales = len(gt_ms_idx_Bl)
    d_total = int(gt_ms_idx_Bl[0].shape[-1])
    codec_cfg = build_codec_config(args, scale_schedule, d_total, num_scales, packet_meta=packet_meta)

    global_kept_pos = None
    global_stats = None
    if codec_cfg["mask_strategy"] in ("entropy_spatial_global", "rdproxy_spatial_global"):
        global_kept_pos, global_stats = precompute_global_spatial_masks(
            codec_cfg,
            infinity,
            vae,
            scale_schedule,
            text,
            text_tokenizer,
            text_encoder,
            cfg_list=1.0,
            cfg_insertion_layer=[getattr(args, "cfg_insertion_layer", 0)],
        )

    ctx = setup_progressive_context(
        infinity,
        scale_schedule,
        text,
        text_tokenizer,
        text_encoder,
        cfg_list=1.0,
        cfg_insertion_layer=[getattr(args, "cfg_insertion_layer", 0)],
    )

    trans_list = []
    help_list = []
    cumulative_bits = 0
    cumulative_total_bits = []
    scale_stats = []
    prev_bits = None
    prev_hw = None

    try:
        for si, pn in enumerate(scale_schedule):
            _, Hs, Ws = pn
            Hs, Ws = int(Hs), int(Ws)
            L = int(Hs * Ws)
            raw_logits = forward_scale_raw_logits(ctx, si)
            prob0 = logits_to_prob0(raw_logits)
            d_eff = int(codec_cfg["active_bits"][si])
            gt_flat = gt_ms_idx_Bl[si].reshape(1, L, d_total).to(device=prob0.device, dtype=torch.uint8)
            rec_bits = torch.zeros((1, L, d_total), device=prob0.device, dtype=torch.uint8)
            rec_bits[:, :, :d_eff] = map_bits_from_prob0(prob0[:, :, :d_eff])

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

            payloads = []
            flags = []
            payload_bits_this_scale = 0

            if plan["unit_mode"] == "channel":
                kept_channels = [int(x) for x in plan.get("kept_channels", [])]
                kept_tensor = torch.tensor(kept_channels, device=prob0.device, dtype=torch.long)
                if kept_channels:
                    for pos in range(L):
                        gt_token = gt_flat[0, pos, kept_tensor].tolist()
                        prob0_token = prob0[0, pos, kept_tensor].detach().cpu().tolist()
                        payload, flag, bit_len = _encode_token(gt_token, prob0_token)
                        payloads.append(payload)
                        flags.append(flag)
                        payload_bits_this_scale += int(bit_len)
                    rec_bits[:, :, kept_tensor] = gt_flat[:, :, kept_tensor]
            else:
                kept_pos = plan.get("kept_pos", torch.zeros((0,), device=prob0.device, dtype=torch.long))
                if kept_pos.numel() > 0:
                    for pos in kept_pos.tolist():
                        gt_token = gt_flat[0, pos, :d_eff].tolist()
                        prob0_token = prob0[0, pos, :d_eff].detach().cpu().tolist()
                        payload, flag, bit_len = _encode_token(gt_token, prob0_token)
                        payloads.append(payload)
                        flags.append(flag)
                        payload_bits_this_scale += int(bit_len)
                    rec_bits.index_copy_(1, kept_pos, gt_flat[:, kept_pos, :])

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

            side_bits_this_scale = len(flags) * int(codec_cfg["flag_bits"])
            total_bits_this_scale = int(payload_bits_this_scale + side_bits_this_scale)
            cumulative_bits += total_bits_this_scale
            cumulative_total_bits.append(int(cumulative_bits))

            trans_list.append(payloads)
            help_list.append(flags)
            scale_stats.append(
                {
                    "scale_idx": int(si),
                    "hw": [Hs, Ws],
                    "d_eff": int(d_eff),
                    "unit_mode": str(plan["unit_mode"]),
                    "num_payload_units": int(len(payloads)),
                    "payload_bits": int(payload_bits_this_scale),
                    "side_bits": int(side_bits_this_scale),
                    "total_bits": int(total_bits_this_scale),
                    "keep_ratio_in_scale": float(plan["keep_mask"].float().mean().item()),
                    "fill_mode": str(codec_cfg["fill_mode"]),
                    "diag": plan.get("diag"),
                }
            )
    finally:
        cleanup_progressive_context(ctx)

    packet_meta = {
        "mask_strategy": codec_cfg["mask_strategy"],
        "mask_params": codec_cfg["mask_params"],
        "k_transmit": int(codec_cfg["k_transmit"]),
        "active_bits_spec": codec_cfg["active_bits_spec"],
        "active_bits": codec_cfg["active_bits"],
        "flag_bits": int(codec_cfg["flag_bits"]),
        "fixed_hw": list(codec_cfg["fixed_hw"]),
        "fill_mode": str(codec_cfg["fill_mode"]),
        "scale_stats": scale_stats,
        "cumulative_total_bits": cumulative_total_bits,
        "bpp_list": cumulative_total_bits,
    }
    if global_stats is not None:
        packet_meta["global_stats"] = global_stats

    return trans_list, help_list, packet_meta
