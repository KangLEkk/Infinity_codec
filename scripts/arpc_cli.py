"""ARPC CLI

Paper-aligned command line interface for the ARPC (Autoregressive-based Progressive
Coding) codec built on Infinity.

Subcommands
-----------
- compress   : image -> .arpc bitstream
- decompress : .arpc bitstream -> reconstructed image

Features
--------
- Progressive transmission (first k scales are entropy-coded)
- Optional **no-training entropy masking** for stage-1 (reduce handcrafted GM-BMSRQ)

Examples
--------
Compress:

  python scripts/arpc_cli.py compress \
    --image input.png \
    --out out.arpc \
    --prompt "a clear photo" \
    --k_transmit 5 \
    --pn 1M \
    --vae_ckpt /path/to/vae.ckpt \
    --model_ckpt /path/to/infinity_ar.ckpt \
    --mask_strategy entropy_channel --keep_ratio 0.5

Decompress:

  python scripts/arpc_cli.py decompress \
    --stream out.arpc \
    --out recon.png \
    --vae_ckpt /path/to/vae.ckpt \
    --model_ckpt /path/to/infinity_ar.ckpt
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np
import torch
from PIL import Image

from codec.arpc_codec import ARPCCodec
from infinity.models.bsq_vae.vae import vae_model
from infinity.models.infinity_patched import Infinity as InfinityPatched
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w


# -----------------------------
# Image helpers (no torchvision)
# -----------------------------

def _to_tensor_minus1_1(pil: Image.Image) -> torch.Tensor:
    """PIL RGB -> float tensor in [-1,1], shape [1,3,H,W]."""
    arr = np.asarray(pil).astype(np.float32) / 255.0
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("expected RGB image")
    arr = arr.transpose(2, 0, 1)  # 3,H,W
    ten = torch.from_numpy(arr)
    ten = ten.mul_(2.0).sub_(1.0)
    return ten.unsqueeze(0)


def _resize_center_crop(pil: Image.Image, tgt_h: int, tgt_w: int) -> Image.Image:
    """Resize (keep aspect) then center-crop to target size."""
    w, h = pil.size
    if w / h <= tgt_w / tgt_h:
        new_w = tgt_w
        new_h = int(round(tgt_w / (w / h)))
    else:
        new_h = tgt_h
        new_w = int(round((w / h) * tgt_h))
    pil = pil.resize((new_w, new_h), resample=Image.LANCZOS)
    arr = np.asarray(pil)
    y0 = (arr.shape[0] - tgt_h) // 2
    x0 = (arr.shape[1] - tgt_w) // 2
    arr = arr[y0 : y0 + tgt_h, x0 : x0 + tgt_w]
    return Image.fromarray(arr)


# -----------------------------
# Model / schedule
# -----------------------------

def _model_kwargs_from_type(model_type: str) -> Dict:
    if model_type == "infinity_2b":
        return dict(depth=32, embed_dim=2048, num_heads=2048 // 128, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8)
    if model_type == "infinity_8b":
        return dict(depth=40, embed_dim=3584, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8)
    if model_type == "infinity_layer12":
        return dict(depth=12, embed_dim=768, num_heads=8, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    if model_type == "infinity_layer16":
        return dict(depth=16, embed_dim=1152, num_heads=12, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    if model_type == "infinity_layer24":
        return dict(depth=24, embed_dim=1536, num_heads=16, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    if model_type == "infinity_layer32":
        return dict(depth=32, embed_dim=2080, num_heads=20, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    if model_type == "infinity_layer40":
        return dict(depth=40, embed_dim=2688, num_heads=24, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    if model_type == "infinity_layer48":
        return dict(depth=48, embed_dim=3360, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    raise ValueError(f"unknown model_type: {model_type}")


@torch.no_grad()
def load_text_encoder(text_encoder_ckpt: str, device: torch.device):
    from transformers import AutoTokenizer, T5EncoderModel

    tok = AutoTokenizer.from_pretrained(text_encoder_ckpt)
    enc = T5EncoderModel.from_pretrained(text_encoder_ckpt)
    enc.eval().to(device)
    return tok, enc


@torch.no_grad()
def load_vae(vae_ckpt: str, vae_type: int, apply_spatial_patchify: int, device: torch.device):
    schedule_mode = "dynamic"
    codebook_dim = int(vae_type)
    codebook_size = 2 ** codebook_dim
    if apply_spatial_patchify:
        patch_size = 8
        encoder_ch_mult = [1, 2, 4, 4]
        decoder_ch_mult = [1, 2, 4, 4]
    else:
        patch_size = 16
        encoder_ch_mult = [1, 2, 4, 4, 4]
        decoder_ch_mult = [1, 2, 4, 4, 4]

    vae = vae_model(
        vqgan_ckpt=vae_ckpt,
        schedule_mode=schedule_mode,
        codebook_dim=codebook_dim,
        codebook_size=codebook_size,
        test_mode=True,
        patch_size=patch_size,
        encoder_ch_mult=encoder_ch_mult,
        decoder_ch_mult=decoder_ch_mult,
    ).to(device)
    vae.eval()
    return vae


@torch.no_grad()
def load_infinity(model_ckpt: str, vae, args, device: torch.device):
    text_maxlen = int(args.tlen)
    model_kwargs = _model_kwargs_from_type(args.model_type)

    with torch.cuda.amp.autocast(enabled=bool(args.bf16), dtype=torch.bfloat16, cache_enabled=True):
        gpt = InfinityPatched(
            vae_local=vae,
            text_channels=int(args.text_channels),
            text_maxlen=text_maxlen,
            shared_aln=True,
            raw_scale_schedule=None,
            checkpointing="full-block",
            customized_flash_attn=False,
            fused_norm=True,
            pad_to_multiplier=128,
            use_flex_attn=bool(args.use_flex_attn),
            add_lvl_embeding_only_first_block=int(args.add_lvl_embeding_only_first_block),
            use_bit_label=int(args.use_bit_label),
            rope2d_each_sa_layer=int(args.rope2d_each_sa_layer),
            rope2d_normalized_by_hw=int(args.rope2d_normalized_by_hw),
            pn=str(args.pn),
            apply_spatial_patchify=int(args.apply_spatial_patchify),
            inference_mode=True,
            train_h_div_w_list=[float(args.h_div_w_template)],
            **model_kwargs,
        ).to(device)

    gpt.eval().requires_grad_(False)

    ckpt = torch.load(model_ckpt, map_location=device)
    state = ckpt
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "gpt", "ema"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break

    missing, unexpected = gpt.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] missing keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] unexpected keys: {len(unexpected)}")
    return gpt


def _get_target_hw(pn: str, ratio: float) -> tuple[int, int]:
    ratio = float(ratio)
    if ratio not in dynamic_resolution_h_w:
        ratio = 1.0
    if pn not in dynamic_resolution_h_w[ratio]:
        raise ValueError(f"pn={pn} not supported in dynamic_resolution_h_w")
    h, w = dynamic_resolution_h_w[ratio][pn]["pixel"]
    return int(h), int(w)


def build_codec(args, device: torch.device) -> ARPCCodec:
    text_tokenizer, text_encoder = load_text_encoder(args.text_encoder_ckpt, device)
    vae = load_vae(args.vae_ckpt, args.vae_type, args.apply_spatial_patchify, device)
    # vae.quantizer.schedule_mode = "dynamic"

    model = load_infinity(args.model_ckpt, vae, args, device)

    codec = ARPCCodec(
        vae=vae,
        model=model,
        text_tokenizer=text_tokenizer,
        text_encoder=text_encoder,
        tlen=int(args.tlen),
        device=str(device),
    )
    return codec


# -----------------------------
# Commands
# -----------------------------

def cmd_compress(args):
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "cpu" else "cuda"))
    codec = build_codec(args, device)

    pil = Image.open(args.image).convert("RGB")
    tgt_h, tgt_w = _get_target_hw(args.pn, args.h_div_w_template)
    pil = _resize_center_crop(pil, tgt_h, tgt_w)
    img = _to_tensor_minus1_1(pil).to(device)

    # build entropy-mask params
    mask_strategy = str(args.mask_strategy).lower()
    mask_params: Dict = {}
    if mask_strategy in ("entropy_channel", "entropy_spatial"):
        if args.entropy_thr is not None:
            mask_params["entropy_thr"] = float(args.entropy_thr)
        else:
            mask_params["keep_ratio"] = float(args.keep_ratio)
    elif mask_strategy == "entropy_scale":
        mask_params["min_keep_ratio"] = float(args.min_keep_ratio)
        mask_params["max_keep_ratio"] = float(args.max_keep_ratio)
        mask_params["gamma"] = float(args.gamma)

    codec.compress(
        img_B3HW=img,
        prompt=args.prompt,
        out_path=args.out,
        k_transmit=int(args.k_transmit),
        active_bits_spec=str(args.active_bits),
        mask_strategy=mask_strategy,
        mask_params=mask_params,
        seed=int(args.seed),
    )

    try:
        nbytes = os.path.getsize(args.out)
        bpp = (nbytes * 8.0) / (tgt_h * tgt_w)
        print(f"[ARPC] wrote {nbytes} bytes -> ~{bpp:.6f} bpp @ {tgt_h}x{tgt_w}")
    except Exception:
        pass


def cmd_decompress(args):
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "cpu" else "cuda"))
    codec = build_codec(args, device)

    codec.decompress(
        stream_path=args.stream,
        out_path=args.out,
        prompt=args.prompt,
        seed=int(args.seed),
    )
    print(f"[ARPC] reconstructed -> {args.out}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--prompt", type=str, default="a clear photo")

    # schedule / model
    common.add_argument("--pn", type=str, default="1M", choices=["0.06M", "0.25M", "0.60M", "1M"])
    common.add_argument("--h_div_w_template", type=float, default=1.0)
    common.add_argument("--vae_type", type=int, default=32)
    common.add_argument("--apply_spatial_patchify", type=int, default=0, choices=[0, 1])

    common.add_argument("--vae_ckpt", type=str, required=True)
    common.add_argument("--model_ckpt", type=str, required=True)
    common.add_argument("--text_encoder_ckpt", type=str, default="google/flan-t5-xl")

    common.add_argument("--tlen", type=int, default=512)
    common.add_argument("--text_channels", type=int, default=2048)
    common.add_argument("--model_type", type=str, default="infinity_2b")
    common.add_argument("--use_bit_label", type=int, default=1, choices=[0, 1])
    common.add_argument("--use_flex_attn", type=int, default=0, choices=[0, 1])
    common.add_argument("--bf16", type=int, default=1, choices=[0, 1])
    common.add_argument("--add_lvl_embeding_only_first_block", type=int, default=0, choices=[0, 1])
    common.add_argument("--rope2d_each_sa_layer", type=int, default=1, choices=[0, 1])
    common.add_argument("--rope2d_normalized_by_hw", type=int, default=2, choices=[0, 1, 2])

    # compress
    pc = sub.add_parser("compress", parents=[common])
    pc.add_argument("--image", type=str, required=True)
    pc.add_argument("--out", type=str, required=True)
    pc.add_argument("--k_transmit", type=int, default=5)
    pc.add_argument(
        "--active_bits",
        type=str,
        default="default",
        help=(
            "Active bits per scale. Examples: 'default' or '16,16,16,...' or '8x3,12x5,16x5' (run-length)."
        ),
    )

    # entropy mask
    pc.add_argument(
        "--mask_strategy",
        type=str,
        default="none",
        choices=["none", "entropy_channel", "entropy_scale", "entropy_spatial"],
        help="Optional no-training entropy masking strategy.",
    )
    pc.add_argument("--keep_ratio", type=float, default=0.5, help="Keep ratio for entropy_channel / entropy_spatial (ignored if --entropy_thr is set).")
    pc.add_argument("--entropy_thr", type=float, default=None, help="Keep everything with entropy >= thr (bits). Overrides keep_ratio.")

    pc.add_argument("--min_keep_ratio", type=float, default=0.2, help="entropy_scale: minimum keep ratio.")
    pc.add_argument("--max_keep_ratio", type=float, default=1.0, help="entropy_scale: maximum keep ratio.")
    pc.add_argument("--gamma", type=float, default=1.0, help="entropy_scale: curve power for keep ratio.")

    # decompress
    pd = sub.add_parser("decompress", parents=[common])
    pd.add_argument("--stream", type=str, required=True)
    pd.add_argument("--out", type=str, required=True)

    args = p.parse_args()
    args.bf16 = bool(int(args.bf16))
    args.use_flex_attn = bool(int(args.use_flex_attn))

    if args.cmd == "compress":
        cmd_compress(args)
    else:
        cmd_decompress(args)


if __name__ == "__main__":
    main()
