"""ARPC CLI — NONE masking + ZERO header + fixed 1024x1024

This CLI is specialized for your ultra-low bitrate setup:

- mask_strategy = 'none' (transmit ALL active bits)
- active_bits   = [d] * K (default)
- K / d / schedule are shared defaults (not stored in bitstream)
- seed / version / prompt are NOT stored in the bitstream
- Bitstream contains ONLY the arithmetic-coded payload bytes (0-byte header)

IMPORTANT
---------
Because the stream has **no header**, both encoder and decoder must agree on:
- fixed resolution (default: 1024x1024)
- k_transmit (default: 5)

`prompt` is provided out-of-band (CLI argument).
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import shutil
from typing import Dict

import numpy as np
import torch
from PIL import Image

# from codec.arpc_codec_none_zeroheader_fixed1024 import ARPCNoneZeroHeaderFixed1024
from codec.arpc_codec_zeroheader_fixed1024_entropy_masks_varfill import ARPCNoneZeroHeaderFixed1024
from infinity.models.bsq_vae.vae import vae_model
from infinity.models.infinity_patched import Infinity as InfinityModel


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
# Model builders
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
def load_infinity(
    rope2d_each_sa_layer,
    rope2d_normalized_by_hw,
    use_scale_schedule_embedding,
    pn,
    use_bit_label,
    add_lvl_embeding_only_first_block,
    model_path='',
    scale_schedule=None,
    vae=None,
    device='cuda',
    model_kwargs=None,
    text_channels=2048,
    apply_spatial_patchify=0,
    use_flex_attn=False,
    bf16=False,
    checkpoint_type='torch',
):
    print('[Loading Infinity]')
    text_maxlen = 512
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
        gpt = InfinityModel(
            vae_local=vae, text_channels=text_channels, text_maxlen=text_maxlen,
            shared_aln=True, raw_scale_schedule=scale_schedule,
            checkpointing='full-block',
            customized_flash_attn=False,
            fused_norm=True,
            pad_to_multiplier=128,
            use_flex_attn=use_flex_attn,
            add_lvl_embeding_only_first_block=add_lvl_embeding_only_first_block,
            use_bit_label=use_bit_label,
            rope2d_each_sa_layer=rope2d_each_sa_layer,
            rope2d_normalized_by_hw=rope2d_normalized_by_hw,
            pn=pn,
            apply_spatial_patchify=apply_spatial_patchify,
            inference_mode=True,
            train_h_div_w_list=[1.0],
            **model_kwargs,
        ).to(device=device)
        print(f'[you selected Infinity with model_kwargs={model_kwargs}] model size: {sum(p.numel() for p in gpt.parameters())/1e9:.2f}B, bf16={bf16}')

        if bf16:
            for block in gpt.unregistered_blocks:
                block.bfloat16()

        gpt.eval()
        gpt.requires_grad_(False)

        gpt.cuda()
        torch.cuda.empty_cache()

        print('[Load Infinity weights]')
        if checkpoint_type == 'torch':
            state_dict = torch.load(model_path, map_location=device)
            print(gpt.load_state_dict(state_dict))
        elif checkpoint_type == 'torch_shard':
            from transformers.modeling_utils import load_sharded_checkpoint
            load_sharded_checkpoint(gpt, model_path, strict=False)
        gpt.rng = torch.Generator(device=device)
        return gpt


def load_transformer(vae, args, device: torch.device):
    model_path = args.model_ckpt
    if args.checkpoint_type == 'torch':
        # if osp.exists(args.cache_dir):
        #     local_model_path = osp.join(args.cache_dir, 'tmp', model_path.replace('/', '_'))
        # else:
        #     local_model_path = model_path
        local_model_path = model_path
        if args.enable_model_cache:
            slim_model_path = model_path.replace('ar-', 'slim-')
            local_slim_model_path = local_model_path.replace('ar-', 'slim-')
            os.makedirs(osp.dirname(local_slim_model_path), exist_ok=True)
            print(f'model_path: {model_path}, slim_model_path: {slim_model_path}')
            print(f'local_model_path: {local_model_path}, local_slim_model_path: {local_slim_model_path}')
            if not osp.exists(local_slim_model_path):
                if osp.exists(slim_model_path):
                    print(f'copy {slim_model_path} to {local_slim_model_path}')
                    shutil.copyfile(slim_model_path, local_slim_model_path)
                else:
                    if not osp.exists(local_model_path):
                        print(f'copy {model_path} to {local_model_path}')
                        shutil.copyfile(model_path, local_model_path)
                    save_slim_model(local_model_path, save_file=local_slim_model_path, device=device)
                    print(f'copy {local_slim_model_path} to {slim_model_path}')
                    if not osp.exists(slim_model_path):
                        shutil.copyfile(local_slim_model_path, slim_model_path)
                        os.remove(local_model_path)
                        os.remove(model_path)
            slim_model_path = local_slim_model_path
        else:
            slim_model_path = model_path
        print(f'load checkpoint from {slim_model_path}')
    elif args.checkpoint_type == 'torch_shard':
        slim_model_path = model_path
    else:
        raise ValueError(f'unknown checkpoint_type: {args.checkpoint_type}')

    kwargs_model = _model_kwargs_from_type(args.model_type)
    return load_infinity(
        rope2d_each_sa_layer=args.rope2d_each_sa_layer,
        rope2d_normalized_by_hw=args.rope2d_normalized_by_hw,
        use_scale_schedule_embedding=getattr(args, 'use_scale_schedule_embedding', 0),
        pn=args.pn,
        use_bit_label=args.use_bit_label,
        add_lvl_embeding_only_first_block=args.add_lvl_embeding_only_first_block,
        model_path=slim_model_path,
        scale_schedule=None,
        vae=vae,
        device=device,
        model_kwargs=kwargs_model,
        text_channels=args.text_channels,
        apply_spatial_patchify=args.apply_spatial_patchify,
        use_flex_attn=args.use_flex_attn,
        bf16=args.bf16,
        checkpoint_type=args.checkpoint_type,
    )

def save_slim_model(infinity_model_path, save_file=None, device='cpu', key='gpt_fsdp'):
    print('[Save slim model]')
    full_ckpt = torch.load(infinity_model_path, map_location='cpu', weights_only=False)
    infinity_slim = full_ckpt['trainer'][key]
    if not save_file:
        save_file = osp.splitext(infinity_model_path)[0] + '-slim.pth'
    print(f'Save to {save_file}')
    torch.save(infinity_slim, save_file)
    print('[Save slim model] done')
    return save_file

def build_codec(args, device: torch.device) -> ARPCNoneZeroHeaderFixed1024:
    text_tokenizer, text_encoder = load_text_encoder(args.text_encoder_ckpt, device)
    vae = load_vae(args.vae_ckpt, args.vae_type, args.apply_spatial_patchify, device)
    model = load_transformer(vae, args, device)

    codec = ARPCNoneZeroHeaderFixed1024(
        vae=vae,
        model=model,
        text_tokenizer=text_tokenizer,
        text_encoder=text_encoder,
        device=str(device),
        tlen=int(args.tlen),
        fixed_hw=(int(args.fixed_hw), int(args.fixed_hw)),
        k_transmit=int(args.k_transmit),
        seed=int(args.seed),
    )

    # Optional masking config (works with entropy-mask codec variants).
    # NOTE: stream is ZERO-header, so encoder/decoder must use the SAME settings.
    if hasattr(codec, "mask_strategy"):
        codec.mask_strategy = str(getattr(args, "mask_strategy", "none"))
    if hasattr(codec, "mask_params"):
        base = {}
        try:
            s = str(getattr(args, "mask_params", "{}") or "{}")
            base = json.loads(s)
            if not isinstance(base, dict):
                base = {}
        except Exception:
            base = {}
        kr = str(getattr(args, "keep_ratio", "") or "").strip()
        if kr and codec.mask_strategy in ("entropy_channel", "entropy_spatial"):
            try:
                base["keep_ratio"] = float(kr)
            except Exception:
                pass
        codec.mask_params = base
    if hasattr(codec, "active_bits_spec"):
        codec.active_bits_spec = str(getattr(args, "active_bits", "all") or "all")

    return codec


# -----------------------------
# Commands
# -----------------------------

def cmd_compress(args):
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "cpu" else "cuda"))
    codec = build_codec(args, device)

    pil = Image.open(args.image).convert("RGB")
    pil = _resize_center_crop(pil, int(args.fixed_hw), int(args.fixed_hw))
    img = _to_tensor_minus1_1(pil).to(device)

    codec.compress(
        img_B3HW=img,
        out_stream_path=args.out,
        prompt=args.prompt,
    )

    nbytes = os.path.getsize(args.out)
    bpp = (nbytes * 8.0) / (int(args.fixed_hw) * int(args.fixed_hw))
    print(f"[ARPC-ZH] wrote {nbytes} bytes -> ~{bpp:.8f} bpp @ {args.fixed_hw}x{args.fixed_hw} | k_transmit={args.k_transmit}")


def cmd_decompress(args):
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "cpu" else "cuda"))
    codec = build_codec(args, device)

    codec.decompress(
        stream_path=args.stream,
        out_path=args.out,
        prompt=args.prompt,
    )
    print(f"[ARPC-ZH] reconstructed -> {args.out} | fixed_hw={args.fixed_hw} | k_transmit={args.k_transmit}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    common.add_argument("--seed", type=int, default=0, help="Not transmitted. Keep encoder/decoder consistent.")
    common.add_argument("--prompt", type=str, default="a clear photo", help="Not transmitted. Out-of-band text conditioning.")

    # fixed stream assumptions (ZERO header)
    common.add_argument("--fixed_hw", type=int, default=1024, help="Fixed square resolution (must match encoder/decoder).")
    common.add_argument("--k_transmit", type=int, default=5, help="Fixed k_transmit (must match encoder/decoder).")

    # masking strategies (ZERO header; must match encoder/decoder)
    common.add_argument("--mask_strategy", type=str, default="none",
                        help="Mask strategy: none | entropy_channel | entropy_scale | entropy_spatial (0-header; must match encoder/decoder).")
    common.add_argument("--mask_params", type=str, default="{}",
                        help="JSON dict of params for the mask strategy (0-header; must match encoder/decoder).")
    common.add_argument("--keep_ratio", type=str, default="",
                        help="Convenience override: set keep_ratio in mask_params (e.g., 0.1). Only used by entropy_channel/entropy_spatial/entropy_spatial_global.")
    common.add_argument("--active_bits", type=str, default="all",
                        help="Active bits spec shared by encoder/decoder. e.g. 'all' or '16'.")

    # model / vae
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

    # keep these for compatibility with your InfinityPatched init (ignored by fixed_hw in preprocessing)
    common.add_argument("--pn", type=str, default="1M")
    common.add_argument("--h_div_w_template", type=float, default=1.0)
    common.add_argument("--add_lvl_embeding_only_first_block", type=int, default=0, choices=[0, 1])
    common.add_argument("--rope2d_each_sa_layer", type=int, default=1, choices=[0, 1])
    common.add_argument("--rope2d_normalized_by_hw", type=int, default=2, choices=[0, 1, 2])

    # compress
    pc = sub.add_parser("compress", parents=[common])
    pc.add_argument("--image", type=str, required=True)
    pc.add_argument("--out", type=str, required=True)

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
