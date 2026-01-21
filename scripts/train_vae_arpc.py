"""Stage-1: train / finetune the Infinity BSQ-VAE for ARPC.

This script implements the paper's tokenizer training ingredients:
  - GM-BMSRQ: three-group channel/bit masking per scale (8/12/16 for c=16)
  - SRD: scale random dropout (p=0.2, start from the 4th scale)

It is intentionally lightweight (pure PyTorch) so you can run it with
`torchrun` on your cluster without depending on the GPT trainer.

Example
-------
torchrun --nproc_per_node 8 scripts/train_vae_arpc.py \
  --data_root /path/to/images \
  --out_dir ./arpc_vae_ckpt \
  --codebook_dim 16 \
  --image_size 256 \
  --phase1_steps 500000 \
  --phase2_steps 300000 \
  --phase2_resos 256,512,1024 \
  --batch_size 16 --lr 1e-4 --amp
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import random
import time
from dataclasses import asdict
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from PIL import Image

from infinity.models.bsq_vae.vae import vae_model
from infinity.utils.arpc_util import parse_active_bits_spec


# ----------------------------- utils -----------------------------


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank0() -> bool:
    return (not is_dist()) or dist.get_rank() == 0


def setup_dist(backend: str = "nccl"):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


def barrier():
    if is_dist():
        dist.barrier()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """RGB PIL -> float tensor in [-1, 1], shape [3,H,W]."""
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)  # 3,H,W
    return t * 2.0 - 1.0


def random_resized_crop(img: Image.Image, size: int) -> Image.Image:
    # torchvision-free random crop/resize (square)
    w, h = img.size
    if w < size or h < size:
        scale = max(size / w, size / h)
        img = img.resize((int(round(w * scale)), int(round(h * scale))), Image.BICUBIC)
        w, h = img.size

    # random crop
    x0 = 0 if w == size else random.randint(0, w - size)
    y0 = 0 if h == size else random.randint(0, h - size)
    crop = img.crop((x0, y0, x0 + size, y0 + size))
    return crop


class ImageFolderDataset(Dataset):
    def __init__(self, data_root: str, filelist: str = "", max_images: int = 0):
        self.data_root = data_root
        if filelist:
            with open(filelist, "r", encoding="utf-8") as f:
                self.files = [ln.strip() for ln in f.readlines() if ln.strip()]
        else:
            exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
            files = []
            for e in exts:
                files.extend(glob.glob(os.path.join(data_root, "**", e), recursive=True))
            self.files = sorted(files)

        if max_images and max_images > 0:
            self.files = self.files[: int(max_images)]

        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {data_root} (filelist={filelist})")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        fp = self.files[idx]
        try:
            img = Image.open(fp)
            img = img.convert("RGB")
        except Exception:
            # return a random other image
            ridx = random.randint(0, len(self.files) - 1)
            fp = self.files[ridx]
            img = Image.open(fp).convert("RGB")
        return img


def save_ckpt(out_dir: str, step: int, model: torch.nn.Module, opt: torch.optim.Optimizer):
    if not rank0():
        return
    os.makedirs(out_dir, exist_ok=True)
    # Match Infinity's expected format: a dict with key "vae".
    ckpt = {
        "step": int(step),
        "vae": model.state_dict(),
        "optim": opt.state_dict(),
    }
    path = os.path.join(out_dir, f"vae_arpc_step{step:07d}.pth")
    torch.save(ckpt, path)
    # update "latest"
    torch.save(ckpt, os.path.join(out_dir, "vae_arpc_latest.pth"))


def load_ckpt(path: str, model: torch.nn.Module, opt: Optional[torch.optim.Optimizer] = None) -> int:
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt.get("vae", ckpt.get("model"))
    if sd is None:
        raise KeyError(f"Invalid ckpt format: {path} (missing 'vae'/'model')")
    model.load_state_dict(sd, strict=True)
    if opt is not None and "optim" in ckpt:
        opt.load_state_dict(ckpt["optim"])
    return int(ckpt.get("step", 0))


# ----------------------------- main -----------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--filelist", type=str, default="")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--max_images", type=int, default=0)

    parser.add_argument("--codebook_dim", type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--schedule_mode", type=str, default="dynamic")

    parser.add_argument("--gm_bits", type=str, default="default")
    parser.add_argument("--srd_prob", type=float, default=0.2)
    parser.add_argument("--srd_start_scale", type=int, default=3)
    parser.add_argument("--srd_mode", type=str, default="level")

    parser.add_argument("--phase1_steps", type=int, default=500000)
    parser.add_argument("--phase2_steps", type=int, default=300000)
    parser.add_argument("--phase2_resos", type=str, default="256,512,1024")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)

    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=5000)
    args = parser.parse_args()

    setup_dist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(args.seed + (dist.get_rank() if is_dist() else 0) * 17)

    # dataset
    ds = ImageFolderDataset(args.data_root, filelist=args.filelist, max_images=args.max_images)
    sampler = DistributedSampler(ds, shuffle=True) if is_dist() else None
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    it = iter(dl)

    # VAE
    codebook_dim = int(args.codebook_dim)
    codebook_size = 2 ** codebook_dim
    gm_bits = parse_active_bits_spec(args.gm_bits, codebook_dim)

    # match upstream defaults
    if args.patch_size == 8:
        encoder_ch_mult = [1, 2, 4, 4]
        decoder_ch_mult = [1, 2, 4, 4]
    else:
        encoder_ch_mult = [1, 2, 4, 4, 4]
        decoder_ch_mult = [1, 2, 4, 4, 4]

    model = vae_model(
        vqgan_ckpt=None,
        schedule_mode=args.schedule_mode,
        codebook_dim=codebook_dim,
        codebook_size=codebook_size,
        test_mode=False,
        patch_size=args.patch_size,
        encoder_ch_mult=encoder_ch_mult,
        decoder_ch_mult=decoder_ch_mult,
        gm_active_bits_per_scale=gm_bits,
        srd_prob=float(args.srd_prob),
        srd_start_scale=int(args.srd_start_scale),
        srd_mode=str(args.srd_mode),
    ).to(device)
    model.train()

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    start_step = 0
    if args.resume:
        start_step = load_ckpt(args.resume, model, opt)
        if rank0():
            print(f"[resume] step={start_step} from {args.resume}")

    if is_dist():
        model = DDP(model, device_ids=[device.index], find_unused_parameters=False)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    total_steps = int(args.phase1_steps) + int(args.phase2_steps)
    phase2_resos = [int(x) for x in args.phase2_resos.split(",") if x.strip()]

    t0 = time.time()
    running = {"loss": 0.0, "recon": 0.0, "aux": 0.0}

    for step in range(start_step, total_steps):
        if sampler is not None and step % len(dl) == 0:
            sampler.set_epoch(step // max(1, len(dl)))

        # choose resolution
        if step < args.phase1_steps:
            reso = int(args.image_size)
        else:
            reso = int(random.choice(phase2_resos))

        try:
            imgs = next(it)
        except StopIteration:
            it = iter(dl)
            imgs = next(it)

        # per-sample crop+to tensor
        batch_t = []
        for im in imgs:
            im2 = random_resized_crop(im, reso)
            batch_t.append(pil_to_tensor(im2))
        x = torch.stack(batch_t, dim=0).to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=args.amp, dtype=torch.bfloat16):
            # encode -> decode
            h, z, _, _, all_loss, _, _ = (
                model.module.encode(x, scale_schedule=None)
                if isinstance(model, DDP)
                else model.encode(x, scale_schedule=None)
            )
            x_hat = model.module.decode(z) if isinstance(model, DDP) else model.decode(z)
            recon = (x_hat - x).abs().mean()
            aux_loss = all_loss.mean() if torch.is_tensor(all_loss) else torch.tensor(0.0, device=device)
            loss = recon + aux_loss

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if args.grad_clip and args.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()

        # logging
        running["loss"] += float(loss.detach())
        running["recon"] += float(recon.detach())
        running["aux"] += float(aux_loss.detach())

        if (step + 1) % args.log_every == 0 and rank0():
            dt = time.time() - t0
            it_s = (step + 1 - start_step) / max(1e-6, dt)
            msg = (
                f"step {step+1}/{total_steps} | "
                f"loss {running['loss']/args.log_every:.4f} "
                f"recon {running['recon']/args.log_every:.4f} "
                f"aux {running['aux']/args.log_every:.4f} | "
                f"{it_s:.2f} it/s | reso {reso}"
            )
            print(msg)
            for k in running:
                running[k] = 0.0

        if (step + 1) % args.save_every == 0:
            barrier()
            save_ckpt(args.out_dir, step + 1, model.module if isinstance(model, DDP) else model, opt)
            barrier()

    barrier()
    save_ckpt(args.out_dir, total_steps, model.module if isinstance(model, DDP) else model, opt)
    barrier()


if __name__ == "__main__":
    main()
