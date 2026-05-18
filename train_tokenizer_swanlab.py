import os
import argparse
import time
import logging
from copy import deepcopy
import numpy as np

from PIL import Image

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from timm.scheduler.cosine_lr import CosineLRScheduler

from bitvae.utils.distributed import init_distributed_mode, reduce_losses, average_losses
from bitvae.utils.logger import create_logger
from bitvae.models import ImageDiscriminator
from bitvae.data import ImageData
from bitvae.modules.loss import get_disc_loss, adopt_weight
from bitvae.utils.misc import get_last_ckpt
from bitvae.utils.init_models import resume_from_ckpt
from bitvae.utils.arguments import MainArgs, add_model_specific_args

logger = logging.getLogger(__name__)

try:
    import swanlab
except Exception:
    swanlab = None


# =========================================================
# Utility
# =========================================================

def _safe_float(x):
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return float(x.detach().cpu().item())
        return float(x.detach().mean().cpu().item())
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(x)
    except Exception:
        return None


def _tensor_to_pil(x_chw: torch.Tensor) -> Image.Image:
    x = x_chw.detach().float().clamp(-1, 1)
    x = (x + 1.0) * 0.5
    x = (x * 255.0).round().to(torch.uint8)
    x = x.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x)


def _concat_horiz(pils):
    w = sum(im.size[0] for im in pils)
    h = max(im.size[1] for im in pils)
    out = Image.new("RGB", (w, h))
    xoff = 0
    for im in pils:
        out.paste(im, (xoff, 0))
        xoff += im.size[0]
    return out


@torch.no_grad()
def run_val_visualization(
    d_vae,
    val_iter_holder,
    val_loader,
    device,
    save_root: str,
    global_step: int,
    rank: int,
    prefix_scales=(2, 3, 4, 5),
    num_images: int = 4,
):
    if rank != 0 or val_loader is None:
        return

    try:
        batch = next(val_iter_holder[0])
    except StopIteration:
        if hasattr(val_loader, "sampler") and hasattr(val_loader.sampler, "set_epoch"):
            val_loader.sampler.set_epoch(global_step)
        val_iter_holder[0] = iter(val_loader)
        batch = next(val_iter_holder[0])

    x = batch["image"].to(device, non_blocking=True)
    b = min(int(num_images), int(x.shape[0]))
    x = x[:b]

    def _unwrap_ddp(m):
        return m.module if hasattr(m, "module") else m

    was_training = d_vae.training
    d_vae.eval()
    d_vae_u = _unwrap_ddp(d_vae)
    x_full, x_prefix = d_vae_u.reconstruct_prefix_scales(x, prefix_scales=prefix_scales)
    if was_training:
        d_vae.train()

    ks = sorted(set(int(k) for k in prefix_scales if int(k) >= 1))
    out_dir = os.path.join(save_root, "val_vis", f"step_{global_step:07d}")
    os.makedirs(out_dir, exist_ok=True)

    b = min(x.shape[0], 4)
    max_k = max(x_prefix.keys()) if len(x_prefix) else None

    for i in range(b):
        pils = [_tensor_to_pil(x[i])]
        for k in ks:
            if max_k is None:
                continue
            kk = min(k, max_k)
            if kk in x_prefix:
                pils.append(_tensor_to_pil(x_prefix[kk][i]))
        if x_full is not None:
            pils.append(_tensor_to_pil(x_full[i]))
        canvas = _concat_horiz(pils)
        canvas.save(os.path.join(out_dir, f"img_{i:02d}.png"))


def lecam_reg_zero(real_pred, fake_pred, thres=0.1):
    assert real_pred.ndim == 0
    reg = torch.mean(F.relu(torch.abs(real_pred) - thres).pow(2)) + torch.mean(F.relu(torch.abs(fake_pred) - thres).pow(2))
    return reg


def build_scheduler(optimizer, args, num_updates):
    if args.disable_sch:
        return None
    return CosineLRScheduler(
        optimizer,
        t_initial=num_updates,
        lr_min=args.lr_min,
        warmup_t=args.warmup_steps,
        warmup_lr_init=args.warmup_lr_init,
        t_in_epochs=False,
    )


def get_nominal_per_gpu_batch_size(args, dataloaders):
    for loader in dataloaders:
        bs = getattr(loader, "batch_size", None)
        if bs is not None:
            return int(bs)
    for key in ["batch_size", "train_batch_size", "bs"]:
        if hasattr(args, key):
            return int(getattr(args, key))
    raise ValueError("Cannot infer batch size from dataloader or args.")


def compute_scaled_lr(base_lr: float, nominal_global_batch_size: int, ref_global_batch_size: int, mode: str = "linear"):
    if ref_global_batch_size <= 0:
        raise ValueError("ref_global_batch_size must be > 0")
    if mode == "none":
        factor = 1.0
    elif mode == "sqrt":
        factor = (nominal_global_batch_size / float(ref_global_batch_size)) ** 0.5
    else:
        factor = nominal_global_batch_size / float(ref_global_batch_size)
    return base_lr * factor, factor


def maybe_init_swanlab(args, rank, logger, run_name, config_dict):
    if rank != 0 or not getattr(args, "use_swanlab", False):
        return None
    if swanlab is None:
        logger.warning("use_swanlab=True but swanlab is not installed. Please `pip install swanlab`.")
        return None

    logdir = os.path.join(args.default_root_dir, "swanlog")
    os.makedirs(logdir, exist_ok=True)

    run = swanlab.init(
        project=args.swanlab_project,
        experiment_name=args.swanlab_experiment_name or run_name,
        description=args.swanlab_description,
        config=config_dict,
        logdir=logdir,
        mode=args.swanlab_mode,
    )
    logger.info(f"SwanLab initialized: project={args.swanlab_project}, experiment={args.swanlab_experiment_name or run_name}")
    return run


def swanlab_log_dict(run, metrics: dict, step: int):
    if run is None:
        return
    payload = {}
    for k, v in metrics.items():
        val = _safe_float(v)
        if val is not None:
            payload[k] = val
    if payload:
        swanlab.log(payload, step=step)


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser = MainArgs.add_main_args(parser)
    parser = ImageData.add_data_specific_args(parser)
    args, unknown = parser.parse_known_args()
    args, parser, d_vae_model = add_model_specific_args(args, parser)

    # -------- train-script specific args --------
    parser.add_argument("--use_swanlab", action="store_true")
    parser.add_argument("--swanlab_project", type=str, default="Tokenizer")
    parser.add_argument("--swanlab_experiment_name", type=str, default="")
    parser.add_argument("--swanlab_description", type=str, default="")
    parser.add_argument("--swanlab_mode", type=str, default="cloud", choices=["cloud", "local", "offline", "disabled"])

    parser.add_argument("--base_lr", type=float, default=1e-4, help="Reference/base LR before batch-size scaling.")
    parser.add_argument("--base_batch_size", type=int, default=8, help="Reference global batch size used by base_lr.")
    parser.add_argument("--lr_scale_mode", type=str, default="linear", choices=["linear", "sqrt", "none"])
    parser.add_argument("--disc_base_lr", type=float, default=None, help="Optional discriminator base LR. If omitted, use scaled_lr * dis_lr_multiplier.")

    args = parser.parse_args()
    args.resolution = (args.resolution[0], args.resolution[0]) if len(args.resolution) == 1 else args.resolution

    print(f"{args.default_root_dir=}")

    # Setup DDP
    init_distributed_mode(args)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)

    # Setup folders / logger
    if rank == 0:
        os.makedirs(args.default_root_dir, exist_ok=True)
        checkpoint_dir = f"{args.default_root_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(args.default_root_dir)
        logger.info(f"Experiment directory created at {args.default_root_dir}")
    else:
        checkpoint_dir = f"{args.default_root_dir}/checkpoints"
        logger = create_logger(None)

    # Data
    data = ImageData(args)
    dataloaders = data.train_dataloader()
    dataloader_iters = [iter(loader) for loader in dataloaders]
    data_epochs = [0 for _ in dataloaders]
    val_loader = data.val_dataloader() if hasattr(data, "val_dataloader") else None
    val_iter_holder = [iter(val_loader)] if val_loader is not None else [None]

    # -------- LR scaling based on nominal batch size --------
    nominal_per_gpu_batch_size = get_nominal_per_gpu_batch_size(args, dataloaders)
    nominal_global_batch_size = nominal_per_gpu_batch_size * world_size
    ref_global_batch_size = int(args.base_batch_size)
    base_lr = float(args.base_lr) if args.base_lr is not None else float(args.lr)
    scaled_lr, lr_scale_factor = compute_scaled_lr(
        base_lr=base_lr,
        nominal_global_batch_size=nominal_global_batch_size,
        ref_global_batch_size=ref_global_batch_size,
        mode=args.lr_scale_mode,
    )
    scaled_disc_lr = float(args.disc_base_lr) if args.disc_base_lr is not None else (scaled_lr * args.dis_lr_multiplier)

    if rank == 0:
        logger.info(
            f"LR scaling: base_lr={base_lr:.6g}, scaled_lr={scaled_lr:.6g}, "
            f"scale_mode={args.lr_scale_mode}, nominal_global_batch_size={nominal_global_batch_size}, "
            f"ref_global_batch_size={ref_global_batch_size}, lr_scale_factor={lr_scale_factor:.6g}, "
            f"scaled_disc_lr={scaled_disc_lr:.6g}"
        )

    # Models
    d_vae = d_vae_model(args).to(device)
    d_vae.logger = logger
    image_disc = ImageDiscriminator(args).to(device)

    # Optimizers
    if args.optim_type == "Adam":
        optim = torch.optim.Adam
    elif args.optim_type == "AdamW":
        optim = torch.optim.AdamW
    else:
        raise ValueError(f"Unsupported optim_type: {args.optim_type}")

    if args.disc_optim_type is None:
        disc_optim = optim
    elif args.disc_optim_type == "rmsprop":
        disc_optim = torch.optim.RMSprop
    else:
        raise ValueError(f"Unsupported disc_optim_type: {args.disc_optim_type}")

    opt_vae = optim(d_vae.parameters(), lr=scaled_lr, betas=(args.beta1, args.beta2))
    if disc_optim == torch.optim.RMSprop:
        opt_image_disc = disc_optim(image_disc.parameters(), lr=scaled_disc_lr)
    else:
        opt_image_disc = disc_optim(image_disc.parameters(), lr=scaled_disc_lr, betas=(args.beta1, args.beta2))

    sch_vae = build_scheduler(opt_vae, args, args.max_steps)
    sch_image_disc = build_scheduler(opt_image_disc, args, args.max_steps)

    model_optims = {
        "vae": d_vae,
        "image_disc": image_disc,
        "opt_vae": opt_vae,
        "opt_image_disc": opt_image_disc,
        "sch_vae": sch_vae,
        "sch_image_disc": sch_image_disc,
    }

    # Resume / pretrained
    ckpt_path = get_last_ckpt(args.default_root_dir)
    init_step = 0
    load_optimizer = not args.not_load_optimizer
    if ckpt_path:
        logger.info(f"Resuming from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model_optims, init_step = resume_from_ckpt(state_dict, model_optims, load_optimizer=True)
    elif args.pretrained is not None:
        state_dict = torch.load(args.pretrained, map_location="cpu", weights_only=True)
        if args.pretrained_mode == "full":
            model_optims, _ = resume_from_ckpt(state_dict, model_optims, load_optimizer=load_optimizer)
        logger.info(f"Successfully loaded ckpt {args.pretrained}, pretrained_mode {args.pretrained_mode}")

    d_vae = DDP(d_vae.to(device), device_ids=[args.gpu], bucket_cap_mb=args.bucket_cap_mb)
    image_disc = DDP(image_disc.to(device), device_ids=[args.gpu], bucket_cap_mb=args.bucket_cap_mb)
    disc_loss = get_disc_loss(args.disc_loss_type)

    # SwanLab
    run_name = os.path.basename(os.path.normpath(args.default_root_dir))
    swan_config = vars(deepcopy(args))
    swan_config.update({
        "nominal_per_gpu_batch_size": nominal_per_gpu_batch_size,
        "nominal_global_batch_size": nominal_global_batch_size,
        "ref_global_batch_size": ref_global_batch_size,
        "base_lr": base_lr,
        "scaled_lr": scaled_lr,
        "scaled_disc_lr": scaled_disc_lr,
        "lr_scale_factor": lr_scale_factor,
    })
    swan_run = maybe_init_swanlab(args, rank, logger, run_name, swan_config)

    if args.multiscale_training:
        scale_idx_list = np.load("bitvae/utils/random_numbers.npy")

    start_time = time.time()
    for global_step in range(init_step, args.max_steps):
        loss_dicts = []

        if global_step == args.discriminator_iter_start - args.disc_pretrain_iter:
            logger.info("discriminator begins pretraining")
        if global_step == args.discriminator_iter_start:
            log_str = "add GAN loss into training"
            if args.disc_pretrain_iter > 0:
                log_str += ", discriminator ends pretraining"
            logger.info(log_str)

        for idx in range(len(dataloader_iters)):
            try:
                _batch = next(dataloader_iters[idx])
            except StopIteration:
                data_epochs[idx] += 1
                logger.info(f"Reset the {idx}th dataloader as epoch {data_epochs[idx]}")
                dataloaders[idx].sampler.set_epoch(data_epochs[idx])
                dataloader_iters[idx] = iter(dataloaders[idx])
                _batch = next(dataloader_iters[idx])

            x = _batch["image"].to(device, non_blocking=True)
            _type = _batch["type"][0]

            if args.multiscale_training:
                scale_idx = scale_idx_list[global_step]
                if scale_idx == 0:
                    x = F.interpolate(x, size=(256, 256), mode="area")
                elif scale_idx == 1:
                    rdn_idx = torch.randperm(len(x), device=x.device)[:4]
                    x = x[rdn_idx]
                    x = F.interpolate(x, size=(512, 512), mode="area")
                elif scale_idx == 2:
                    rdn_idx = torch.randperm(len(x), device=x.device)[:2]
                    x = x[rdn_idx]
                else:
                    raise ValueError(f"scale_idx {scale_idx} is not supported")

            if _type != "image":
                continue

            x_recon, flat_frames, flat_frames_recon, vae_loss_dict = d_vae(x, global_step, image_disc=image_disc)
            g_loss = sum(v for k, v in vae_loss_dict.items() if not k.startswith("metric/"))

            opt_vae.zero_grad()
            g_loss.backward()

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(d_vae.parameters(), args.max_grad_norm)

            if sch_vae is not None:
                sch_vae.step(global_step)
            elif args.lr_drop and global_step in args.lr_drop:
                logger.info(f"multiply lr of VQ-VAE by {args.lr_drop_rate} at iteration {global_step}")
                for opt_vae_param_group in opt_vae.param_groups:
                    opt_vae_param_group["lr"] *= args.lr_drop_rate
            opt_vae.step()
            opt_vae.zero_grad()

            disc_loss_dict = {}
            disc_factor = adopt_weight(global_step, threshold=args.discriminator_iter_start - args.disc_pretrain_iter)
            discloss = torch.tensor(0.0, device=x.device)
            d_image_loss = torch.tensor(0.0, device=x.device)

            for disc_step in range(args.disc_optim_steps):
                require_optim = False
                if _type == "image" and args.image_disc_weight > 0:
                    require_optim = True
                    logits_image_real = image_disc(x, pool_name="real")
                    logits_image_fake = image_disc(x_recon.detach(), pool_name="fake")
                    d_image_loss = disc_loss(logits_image_real, logits_image_fake)
                    disc_loss_dict["train/logits_image_real"] = logits_image_real.mean().detach()
                    disc_loss_dict["train/logits_image_fake"] = logits_image_fake.mean().detach()
                    disc_loss_dict["train/d_image_loss"] = d_image_loss.mean().detach()
                    discloss = d_image_loss * args.image_disc_weight
                    opt_discs, sch_discs = [opt_image_disc], [sch_image_disc]

                    if global_step >= args.discriminator_iter_start and args.use_lecam_reg_zero:
                        lecam_zero_loss = lecam_reg_zero(logits_image_real.mean(), logits_image_fake.mean())
                        disc_loss_dict["train/lecam_zero_loss"] = lecam_zero_loss.mean().detach()
                        discloss += lecam_zero_loss * args.lecam_weight

                discloss = disc_factor * discloss

                if require_optim:
                    for opt_disc in opt_discs:
                        opt_disc.zero_grad()
                    discloss.backward()

                    if args.max_grad_norm_disc > 0:
                        torch.nn.utils.clip_grad_norm_(image_disc.parameters(), args.max_grad_norm_disc)

                    for sch_disc in sch_discs:
                        if sch_disc is not None:
                            sch_disc.step(global_step)
                        elif args.lr_drop and global_step in args.lr_drop:
                            for opt_disc in opt_discs:
                                logger.info(f"multiply lr of discriminator by {args.lr_drop_rate} at iteration {global_step}")
                                for opt_disc_param_group in opt_disc.param_groups:
                                    opt_disc_param_group["lr"] *= args.lr_drop_rate
                    for opt_disc in opt_discs:
                        opt_disc.step()
                        opt_disc.zero_grad()

            loss_dict = {**vae_loss_dict, **disc_loss_dict}
            reduced_loss_dict = reduce_losses(loss_dict) if ((global_step + 1) % args.log_every == 0) else {}
            loss_dicts.append(reduced_loss_dict)

        if (global_step + 1) % args.log_every == 0:
            avg_loss_dict = average_losses(loss_dicts)
            train_log = {
                **avg_loss_dict,
                "train/lr_vae": opt_vae.param_groups[0]["lr"],
                "train/lr_disc": opt_image_disc.param_groups[0]["lr"],
                "train/lr_scale_factor": lr_scale_factor,
                "train/nominal_global_batch_size": nominal_global_batch_size,
            }
            torch.cuda.synchronize()
            end_time = time.time()
            iter_speed = (end_time - start_time) / args.log_every
            train_log["train/iter_speed_sec"] = iter_speed

            if rank == 0:
                # concise console summary
                logger.info(
                    "global_step=%d, recon=%.4f, perceptual=%.4f, rate=%.4f, dino=%.4f, commit=%.4f, "
                    "rate_sym=%.4f, logit_r=%.4f, logit_f=%.4f, L_disc=%.4f, "
                    "lr=%.6g, iter_speed=%.2fs",
                    global_step,
                    _safe_float(train_log.get("train/recon_loss", 0.0)),
                    _safe_float(train_log.get("train/perceptual_loss", 0.0)),
                    _safe_float(train_log.get("train/rate_loss", 0.0)),
                    _safe_float(train_log.get("train/dino_loss", 0.0)),
                    _safe_float(train_log.get("train/commitment_loss", 0.0)),
                    _safe_float(train_log.get("metric/rate_symbol_loss", train_log.get("train/rate_symbol_loss", 0.0))),
                    _safe_float(train_log.get("train/logits_image_real", 0.0)),
                    _safe_float(train_log.get("train/logits_image_fake", 0.0)),
                    _safe_float(train_log.get("train/d_image_loss", 0.0)),
                    opt_vae.param_groups[0]["lr"],
                    iter_speed,
                )

                # full structured logging to SwanLab
                swanlab_log_dict(swan_run, train_log, step=global_step)

            start_time = time.time()

        if (args.visu_every > 0) and ((global_step + 1) % args.visu_every == 0):
            run_val_visualization(
                d_vae=d_vae,
                val_iter_holder=val_iter_holder,
                val_loader=val_loader,
                device=device,
                save_root=args.default_root_dir,
                global_step=global_step,
                rank=rank,
                prefix_scales=(2, 3, 4, 5),
                num_images=4,
            )

        if (global_step + 1) % args.ckpt_every == 0 and global_step != init_step:
            if rank == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"model_step_{global_step}.ckpt")
                save_dict = {}
                for k in model_optims:
                    save_dict[k] = None if model_optims[k] is None else model_optims[k].module.state_dict() if hasattr(model_optims[k], "module") else model_optims[k].state_dict()
                torch.save({"step": global_step, **save_dict}, checkpoint_path)
                logger.info(f"Checkpoint saved at step {global_step}")

    if rank == 0 and swan_run is not None:
        try:
            swanlab.finish()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
