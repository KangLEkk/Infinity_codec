import argparse
import math

import torch
import numpy as np
from einops import rearrange
from torch import Tensor, nn
from torchvision import transforms
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
import timm

try:
    from torch.func import functional_call
except Exception:
    functional_call = None

from bitvae.modules.quantizer import MultiScaleBSQ
from bitvae.modules.entropy import OneShotScaleCausalPrior, OneShotScaleCausalPriorConfig, BitwiseMaskedCNN, BitwiseMaskedCNNConfig, bernoulli_nll_bits_from_logits
from bitvae.modules import Conv, adopt_weight, LPIPS, Normalize
from bitvae.utils.misc import ptdtype


def swish(x: Tensor) -> Tensor:
    try:
        return x * torch.sigmoid(x)
    except:
        device = x.device
        x = x.cpu().pin_memory()
        return (x*torch.sigmoid(x)).to(device=device)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_type='group'):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels, norm_type)

        self.conv1 = Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = Normalize(out_channels, norm_type)

        self.conv2 = Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            self.nin_shortcut = Conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h



class Downsample(nn.Module):
    def __init__(self, in_channels, spatial_down=False):
        super().__init__()
        assert spatial_down == True
        self.pad = (0, 1, 0, 1)
        self.conv = Conv(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        x = nn.functional.pad(x, self.pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, spatial_up=False):
        super().__init__()
        assert spatial_up == True

        self.scale_factor = 2
        self.conv = Conv(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
        in_channels = 3,
        patch_size=8,
        norm_type='group',
        use_checkpoint=False,
    ):
        super().__init__()
        self.max_down = np.log2(patch_size)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.use_checkpoint = use_checkpoint
        # downsampling
        # self.conv_in = Conv(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        self.conv_in = Conv(in_channels, ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, norm_type=norm_type))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn

            spatial_down = True if i_level < self.max_down else False
            if spatial_down:
                down.downsample = Downsample(block_in, spatial_down=spatial_down)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, norm_type=norm_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, norm_type=norm_type)

        # end
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = Conv(block_in, z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, return_hidden=False):
        if not self.use_checkpoint:
            return self._forward(x, return_hidden=return_hidden)
        else:
            return checkpoint.checkpoint(self._forward, x, return_hidden, use_reentrant=False)

    def _forward(self, x: Tensor, return_hidden=False) -> Tensor:
        # downsampling
        h0 = self.conv_in(x)
        hs = [h0]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if hasattr(self.down[i_level], "downsample"):
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        hs_mid = [h]
        h = self.mid.block_1(h)
        h = self.mid.block_2(h)
        hs_mid.append(h)
        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        if return_hidden:
            return h, hs, hs_mid
        else:
            return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
        out_ch = 3, 
        patch_size=8,
        norm_type="group",
        use_checkpoint=False,
    ):
        super().__init__()
        self.max_up = np.log2(patch_size)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.ffactor = 2 ** (self.num_resolutions - 1)
        self.use_checkpoint = use_checkpoint

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]

        # z to block_in
        self.conv_in = Conv(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, norm_type=norm_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, norm_type=norm_type)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, norm_type=norm_type))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            # https://github.com/black-forest-labs/flux/blob/b4f689aaccd40de93429865793e84a734f4a6254/src/flux/modules/autoencoder.py#L228
            spatial_up = True if 1 <= i_level <= self.max_up else False
            if spatial_up:
                up.upsample = Upsample(block_in, spatial_up=spatial_up)
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = Conv(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        if not self.use_checkpoint:
            return self._forward(z)
        else:
            return checkpoint.checkpoint(self._forward, z, use_reentrant=False)

    def _forward(self, z: Tensor) -> Tensor:
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if hasattr(self.up[i_level], "upsample"):
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h

class LatentDinoProjector(nn.Module):
    # Project early-scale latent (B,C,H,W) to a DINO embedding vector without decoding.
    def __init__(self, in_channels: int, out_dim: int, hidden: int = 256, mlp_hidden: int = 512, norm_type: str = "group"):
        super().__init__()
        self.conv = nn.Sequential(
            Conv(in_channels, hidden, kernel_size=1, stride=1, padding=0),
            Normalize(hidden, norm_type),
            nn.GELU(),
            Conv(hidden, hidden, kernel_size=3, stride=1, padding=1),
            Normalize(hidden, norm_type),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, out_dim),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.mlp(self.conv(z))

def load_dinov2_local(model_name: str, ckpt_path: str, device="cuda"):
    # 1) build model skeleton
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=0,     # remove cls head
        dynamic_img_size=True,
        global_pool=""     # keep tokens
    )

    # 2) load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 3) unwrap common checkpoint dict structures
    if isinstance(ckpt, dict):
        for k in ["model", "state_dict", "teacher", "student", "net", "module"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break

    # 4) strip common prefixes
    new_sd = {}
    for k, v in ckpt.items():
        for prefix in ("module.", "model.", "backbone.", "encoder."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        new_sd[k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"[{model_name}] missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) < 30:
        print("missing:", missing)
    if len(unexpected) < 30:
        print("unexpected:", unexpected)

    model.eval().to(device)
    return model  

class AutoEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = Encoder(
            ch=args.base_ch,
            ch_mult=args.encoder_ch_mult,
            num_res_blocks=args.num_res_blocks,
            z_channels=args.codebook_dim,
            patch_size=args.patch_size,
            use_checkpoint=args.use_checkpoint,
        )
        self.decoder = Decoder(
            ch=args.base_ch,
            ch_mult=args.decoder_ch_mult,
            num_res_blocks=args.num_res_blocks,
            z_channels=args.codebook_dim,
            patch_size=args.patch_size,
            use_checkpoint=args.use_checkpoint,
        )

        self.gan_feat_weight = args.gan_feat_weight
        self.recon_loss_type = args.recon_loss_type
        self.l1_weight = args.l1_weight
        self.kl_weight = args.kl_weight
        self.lfq_weight = args.lfq_weight
        self.image_gan_weight = args.image_gan_weight # image GAN loss weight
        self.perceptual_weight = args.perceptual_weight

        self.compute_all_commitment = args.compute_all_commitment # compute commitment between input and rq-output

        self.perceptual_model = LPIPS(upcast_tf32=args.upcast_tf32).eval()
        for p in self.perceptual_model.parameters():
            p.requires_grad_(False)

        if args.quantizer_type == 'MultiScaleBSQ':
            self.quantizer = MultiScaleBSQ(
                dim = args.codebook_dim,                        # this is the input feature dimension, defaults to log2(codebook_size) if not defined
                entropy_loss_weight = args.entropy_loss_weight, # how much weight to place on entropy loss
                diversity_gamma = args.diversity_gamma,         # within entropy loss, how much weight to give to diversity of codes, taken from https://arxiv.org/abs/1911.05894
                commitment_loss_weight=args.commitment_loss_weight, # loss weight of commitment loss
                new_quant=args.new_quant,
                use_decay_factor=args.use_decay_factor,
                use_stochastic_depth=args.use_stochastic_depth,
                drop_rate=args.drop_rate,
                schedule_mode=args.schedule_mode,
                keep_first_quant=args.keep_first_quant,
                keep_last_quant=args.keep_last_quant,
                remove_residual_detach=args.remove_residual_detach,
                use_out_phi=args.use_out_phi,
                use_out_phi_res=args.use_out_phi_res,
                random_flip = args.random_flip,
                flip_prob = args.flip_prob,
                flip_mode = args.flip_mode,
                max_flip_lvl = args.max_flip_lvl,
                random_flip_1lvl = args.random_flip_1lvl,
                flip_lvl_idx = args.flip_lvl_idx,
                drop_when_test = args.drop_when_test,
                drop_lvl_idx = args.drop_lvl_idx,
                drop_lvl_num = args.drop_lvl_num,
                random_short_schedule = args.random_short_schedule,
                short_schedule_prob = args.short_schedule_prob,
                disable_flip_prob = args.disable_flip_prob,
                zeta = args.zeta,
                gamma = args.gamma,
                uniform_short_schedule = args.uniform_short_schedule
            )
        else:
            raise NotImplementedError(f"{args.quantizer_type} not supported")
        self.commitment_loss_weight = args.commitment_loss_weight

        # --- Optional entropy model for RD fine-tuning (now replaced by joint next-scale prior) ---
        self.use_entropy_model = getattr(args, 'use_entropy_model', False)
        self.rate_lambda = float(getattr(args, 'rate_lambda', 0.0))
        self.rate_scale_mode = getattr(args, 'rate_scale_mode', 'uniform')
        self.rate_scale_alpha = float(getattr(args, 'rate_scale_alpha', 2.0))
        self.rate_scale_weights = getattr(args, 'rate_scale_weights', None)
        self.entropy_cond = getattr(args, 'entropy_cond', 'prev_sum')

        # --- Joint next-scale prior shaping ---
        self.entropy_fit_lambda = float(getattr(args, 'entropy_fit_lambda', 1.0))  # kept only for compatibility
        self.predict_lambda = float(getattr(args, 'predict_lambda', 1.0))
        self.predict_max_scale = int(getattr(args, 'predict_max_scale', 6))  # <=0 means all
        self.predict_every = int(getattr(args, 'predict_every', 1))
        self.use_soft_bits = bool(getattr(args, 'use_soft_bits', True))
        self.bit_tau = float(getattr(args, 'bit_tau', 1.0))
        self.bit_tau_min = float(getattr(args, 'bit_tau_min', 0.3))
        self.bit_tau_decay = float(getattr(args, 'bit_tau_decay', 20000.0))
        self.entropy_input_mode = getattr(args, 'entropy_input_mode', 'st')  # kept for compatibility
        self.soft_target_mode = getattr(args, 'soft_target_mode', 'soft')    # soft|st
        self.predict_objective = getattr(args, 'predict_objective', 'soft_nll')
        self.mi_alpha = float(getattr(args, 'mi_alpha', 0.0))
        self.tc_weight = float(getattr(args, 'tc_weight', 0.0))
        self.tc_mode = getattr(args, 'tc_mode', 'ysoft')

        if self.use_entropy_model and self.rate_lambda > 0:
            cfg = OneShotScaleCausalPriorConfig(
                bits_channels=args.codebook_dim,
                cond_channels=args.codebook_dim,   # 始终吃 D 通道 cond，非 prev_sum 时传零图
                model_dim=getattr(args, 'entropy_hidden', 256),
                depth=getattr(args, 'entropy_resblocks', 4),
                num_heads=getattr(args, 'entropy_heads', 8),
                mlp_ratio=getattr(args, 'entropy_mlp_ratio', 4.0),
                dropout=getattr(args, 'entropy_dropout', 0.0),
                max_scales=32,
                use_scale_embedding=True,
                use_pos2d=bool(getattr(args, 'entropy_use_pos2d', True)),
                checkpoint_blocks=bool(getattr(args, 'entropy_checkpoint_blocks', False)),
                cond_norm=True,
            )
            self.entropy_model = OneShotScaleCausalPrior(cfg)
        else:
            self.entropy_model = None

        # --- Optional entropy regularization to suppress late-scale uncertainty ---
        self.prior_entropy_weight = float(getattr(args, 'prior_entropy_weight', 0.0))
        self.prior_entropy_start_scale = int(getattr(args, 'prior_entropy_start_scale', 7))
        self.prior_stopgrad_cond = bool(getattr(args, 'prior_stopgrad_cond', False))

        # --- Coarse-prefix reconstruction to push information into early scales ---
        self.coarse_prefix_scales = getattr(args, 'coarse_prefix_scales', None)
        self.coarse_prefix_weight = float(getattr(args, 'coarse_prefix_weight', 0.0))
        self.coarse_prefix_decay = float(getattr(args, 'coarse_prefix_decay', 0.7))

        # --- DINO distillation (latent -> projector -> align to DINO(x) or cached features) ---
        self.dino_weight = float(getattr(args, 'dino_weight', 0.0))
        self.dino_max_scale = int(getattr(args, 'dino_max_scale', 6))  # <=0 means all
        self.dino_every = int(getattr(args, 'dino_every', 4))
        self.dino_use_cached = bool(getattr(args, 'dino_use_cached', False))
        self.dino_cache_key = getattr(args, 'dino_cache_key', 'dino_feat')

        self.dino_model_name = getattr(args, 'dino_model', 'dinov2_vits14')
        self.dino_input_size = int(getattr(args, 'dino_input_size', 224))
        self.dino_input_is_01 = bool(getattr(args, 'dino_input_is_01', False))
        self.dino_feat_dim = int(getattr(args, 'dino_feat_dim', 384))
        self.dino_proj_hidden = int(getattr(args, 'dino_proj_hidden', 256))
        self.dino_proj_mlp_hidden = int(getattr(args, 'dino_proj_mlp_hidden', 512))
        self.dino_norm_type = getattr(args, 'dino_norm_type', 'group')
        self.dino_amp = bool(getattr(args, 'dino_amp', False))
        self.dino_teacher_on_cpu = bool(getattr(args, 'dino_teacher_on_cpu', False))

        self._dino_teacher = None
        if self.dino_weight > 0:
            self.dino_projector = LatentDinoProjector(
                in_channels=args.codebook_dim,
                out_dim=self.dino_feat_dim,
                hidden=self.dino_proj_hidden,
                mlp_hidden=self.dino_proj_mlp_hidden,
                norm_type=self.dino_norm_type,
            )
        else:
            self.dino_projector = None


    def _get_bit_tau(self, global_step: int) -> float:
        tau0 = float(self.bit_tau)
        tau_min = float(self.bit_tau_min)
        decay = float(self.bit_tau_decay)
        if decay <= 0:
            return float(max(tau_min, tau0))
        tau = tau0 * math.exp(-float(global_step) / decay)
        return float(max(tau_min, tau))

    def _make_soft_bits(self, bits_hard: torch.Tensor, pre_quant: torch.Tensor, global_step: int):
        B, T, H, W, D = bits_hard.shape
        tau = self._get_bit_tau(global_step)
        y_soft = torch.sigmoid(pre_quant / max(tau, 1e-6))
        y_hard = bits_hard
        y_st = y_hard + y_soft - y_soft.detach()

        def _to_bt_dhw(y):
            y = y.permute(0, 1, 4, 2, 3).contiguous()
            return y.view(B * T, D, H, W)

        return _to_bt_dhw(y_soft), _to_bt_dhw(y_st), _to_bt_dhw(y_hard)

    @staticmethod
    def _bernoulli_soft_nll_bits_from_logits(
        logits: torch.Tensor,
        target_prob: torch.Tensor,
        reduce: str = 'sum'
    ) -> torch.Tensor:
        nll_nats = F.binary_cross_entropy_with_logits(logits, target_prob, reduction=reduce)
        return nll_nats / math.log(2.0)

    @staticmethod
    def _bernoulli_hard_forward_soft_grad_nll_bits_from_logits(
        logits: torch.Tensor,
        y_hard: torch.Tensor,
        y_soft: torch.Tensor,
        reduce: str = 'sum'
    ) -> torch.Tensor:
        """Forward equals hard-bit NLL, but gradients follow soft target."""
        loss_soft = AutoEncoder._bernoulli_soft_nll_bits_from_logits(logits, y_soft, reduce=reduce)
        loss_hard = bernoulli_nll_bits_from_logits(logits, y_hard.detach(), reduce=reduce)
        return loss_soft + (loss_hard - loss_soft).detach()

    @staticmethod
    def _bernoulli_marginal_entropy_bits(y_prob: torch.Tensor, reduce: str = 'sum') -> torch.Tensor:
        """Approx marginal entropy H(Y) under independent Bernoulli per bit-dim."""
        eps = 1e-6
        q = y_prob.mean(dim=(0, 2, 3)).clamp(min=eps, max=1.0 - eps)  # (D,)
        h = -(q * torch.log(q) + (1.0 - q) * torch.log(1.0 - q)) / math.log(2.0)
        if reduce == 'sum':
            n = float(y_prob.shape[0] * y_prob.shape[2] * y_prob.shape[3])
            return h.sum() * n
        elif reduce == 'mean':
            return h.mean()
        else:
            return h.sum()

    @staticmethod
    def _total_correlation_proxy(feat: torch.Tensor) -> torch.Tensor:
        """Cheap redundancy proxy via off-diagonal correlation penalty."""
        eps = 1e-6
        BT, D, H, W = feat.shape
        z = feat.permute(0, 2, 3, 1).contiguous().view(BT * H * W, D)
        z = z - z.mean(dim=0, keepdim=True)
        z = z / (z.std(dim=0, keepdim=True) + eps)
        n = z.shape[0]
        corr = (z.t() @ z) / float(n)
        eye = torch.eye(D, device=feat.device, dtype=feat.dtype)
        off = corr - eye
        return (off * off).sum() / float(D * D)

    # -------------------- DINO helpers --------------------
    def _maybe_init_dino_teacher(self):
        if self.dino_use_cached:
            return
        if self._dino_teacher is not None:
            return
        name = str(self.dino_model_name)
        self._dino_teacher = load_dinov2_local(
            "vit_large_patch14_dinov2",
            "/workspace/CKPT/DINOv2/large/pytorch_model.bin",
            device="cuda"
        )

        # large = load_dinov2_local(
        #     "vit_large_patch14_dinov2",
        #     "/path/to/vit_large_patch14_dinov2.pth",
        #     device="cuda"
        # )
        # try:
        #     if name.startswith("dinov2_"):
        #         self._dino_teacher = torch.hub.load("facebookresearch/dinov2", name)
        #     elif name.startswith("dino_"):
        #         self._dino_teacher = torch.hub.load("facebookresearch/dino:main", name)
        #     else:
        #         self._dino_teacher = torch.hub.load("facebookresearch/dinov2", name)
        # except Exception as e:
        #     raise RuntimeError(
        #         f"Failed to load DINO teacher '{name}' via torch.hub. "
        #         f"Either enable --dino_use_cached to provide precomputed features, "
        #         f"or ensure the model repo is accessible. Original error: {e}"
        #     )

        self._dino_teacher.eval()
        for p in self._dino_teacher.parameters():
            p.requires_grad_(False)

        if self.dino_teacher_on_cpu:
            self._dino_teacher.to("cpu")
        else:
            self._dino_teacher.to(next(self.parameters()).device)

    @staticmethod
    def _dino_imagenet_norm(x01: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=x01.device, dtype=x01.dtype)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=x01.device, dtype=x01.dtype)[None, :, None, None]
        return (x01 - mean) / std

    def _prep_dino_input(self, x: torch.Tensor) -> torch.Tensor:
        x01 = x.clamp(0.0, 1.0) if self.dino_input_is_01 else ((x + 1.0) * 0.5).clamp(0.0, 1.0)
        if self.dino_input_size and (x01.shape[-1] != self.dino_input_size or x01.shape[-2] != self.dino_input_size):
            x01 = F.interpolate(x01, size=(self.dino_input_size, self.dino_input_size), mode="bilinear", align_corners=False)
        x01 = self._dino_imagenet_norm(x01)
        return x01
    
    @torch.no_grad()
    def _extract_dino_feat(self, x: torch.Tensor) -> torch.Tensor:
        self._maybe_init_dino_teacher()
        assert self._dino_teacher is not None, "DINO teacher is not initialized."

        if self.dino_teacher_on_cpu:
            x_in = self._prep_dino_input(x).to("cpu")
        else:
            x_in = self._prep_dino_input(x)

        if self.dino_amp and (not self.dino_teacher_on_cpu):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                feat = self._dino_teacher.forward_features(x_in) if hasattr(self._dino_teacher, "forward_features") else self._dino_teacher(x_in)
        else:
            feat = self._dino_teacher.forward_features(x_in) if hasattr(self._dino_teacher, "forward_features") else self._dino_teacher(x_in)

        if isinstance(feat, dict):
            if "x_norm_clstoken" in feat:
                out = feat["x_norm_clstoken"]
            elif "cls_token" in feat:
                out = feat["cls_token"]
            elif "x_norm_patchtokens" in feat:
                out = feat["x_norm_patchtokens"].mean(dim=1)
            else:
                out = next(v for v in feat.values() if torch.is_tensor(v))
                if out.ndim > 2:
                    out = out.mean(dim=1)
        else:
            out = feat
            if out.ndim > 2:
                out = out.mean(dim=1)

        out = out.to(device=x.device, dtype=torch.float32)
        if out.shape[-1] != self.dino_feat_dim:
            if out.shape[-1] > self.dino_feat_dim:
                out = out[:, : self.dino_feat_dim]
            else:
                pad = self.dino_feat_dim - out.shape[-1]
                out = F.pad(out, (0, pad), mode="constant", value=0.0)
        return out
    
    def forward(self, x, global_step, image_disc=None, is_train=True):
        cached_dino_feat = None
        if isinstance(x, (tuple, list)):
            cached_dino_feat = x[1] if len(x) > 1 else None
            x = x[0]
        elif isinstance(x, dict):
            cached_dino_feat = x.get(getattr(self, 'dino_cache_key', 'dino_feat'), None)
            x = x.get('image', x.get('x', x.get('img', None)))
        assert x is not None and x.ndim == 4  # input is image BCHW

        enc_dtype = ptdtype[self.args.encoder_dtype]

        with torch.amp.autocast("cuda", dtype=enc_dtype):
            h = self.encoder(x, return_hidden=False) # B C H W
        h = h.to(dtype=torch.float32)

        # Multiscale LFQ
        need_dino_scale_data = (
            self.dino_projector is not None
            and self.dino_weight > 0
            and ((global_step % max(1, self.dino_every)) == 0)
        )
        need_scale_data = (
            (self.entropy_model is not None)
            or (self.coarse_prefix_weight > 0 and self.coarse_prefix_scales is not None)
            or need_dino_scale_data
        )
        if need_scale_data:
            z, all_indices, all_bit_indices, all_loss, scale_schedule, quantized_full_list, pre_quant_list = self.quantizer(h, return_scale_data=True)
        else:
            z, all_indices, all_bit_indices, all_loss = self.quantizer(h)
            scale_schedule, quantized_full_list, pre_quant_list = None, None, None
        # print(torch.unique(torch.round(z * 10**4)/10**4)) # keep 4 decimal places
        x_recon = self.decoder(z)
        vq_output = {
            "commitment_loss": torch.mean(all_loss) * self.lfq_weight, # here commitment loss is sum of commitment loss and entropy penalty
            "encodings": all_indices,
            "bit_encodings": all_bit_indices,
        }
        if self.compute_all_commitment:
            # compute commitment loss between input and rq-output
            vq_output["all_commitment_loss"] = F.mse_loss(h, z.detach(), reduction="mean") * self.commitment_loss_weight * self.lfq_weight
        else:
            # disable backward prop
            vq_output["all_commitment_loss"] = F.mse_loss(h.detach(), z.detach(), reduction="mean") * self.commitment_loss_weight * self.lfq_weight

        assert x.shape == x_recon.shape, f"x.shape {x.shape}, x_recon.shape {x_recon.shape}"

        if is_train == False:
            return x_recon, vq_output

        if self.recon_loss_type == 'l1':
            recon_loss = F.l1_loss(x_recon, x) * self.l1_weight
        else:
            recon_loss = F.mse_loss(x_recon, x) * self.l1_weight

        flat_frames = x
        flat_frames_recon = x_recon

        perceptual_loss = self.perceptual_model(flat_frames, flat_frames_recon).mean() * self.perceptual_weight

        loss_dict = {
            "train/perceptual_loss": perceptual_loss,
            "train/recon_loss": recon_loss,
            "train/commitment_loss": vq_output['commitment_loss'],
            "train/all_commitment_loss": vq_output['all_commitment_loss'],
        }

        # --- DINO distillation (no decoding): align early-scale latent prefix to DINO(x) embedding ---
        if self.dino_projector is not None and self.dino_weight > 0 and quantized_full_list is not None:
            if (global_step % max(1, self.dino_every)) == 0:
                if self.dino_use_cached and cached_dino_feat is not None:
                    feat_t = cached_dino_feat.to(device=x.device, dtype=torch.float32)
                    if feat_t.ndim > 2:
                        feat_t = feat_t.mean(dim=1)
                else:
                    feat_t = self._extract_dino_feat(x)

                k = int(self.dino_max_scale)
                if k <= 0:
                    k = len(quantized_full_list)
                k = max(1, min(k, len(quantized_full_list)))

                z_pref = None
                for i in range(k):
                    z_pref = quantized_full_list[i] if z_pref is None else (z_pref + quantized_full_list[i])

                feat_s = self.dino_projector(z_pref)
                feat_s = F.normalize(feat_s, dim=-1)
                feat_t = F.normalize(feat_t.detach(), dim=-1)

                dino_cos = (feat_s * feat_t).sum(dim=-1).mean()
                dino_loss = (1.0 - dino_cos)

                loss_dict["metric/dino_cos"] = dino_cos.detach()
                loss_dict["train/dino_loss"] = dino_loss * self.dino_weight

        
        # --- RD: one-shot all-scale next-scale prior (single forward) ---
        if self.entropy_model is not None and scale_schedule is not None:
            K = len(all_bit_indices)

            if self.rate_scale_mode == 'uniform':
                w = [1.0 for _ in range(K)]
            elif self.rate_scale_mode == 'linear':
                w = [float(i + 1) / float(max(1, K)) for i in range(K)]
            elif self.rate_scale_mode == 'power':
                alpha = self.rate_scale_alpha
                w = [(float(i + 1) / float(max(1, K))) ** alpha for i in range(K)]
            elif self.rate_scale_mode == 'custom' and self.rate_scale_weights is not None and len(self.rate_scale_weights) >= K:
                w = [float(self.rate_scale_weights[i]) for i in range(K)]
            else:
                w = [1.0 for _ in range(K)]

            img_pixels = float(x.shape[-2] * x.shape[-1])
            prior_bits_total_w = torch.zeros((), device=x.device, dtype=torch.float32)
            do_predict = (self.predict_lambda > 0) and ((global_step % max(1, self.predict_every)) == 0)

            cond_maps = []
            scale_ids = []
            y_soft_list = []
            y_st_list = []
            y_hard_list = []
            pre_q_bt_list = []
            batch_list = []
            use_scale_list = []

            prev_full = None

            # -------- step 1: prepare cond maps / targets for all scales --------
            for si, bits_si in enumerate(all_bit_indices):
                if bits_si is None:
                    continue

                B0, Ts, Hs, Ws, D = bits_si.shape
                assert D == self.args.codebook_dim

                # cond for current scale = previous cumulative latent only
                if self.entropy_cond == 'prev_sum':
                    if prev_full is None:
                        cond_pred = torch.zeros((B0, D, Hs, Ws), device=x.device, dtype=torch.float32)
                    else:
                        cond_pred = F.interpolate(prev_full, size=(Hs, Ws), mode='area')
                else:
                    cond_pred = torch.zeros((B0, D, Hs, Ws), device=x.device, dtype=torch.float32)

                if self.prior_stopgrad_cond:
                    cond_pred = cond_pred.detach()

                # expand to BT
                cond_bt = cond_pred[:, None].expand(-1, Ts, -1, -1, -1).contiguous().view(B0 * Ts, D, Hs, Ws)

                bits_hard = bits_si.to(dtype=torch.float32)

                if self.use_soft_bits:
                    if pre_quant_list is None or si >= len(pre_quant_list) or pre_quant_list[si] is None:
                        raise RuntimeError(
                            "pre_quant_list is required for soft/STE bit training. "
                            "Patch MultiScaleBSQ/BSQ to return pre-quant values."
                        )
                    pre_q = pre_quant_list[si].to(dtype=torch.float32)
                    y_soft, y_st, y_hard = self._make_soft_bits(bits_hard, pre_q, global_step)
                    pre_q_bt = pre_q.permute(0, 1, 4, 2, 3).contiguous().view(B0 * Ts, D, Hs, Ws)
                else:
                    y_hard = bits_hard.permute(0, 1, 4, 2, 3).contiguous().view(B0 * Ts, D, Hs, Ws)
                    y_soft, y_st = y_hard, y_hard
                    pre_q_bt = None

                cond_maps.append(cond_bt)
                scale_ids.append(si)
                y_soft_list.append(y_soft)
                y_st_list.append(y_st)
                y_hard_list.append(y_hard)
                pre_q_bt_list.append(pre_q_bt)
                batch_list.append(B0)

                # 这里控制“哪些尺度参与 loss”
                # 但 forward 仍然是一次性把所有尺度都预测出来
                use_this_scale = do_predict and (self.predict_max_scale <= 0 or (si + 1) <= self.predict_max_scale)
                use_scale_list.append(use_this_scale)

                # roll prefix for next scale
                if quantized_full_list is not None and si < len(quantized_full_list):
                    q_full = quantized_full_list[si]
                    prev_full = q_full if prev_full is None else (prev_full + q_full)

            # -------- step 2: one-shot prior forward over all scales --------
            if len(cond_maps) > 0:
                logits_pred_list = self.entropy_model(cond_maps=cond_maps, scale_ids=scale_ids)
            else:
                logits_pred_list = []

            # -------- step 3: compute per-scale losses --------
            for li, si in enumerate(scale_ids):
                logits_pred = logits_pred_list[li]
                y_soft = y_soft_list[li]
                y_st = y_st_list[li]
                y_hard = y_hard_list[li]
                B0 = batch_list[li]
                pre_q_bt = pre_q_bt_list[li]

                if self.predict_objective == 'soft_nll':
                    target = y_st if self.soft_target_mode == 'st' else y_soft
                    pred_bits = self._bernoulli_soft_nll_bits_from_logits(
                        logits_pred, target, reduce='sum'
                    ) / float(B0)

                elif self.predict_objective == 'strict_entropy':
                    pred_bits = self._bernoulli_hard_forward_soft_grad_nll_bits_from_logits(
                        logits_pred, y_hard, y_soft, reduce='sum'
                    ) / float(B0)

                elif self.predict_objective == 'mi':
                    h_cond = self._bernoulli_hard_forward_soft_grad_nll_bits_from_logits(
                        logits_pred, y_hard, y_soft, reduce='sum'
                    )
                    h_marg = self._bernoulli_marginal_entropy_bits(y_soft, reduce='sum')
                    pred_bits = (h_cond - self.mi_alpha * h_marg) / float(B0)

                else:
                    target = y_st if self.soft_target_mode == 'st' else y_soft
                    pred_bits = self._bernoulli_soft_nll_bits_from_logits(
                        logits_pred, target, reduce='sum'
                    ) / float(B0)

                if self.tc_weight > 0:
                    if self.tc_mode == 'preq' and self.use_soft_bits and (pre_q_bt is not None):
                        feat_tc = pre_q_bt
                    else:
                        feat_tc = y_soft
                    tc = self._total_correlation_proxy(feat_tc)
                    pred_bits = pred_bits + (self.tc_weight * tc)
                    loss_dict[f'metric/prior_tc_s{si+1}'] = tc.detach()

                if self.prior_entropy_weight > 0 and (si + 1) >= max(1, self.prior_entropy_start_scale):
                    h_marg = self._bernoulli_marginal_entropy_bits(y_soft, reduce='sum') / float(B0)
                    pred_bits = pred_bits + (self.prior_entropy_weight * h_marg)
                    loss_dict[f'metric/prior_marginal_entropy_s{si+1}'] = h_marg.detach()

                loss_dict[f'metric/rate_pred_bits_s{si+1}'] = pred_bits.detach()

                if use_scale_list[li]:
                    prior_bits_total_w = prior_bits_total_w + (w[si] * pred_bits)

            loss_dict['train/entropy_fit_loss'] = torch.zeros((), device=x.device, dtype=torch.float32)

            if do_predict:
                bpp_prior_w = (prior_bits_total_w / img_pixels)
                loss_dict['metric/bpp_pred_weighted'] = bpp_prior_w.detach()
                loss_dict['metric/bpp_fit_weighted'] = bpp_prior_w.detach()
                loss_dict['train/rate_loss'] = bpp_prior_w * self.rate_lambda * float(self.predict_lambda)
            else:
                loss_dict['train/rate_loss'] = torch.zeros((), device=x.device, dtype=torch.float32)


        # --- Coarse-prefix reconstruction losses (decode with only early scales) ---
        if self.coarse_prefix_weight > 0 and self.coarse_prefix_scales is not None and quantized_full_list is not None:
            # scales are 1-based in args
            prefix_scales = [int(s) for s in self.coarse_prefix_scales if int(s) >= 1]
            prefix_scales = sorted(set(prefix_scales))
            if len(prefix_scales) > 0:
                # precompute cumulative sums once
                cum = []
                running = None
                for q in quantized_full_list:
                    running = q if running is None else (running + q)
                    cum.append(running)

                coarse_recon_total = torch.zeros((), device=x.device, dtype=torch.float32)
                coarse_lpips_total = torch.zeros((), device=x.device, dtype=torch.float32)
                coarse_gan_total = torch.zeros((), device=x.device, dtype=torch.float32)
                last_prefix_scale = prefix_scales[-1]

                disc_factor = adopt_weight(
                    global_step,
                    threshold=self.args.discriminator_iter_start,
                    warmup=self.args.disc_warmup
                )

                for j, k in enumerate(prefix_scales):
                    kk = min(k, len(cum))
                    z_k = cum[kk - 1]
                    x_k = self.decoder(z_k)

                    if self.recon_loss_type == 'l1':
                        l_recon_k = F.l1_loss(x_k, x) * self.l1_weight
                    else:
                        l_recon_k = F.mse_loss(x_k, x) * self.l1_weight

                    l_lpips_k = self.perceptual_model(x, x_k).mean() * self.perceptual_weight

                    # GAN only on the largest prefix scale
                    if self.image_gan_weight > 0 and image_disc is not None and kk == last_prefix_scale:
                        logits_fake_k = image_disc(x_k)
                        l_gan_k = -torch.mean(logits_fake_k) * self.image_gan_weight * disc_factor
                    else:
                        l_gan_k = torch.zeros((), device=x.device, dtype=torch.float32)

                    w_k = (self.coarse_prefix_decay ** j)
                    coarse_recon_total = coarse_recon_total + w_k * l_recon_k
                    coarse_lpips_total = coarse_lpips_total + w_k * l_lpips_k
                    coarse_gan_total = coarse_gan_total + w_k * l_gan_k

                    loss_dict[f'metric/coarse_recon_loss_k{kk}'] = l_recon_k.detach()
                    loss_dict[f'metric/coarse_lpips_loss_k{kk}'] = l_lpips_k.detach()
                    loss_dict[f'metric/coarse_gan_loss_k{kk}'] = l_gan_k.detach()

                coarse_loss_total = coarse_recon_total + coarse_lpips_total + coarse_gan_total
                loss_dict['metric/coarse_recon_loss'] = coarse_recon_total.detach()
                loss_dict['metric/coarse_lpips_loss'] = coarse_lpips_total.detach()
                loss_dict['metric/coarse_gan_loss'] = coarse_gan_total.detach()
                loss_dict['metric/coarse_total_loss'] = coarse_loss_total.detach()
                loss_dict['train/coarse_loss'] = coarse_loss_total * self.coarse_prefix_weight
        ### GAN loss
        disc_factor = adopt_weight(global_step, threshold=self.args.discriminator_iter_start, warmup=self.args.disc_warmup)
        if self.image_gan_weight > 0: # image GAN loss
            logits_image_fake = image_disc(flat_frames_recon)
            g_image_loss = -torch.mean(logits_image_fake) * self.image_gan_weight * disc_factor # disc_factor=0 before self.args.discriminator_iter_start
            loss_dict["train/g_image_loss"] = g_image_loss

        return (x_recon.detach(), flat_frames.detach(), flat_frames_recon.detach(), loss_dict)

    @torch.no_grad()
    def reconstruct_prefix_scales(self, x: torch.Tensor, prefix_scales=(2, 3, 4, 5)):
        """Reconstruct with only early scales (prefix sums of multiscale quantization).

        Returns:
        x_full: reconstruction using all scales (decoder(z_all)).
        x_prefix: dict[int, Tensor] mapping k->reconstruction using first k scales.
        """
        self.eval()
        assert x.ndim == 4, "x must be BCHW"
        enc_dtype = ptdtype[self.args.encoder_dtype]
        with torch.amp.autocast("cuda", dtype=enc_dtype):
            h = self.encoder(x, return_hidden=False)
        h = h.to(dtype=torch.float32)

        # always request per-scale full-resolution contributions
        z, _, _, _, _, quantized_full_list, _ = self.quantizer(h, return_scale_data=True)
        x_full = self.decoder(z)

        # cumulative sums in latent space
        cum = []
        running = None
        for q in quantized_full_list:
            running = q if running is None else (running + q)
            cum.append(running)

        x_prefix = {}
        for k in prefix_scales:
            kk = int(k)
            if kk < 1:
                continue
            kk = min(kk, len(cum))
            x_prefix[kk] = self.decoder(cum[kk - 1])
        return x_full, x_prefix


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--codebook_size", type=int, default=16384)

        parser.add_argument("--base_ch", type=int, default=128)
        parser.add_argument("--num_res_blocks", type=int, default=2)  # num_res_blocks for encoder, num_res_blocks+1 for decoder
        parser.add_argument("--encoder_ch_mult", type=int, nargs='+', default=[1, 1, 2, 2, 4])
        parser.add_argument("--decoder_ch_mult", type=int, nargs='+', default=[1, 1, 2, 2, 4])
        # RD / entropy shaping (A)
        # RD / next-scale prior shaping
        parser.add_argument("--entropy_fit_lambda", type=float, default=1.0)
        parser.add_argument("--predict_lambda", type=float, default=1.0)
        parser.add_argument("--predict_max_scale", type=int, default=6)
        parser.add_argument("--predict_every", type=int, default=1)
        parser.add_argument("--use_soft_bits", action="store_true")
        parser.add_argument("--bit_tau", type=float, default=1.0)
        parser.add_argument("--bit_tau_min", type=float, default=0.3)
        parser.add_argument("--bit_tau_decay", type=float, default=20000.0)
        parser.add_argument("--entropy_input_mode", type=str, default="st", choices=["hard", "st", "soft"])
        parser.add_argument("--soft_target_mode", type=str, default="soft", choices=["soft", "st"])
        parser.add_argument("--entropy_hidden", type=int, default=256)
        parser.add_argument("--entropy_resblocks", type=int, default=4)   # 这里现在表示 prior depth
        parser.add_argument("--entropy_heads", type=int, default=8)
        parser.add_argument("--entropy_mlp_ratio", type=float, default=4.0)
        parser.add_argument("--entropy_dropout", type=float, default=0.0)
        parser.add_argument("--entropy_use_pos2d", action="store_true")
        parser.add_argument("--entropy_checkpoint_blocks", action="store_true")
        parser.add_argument(
            "--predict_objective",
            type=str,
            default="soft_nll",
            choices=["soft_nll", "strict_entropy", "mi"],
            help="soft_nll: CE(logits, y_soft/y_st). strict_entropy: forward uses hard bits but gradient uses y_soft. mi: H_cond - mi_alpha*H_marg."
        )
        parser.add_argument(
            "--mi_alpha",
            type=float,
            default=0.0,
            help="Weight for marginal-entropy term when predict_objective=mi."
        )
        parser.add_argument(
            "--tc_weight",
            type=float,
            default=0.0,
            help="Weight for total-correlation / redundancy penalty."
        )
        parser.add_argument(
            "--tc_mode",
            type=str,
            default="ysoft",
            choices=["ysoft", "preq"],
            help="Feature used for TC penalty: ysoft or preq."
        )

        parser.add_argument(
            "--prior_entropy_weight",
            type=float,
            default=0.0,
            help="Later-scale weighted marginal entropy penalty on current-scale soft bits."
        )
        parser.add_argument(
            "--prior_entropy_start_scale",
            type=int,
            default=7,
            help="Apply prior_entropy_weight from this 1-based scale onward."
        )
        parser.add_argument(
            "--prior_stopgrad_cond",
            action="store_true",
            help="Stop gradient from next-scale prior into prefix condition F_{k-1}."
        )

        # DINO distillation (latent->projector, align to DINO(x) or cached embedding)
        parser.add_argument("--dino_weight", type=float, default=0.0)
        parser.add_argument("--dino_max_scale", type=int, default=6, help="Use first k scales to build latent prefix for distillation (<=0 means all).")
        parser.add_argument("--dino_every", type=int, default=4, help="Compute distillation loss every N steps for efficiency.")
        parser.add_argument("--dino_use_cached", action="store_true", help="If set, expect cached DINO embeddings to be provided alongside images.")
        parser.add_argument("--dino_cache_key", type=str, default="dino_feat", help="Key name for cached DINO embeddings when batch is a dict.")
        parser.add_argument("--dino_model", type=str, default="dinov2_vits14", help="torch.hub model name: dinov2_vits14, dino_vits16, etc.")
        parser.add_argument("--dino_input_size", type=int, default=224)
        parser.add_argument("--dino_input_is_01", action="store_true", help="Set if training images are already in [0,1] (skip [-1,1]->[0,1] shift).")
        parser.add_argument("--dino_feat_dim", type=int, default=384, help="Expected DINO embedding dim (dinov2_vits14=384).")
        parser.add_argument("--dino_proj_hidden", type=int, default=256)
        parser.add_argument("--dino_proj_mlp_hidden", type=int, default=512)
        parser.add_argument("--dino_norm_type", type=str, default="group")
        parser.add_argument("--dino_amp", action="store_true", help="Use autocast fp16 for teacher forward on CUDA.")
        parser.add_argument("--dino_teacher_on_cpu", action="store_true", help="Run DINO teacher on CPU (slower but avoids GPU memory).")

        # parser.add_argument("--entropy_fusion_mode", type=str, default="gated_concat",
        #             choices=["add", "concat", "gated_concat", "film"])
        # parser.add_argument("--entropy_cond_norm", action="store_true")
        # parser.add_argument("--entropy_gate_init", type=float, default=0.0)
        
        return parser
