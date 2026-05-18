

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import timm
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from torch import Tensor, nn

try:
    from torch.func import functional_call
except Exception:
    functional_call = None

from bitvae.modules.quantizer import MultiScaleBSQ
from bitvae.modules import Conv, LPIPS, Normalize, adopt_weight
from bitvae.utils.misc import ptdtype


# =========================================================
# Basic helpers
# =========================================================

def swish(x: Tensor) -> Tensor:
    try:
        return x * torch.sigmoid(x)
    except Exception:
        device = x.device
        x = x.cpu().pin_memory()
        return (x * torch.sigmoid(x)).to(device=device)


def _pick_gn_groups(channels: int, max_groups: int = 8) -> int:
    for g in reversed(range(1, max_groups + 1)):
        if channels % g == 0:
            return g
    return 1


def _build_2d_sincos_pos_embed(H: int, W: int, C: int, device, dtype) -> torch.Tensor:
    """Return (H*W, C) 2D sin-cos positional embedding.
    C must be divisible by 4.
    """
    assert C % 4 == 0, f"pos dim {C} must be divisible by 4"
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    omega = torch.arange(C // 4, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(1, (C // 4 - 1))))

    y = y.reshape(-1, 1) * omega.reshape(1, -1)
    x = x.reshape(-1, 1) * omega.reshape(1, -1)
    pe = torch.cat([torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)], dim=1)
    return pe.to(dtype=dtype)


def _build_binary_prototypes(group_size: int, device, dtype) -> torch.Tensor:
    """All binary patterns in [0,1]^m, shape (2^m, m)."""
    vocab = 1 << group_size
    vals = torch.arange(vocab, device=device, dtype=torch.long)
    bits = []
    for shift in reversed(range(group_size)):
        bits.append(((vals >> shift) & 1).to(dtype=dtype))
    return torch.stack(bits, dim=-1)


# =========================================================
# RDVQ-style one-shot group-symbol prior
# =========================================================

class PriorTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


@dataclass
class RDVQStyleGroupScalePriorConfig:
    bits_channels: int
    cond_channels: int
    group_size: int = 4
    model_dim: int = 256
    depth: int = 4
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    max_scales: int = 32
    use_scale_embedding: bool = True
    use_pos2d: bool = True
    checkpoint_blocks: bool = False
    cond_norm: bool = True
    use_ctx_group_embedding: bool = True
    base_slices: int = 2
    max_slices: int = 8


class RDVQStyleGroupScalePrior(nn.Module):
    """
    RDVQ-style one-shot all-scale prior.

    Main ideas:
      1) prefix condition from previous cumulative latent (coarse-to-fine)
      2) teacher-forced current-scale shifted group-symbol context
      3) one-shot masked transformer over all scales
      4) dual heads:
         - symbol head (main RD objective)
         - bit head (auxiliary stabilizer)

    Inputs:
      - cond_maps: list[(BT,C,H,W)]
      - scale_ids: list[int]
      - ctx_group_ids: optional list[(BT,G,H,W)] with shifted group ids, pad id = vocab_size

    Outputs:
      - bit_logits_list: list[(BT,D,H,W)]
      - sym_logits_list: list[(BT,G,V,H,W)]
    """

    def __init__(self, cfg: RDVQStyleGroupScalePriorConfig):
        super().__init__()
        self.cfg = cfg
        self.group_size = int(cfg.group_size)
        assert cfg.bits_channels % self.group_size == 0, "bits_channels must be divisible by group_size"
        self.num_groups = cfg.bits_channels // self.group_size
        self.vocab_size = 1 << self.group_size

        self.cond_gain = nn.Parameter(torch.tensor(1.0))

        gn = _pick_gn_groups(cfg.model_dim, 8)
        self.cond_proj = nn.Sequential(
            nn.Conv2d(cfg.cond_channels, cfg.model_dim, kernel_size=3, padding=1),
            nn.GroupNorm(gn, cfg.model_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(cfg.model_dim, cfg.model_dim, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )

        self.use_ctx_group_embedding = bool(cfg.use_ctx_group_embedding)
        if self.use_ctx_group_embedding:
            self.ctx_group_embed = nn.Embedding(self.vocab_size + 1, cfg.model_dim)  # +1 pad token
            self.ctx_proj = nn.Sequential(
                nn.Conv2d(cfg.model_dim, cfg.model_dim, kernel_size=1),
                nn.GroupNorm(gn, cfg.model_dim),
                nn.SiLU(inplace=True),
            )
        else:
            self.ctx_group_embed = None
            self.ctx_proj = None

        self.scale_emb = nn.Embedding(cfg.max_scales, cfg.model_dim) if cfg.use_scale_embedding else None

        self.blocks = nn.ModuleList([
            PriorTransformerBlock(
                dim=cfg.model_dim,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.depth)
        ])
        self.norm = nn.LayerNorm(cfg.model_dim)

        self.bit_head = nn.Linear(cfg.model_dim, cfg.bits_channels)
        self.sym_head = nn.Linear(cfg.model_dim, self.num_groups * self.vocab_size)

    def _build_slice_order(self, H: int, W: int, num_slices: int, device) -> torch.Tensor:
        """Return local order ids of shape (H*W,).
        We partition width into slices; positions in the same slice share the same order.
        This is a light approximation of dependency-aware slice ordering.
        """
        xs = torch.arange(W, device=device, dtype=torch.long)
        slice_x = torch.div(xs * num_slices, max(1, W), rounding_mode="floor")
        order_2d = slice_x.unsqueeze(0).expand(H, W)
        return order_2d.reshape(-1)

    def _get_attn_mask(self, metas: List[Tuple[int, int, int]], device) -> torch.Tensor:
        """metas = [(H, W, scale_id), ...]"""
        orders = []
        offset = 0
        for H, W, scale_id in metas:
            num_slices = min(self.cfg.max_slices, self.cfg.base_slices + int(scale_id))
            local_order = self._build_slice_order(H, W, num_slices, device=device) + offset
            orders.append(local_order)
            offset += num_slices
        order = torch.cat(orders, dim=0)  # (L_total,)
        # True means masked / not allowed
        return order[:, None] < order[None, :]

    def forward(
        self,
        cond_maps: List[torch.Tensor],
        scale_ids: Optional[List[int]] = None,
        ctx_group_ids: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if len(cond_maps) == 0:
            return [], []

        if scale_ids is None:
            scale_ids = list(range(len(cond_maps)))
        if ctx_group_ids is None:
            ctx_group_ids = [None for _ in cond_maps]

        B = cond_maps[0].shape[0]
        tokens = []
        metas = []

        for li, cond in enumerate(cond_maps):
            assert cond.ndim == 4, f"cond must be (B,C,H,W), got {cond.shape}"

            if self.cfg.cond_norm:
                rms = cond.pow(2).mean(dim=1, keepdim=True).add(1e-6).sqrt()
                cond = cond / rms
            cond = cond * self.cond_gain.to(dtype=cond.dtype)

            feat = self.cond_proj(cond)  # (B,C,H,W)
            _, C, H, W = feat.shape

            ctx_ids = ctx_group_ids[li]
            if self.use_ctx_group_embedding and ctx_ids is not None:
                # ctx_ids: (B,G,H,W), pad id = vocab_size
                ctx_emb = self.ctx_group_embed(ctx_ids.clamp(min=0, max=self.vocab_size))  # (B,G,H,W,C)
                ctx_emb = ctx_emb.mean(dim=1).permute(0, 3, 1, 2).contiguous()            # (B,C,H,W)
                feat = feat + self.ctx_proj(ctx_emb)

            tok = feat.flatten(2).transpose(1, 2).contiguous()  # (B,HW,C)

            if self.cfg.use_pos2d:
                pos = _build_2d_sincos_pos_embed(H, W, C, tok.device, tok.dtype)
                tok = tok + pos.unsqueeze(0)

            if self.scale_emb is not None:
                sid = int(scale_ids[li])
                sid = max(0, min(self.cfg.max_scales - 1, sid))
                tok = tok + self.scale_emb.weight[sid].to(dtype=tok.dtype).view(1, 1, C)

            tokens.append(tok)
            metas.append((H, W, int(scale_ids[li])))

        x = torch.cat(tokens, dim=1)  # (B, L_total, C)
        attn_mask = self._get_attn_mask(metas, x.device)

        for blk in self.blocks:
            if self.cfg.checkpoint_blocks and self.training:
                x = checkpoint.checkpoint(lambda inp, mask: blk(inp, mask), x, attn_mask, use_reentrant=False)
            else:
                x = blk(x, attn_mask)

        x = self.norm(x)
        bit_logits_all = self.bit_head(x)  # (B, L_total, D)
        sym_logits_all = self.sym_head(x)  # (B, L_total, G*V)

        bit_outs: List[torch.Tensor] = []
        sym_outs: List[torch.Tensor] = []
        ptr = 0

        for H, W, _ in metas:
            L = H * W
            bit_cur = bit_logits_all[:, ptr:ptr + L, :]
            bit_cur = bit_cur.transpose(1, 2).contiguous().view(B, self.cfg.bits_channels, H, W)
            bit_outs.append(bit_cur)

            sym_cur = sym_logits_all[:, ptr:ptr + L, :]
            sym_cur = sym_cur.view(B, L, self.num_groups, self.vocab_size)
            sym_cur = sym_cur.permute(0, 2, 3, 1).contiguous().view(B, self.num_groups, self.vocab_size, H, W)
            sym_outs.append(sym_cur)
            ptr += L

        return bit_outs, sym_outs


# =========================================================
# Encoder / Decoder
# =========================================================

class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_type="group"):
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
        h = self.norm1(x)
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
        assert spatial_down is True
        self.pad = (0, 1, 0, 1)
        self.conv = Conv(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        x = F.pad(x, self.pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, spatial_up=False):
        super().__init__()
        assert spatial_up is True
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
        ch_mult: List[int],
        num_res_blocks: int,
        z_channels: int,
        in_channels=3,
        patch_size=8,
        norm_type="group",
        use_checkpoint=False,
    ):
        super().__init__()
        self.max_down = np.log2(patch_size)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.use_checkpoint = use_checkpoint

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

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, norm_type=norm_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, norm_type=norm_type)

        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = Conv(block_in, z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, return_hidden=False):
        if not self.use_checkpoint:
            return self._forward(x, return_hidden=return_hidden)
        return checkpoint.checkpoint(self._forward, x, return_hidden, use_reentrant=False)

    def _forward(self, x: Tensor, return_hidden=False) -> Tensor:
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

        h = hs[-1]
        hs_mid = [h]
        h = self.mid.block_1(h)
        h = self.mid.block_2(h)
        hs_mid.append(h)
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        if return_hidden:
            return h, hs, hs_mid
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        ch_mult: List[int],
        num_res_blocks: int,
        z_channels: int,
        out_ch=3,
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

        block_in = ch * ch_mult[self.num_resolutions - 1]
        self.conv_in = Conv(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, norm_type=norm_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, norm_type=norm_type)

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
            spatial_up = True if 1 <= i_level <= self.max_up else False
            if spatial_up:
                up.upsample = Upsample(block_in, spatial_up=spatial_up)
            self.up.insert(0, up)

        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = Conv(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        if not self.use_checkpoint:
            return self._forward(z)
        return checkpoint.checkpoint(self._forward, z, use_reentrant=False)

    def _forward(self, z: Tensor) -> Tensor:
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.block_2(h)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if hasattr(self.up[i_level], "upsample"):
                h = self.up[i_level].upsample(h)
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


# =========================================================
# Optional DINO distillation
# =========================================================

class LatentDinoProjector(nn.Module):
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


def load_timm_dino_local(model_name: str, ckpt_path: str, device="cuda"):
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=0,
        dynamic_img_size=True,
        global_pool="",
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for k in ["model", "state_dict", "teacher", "student", "net", "module"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break

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


def load_dinov3_hf(model_name: str, device="cuda"):
    try:
        from transformers import AutoModel
    except Exception as exc:
        raise RuntimeError(
            "DINOv3 Hugging Face loading requires a recent transformers install "
            "(official DINOv3 support starts from transformers>=4.56.0)."
        ) from exc

    model = AutoModel.from_pretrained(model_name)
    model.eval().to(device)
    return model


def load_dinov3_hub(model_name: str, repo_dir: str, weights: Optional[str], device="cuda"):
    if not repo_dir:
        raise ValueError("DINOv3 torch.hub loading requires --dino_repo_dir.")
    kwargs = {"source": "local"}
    if weights:
        kwargs["weights"] = weights
    model = torch.hub.load(repo_dir, model_name, **kwargs)
    model.eval().to(device)
    return model


# =========================================================
# Main AutoEncoder with RDVQ-style group-symbol rate branch
# =========================================================

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

        # losses
        self.gan_feat_weight = args.gan_feat_weight
        self.recon_loss_type = args.recon_loss_type
        self.l1_weight = args.l1_weight
        self.kl_weight = args.kl_weight
        self.lfq_weight = args.lfq_weight
        self.image_gan_weight = args.image_gan_weight
        self.perceptual_weight = args.perceptual_weight
        self.compute_all_commitment = args.compute_all_commitment

        self.perceptual_model = LPIPS(upcast_tf32=args.upcast_tf32).eval()
        for p in self.perceptual_model.parameters():
            p.requires_grad_(False)

        if args.quantizer_type != "MultiScaleBSQ":
            raise NotImplementedError(f"{args.quantizer_type} not supported")

        self.quantizer = MultiScaleBSQ(
            dim=args.codebook_dim,
            entropy_loss_weight=args.entropy_loss_weight,
            diversity_gamma=args.diversity_gamma,
            commitment_loss_weight=args.commitment_loss_weight,
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
            random_flip=args.random_flip,
            flip_prob=args.flip_prob,
            flip_mode=args.flip_mode,
            max_flip_lvl=args.max_flip_lvl,
            random_flip_1lvl=args.random_flip_1lvl,
            flip_lvl_idx=args.flip_lvl_idx,
            drop_when_test=args.drop_when_test,
            drop_lvl_idx=args.drop_lvl_idx,
            drop_lvl_num=args.drop_lvl_num,
            random_short_schedule=args.random_short_schedule,
            short_schedule_prob=args.short_schedule_prob,
            disable_flip_prob=args.disable_flip_prob,
            zeta=args.zeta,
            gamma=args.gamma,
            uniform_short_schedule=args.uniform_short_schedule,
        )
        self.commitment_loss_weight = args.commitment_loss_weight

        # RDVQ-style group-symbol entropy model.
        self.use_group_rate = bool(getattr(args, "use_group_rate", False))
        self.rate_lambda = float(getattr(args, "rate_lambda", 0.0))
        self.group_size = int(getattr(args, "group_size", 4))

        if self.use_group_rate and self.rate_lambda > 0:
            prior_cfg = RDVQStyleGroupScalePriorConfig(
                bits_channels=args.codebook_dim,
                cond_channels=args.codebook_dim,
                group_size=self.group_size,
                model_dim=int(getattr(args, "entropy_hidden", 256)),
                depth=int(getattr(args, "entropy_resblocks", 4)),
                num_heads=int(getattr(args, "entropy_heads", 8)),
                mlp_ratio=float(getattr(args, "entropy_mlp_ratio", 4.0)),
                dropout=float(getattr(args, "entropy_dropout", 0.0)),
                max_scales=int(getattr(args, "entropy_max_scales", 32)),
                use_scale_embedding=True,
                use_pos2d=bool(getattr(args, "entropy_use_pos2d", True)),
                checkpoint_blocks=bool(getattr(args, "entropy_checkpoint_blocks", False)),
                cond_norm=bool(getattr(args, "entropy_cond_norm", True)),
                use_ctx_group_embedding=True,
                base_slices=int(getattr(args, "prior_base_slices", 2)),
                max_slices=int(getattr(args, "prior_max_slices", 8)),
            )
            self.entropy_model = RDVQStyleGroupScalePrior(prior_cfg)
        else:
            self.entropy_model = None

        # coarse prefix sample
        self.coarse_prefix_scales = getattr(args, "coarse_prefix_scales", None)
        self.coarse_prefix_sample = bool(getattr(args, "coarse_prefix_sample", False))
        self.coarse_prefix_full_prob = float(getattr(args, "coarse_prefix_full_prob", 0.5))

        # DINO
        self.dino_weight = float(getattr(args, "dino_weight", 0.0))
        self.dino_max_scale = int(getattr(args, "dino_max_scale", 6))
        self.dino_every = int(getattr(args, "dino_every", 4))
        self.dino_use_cached = bool(getattr(args, "dino_use_cached", False))
        self.dino_cache_key = getattr(args, "dino_cache_key", "dino_feat")
        self.dino_scales = getattr(args, "dino_scales", None)
        self.dino_scale_decay = float(getattr(args, "dino_scale_decay", 0.7))
        self.dino_backend = getattr(args, "dino_backend", "auto")
        self.dino_model_name = getattr(args, "dino_model", "facebook/dinov3-vitb16-pretrain-lvd1689m")
        self.dino_weights = getattr(args, "dino_weights", None)
        self.dino_repo_dir = getattr(args, "dino_repo_dir", "")
        self.dino_input_size = int(getattr(args, "dino_input_size", 256))
        self.dino_input_is_01 = bool(getattr(args, "dino_input_is_01", False))
        self.dino_feat_dim = int(getattr(args, "dino_feat_dim", 768))
        self.dino_proj_hidden = int(getattr(args, "dino_proj_hidden", 256))
        self.dino_proj_mlp_hidden = int(getattr(args, "dino_proj_mlp_hidden", 512))
        self.dino_norm_type = getattr(args, "dino_norm_type", "group")
        self.dino_amp = bool(getattr(args, "dino_amp", False))
        self.dino_teacher_on_cpu = bool(getattr(args, "dino_teacher_on_cpu", False))
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

    # ------------------------- helpers -------------------------
    def _make_soft_bits(self, bits_hard: torch.Tensor, pre_quant: torch.Tensor):
        B, T, H, W, D = bits_hard.shape
        y_soft = torch.sigmoid(pre_quant)
        y_hard = bits_hard.to(dtype=torch.float32)

        def _to_bt_dhw(y):
            y = y.permute(0, 1, 4, 2, 3).contiguous()
            return y.view(B * T, D, H, W)

        return _to_bt_dhw(y_soft), _to_bt_dhw(y_hard)

    def _bits_to_group_ids(self, bits_bt_dhw: torch.Tensor) -> torch.Tensor:
        """Convert bit maps (BT,D,H,W) into group-symbol ids (BT,G,H,W)."""
        BT, D, H, W = bits_bt_dhw.shape
        group_size = self.group_size
        assert D % group_size == 0, f"codebook_dim={D} must be divisible by group_size={group_size}"
        num_groups = D // group_size
        bits = bits_bt_dhw.to(dtype=torch.long)
        bits = bits.view(BT, num_groups, group_size, H, W).permute(0, 1, 3, 4, 2).contiguous()
        weights = 2 ** torch.arange(group_size - 1, -1, -1, device=bits.device, dtype=torch.long)
        return (bits * weights.view(1, 1, 1, 1, group_size)).sum(dim=-1)

    def _soft_bits_to_group_probs(self, bits_prob_bt_dhw: torch.Tensor) -> torch.Tensor:
        """Convert independent bit probabilities into soft group-symbol probabilities.

        Returns (BT,G,V,H,W), with V=2**group_size. This is the only target used
        for the symbol rate loss; no hard-forward correction is applied.
        """
        BT, D, H, W = bits_prob_bt_dhw.shape
        group_size = self.group_size
        assert D % group_size == 0, f"codebook_dim={D} must be divisible by group_size={group_size}"
        num_groups = D // group_size
        vocab_size = 1 << group_size

        bits_prob = bits_prob_bt_dhw.clamp(1e-6, 1.0 - 1e-6)
        bits_prob = bits_prob.view(BT, num_groups, group_size, H, W).permute(0, 1, 3, 4, 2).contiguous()

        proto = _build_binary_prototypes(group_size, bits_prob.device, bits_prob.dtype)
        proto = proto.view(1, 1, 1, 1, vocab_size, group_size)
        p = bits_prob.unsqueeze(-2)
        log_prob = proto * torch.log(p) + (1.0 - proto) * torch.log1p(-p)
        log_prob = log_prob.sum(dim=-1)

        probs = torch.exp(log_prob)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return probs.permute(0, 1, 4, 2, 3).contiguous()

    @staticmethod
    def _soft_group_ce_bits_from_logits(
        logits_bgvhw: torch.Tensor,
        target_probs_bgvhw: torch.Tensor,
        reduce: str = "sum",
    ) -> torch.Tensor:
        log_probs = F.log_softmax(logits_bgvhw, dim=2)
        ce_bits = -(target_probs_bgvhw * log_probs).sum(dim=2) / math.log(2.0)
        if reduce == "sum":
            return ce_bits.sum()
        if reduce == "mean":
            return ce_bits.mean()
        return ce_bits

    @staticmethod
    def _shift_group_ids_raster(group_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
        """
        Teacher-forced symbol context.
        group_ids: (BT,G,H,W)
        returns shifted ids with pad at the first raster position.
        """
        BT, G, H, W = group_ids.shape
        seq = group_ids.view(BT, G, H * W)
        pad = torch.full((BT, G, 1), fill_value=pad_id, device=group_ids.device, dtype=group_ids.dtype)
        seq = torch.cat([pad, seq[:, :, :-1]], dim=-1)
        return seq.view(BT, G, H, W)

    @staticmethod
    def _build_cumulative_latents(quantized_full_list):
        if quantized_full_list is None:
            return None
        cum = []
        running = None
        for q in quantized_full_list:
            running = q if running is None else (running + q)
            cum.append(running)
        return cum

    @staticmethod
    def _sanitize_1based_scales(scales, max_scale: int):
        if scales is None:
            return []
        out = []
        seen = set()
        for s in scales:
            kk = int(s)
            if kk < 1:
                continue
            kk = min(kk, max_scale)
            if kk not in seen:
                out.append(kk)
                seen.add(kk)
        return sorted(out)

    def _sample_prefix_scale(self, candidate_scales, full_k: int, device):
        cand = list(candidate_scales)
        if full_k not in cand:
            cand.append(full_k)
        cand = sorted(set(int(k) for k in cand if int(k) >= 1))
        if len(cand) == 1:
            probs = torch.ones(1, device=device, dtype=torch.float32)
            return cand[0], probs

        p_full = float(min(max(self.coarse_prefix_full_prob, 0.0), 1.0))
        probs = torch.full(
            (len(cand),),
            (1.0 - p_full) / float(len(cand) - 1),
            device=device,
            dtype=torch.float32,
        )
        full_idx = cand.index(full_k)
        probs[full_idx] = p_full
        probs = probs / probs.sum()
        idx = int(torch.multinomial(probs, num_samples=1).item())
        return cand[idx], probs

    # -------------------- DINO helpers --------------------
    def _maybe_init_dino_teacher(self):
        if self.dino_use_cached:
            return
        if self._dino_teacher is not None:
            return

        name = str(self.dino_model_name)
        backend = str(self.dino_backend).lower()
        device = next(self.parameters()).device

        if backend == "auto":
            if name.startswith("facebook/dinov3-"):
                backend = "hf"
            elif name.startswith("dinov3_"):
                backend = "hub"
            else:
                backend = "timm"

        if backend == "hf":
            self._dino_teacher = load_dinov3_hf(name, device=device)
        elif backend == "hub":
            self._dino_teacher = load_dinov3_hub(
                name,
                repo_dir=self.dino_repo_dir,
                weights=self.dino_weights,
                device=device,
            )
        elif backend == "timm":
            ckpt_path = self.dino_weights or "/workspace/CKPT/DINOv2/large/pytorch_model.bin"
            self._dino_teacher = load_timm_dino_local(
                name,
                ckpt_path,
                device=device,
            )
        else:
            raise ValueError(f"Unsupported dino_backend={self.dino_backend!r}")

        self._dino_teacher.eval()
        for p in self._dino_teacher.parameters():
            p.requires_grad_(False)
        if self.dino_teacher_on_cpu:
            self._dino_teacher.to("cpu")
        else:
            self._dino_teacher.to(device)

    @staticmethod
    def _dino_imagenet_norm(x01: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=x01.device, dtype=x01.dtype)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=x01.device, dtype=x01.dtype)[None, :, None, None]
        return (x01 - mean) / std

    def _prep_dino_input(self, x: torch.Tensor) -> torch.Tensor:
        x01 = x.clamp(0.0, 1.0) if self.dino_input_is_01 else ((x + 1.0) * 0.5).clamp(0.0, 1.0)
        if self.dino_input_size and (x01.shape[-1] != self.dino_input_size or x01.shape[-2] != self.dino_input_size):
            x01 = F.interpolate(x01, size=(self.dino_input_size, self.dino_input_size), mode="bilinear", align_corners=False)
        return self._dino_imagenet_norm(x01)

    @torch.no_grad()
    def _extract_dino_feat(self, x: torch.Tensor) -> torch.Tensor:
        self._maybe_init_dino_teacher()
        assert self._dino_teacher is not None

        if self.dino_teacher_on_cpu:
            x_in = self._prep_dino_input(x).to("cpu")
        else:
            x_in = self._prep_dino_input(x)

        if self.dino_amp and (not self.dino_teacher_on_cpu):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                if hasattr(self._dino_teacher, "forward_features"):
                    feat = self._dino_teacher.forward_features(x_in)
                elif str(self.dino_backend).lower() == "hf" or str(self.dino_model_name).startswith("facebook/dinov3-"):
                    feat = self._dino_teacher(pixel_values=x_in)
                else:
                    feat = self._dino_teacher(x_in)
        else:
            if hasattr(self._dino_teacher, "forward_features"):
                feat = self._dino_teacher.forward_features(x_in)
            elif str(self.dino_backend).lower() == "hf" or str(self.dino_model_name).startswith("facebook/dinov3-"):
                feat = self._dino_teacher(pixel_values=x_in)
            else:
                feat = self._dino_teacher(x_in)

        if hasattr(feat, "pooler_output") and torch.is_tensor(feat.pooler_output):
            out = feat.pooler_output
        elif hasattr(feat, "last_hidden_state") and torch.is_tensor(feat.last_hidden_state):
            out = feat.last_hidden_state
            if out.ndim > 2:
                out = out[:, 0]
        elif isinstance(feat, dict):
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
                out = F.pad(out, (0, self.dino_feat_dim - out.shape[-1]), mode="constant", value=0.0)
        return out

    # ------------------------- forward -------------------------
    def forward(self, x, global_step, image_disc=None, is_train=True):
        cached_dino_feat = None
        if isinstance(x, (tuple, list)):
            cached_dino_feat = x[1] if len(x) > 1 else None
            x = x[0]
        elif isinstance(x, dict):
            cached_dino_feat = x.get(getattr(self, "dino_cache_key", "dino_feat"), None)
            x = x.get("image", x.get("x", x.get("img", None)))
        assert x is not None and x.ndim == 4

        enc_dtype = ptdtype[self.args.encoder_dtype]
        with torch.amp.autocast("cuda", dtype=enc_dtype):
            h = self.encoder(x, return_hidden=False)
        h = h.to(dtype=torch.float32)

        need_dino_scale_data = (
            self.dino_projector is not None
            and self.dino_weight > 0
            and ((global_step % max(1, self.dino_every)) == 0)
        )
        need_rate_data = self.entropy_model is not None and self.rate_lambda > 0
        need_prefix_scale_data = self.coarse_prefix_sample
        need_scale_data = need_rate_data or need_prefix_scale_data or need_dino_scale_data

        if need_scale_data:
            z, all_indices, all_bit_indices, all_loss, scale_schedule, quantized_full_list, pre_quant_list = self.quantizer(h, return_scale_data=True)
        else:
            z, all_indices, all_bit_indices, all_loss = self.quantizer(h)
            scale_schedule, quantized_full_list, pre_quant_list = None, None, None

        cum_latents = self._build_cumulative_latents(quantized_full_list)
        sampled_prefix_k = None
        sampled_prefix_probs = None
        z_decode = z

        if is_train and self.coarse_prefix_sample and cum_latents is not None and len(cum_latents) > 0:
            prefix_scales = self._sanitize_1based_scales(self.coarse_prefix_scales, len(cum_latents))
            sampled_prefix_k, sampled_prefix_probs = self._sample_prefix_scale(prefix_scales, len(cum_latents), x.device)
            if sampled_prefix_k != len(cum_latents):
                z_decode = cum_latents[sampled_prefix_k - 1]

        x_recon = self.decoder(z_decode)

        vq_output = {
            "commitment_loss": torch.mean(all_loss) * self.lfq_weight,
            "encodings": all_indices,
            "bit_encodings": all_bit_indices,
        }
        if self.compute_all_commitment:
            vq_output["all_commitment_loss"] = F.mse_loss(h, z.detach(), reduction="mean") * self.commitment_loss_weight * self.lfq_weight
        else:
            vq_output["all_commitment_loss"] = F.mse_loss(h.detach(), z.detach(), reduction="mean") * self.commitment_loss_weight * self.lfq_weight

        if not is_train:
            return x_recon, vq_output

        recon_loss = F.l1_loss(x_recon, x) * self.l1_weight if self.recon_loss_type == "l1" else F.mse_loss(x_recon, x) * self.l1_weight
        perceptual_loss = self.perceptual_model(x, x_recon).mean() * self.perceptual_weight

        loss_dict = {
            "train/perceptual_loss": perceptual_loss,
            "train/recon_loss": recon_loss,
            "train/commitment_loss": vq_output["commitment_loss"],
            "train/all_commitment_loss": vq_output["all_commitment_loss"],
        }

        if sampled_prefix_k is not None:
            loss_dict["metric/sample_prefix_k"] = torch.tensor(float(sampled_prefix_k), device=x.device, dtype=torch.float32)
            loss_dict["metric/sample_prefix_is_full"] = torch.tensor(float(sampled_prefix_k == len(cum_latents)), device=x.device, dtype=torch.float32)
            prefix_scales_dbg = self._sanitize_1based_scales(self.coarse_prefix_scales, len(cum_latents))
            if len(cum_latents) not in prefix_scales_dbg:
                prefix_scales_dbg = sorted(prefix_scales_dbg + [len(cum_latents)])
            full_idx = prefix_scales_dbg.index(len(cum_latents))
            loss_dict["metric/sample_prefix_full_prob"] = sampled_prefix_probs[full_idx].detach()

        # -------------------- DINO distillation --------------------
        if self.dino_projector is not None and self.dino_weight > 0 and cum_latents is not None:
            if (global_step % max(1, self.dino_every)) == 0:
                if self.dino_use_cached and cached_dino_feat is not None:
                    feat_t = cached_dino_feat.to(device=x.device, dtype=torch.float32)
                    if feat_t.ndim > 2:
                        feat_t = feat_t.mean(dim=1)
                else:
                    feat_t = self._extract_dino_feat(x)

                if self.dino_scales is not None:
                    dino_scales = self._sanitize_1based_scales(self.dino_scales, len(cum_latents))
                elif self.coarse_prefix_scales is not None:
                    dino_scales = self._sanitize_1based_scales(self.coarse_prefix_scales, len(cum_latents))
                else:
                    k = int(self.dino_max_scale)
                    if k <= 0:
                        k = len(cum_latents)
                    k = max(1, min(k, len(cum_latents)))
                    dino_scales = list(range(1, k + 1))

                feat_t = F.normalize(feat_t.detach(), dim=-1)
                dino_loss_total = torch.zeros((), device=x.device, dtype=torch.float32)
                dino_cos_total = torch.zeros((), device=x.device, dtype=torch.float32)
                dino_wsum = 0.0

                for j, kk in enumerate(dino_scales):
                    z_pref = cum_latents[kk - 1]
                    feat_s = self.dino_projector(z_pref)
                    feat_s = F.normalize(feat_s, dim=-1)
                    dino_cos_k = (feat_s * feat_t).sum(dim=-1).mean()
                    dino_loss_k = (1.0 - dino_cos_k)
                    w_k = float(self.dino_scale_decay ** j)
                    dino_loss_total = dino_loss_total + w_k * dino_loss_k
                    dino_cos_total = dino_cos_total + w_k * dino_cos_k
                    dino_wsum += w_k
                    loss_dict[f"metric/dino_cos_k{kk}"] = dino_cos_k.detach()

                dino_loss = dino_loss_total / max(dino_wsum, 1e-8)
                dino_cos = dino_cos_total / max(dino_wsum, 1e-8)
                loss_dict["metric/dino_cos"] = dino_cos.detach()
                loss_dict["train/dino_loss"] = dino_loss * self.dino_weight

        # -------------------- RDVQ-style rate branch --------------------
        if self.entropy_model is not None and scale_schedule is not None:
            img_pixels = float(x.shape[-2] * x.shape[-1])
            cond_maps: List[torch.Tensor] = []
            ctx_group_ids: List[Optional[torch.Tensor]] = []
            scale_ids: List[int] = []
            y_hard_list: List[torch.Tensor] = []
            group_target_list: List[torch.Tensor] = []
            group_id_list: List[torch.Tensor] = []
            batch_list: List[int] = []

            prev_full = None

            for si, bits_si in enumerate(all_bit_indices):
                if bits_si is None:
                    continue

                B0, Ts, Hs, Ws, D = bits_si.shape
                assert D == self.args.codebook_dim

                if prev_full is None:
                    cond_pred = torch.zeros((B0, D, Hs, Ws), device=x.device, dtype=torch.float32)
                else:
                    cond_pred = F.interpolate(prev_full, size=(Hs, Ws), mode="area")

                cond_bt = cond_pred[:, None].expand(-1, Ts, -1, -1, -1).contiguous().view(B0 * Ts, D, Hs, Ws)
                bits_hard = bits_si.to(dtype=torch.float32)

                if pre_quant_list is None or si >= len(pre_quant_list) or pre_quant_list[si] is None:
                    raise RuntimeError(
                        "pre_quant_list is required for RDVQ soft group-rate training. "
                        "Call MultiScaleBSQ with return_scale_data=True."
                    )
                y_soft, y_hard = self._make_soft_bits(bits_hard, pre_quant_list[si].to(dtype=torch.float32))

                group_ids = self._bits_to_group_ids(y_hard)
                group_probs = self._soft_bits_to_group_probs(y_soft)
                ctx_ids = self._shift_group_ids_raster(group_ids, self.entropy_model.vocab_size)

                cond_maps.append(cond_bt)
                ctx_group_ids.append(ctx_ids)
                scale_ids.append(si)
                y_hard_list.append(y_hard)
                group_target_list.append(group_probs)
                group_id_list.append(group_ids)
                batch_list.append(B0)

                if quantized_full_list is not None and si < len(quantized_full_list):
                    q_full = quantized_full_list[si]
                    prev_full = q_full if prev_full is None else (prev_full + q_full)

            if len(cond_maps) > 0:
                _, sym_logits_list = self.entropy_model(
                    cond_maps=cond_maps,
                    scale_ids=scale_ids,
                    ctx_group_ids=ctx_group_ids,
                )
            else:
                sym_logits_list = []

            symbol_bits_total = torch.zeros((), device=x.device, dtype=torch.float32)
            for li, si in enumerate(scale_ids):
                sym_logits = sym_logits_list[li]
                group_probs = group_target_list[li]
                y_hard = y_hard_list[li]
                B0 = batch_list[li]

                symbol_bits = self._soft_group_ce_bits_from_logits(sym_logits, group_probs, reduce="sum") / float(B0)
                symbol_bits_total = symbol_bits_total + symbol_bits

                loss_dict[f"metric/rate_symbol_bits_s{si+1}"] = symbol_bits.detach()
                loss_dict[f"metric/group_id_mean_s{si+1}"] = group_id_list[li].to(dtype=torch.float32).mean().detach()
                loss_dict[f"metric/group_target_entropy_s{si+1}"] = (
                    -(group_probs.clamp_min(1e-8) * torch.log2(group_probs.clamp_min(1e-8))).sum(dim=2).mean()
                ).detach()
                loss_dict[f"metric/bit_target_mean_s{si+1}"] = y_hard.mean().detach()

            bpp = symbol_bits_total / img_pixels
            rate_loss = bpp * self.rate_lambda
            loss_dict["metric/bpp_pred"] = bpp.detach()
            loss_dict["metric/bpp_pred_weighted"] = bpp.detach()
            loss_dict["metric/rate_symbol_loss"] = rate_loss.detach()
            loss_dict["train/rate_loss"] = rate_loss


        # -------------------- GAN --------------------
        disc_factor = adopt_weight(global_step, threshold=self.args.discriminator_iter_start, warmup=self.args.disc_warmup)
        if self.image_gan_weight > 0 and image_disc is not None:
            logits_image_fake = image_disc(x_recon)
            g_image_loss = -torch.mean(logits_image_fake) * self.image_gan_weight * disc_factor
            loss_dict["train/g_image_loss"] = g_image_loss

        return x_recon.detach(), x.detach(), x_recon.detach(), loss_dict

    @torch.no_grad()
    def reconstruct_prefix_scales(self, x: torch.Tensor, prefix_scales=(2, 3, 4, 5)):
        self.eval()
        assert x.ndim == 4
        enc_dtype = ptdtype[self.args.encoder_dtype]
        with torch.amp.autocast("cuda", dtype=enc_dtype):
            h = self.encoder(x, return_hidden=False)
        h = h.to(dtype=torch.float32)

        z, _, _, _, _, quantized_full_list, _ = self.quantizer(h, return_scale_data=True)
        x_full = self.decoder(z)

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
        parser.add_argument("--num_res_blocks", type=int, default=2)
        parser.add_argument("--encoder_ch_mult", type=int, nargs="+", default=[1, 1, 2, 2, 4])
        parser.add_argument("--decoder_ch_mult", type=int, nargs="+", default=[1, 1, 2, 2, 4])

        # RD / prior
        parser.add_argument("--use_group_rate", action="store_true")
        parser.add_argument("--group_size", type=int, default=4)
        parser.add_argument("--entropy_hidden", type=int, default=256)
        parser.add_argument("--entropy_resblocks", type=int, default=4)
        parser.add_argument("--entropy_heads", type=int, default=8)
        parser.add_argument("--entropy_mlp_ratio", type=float, default=4.0)
        parser.add_argument("--entropy_dropout", type=float, default=0.0)
        parser.add_argument("--entropy_max_scales", type=int, default=32)
        parser.add_argument("--entropy_use_pos2d", action="store_true", default=True)
        parser.add_argument("--no_entropy_use_pos2d", dest="entropy_use_pos2d", action="store_false")
        parser.add_argument("--entropy_checkpoint_blocks", action="store_true")
        parser.add_argument("--entropy_cond_norm", action="store_true", default=True)
        parser.add_argument("--no_entropy_cond_norm", dest="entropy_cond_norm", action="store_false")
        parser.add_argument("--prior_base_slices", type=int, default=2)
        parser.add_argument("--prior_max_slices", type=int, default=8)

        parser.add_argument("--coarse_prefix_sample", action="store_true")
        parser.add_argument("--coarse_prefix_full_prob", type=float, default=0.5)

        parser.add_argument("--dino_weight", type=float, default=0.0)
        parser.add_argument("--dino_max_scale", type=int, default=6)
        parser.add_argument("--dino_every", type=int, default=4)
        parser.add_argument("--dino_use_cached", action="store_true")
        parser.add_argument("--dino_cache_key", type=str, default="dino_feat")
        parser.add_argument("--dino_scales", type=int, nargs="+", default=None)
        parser.add_argument("--dino_scale_decay", type=float, default=0.7)
        parser.add_argument("--dino_backend", type=str, default="auto", choices=["auto", "hf", "hub", "timm"])
        parser.add_argument("--dino_model", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m")
        parser.add_argument("--dino_weights", type=str, default=None)
        parser.add_argument("--dino_repo_dir", type=str, default="")
        parser.add_argument("--dino_input_size", type=int, default=256)
        parser.add_argument("--dino_input_is_01", action="store_true")
        parser.add_argument("--dino_feat_dim", type=int, default=768)
        parser.add_argument("--dino_proj_hidden", type=int, default=256)
        parser.add_argument("--dino_proj_mlp_hidden", type=int, default=512)
        parser.add_argument("--dino_norm_type", type=str, default="group")
        parser.add_argument("--dino_amp", action="store_true")
        parser.add_argument("--dino_teacher_on_cpu", action="store_true")
        return parser
