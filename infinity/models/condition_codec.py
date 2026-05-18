from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


def _image_to_luma(
    img_B3HW: torch.Tensor,
    out_size: int = 128,
) -> torch.Tensor:
    if img_B3HW.ndim != 4 or img_B3HW.shape[1] != 3:
        raise ValueError(f"Expected image tensor [B,3,H,W], got {tuple(img_B3HW.shape)}")
    x = (img_B3HW.float() + 1.0) * 0.5
    x = x.clamp_(0.0, 1.0)
    gray = x[:, 0:1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114
    if out_size and (gray.shape[-2] != out_size or gray.shape[-1] != out_size):
        gray = F.interpolate(gray, size=(out_size, out_size), mode="area")
    return gray


def image_to_luma_boundary(
    img_B3HW: torch.Tensor,
    out_size: int = 128,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fast fixed boundary extractor used as encoder-side side information.

    The input image is expected in Infinity's usual [-1, 1] range.  The output
    is a soft edge map in [0, 1] at a small fixed resolution so the condition
    codec stays cheap.
    """
    gray = _image_to_luma(img_B3HW, out_size=out_size)

    kx = gray.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    ky = gray.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx.square() + gy.square() + eps)
    denom = mag.flatten(1).quantile(0.95, dim=1).clamp_min(eps).view(-1, 1, 1, 1)
    return (mag / denom).clamp_(0.0, 1.0)


def image_to_luma_depth(
    img_B3HW: torch.Tensor,
    out_size: int = 128,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Dependency-free low-frequency depth/layout proxy.

    Real monocular depth models are better when precomputed offline, but they
    are too expensive and dependency-heavy to run inside every training step.
    This proxy keeps the same one-channel codec contract while emphasizing
    smooth scene layout instead of high-frequency edges, which is much more
    rate-friendly at ultra-low side bitrates.
    """
    gray = _image_to_luma(img_B3HW, out_size=out_size)
    tiny = max(2, int(out_size) // 8)
    low = F.interpolate(gray, size=(tiny, tiny), mode="area")
    low = F.interpolate(low, size=(int(out_size), int(out_size)), mode="bilinear", align_corners=False)

    inv_luma = 1.0 - low
    flat = inv_luma.flatten(1)
    lo = flat.amin(dim=1).view(-1, 1, 1, 1)
    hi = flat.amax(dim=1).view(-1, 1, 1, 1)
    inv_luma = (inv_luma - lo) / (hi - lo).clamp_min(eps)

    detail = (gray - low).abs()
    detail_denom = detail.flatten(1).quantile(0.95, dim=1).clamp_min(eps).view(-1, 1, 1, 1)
    flatness = 1.0 - (detail / detail_denom).clamp_(0.0, 1.0)
    depth = 0.85 * inv_luma + 0.15 * flatness
    depth = F.avg_pool2d(depth, kernel_size=3, stride=1, padding=1)
    return depth.clamp_(0.0, 1.0)


def normalize_depth_B1HW(depth_B1HW: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if depth_B1HW.ndim == 3:
        depth_B1HW = depth_B1HW.unsqueeze(1)
    if depth_B1HW.ndim != 4 or depth_B1HW.shape[1] != 1:
        raise ValueError(f"Expected depth tensor [B,1,H,W], got {tuple(depth_B1HW.shape)}")
    depth = depth_B1HW.float()
    flat = depth.flatten(1)
    lo = flat.amin(dim=1).view(-1, 1, 1, 1)
    hi = flat.amax(dim=1).view(-1, 1, 1, 1)
    return ((depth - lo) / (hi - lo).clamp_min(eps)).clamp_(0.0, 1.0)


class TransformersDepthExtractor:
    """Small frozen monocular depth extractor for online encoder-side condition maps."""

    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
        device: Optional[torch.device | str] = None,
        dtype: str = "fp16",
        cache_dir: str = "",
    ):
        self.model_name = str(model_name)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = str(dtype or "fp16").lower()
        if dtype in {"fp16", "float16", "half"} and self.device.type == "cuda":
            self.dtype = torch.float16
        elif dtype in {"bf16", "bfloat16"} and self.device.type == "cuda":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
        self.cache_dir = str(cache_dir or "")
        self.processor = None
        self.model = None

    def _load(self):
        if self.model is not None:
            return
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except Exception as exc:
            raise ImportError(
                "Online depth condition requires transformers with Depth Anything V2 support. "
                "Install/upgrade transformers or use --depth_condition_source proxy."
            ) from exc

        kwargs = {"cache_dir": self.cache_dir} if self.cache_dir else {}
        self.processor = AutoImageProcessor.from_pretrained(self.model_name, **kwargs)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_name, **kwargs)
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, img_B3HW: torch.Tensor, out_size: Optional[int | Tuple[int, int]] = None) -> torch.Tensor:
        self._load()
        if img_B3HW.ndim != 4 or img_B3HW.shape[1] != 3:
            raise ValueError(f"Expected image tensor [B,3,H,W], got {tuple(img_B3HW.shape)}")
        x = (img_B3HW.detach().float().clamp(-1.0, 1.0) + 1.0) * 127.5
        images = x.round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        inputs = self.processor(images=list(images), return_tensors="pt")
        inputs = {
            k: (v.to(device=self.device, dtype=self.dtype) if k == "pixel_values" else v.to(device=self.device))
            for k, v in inputs.items()
        }
        output = self.model(**inputs)
        depth = output.predicted_depth.unsqueeze(1).float()
        if out_size is None or out_size == 0:
            size = tuple(int(x) for x in img_B3HW.shape[-2:])
        elif isinstance(out_size, int):
            size = (int(out_size), int(out_size))
        else:
            size = (int(out_size[0]), int(out_size[1]))
        if depth.shape[-2:] != size:
            depth = F.interpolate(depth, size=size, mode="bicubic", align_corners=False)
        return normalize_depth_B1HW(depth).to(device=img_B3HW.device)


class TransformersSAMSegmentationExtractor:
    """Optional online SAM mask generator reduced to a one-channel condition map."""

    def __init__(
        self,
        model_name: str = "facebook/sam-vit-base",
        device: Optional[torch.device | str] = None,
        dtype: str = "fp16",
        cache_dir: str = "",
        output_mode: str = "region_boundary",
        max_masks: int = 16,
        points_per_batch: int = 32,
    ):
        self.model_name = str(model_name)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.requested_dtype = str(dtype or "fp32").lower()
        self.dtype = torch.float32
        self.cache_dir = str(cache_dir or "")
        self.output_mode = str(output_mode or "region_boundary").lower()
        self.max_masks = int(max(1, max_masks))
        self.points_per_batch = int(max(1, points_per_batch))
        self.pipe = None

    def _pipeline_device(self):
        if self.device.type != "cuda":
            return -1
        return 0 if self.device.index is None else int(self.device.index)

    def _load(self):
        if self.pipe is not None:
            return
        try:
            from transformers import pipeline
        except Exception as exc:
            raise ImportError(
                "Online SAM/segmentation condition requires transformers mask-generation pipeline. "
                "Install/upgrade transformers or choose another spatial_cond_type."
            ) from exc

        kwargs = dict(task="mask-generation", model=self.model_name, device=self._pipeline_device())
        if self.cache_dir:
            kwargs["model_kwargs"] = {"cache_dir": self.cache_dir}
        try:
            self.pipe = pipeline(**kwargs)
        except TypeError:
            kwargs.pop("model_kwargs", None)
            self.pipe = pipeline(**kwargs)

    @staticmethod
    def _mask_to_region_boundary(mask, height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = arr[..., 0]
        mt = torch.from_numpy((arr.astype(np.float32) > 0.5).astype(np.float32)).view(1, 1, arr.shape[0], arr.shape[1])
        if mt.shape[-2:] != (height, width):
            mt = F.interpolate(mt, size=(height, width), mode="nearest")
        dilated = F.max_pool2d(mt, kernel_size=3, stride=1, padding=1)
        eroded = 1.0 - F.max_pool2d(1.0 - mt, kernel_size=3, stride=1, padding=1)
        boundary = (dilated - eroded).clamp_(0.0, 1.0)
        return mt.clamp_(0.0, 1.0), boundary

    @torch.no_grad()
    def __call__(self, img_B3HW: torch.Tensor, out_size: Optional[int | Tuple[int, int]] = None) -> torch.Tensor:
        self._load()
        if img_B3HW.ndim != 4 or img_B3HW.shape[1] != 3:
            raise ValueError(f"Expected image tensor [B,3,H,W], got {tuple(img_B3HW.shape)}")
        if out_size is None or out_size == 0:
            size = tuple(int(x) for x in img_B3HW.shape[-2:])
        elif isinstance(out_size, int):
            size = (int(out_size), int(out_size))
        else:
            size = (int(out_size[0]), int(out_size[1]))

        x = (img_B3HW.detach().float().clamp(-1.0, 1.0) + 1.0) * 127.5
        images = x.round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        cond_maps = []
        for image in images:
            height, width = int(image.shape[0]), int(image.shape[1])
            pil_image = Image.fromarray(image).convert("RGB")
            try:
                out = self.pipe(pil_image, points_per_batch=self.points_per_batch)
            except TypeError:
                out = self.pipe(pil_image)
            masks = out.get("masks", []) if isinstance(out, dict) else []
            region = torch.zeros(1, 1, height, width)
            boundary = torch.zeros_like(region)
            for mask in list(masks)[:self.max_masks]:
                m_region, m_boundary = self._mask_to_region_boundary(mask, height, width)
                region = torch.maximum(region, m_region)
                boundary = torch.maximum(boundary, m_boundary)
            if not masks:
                fallback_img = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
                fallback = image_to_luma_boundary(fallback_img, out_size=0)
                region = torch.zeros_like(fallback)
                boundary = fallback
            if self.output_mode in {"region", "mask", "masks"}:
                cond = region
            elif self.output_mode in {"boundary", "edge", "edges"}:
                cond = boundary
            else:
                cond = (0.65 * region + 0.35 * boundary).clamp_(0.0, 1.0)
            if cond.shape[-2:] != size:
                cond = F.interpolate(cond, size=size, mode="area")
            cond_maps.append(cond)
        return torch.cat(cond_maps, dim=0).to(device=img_B3HW.device)


def image_to_spatial_condition(
    img_B3HW: torch.Tensor,
    cond_type: str = "boundary",
    out_size: int = 128,
) -> torch.Tensor:
    cond_type = str(cond_type or "boundary").lower()
    if cond_type in {"boundary", "edge", "edges"}:
        return image_to_luma_boundary(img_B3HW, out_size=out_size)
    if cond_type in {"depth", "depth_proxy", "depth_model", "depth_anything", "layout"}:
        return image_to_luma_depth(img_B3HW, out_size=out_size)
    if cond_type in {"sam", "seg", "segmentation", "segment"}:
        return image_to_luma_boundary(img_B3HW, out_size=out_size)
    raise ValueError(f"Unsupported spatial condition type {cond_type!r}; use 'boundary', 'depth', or 'sam'.")


def var_token_condition_from_map(
    vae,
    condition_B1HW: torch.Tensor,
    scale_schedule: List[Tuple[int, int, int]],
    num_scales: int = 2,
    image_hw: Optional[Tuple[int, int]] = None,
) -> dict:
    """Encode a condition map with the frozen VAR/VAE tokenization path.

    This intentionally mirrors Infinity's image codec: the condition is first
    represented as LFQ/BSQ bit tokens over the existing multiscale schedule, and
    only the first few coarse scales are transmitted.  With a 16-bit LFQ and the
    default 1x1+2x2 scales this costs 80 raw bits per image, which is about
    7.6e-5 bpp for a 1024x1024 image.
    """
    if condition_B1HW.ndim != 4 or condition_B1HW.shape[1] != 1:
        raise ValueError(f"Expected condition tensor [B,1,H,W], got {tuple(condition_B1HW.shape)}")
    if not scale_schedule:
        raise ValueError("scale_schedule must not be empty")
    num_scales = int(max(1, min(num_scales, len(scale_schedule))))
    cond = condition_B1HW.float().clamp(0.0, 1.0)
    cond_img = cond.repeat(1, 3, 1, 1).mul(2.0).sub(1.0)

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
        raw_features, _, _ = vae.encode_for_raw_features(cond_img, scale_schedule=scale_schedule)
        B = raw_features.shape[0]
        codes_out = raw_features.unsqueeze(2) if raw_features.dim() == 4 else raw_features
        target_size = tuple(int(x) for x in scale_schedule[-1])
        cum_codes = torch.zeros_like(codes_out)
        bit_indices = []
        for si, pn in enumerate(scale_schedule[:num_scales]):
            pn = tuple(int(x) for x in pn)
            residual = codes_out - cum_codes
            if pn != target_size:
                residual = F.interpolate(residual, size=pn, mode=vae.quantizer.z_interplote_down).contiguous()
            quantized, _, bits, _ = vae.quantizer.lfq(residual)
            bit_indices.append(bits.detach())
            cum_codes = cum_codes + F.interpolate(quantized, size=target_size, mode=vae.quantizer.z_interplote_up).contiguous()

    features = cum_codes.squeeze(-3) if cum_codes.dim() == 5 else cum_codes
    raw_bits = sum(int(x[0].numel()) for x in bit_indices)
    raw_bits_t = features.new_full((features.shape[0],), float(raw_bits))
    if image_hw is None:
        pixels = float(max(1, int(condition_B1HW.shape[-2]) * int(condition_B1HW.shape[-1])))
    else:
        pixels = float(max(1, int(image_hw[0]) * int(image_hw[1])))
    side_bpp = raw_bits_t / pixels

    return {
        "features": features,
        "bit_indices": bit_indices,
        "side_bits_per_image": raw_bits_t,
        "side_bpp": side_bpp,
        "hard_side_bpp": side_bpp,
        "expected_side_bpp": side_bpp,
        "hard_expected_side_bpp": side_bpp,
        "rate_nats_per_image": raw_bits_t * math.log(2.0),
        "recon_loss": features.new_tensor(0.0),
        "latent_shape": [tuple(x.shape[-4:]) for x in bit_indices],
        "num_scales": num_scales,
    }


class BoundaryConditionCodec(nn.Module):
    """Tiny binary side codec for low-resolution boundary maps.

    This is deliberately much smaller than the image VAR.  It only needs to
    transmit structural hints, so a binary bottleneck plus a factorized learned
    Bernoulli prior is a good first rate-efficient baseline.
    """

    def __init__(
        self,
        input_size: int = 128,
        hidden_dim: int = 48,
        latent_dim: int = 12,
        feature_dim: int = 32,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.feature_dim = int(feature_dim)
        if self.input_size % 4 != 0:
            raise ValueError(f"BoundaryConditionCodec input_size must be divisible by 4, got {self.input_size}")

        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, latent_dim, kernel_size=3, padding=1),
        )
        self.prior_logits = nn.Parameter(torch.zeros(1, latent_dim, 1, 1))
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(hidden_dim, feature_dim, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.boundary_head = nn.Conv2d(feature_dim, 1, kernel_size=1)

    def forward(
        self,
        boundary_B1HW: torch.Tensor,
        image_hw: Optional[Tuple[int, int]] = None,
    ) -> dict:
        if boundary_B1HW.ndim != 4 or boundary_B1HW.shape[1] != 1:
            raise ValueError(f"Expected boundary tensor [B,1,H,W], got {tuple(boundary_B1HW.shape)}")
        B = boundary_B1HW.shape[0]
        if boundary_B1HW.shape[-2:] != (self.input_size, self.input_size):
            boundary = F.interpolate(boundary_B1HW.float(), size=(self.input_size, self.input_size), mode="area")
        else:
            boundary = boundary_B1HW.float()
        boundary = boundary.clamp(0.0, 1.0)

        z_logits = self.encoder(boundary)
        z_prob = torch.sigmoid(z_logits)
        z_hard = (z_prob >= 0.5).to(dtype=z_prob.dtype)
        z_st = z_prob + (z_hard - z_prob).detach()

        prior = self.prior_logits.expand_as(z_logits)
        expected_nats = (
            z_prob * F.softplus(-prior) + (1.0 - z_prob) * F.softplus(prior)
        ).flatten(1).sum(dim=1)
        hard_nats = F.binary_cross_entropy_with_logits(prior, z_hard.detach(), reduction="none").flatten(1).sum(dim=1)

        features = self.decoder(z_st)
        recon_logits = self.boundary_head(features)
        recon_loss = F.binary_cross_entropy_with_logits(recon_logits, boundary)

        if image_hw is None:
            pixels = float(self.input_size * self.input_size)
        else:
            pixels = float(max(1, int(image_hw[0]) * int(image_hw[1])))
        side_bpp = expected_nats / math.log(2.0) / pixels
        hard_side_bpp = hard_nats / math.log(2.0) / pixels

        return {
            "features": features,
            "recon_logits": recon_logits,
            "recon_loss": recon_loss,
            "rate_nats_per_image": expected_nats,
            "side_bpp": side_bpp,
            "hard_side_bpp": hard_side_bpp,
            "z_hard": z_hard.detach(),
            "prior_logits": prior.detach(),
            "latent_shape": tuple(z_st.shape[-3:]),
        }


class BoundarySpatialAdapter(nn.Module):
    """Zero-init spatial residual adapters for Infinity token hidden states."""

    def __init__(
        self,
        feature_dim: int,
        embed_dim: int,
        depth: int,
        adapter_init: str = "shared",
        max_scales: int = 1,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        adapter_init = str(adapter_init or "shared").lower().replace("-", "_")
        adapter_init_aliases = {
            "shared": "shared",
            "share": "shared",
            "all_shared": "shared",
            "per_scale": "per_scale_zero",
            "per_scale_zero": "per_scale_zero",
            "scale": "per_scale_zero",
            "scale_zero": "per_scale_zero",
            "separate": "per_scale_zero",
        }
        if adapter_init not in adapter_init_aliases:
            raise ValueError(
                f"Unsupported adapter_init={adapter_init!r}; use 'shared' or 'per_scale_zero'."
            )
        self.adapter_init = adapter_init_aliases[adapter_init]
        self.max_scales = int(max(1, max_scales))
        if self.adapter_init == "shared":
            self.proj = nn.ModuleList([nn.Linear(feature_dim, embed_dim) for _ in range(depth)])
        else:
            self.proj = nn.ModuleList([
                nn.ModuleList([nn.Linear(feature_dim, embed_dim) for _ in range(depth)])
                for _ in range(self.max_scales)
            ])
        self._init_zero()

    def _init_zero(self):
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def _scale_index(self, scale_id: int) -> int:
        scale_id = int(scale_id)
        if scale_id < 0 or scale_id >= self.max_scales:
            raise IndexError(f"scale_id={scale_id} is outside condition adapter max_scales={self.max_scales}")
        return scale_id

    @staticmethod
    def _scale_token_count(pn: Tuple[int, int, int]) -> int:
        pt, ph, pw = pn
        return int(pt) * int(ph) * int(pw)

    def make_tokens(
        self,
        features_BCHW: torch.Tensor,
        scale_schedule: List[Tuple[int, int, int]],
        need_to_pad: int = 0,
    ) -> torch.Tensor:
        tokens = []
        for pt, ph, pw in scale_schedule:
            feat = F.interpolate(features_BCHW.float(), size=(ph, pw), mode="bilinear", align_corners=False)
            feat = feat.flatten(2).transpose(1, 2).contiguous()
            if pt != 1:
                feat = feat[:, None].expand(-1, int(pt), -1, -1).reshape(feat.shape[0], int(pt) * ph * pw, feat.shape[-1])
            tokens.append(feat)
        out = torch.cat(tokens, dim=1)
        if need_to_pad:
            out = F.pad(out, (0, 0, 0, need_to_pad))
        return out

    def make_scale_tokens(
        self,
        features_BCHW: torch.Tensor,
        pn: Tuple[int, int, int],
        need_to_pad: int = 0,
    ) -> torch.Tensor:
        pt, ph, pw = pn
        feat = F.interpolate(features_BCHW.float(), size=(ph, pw), mode="bilinear", align_corners=False)
        feat = feat.flatten(2).transpose(1, 2).contiguous()
        if pt != 1:
            feat = feat[:, None].expand(-1, int(pt), -1, -1).reshape(feat.shape[0], int(pt) * ph * pw, feat.shape[-1])
        if need_to_pad:
            feat = F.pad(feat, (0, 0, 0, need_to_pad))
        return feat

    def forward(
        self,
        condition_tokens_BLF: torch.Tensor,
        block_idx: int,
        dtype: Optional[torch.dtype] = None,
        scale_schedule: Optional[List[Tuple[int, int, int]]] = None,
        scale_id: Optional[int] = None,
    ) -> torch.Tensor:
        idx = int(max(0, min(block_idx, self.depth - 1)))
        if self.adapter_init == "shared":
            delta = self.proj[idx](condition_tokens_BLF)
            return delta if dtype is None else delta.to(dtype=dtype)

        if scale_id is not None:
            delta = self.proj[self._scale_index(scale_id)][idx](condition_tokens_BLF)
            return delta if dtype is None else delta.to(dtype=dtype)

        if not scale_schedule:
            delta = self.proj[0][idx](condition_tokens_BLF)
            return delta if dtype is None else delta.to(dtype=dtype)

        pieces = []
        ptr = 0
        token_len = condition_tokens_BLF.shape[1]
        for si, pn in enumerate(scale_schedule):
            next_ptr = ptr + self._scale_token_count(pn)
            end = min(next_ptr, token_len)
            if end > ptr:
                pieces.append(self.proj[self._scale_index(si)][idx](condition_tokens_BLF[:, ptr:end]))
            ptr = next_ptr
            if ptr >= token_len:
                break
        if ptr < token_len:
            pad_tokens = condition_tokens_BLF[:, ptr:]
            pieces.append(pad_tokens.new_zeros((*pad_tokens.shape[:-1], self.embed_dim)))
        delta = torch.cat(pieces, dim=1) if pieces else condition_tokens_BLF.new_zeros(
            (*condition_tokens_BLF.shape[:-1], self.embed_dim)
        )
        return delta if dtype is None else delta.to(dtype=dtype)
