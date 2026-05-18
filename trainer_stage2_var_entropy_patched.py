import random
import time
import gc
import os
import sys
from functools import partial
from pprint import pformat
from typing import List, Optional, Tuple, Union
import os.path as osp

import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullOptimStateDictConfig, FullStateDictConfig, StateDictType
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import numpy as np
import torch.distributed as tdist
from torch.amp import autocast
import cv2

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.append(_THIS_DIR)

import infinity.utils.dist as dist
from infinity.models import InfinityPatched as Infinity
from infinity.models.condition_codec import (
    TransformersDepthExtractor,
    TransformersSAMSegmentationExtractor,
    image_to_spatial_condition,
    normalize_depth_B1HW,
    var_token_condition_from_map,
)
from infinity.models.ema import update_ema
from infinity.models.bitwise_self_correction import BitwiseSelfCorrection
from infinity.models.tiny_entropy_student import teacher_logits_to_bit_logits_per_scale
from infinity.models.same_scale_refiner import (
    compute_neighborhood_context,
    compute_normalized_uncertainty,
    single_bit_logits_to_pair_logits,
    split_flat_logits_to_bit_logits_per_scale,
)
from infinity.utils import arg_util, misc, swanlab_utils
from infinity.utils.amp_opt import AmpOptimizer
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
fulloptstate_save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)

class InfinityTrainer(object):
    def __init__(
        self, is_visualizer: bool, device, raw_scale_schedule: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local, gpt_wo_ddp: Infinity, gpt: DDP, ema_ratio: float, max_it: int,
        gpt_opt: AmpOptimizer, label_smooth: float, z_loss_ratio: float, eq_loss: int, xen: bool,
        dbg_unused=False,zero=0, vae_type=True, reweight_loss_by_scale=False,
        gpt_wo_ddp_ema=None, gpt_ema=None, use_fsdp_model_ema=False, student_wo_ddp=None, student=None, student_opt=None,
        same_scale_refiner_wo_ddp=None, same_scale_refiner=None, same_scale_refiner_opt=None, other_args=None,
    ):
        super(InfinityTrainer, self).__init__()
        self.dbg_unused = dbg_unused
        
        self.zero = zero
        self.vae_type = vae_type
        
        self.gpt: Union[DDP, FSDP, nn.Module]
        self.gpt, self.vae_local, self.quantize_local = gpt, vae_local, vae_local.quantize
        self.device = device
        self.gpt_opt: AmpOptimizer = gpt_opt
        self.gpt_wo_ddp: Union[Infinity, torch._dynamo.eval_frame.OptimizedModule] = gpt_wo_ddp  # after torch.compile
        self.gpt_wo_ddp_ema = gpt_wo_ddp_ema
        self.gpt_ema = gpt_ema
        self.bitwise_self_correction = BitwiseSelfCorrection(self.vae_local, other_args)
        self.use_fsdp_model_ema = use_fsdp_model_ema
        self.batch_size, self.seq_len = 0, 0
        self.seq_len_each = []
        self.reweight_loss_by_scale = reweight_loss_by_scale
        print(f'self.reweight_loss_by_scale: {self.reweight_loss_by_scale}')
        self.student_wo_ddp = student_wo_ddp
        self.student = student
        self.student_opt = student_opt
        self.enable_student_entropy = bool(getattr(other_args, 'enable_student_entropy', 0)) and (student_wo_ddp is not None)
        self.student_start_step = int(getattr(other_args, 'student_start_step', 0))
        self.student_start_scale = int(getattr(other_args, 'student_start_scale', 1))
        self.student_kd_ratio = float(getattr(other_args, 'student_kd_ratio', 1.0))
        self.student_gt_ratio = float(getattr(other_args, 'student_gt_ratio', 1.0))
        self.student_grad_clip = float(getattr(other_args, 'student_grad_clip', 1.0))
        self.student_codebook_dim = int(getattr(self.gpt_wo_ddp, 'V', self.vae_local.codebook_dim * 2) // 2) if self.enable_student_entropy else 0
        if self.enable_student_entropy:
            print(f'[student] start_step={self.student_start_step}, start_scale={self.student_start_scale}, kd={self.student_kd_ratio}, gt={self.student_gt_ratio}, grad_clip={self.student_grad_clip}')

        self.max_it = int(max_it)
        self.same_scale_refiner_wo_ddp = same_scale_refiner_wo_ddp
        self.same_scale_refiner = same_scale_refiner
        self.same_scale_refiner_opt = same_scale_refiner_opt
        self.enable_same_scale_refiner = bool(getattr(other_args, 'enable_same_scale_refiner', 1)) and (same_scale_refiner_wo_ddp is not None)
        self.same_scale_start_step = int(getattr(other_args, 'same_scale_start_step', 0))
        self.same_scale_refine_ratio = float(getattr(other_args, 'same_scale_refine_ratio', 1.0))
        self.same_scale_safety_ratio = float(getattr(other_args, 'same_scale_safety_ratio', 0.20))
        self.same_scale_reg_ratio = float(getattr(other_args, 'same_scale_reg_ratio', 0.01))
        self.same_scale_safety_margin = float(getattr(other_args, 'same_scale_safety_margin', 0.0))
        self.same_scale_refine_alpha = float(getattr(other_args, 'same_scale_refine_alpha', 1.0))
        self.same_scale_refiner_grad_clip = float(getattr(other_args, 'same_scale_refiner_grad_clip', 1.0))
        self.same_scale_detach_firstpass_input = bool(int(getattr(other_args, 'same_scale_detach_firstpass_input', 1)))
        if self.enable_same_scale_refiner:
            print(
                f'[same_scale] start_step={self.same_scale_start_step}, cal={self.same_scale_refine_ratio}, '
                f'safe={self.same_scale_safety_ratio}, reg={self.same_scale_reg_ratio}, '
                f'alpha={self.same_scale_refine_alpha}, grad_clip={self.same_scale_refiner_grad_clip}'
            )
        self.enable_boundary_condition = bool(getattr(other_args, 'enable_boundary_condition', 0))
        self.spatial_cond_type = str(getattr(other_args, 'spatial_cond_type', 'boundary') or 'boundary').lower()
        self.condition_codec_type = str(getattr(other_args, 'condition_codec_type', 'binary') or 'binary').lower()
        self.condition_token_scales = int(getattr(other_args, 'condition_token_scales', 2))
        self.condition_token_scales_min = int(getattr(other_args, 'condition_token_scales_min', 0) or 0)
        condition_adapter = getattr(
            self.gpt_wo_ddp._orig_mod if hasattr(self.gpt_wo_ddp, '_orig_mod') else self.gpt_wo_ddp,
            'condition_adapter',
            None,
        )
        self.condition_adapter_init = str(getattr(condition_adapter, 'adapter_init', getattr(other_args, 'condition_adapter_init', 'shared')) or 'shared')
        self.condition_adapter_max_scales = int(getattr(condition_adapter, 'max_scales', getattr(other_args, 'condition_adapter_max_scales', 0)) or 0)
        self.depth_condition_source = str(getattr(other_args, 'depth_condition_source', 'proxy') or 'proxy').lower()
        self.depth_model_name = str(getattr(other_args, 'depth_model_name', 'depth-anything/Depth-Anything-V2-Small-hf'))
        self.depth_model_dtype = str(getattr(other_args, 'depth_model_dtype', 'fp16'))
        self.depth_model_device = str(getattr(other_args, 'depth_model_device', '') or device)
        self.depth_model_cache_dir = str(getattr(other_args, 'depth_model_cache_dir', '') or '')
        self.depth_extractor = None
        self.seg_condition_source = str(getattr(other_args, 'seg_condition_source', 'transformers') or 'transformers').lower()
        self.seg_model_name = str(getattr(other_args, 'seg_model_name', 'facebook/sam-vit-base'))
        self.seg_model_dtype = str(getattr(other_args, 'seg_model_dtype', 'fp16'))
        self.seg_model_device = str(getattr(other_args, 'seg_model_device', '') or device)
        self.seg_model_cache_dir = str(getattr(other_args, 'seg_model_cache_dir', '') or '')
        self.seg_max_masks = int(getattr(other_args, 'seg_max_masks', 16))
        self.seg_points_per_batch = int(getattr(other_args, 'seg_points_per_batch', 32))
        self.seg_output_mode = str(getattr(other_args, 'seg_output_mode', 'region_boundary') or 'region_boundary')
        self.seg_extractor = None
        self.boundary_cond_size = int(getattr(other_args, 'boundary_cond_size', 128))
        self.boundary_cond_rate_ratio = float(getattr(other_args, 'boundary_cond_rate_ratio', 1.0))
        self.boundary_cond_recon_ratio = float(getattr(other_args, 'boundary_cond_recon_ratio', 0.1))
        if self.enable_boundary_condition:
            print(
                f'[spatial_condition] type={self.spatial_cond_type}, codec={self.condition_codec_type}, '
                f'adapter={self.condition_adapter_init}/{self.condition_adapter_max_scales}, '
                f'depth_source={self.depth_condition_source}, token_scales={self.condition_token_scales_min or self.condition_token_scales}-{self.condition_token_scales}, '
                f'seg_source={self.seg_condition_source}, '
                f'size={self.boundary_cond_size}, '
                f'rate={self.boundary_cond_rate_ratio}, recon={self.boundary_cond_recon_ratio}'
            )
        
        self.using_ema = ema_ratio != 0 and self.zero == 0
        self.ema_ratio = abs(ema_ratio)
        self.ema_cpu = ema_ratio < 0
        self.is_visualizer = is_visualizer
        
        gpt_uncompiled = self.gpt_wo_ddp._orig_mod if hasattr(self.gpt_wo_ddp, '_orig_mod') else self.gpt_wo_ddp
        del gpt_uncompiled.rng
        gpt_uncompiled.rng = torch.Generator(device=device)
        del gpt_uncompiled
        
        self.cached_state_not_ema = None
        if self.using_ema:
            self.pi_para_copy_for_parallel_ema = []
            all_tot = tot = 0
            for pi, para in enumerate(self.gpt_opt.paras):          # only learnable parameters need ema update
                if pi % dist.get_world_size() == dist.get_rank():   # model-parallel-style split
                    p_ema = para.data.cpu() if self.ema_cpu else para.data.clone()
                    self.pi_para_copy_for_parallel_ema.append((pi, p_ema))
                    tot += p_ema.numel()
                all_tot += para.numel()
            t = torch.zeros(dist.get_world_size())
            t[dist.get_rank()] = float(tot)
            dist.allreduce(t)
            t = [round(x) for x in t.tolist()]
            print(f'[ema tot #para] min={min(t)/1e6:.2f}, max={max(t)/1e6:.2f}, sum={sum(t)/1e6:.2f}, error={sum(t)-all_tot}')
            # lvl_1L, attn_bias_for_masking, zero_k_bias are never changed
            # check we only have these buffers so that we can skip buffer copy in ema update (only perform param update)
            assert all(any(s in name for s in ('lvl_1L', 'attn_bias_for_masking', 'zero_k_bias')) for name, _ in self.gpt_wo_ddp.named_buffers())
        else:
            self.pi_para_copy_for_parallel_ema = None
        
        self.label_smooth = label_smooth
        self.z_loss_ratio = z_loss_ratio
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='none')
        self.eq_loss = eq_loss
        
        if self.eq_loss:
            self.loss_eq_weight = torch.empty(1, self.raw_L, device=device)
            cur = 0
            for raw_pn in raw_scale_schedule:
                l = raw_pn*raw_pn
                self.loss_eq_weight[0, cur:cur+l] = 1./((raw_pn*raw_pn) if self.eq_loss == 2 else raw_pn)
                cur += l
            self.loss_eq_weight /= self.loss_eq_weight.sum()
        else:
            self.loss_eq_weight = 1.
        
        self.cmap_sim: ListedColormap = sns.color_palette('viridis', as_cmap=True)
        
        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True
        self.generator = np.random.default_rng(0)
    
    def _build_student_targets(self, gt_ms_idx_Bl, scale_schedule, vae_scale_schedule, device, need_gt_bits: bool = True):
        B = gt_ms_idx_Bl[0].shape[0]
        D = gt_ms_idx_Bl[0].shape[-1]
        prefix_maps = []
        gt_bits_bdhw = []
        cum_final = None
        for si, ((pt, ph, pw), idx_bl) in enumerate(zip(scale_schedule, gt_ms_idx_Bl)):
            if pt != 1:
                raise NotImplementedError('Tiny entropy student currently assumes image scales with pt=1.')
            bits = idx_bl.reshape(B, ph, pw, D).contiguous()
            if need_gt_bits:
                gt_bits_bdhw.append(bits.permute(0, 3, 1, 2).contiguous().float())
            if si == 0:
                prefix_maps.append(torch.zeros((B, D, ph, pw), device=device, dtype=torch.float32))
            else:
                pm = F.interpolate(cum_final, size=vae_scale_schedule[si], mode=self.vae_local.quantizer.z_interplote_up).contiguous()
                pm = pm.squeeze(2).to(dtype=torch.float32)
                if getattr(self.gpt_wo_ddp, 'apply_spatial_patchify', 0):
                    pm = F.pixel_unshuffle(pm, 2)
                prefix_maps.append(pm)
            bits_btHWD = bits.unsqueeze(1)
            q = self.vae_local.quantizer.lfq.indices_to_codes(bits_btHWD, label_type='bit_label')
            q_up_final = F.interpolate(q, size=vae_scale_schedule[-1], mode=self.vae_local.quantizer.z_interplote_up).contiguous()
            cum_final = q_up_final if cum_final is None else (cum_final + q_up_final)
        return prefix_maps, gt_bits_bdhw

    def _pool_text_summary(self, text_cond_tuple):
        if not isinstance(text_cond_tuple, tuple):
            return None
        kv_compact, lens, cu_seqlens_k, _ = text_cond_tuple
        pooled = []
        for b, le in enumerate(lens):
            start = int(cu_seqlens_k[b].item())
            end = int(cu_seqlens_k[b + 1].item())
            if end <= start:
                pooled.append(kv_compact.new_zeros(kv_compact.shape[-1]))
            else:
                pooled.append(kv_compact[start:end].mean(dim=0))
        return torch.stack(pooled, dim=0)

    def _scale_hidden_from_text_cond(self, text_cond_tuple):
        if not isinstance(text_cond_tuple, tuple):
            return None
        model = self.gpt_wo_ddp._orig_mod if hasattr(self.gpt_wo_ddp, '_orig_mod') else self.gpt_wo_ddp
        if not (hasattr(model, 'text_norm') and hasattr(model, 'text_proj_for_sos')):
            return None
        kv_compact, _, cu_seqlens_k, max_seqlen_k = text_cond_tuple
        with torch.no_grad():
            kv_norm = model.text_norm(kv_compact.detach()).contiguous()
            scale_hidden = model.text_proj_for_sos((kv_norm, cu_seqlens_k, max_seqlen_k)).float().contiguous()
        return scale_hidden.detach()

    @torch.no_grad()
    def eval_ep(self, ep: int, args: arg_util.Args, ld_val: DataLoader):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        stt = time.time()
        training = self.gpt_wo_ddp.training
        self.gpt_wo_ddp.eval()
        for inp, label_B in ld_val:
            B = label_B.shape[0]
            label_B = label_B.to(args.device, non_blocking=True)
            V = self.vae_local.vocab_size
            inp = inp.to(args.device, non_blocking=True)
            gt_ms_idx_Bl: List[Ten] = self.vae_local.get_GPT_ground_truth(inp)
            
            gt_BL = torch.cat(gt_ms_idx_Bl, dim=1)
            self.gpt_wo_ddp.forward
            logits_BLV = self.gpt_wo_ddp(label_B, self.quantize_local.fuse_multiscale_idx_as_gpt_inp_BL(gt_ms_idx_Bl))
            
            L_mean += self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)) * B
            L_tail += self.val_loss(logits_BLV.data[:, -self.raw_last_l:].reshape(-1, V), gt_BL[:, -self.raw_last_l:].reshape(-1)) * B
            acc_mean += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (100/gt_BL.shape[1])
            acc_tail += (logits_BLV.data[:, -self.raw_last_l:].argmax(dim=-1) == gt_BL[:, -self.raw_last_l:]).sum() * (100/self.raw_last_l)
            tot += B
        self.gpt_wo_ddp.train(training)
        
        stats = L_mean.new_tensor([L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot])
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()
        return L_mean, L_tail, acc_mean, acc_tail, tot, time.time()-stt

    def _get_online_depth_extractor(self):
        if self.depth_extractor is None:
            if dist.is_local_master():
                print(
                    f'[spatial_condition] loading online depth model '
                    f'{self.depth_model_name} on {self.depth_model_device} ({self.depth_model_dtype})',
                    flush=True,
                )
            self.depth_extractor = TransformersDepthExtractor(
                model_name=self.depth_model_name,
                device=self.depth_model_device,
                dtype=self.depth_model_dtype,
                cache_dir=self.depth_model_cache_dir,
            )
        return self.depth_extractor

    def _get_online_seg_extractor(self):
        if self.seg_extractor is None:
            if dist.is_local_master():
                print(
                    f'[spatial_condition] loading online SAM/seg model '
                    f'{self.seg_model_name} on {self.seg_model_device} ({self.seg_model_dtype}), '
                    f'mode={self.seg_output_mode}',
                    flush=True,
                )
            self.seg_extractor = TransformersSAMSegmentationExtractor(
                model_name=self.seg_model_name,
                device=self.seg_model_device,
                dtype=self.seg_model_dtype,
                cache_dir=self.seg_model_cache_dir,
                output_mode=self.seg_output_mode,
                max_masks=self.seg_max_masks,
                points_per_batch=self.seg_points_per_batch,
            )
        return self.seg_extractor

    def _sample_condition_token_scales(self, training_scales: int) -> int:
        hi = max(1, min(int(self.condition_token_scales), int(training_scales)))
        lo_cfg = int(self.condition_token_scales_min)
        lo = hi if lo_cfg <= 0 else max(1, min(lo_cfg, hi))
        if lo >= hi:
            return hi
        return random.randint(lo, hi)

    @torch.no_grad()
    def _make_condition_map(self, inp_B3HW: torch.Tensor, out_size: Optional[int], condition_B1HW: Optional[torch.Tensor] = None) -> torch.Tensor:
        if condition_B1HW is not None:
            cond = condition_B1HW.float()
            if cond.ndim == 3:
                cond = cond.unsqueeze(1)
            if out_size and cond.shape[-2:] != (int(out_size), int(out_size)):
                cond = F.interpolate(cond, size=(int(out_size), int(out_size)), mode='area')
            return normalize_depth_B1HW(cond) if self.spatial_cond_type.startswith('depth') else cond.clamp(0.0, 1.0)

        if self.spatial_cond_type in {'depth', 'depth_model', 'depth_anything'} and self.depth_condition_source in {'transformers', 'hf', 'depth_anything', 'depth_anything_v2'}:
            size = None if not out_size else int(out_size)
            return self._get_online_depth_extractor()(inp_B3HW, out_size=size).detach()

        if self.spatial_cond_type in {'sam', 'seg', 'segmentation', 'segment'} and self.seg_condition_source in {'transformers', 'hf', 'sam'}:
            size = None if not out_size else int(out_size)
            return self._get_online_seg_extractor()(inp_B3HW, out_size=size).detach()

        size = int(out_size) if out_size else 0
        return image_to_spatial_condition(inp_B3HW, cond_type=self.spatial_cond_type, out_size=size).detach()
    
    def train_step(
        self, ep: int, it: int, g_it: int, stepping: bool, clip_decay_ratio: float, metric_lg: misc.MetricLogger, logging_params: bool,
        inp_B3HW: FTen, text_cond_tuple: Union[ITen, FTen], args: arg_util.Args, condition_B1HW: Optional[FTen] = None,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        
        B = inp_B3HW.shape[0]
        T = 1 if inp_B3HW.dim() == 4 else inp_B3HW.shape[2]
        V = self.vae_local.vocab_size
        device = inp_B3HW.device

        h_div_w = inp_B3HW.shape[-2] / inp_B3HW.shape[-1]
        h_div_w_templates = np.array(list(dynamic_resolution_h_w.keys()))
        h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w-h_div_w_templates))]
        scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
        scale_schedule = [(min(t, T//4+1), h, w) for (t, h, w) in scale_schedule]

        ref_loss = inp_B3HW.new_tensor(0.0)
        safe_loss = inp_B3HW.new_tensor(0.0)
        reg_loss = inp_B3HW.new_tensor(0.0)
        refiner_grad_norm = inp_B3HW.new_tensor(0.0)
        cond_rate_loss = inp_B3HW.new_tensor(0.0)
        cond_recon_loss = inp_B3HW.new_tensor(0.0)
        cond_side_bpp = inp_B3HW.new_tensor(0.0)
        cond_hard_side_bpp = inp_B3HW.new_tensor(0.0)

        with self.gpt_opt.amp_ctx:
            with torch.amp.autocast('cuda', enabled=False):
                with torch.no_grad():
                    if args.apply_spatial_patchify:
                        vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
                    else:
                        vae_scale_schedule = scale_schedule
                    raw_features, _, _ = self.vae_local.encode_for_raw_features(inp_B3HW, scale_schedule=vae_scale_schedule)
            
            x_BLC_wo_prefix, gt_ms_idx_Bl = self.bitwise_self_correction.flip_requant(vae_scale_schedule, inp_B3HW, raw_features, device)

            available_scales = min(len(scale_schedule), len(vae_scale_schedule), len(gt_ms_idx_Bl))
            training_scales = min(int(args.always_training_scales), available_scales)
            if training_scales <= 0:
                raise ValueError(
                    f'No valid training scales: always_training_scales={args.always_training_scales}, '
                    f'len(scale_schedule)={len(scale_schedule)}, len(vae_scale_schedule)={len(vae_scale_schedule)}, '
                    f'len(gt_ms_idx_Bl)={len(gt_ms_idx_Bl)}'
                )
            training_seq_len = np.array(scale_schedule)[:training_scales].prod(axis=1).sum()
            x_BLC_wo_prefix = x_BLC_wo_prefix[:, :(training_seq_len - np.array(scale_schedule[0]).prod()), :]

            condition_input = None
            condition_aux = None
            condition_kwargs = {}
            if self.enable_boundary_condition:
                condition_out_size = 0 if self.condition_codec_type == 'vae_token' else self.boundary_cond_size
                condition_input = self._make_condition_map(inp_B3HW, out_size=condition_out_size, condition_B1HW=condition_B1HW)
                if self.condition_codec_type == 'vae_token':
                    condition_token_scales = self._sample_condition_token_scales(training_scales)
                    condition_aux = var_token_condition_from_map(
                        self.vae_local,
                        condition_input,
                        vae_scale_schedule[:training_scales],
                        num_scales=condition_token_scales,
                        image_hw=(int(inp_B3HW.shape[-2]), int(inp_B3HW.shape[-1])),
                    )
                    condition_kwargs = dict(condition_features=condition_aux["features"].detach())
                else:
                    condition_kwargs = dict(
                        condition_input=condition_input,
                        return_condition_aux=True,
                        condition_image_hw=(int(inp_B3HW.shape[-2]), int(inp_B3HW.shape[-1])),
                    )
            self.gpt_wo_ddp.forward
            gpt_out = self.gpt(text_cond_tuple, x_BLC_wo_prefix, scale_schedule=scale_schedule[:training_scales], **condition_kwargs)
            if self.enable_boundary_condition and self.condition_codec_type != 'vae_token':
                logits_BLV, condition_aux = gpt_out
            else:
                logits_BLV = gpt_out
            self.batch_size, self.seq_len = logits_BLV.shape[:2]
            self.seq_len_each = [idx_Bl.shape[1] for idx_Bl in gt_ms_idx_Bl]

            gt_BL = torch.cat(gt_ms_idx_Bl, dim=1)[:, :training_seq_len].contiguous().type(torch.long)
            if args.use_bit_label:
                tmp_bs, tmp_seq_len, _ = logits_BLV.shape
                main_loss_tokens = self.train_loss(logits_BLV.reshape(tmp_bs, tmp_seq_len, -1, 2).permute(0, 3, 1, 2), gt_BL)
                if args.bitloss_type == 'mean':
                    main_loss_tokens = main_loss_tokens.mean(dim=-1)
                elif args.bitloss_type == 'sum':
                    main_loss_tokens = main_loss_tokens.sum(dim=-1)
                else:
                    raise NotImplementedError(f'{args.bitloss_type=}')
            else:
                main_loss_tokens = self.train_loss(logits_BLV.reshape(-1, V), gt_BL.reshape(-1)).reshape(B, -1)

            if self.reweight_loss_by_scale:
                lw = []
                last_scale_area = np.sqrt(int(np.prod(scale_schedule[-1])))
                for (pt, ph, pw) in scale_schedule[:training_scales]:
                    this_scale_area = np.sqrt(pt * ph * pw)
                    lw.extend([last_scale_area / this_scale_area for _ in range(pt * ph * pw)])
                lw = torch.tensor(lw, device=main_loss_tokens.device)[None, ...]
                lw = lw / lw.sum()
            else:
                lw = 1. / self.seq_len
            loss = main_loss_tokens.mul(lw).sum(dim=-1).mean()
            if condition_aux is not None:
                cond_rate_loss = (condition_aux["rate_nats_per_image"] / max(1, int(training_seq_len))).mean()
                cond_recon_loss = condition_aux["recon_loss"]
                cond_side_bpp = condition_aux["side_bpp"].mean()
                cond_hard_side_bpp = condition_aux["hard_side_bpp"].mean()
                if self.condition_codec_type != 'vae_token' and self.boundary_cond_rate_ratio > 0:
                    loss = loss + self.boundary_cond_rate_ratio * cond_rate_loss
                if self.condition_codec_type != 'vae_token' and self.boundary_cond_recon_ratio > 0:
                    loss = loss + self.boundary_cond_recon_ratio * cond_recon_loss

            same_scale_is_active = self.enable_same_scale_refiner and args.use_bit_label and (g_it >= self.same_scale_start_step)
            if same_scale_is_active:
                refiner_model = self.same_scale_refiner if self.same_scale_refiner is not None else self.same_scale_refiner_wo_ddp
                refiner_base = self.same_scale_refiner_wo_ddp if self.same_scale_refiner_wo_ddp is not None else refiner_model
                prefix_maps, _ = self._build_student_targets(gt_ms_idx_Bl[:training_scales], scale_schedule[:training_scales], vae_scale_schedule[:training_scales], device, need_gt_bits=False)
                bit_logits_scales = split_flat_logits_to_bit_logits_per_scale(logits_BLV, scale_schedule[:training_scales])
                scale_hidden = self._scale_hidden_from_text_cond(text_cond_tuple)
                ref_terms, safe_terms, reg_terms = [], [], []
                for si in range(training_scales):
                    bit_logits_map = bit_logits_scales[si]
                    head_in = bit_logits_map.detach() if self.same_scale_detach_firstpass_input else bit_logits_map
                    uncertainty = compute_normalized_uncertainty(head_in)

                    if self.same_scale_refine_ratio > 0:
                        neighbor_ctx = compute_neighborhood_context(torch.sigmoid(head_in), kernel_size=refiner_base.neighborhood_kernel)
                        delta_logits, gate, bounded_delta = refiner_model(
                            prefix_maps[si].detach(),
                            head_in,
                            uncertainty=uncertainty.detach(),
                            neighbor_context=neighbor_ctx.detach(),
                            text_summary=None,
                            scale_hidden=scale_hidden,
                            scale_id=si,
                            return_gate=True,
                        )
                        refined_bit_logits = head_in + self.same_scale_refine_alpha * delta_logits
                        refined_pair = single_bit_logits_to_pair_logits(refined_bit_logits.permute(0, 2, 3, 1).reshape(B, -1, bit_logits_map.shape[1]))
                        base_pair = single_bit_logits_to_pair_logits(head_in.permute(0, 2, 3, 1).reshape(B, -1, bit_logits_map.shape[1]))
                        gt_bits_flat = gt_ms_idx_Bl[si].contiguous().long()
                        refine_token_loss = self.train_loss(refined_pair.permute(0, 3, 1, 2), gt_bits_flat)
                        base_token_loss = self.train_loss(base_pair.permute(0, 3, 1, 2), gt_bits_flat)
                        if args.bitloss_type == 'mean':
                            refine_token_loss = refine_token_loss.mean(dim=-1)
                            base_token_loss = base_token_loss.mean(dim=-1)
                        elif args.bitloss_type == 'sum':
                            refine_token_loss = refine_token_loss.sum(dim=-1)
                            base_token_loss = base_token_loss.sum(dim=-1)
                        ref_terms.append(refine_token_loss.mean())
                        safe_terms.append(F.relu(refine_token_loss - base_token_loss.detach() + self.same_scale_safety_margin).mean())
                        reg_terms.append((gate * bounded_delta.abs()).mean())

                if ref_terms:
                    ref_loss = torch.stack(ref_terms).mean()
                    loss = loss + self.same_scale_refine_ratio * ref_loss
                if safe_terms and self.same_scale_safety_ratio > 0:
                    safe_loss = torch.stack(safe_terms).mean()
                    loss = loss + self.same_scale_safety_ratio * safe_loss
                if reg_terms and self.same_scale_reg_ratio > 0:
                    reg_loss = torch.stack(reg_terms).mean()
                    loss = loss + self.same_scale_reg_ratio * reg_loss

        student_loss = None
        student_grad_norm = loss.new_tensor(0.0)
        student_is_active = self.enable_student_entropy and (g_it >= self.student_start_step)
        if student_is_active:
            teacher_bit_logits_per_scale = teacher_logits_to_bit_logits_per_scale(logits_BLV.detach(), scale_schedule[:training_scales], self.student_codebook_dim)
            prefix_maps, gt_bits_bdhw = self._build_student_targets(gt_ms_idx_Bl[:training_scales], scale_schedule[:training_scales], vae_scale_schedule[:training_scales], device)
            student_terms = []
            student_model = self.student if self.student is not None else self.student_wo_ddp
            for si in range(max(0, self.student_start_scale), training_scales):
                s_logits = student_model(prefix_maps[si].detach(), text_cond_tuple, scale_id=si)
                t_probs = torch.sigmoid(teacher_bit_logits_per_scale[si].detach())
                hard_bits = gt_bits_bdhw[si].to(dtype=s_logits.dtype)
                cur = 0.0
                if self.student_kd_ratio > 0:
                    cur = cur + self.student_kd_ratio * F.binary_cross_entropy_with_logits(s_logits, t_probs)
                if self.student_gt_ratio > 0:
                    cur = cur + self.student_gt_ratio * F.binary_cross_entropy_with_logits(s_logits, hard_bits)
                student_terms.append(cur)
            if len(student_terms):
                student_loss = torch.stack(student_terms).mean()
        
        if student_is_active and student_loss is not None:
            (student_loss * self.gpt_opt.r_accu).backward(retain_graph=False, create_graph=False)
        grad_norm_t, scale_log2_t = self.gpt_opt.backward_clip_step(ep=ep, it=it, g_it=g_it, stepping=stepping, logging_params=logging_params, loss=loss, clip_decay_ratio=clip_decay_ratio, stable=args.stable)
        if stepping and student_is_active and self.student_opt is not None:
            if self.student_grad_clip > 0:
                if isinstance(self.student, FSDP):
                    student_grad_norm = self.student.clip_grad_norm_(self.student_grad_clip)
                else:
                    student_grad_norm = torch.nn.utils.clip_grad_norm_(self.student.parameters() if self.student is not None else self.student_wo_ddp.parameters(), self.student_grad_clip)
            self.student_opt.step()
        if stepping and same_scale_is_active and self.same_scale_refiner_opt is not None:
            if self.same_scale_refiner_grad_clip > 0:
                if isinstance(self.same_scale_refiner, FSDP):
                    refiner_grad_norm = self.same_scale_refiner.clip_grad_norm_(self.same_scale_refiner_grad_clip)
                else:
                    target_params = self.same_scale_refiner.parameters() if self.same_scale_refiner is not None else self.same_scale_refiner_wo_ddp.parameters()
                    refiner_grad_norm = torch.nn.utils.clip_grad_norm_(target_params, self.same_scale_refiner_grad_clip)
            self.same_scale_refiner_opt.step()
        
        if args.use_fsdp_model_ema:
            update_ema(self.gpt_ema, self.gpt)

        if stepping:
            if self.using_ema:
                self.ema_update(g_it)
            if self.dbg_unused:
                ls = []
                for n, p in self.gpt_wo_ddp.named_parameters():
                    if p.grad is None:
                        ls.append(n)
                if len(ls):
                    raise AttributeError(f'unused param: {ls}')
        
            self.gpt_opt.optimizer.zero_grad(set_to_none=True)
            if self.enable_student_entropy and self.student_opt is not None:
                self.student_opt.zero_grad(set_to_none=True)
            if self.enable_same_scale_refiner and self.same_scale_refiner_opt is not None:
                self.same_scale_refiner_opt.zero_grad(set_to_none=True)
        
        if metric_lg.log_every_iter or it == 0 or it in metric_lg.log_iters:
            with torch.no_grad():
                logits_for_log = logits_BLV.detach()
                B, seq_len = logits_for_log.shape[:2]
                if args.use_bit_label:
                    res_loss = self.train_loss(logits_for_log.reshape(B, seq_len, -1, 2).permute(0, 3, 1, 2), gt_BL).mean(dim=-1).mean(0)
                    bitwise_acc = (logits_for_log.reshape(B, seq_len, -1, 2).argmax(dim=-1) == gt_BL).float()
                else:
                    res_loss = self.train_loss(logits_for_log.reshape(-1, V), gt_BL.reshape(-1)).reshape(B, -1).mean(0)
                    pred_BL = logits_for_log.argmax(dim=-1)
                    mask = self.vae_local.quantizer.lfq.mask
                    pred_bits = ((pred_BL[..., None].int() & mask) != 0)
                    gt_bits = ((gt_BL[..., None].int() & mask) != 0)
                    bitwise_acc = (pred_bits == gt_bits).float()
                res_bit_acc = bitwise_acc.mean(-1).mean(0)
                res_token_acc = (bitwise_acc.sum(-1) == self.vae_local.codebook_dim).float().mean(0)
                
                loss_token_mean, acc_bit_mean, acc_token_mean = res_loss.mean().item(), res_bit_acc.mean().item() * 100., res_token_acc.mean().item() * 100.
                ptr = 0
                L_list, acc_bit_list, acc_token_list = [], [], []
                for scale_ind in range(min(training_scales, len(scale_schedule))):
                    start, end = ptr, ptr + np.array(scale_schedule[scale_ind]).prod()
                    L_list.append(res_loss[start:end].mean().item())
                    acc_bit_list.append(res_bit_acc[start:end].mean().item() * 100.)
                    acc_token_list.append(res_token_acc[start:end].mean().item() * 100.)
                    ptr = end
                
                metrics = torch.tensor(L_list + acc_bit_list + acc_token_list + [
                    grad_norm_t.item(), loss_token_mean, acc_bit_mean, acc_token_mean,
                    float(student_loss.item()) if student_loss is not None else 0.0,
                    float(student_grad_norm.item()) if torch.is_tensor(student_grad_norm) else float(student_grad_norm),
                    float(ref_loss.item()), float(safe_loss.item()), float(reg_loss.item()),
                    float(refiner_grad_norm.item()) if torch.is_tensor(refiner_grad_norm) else float(refiner_grad_norm),
                    float(cond_rate_loss.item()), float(cond_recon_loss.item()),
                    float(cond_side_bpp.item()), float(cond_hard_side_bpp.item()),
                ], device=loss.device)
            tdist.all_reduce(metrics, op=tdist.ReduceOp.SUM)
            metrics = metrics.cpu().data.numpy() / dist.get_world_size()
            leng = len(L_list)
            L_list = metrics[:leng]
            acc_bit_list = metrics[leng:2*leng]
            acc_token_list = metrics[2*leng:3*leng]
            (
                grad_norm_t, loss_token_mean, acc_bit_mean, acc_token_mean,
                student_loss_mean, student_grad_norm_f, cal_loss_mean, safe_loss_mean,
                reg_loss_mean, refiner_grad_norm_f, cond_rate_loss_mean,
                cond_recon_loss_mean, cond_side_bpp_mean, cond_hard_side_bpp_mean,
            ) = metrics[-14:]
            Lmean = loss_token_mean
            Ltail = L_list[-1]
            acc_mean = acc_bit_mean if args.use_bit_label else acc_token_mean
            acc_tail = acc_bit_list[-1] if args.use_bit_label else acc_token_list[-1]
            metric_lg.update(
                Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm_t,
                Stu=student_loss_mean, StuGD=student_grad_norm_f, Cal=cal_loss_mean,
                Safe=safe_loss_mean, Reg=reg_loss_mean, RefGD=refiner_grad_norm_f,
                BCR=cond_rate_loss_mean, BCRec=cond_recon_loss_mean, BCBpp=cond_side_bpp_mean,
            )
            swanlab_log_dict = {
                "Overall/L_mean": Lmean,
                'Overall/Acc_bit_mean': acc_bit_mean,
                'Overall/Acc_token_mean': acc_token_mean,
                'Overall/grad_norm_t': grad_norm_t,
                'Student/loss': student_loss_mean,
                'Student/grad_norm': student_grad_norm_f,
                'Student/active': float(student_is_active),
                'SameScale/calibration_loss': cal_loss_mean,
                'SameScale/safety_loss': safe_loss_mean,
                'SameScale/reg_loss': reg_loss_mean,
                'SameScale/refiner_grad_norm': refiner_grad_norm_f,
                'SameScale/active': float(same_scale_is_active),
                'SpatialCond/rate_loss': cond_rate_loss_mean,
                'SpatialCond/recon_loss': cond_recon_loss_mean,
                'SpatialCond/side_bpp_expected': cond_side_bpp_mean,
                'SpatialCond/side_bpp_hard': cond_hard_side_bpp_mean,
                'SpatialCond/active': float(self.enable_boundary_condition),
            }
            for si, (loss_si, acc_bit_si, acc_token_si) in enumerate(zip(L_list, acc_bit_list, acc_token_list)):
                swanlab_log_dict[f'Detail/L_s{si+1:02d}'] = loss_si
                swanlab_log_dict[f'Detail/Acc_bit_s{si+1:02d}'] = acc_bit_si
                swanlab_log_dict[f'Detail/Acc_token_s{si+1:02d}'] = acc_token_si
            swanlab_utils.log(swanlab_log_dict, step=g_it)
        
        return grad_norm_t, scale_log2_t

    def __repr__(self):
        return (
            f'\n'
            f'[VGPTTr.config]: {pformat(self.get_config(), indent=2, width=250)}\n'
            f'[VGPTTr.structure]: {super(InfinityTrainer, self).__repr__().replace(InfinityTrainer.__name__, "")}'
        )
    
    def ema_load(self):
        self.cached_state_not_ema = {k: v.cpu() for k, v in self.gpt_wo_ddp.state_dict().items()}
        for pi, p_ema in self.pi_para_copy_for_parallel_ema:
            self.gpt_opt.paras[pi].data.copy_(p_ema)
        for pi, para in enumerate(self.gpt_opt.paras):
            dist.broadcast(para, src_rank=pi % dist.get_world_size())
    
    def ema_recover(self):
        self.gpt_wo_ddp.load_state_dict(self.cached_state_not_ema)
        del self.cached_state_not_ema
        self.cached_state_not_ema = None
    
    # p_ema = p_ema*0.9 + p*0.1 <==> p_ema.lerp_(p, 0.1)
    # p_ema.mul_(self.ema_ratio).add_(p.mul(self.ema_ratio_1))
    # @profile(precision=4, stream=open('ema_update.log', 'w+'))
    def ema_update(self, g_it): # todo: 将来再用离线ema
        # if self.using_ema and (g_it + 1) in self.ema_upd_it:
        stt = time.time()
        for pi, p_ema in self.pi_para_copy_for_parallel_ema:
            p = self.gpt_opt.paras[pi]
            p_ema.data.mul_(self.ema_ratio).add_(p.data.to(p_ema.device), alpha=1-self.ema_ratio)
        # ii = self.ema_upd_it.index(g_it + 1)
        ii = g_it
        if ii < 3:
            print(f'[ema upd {self.ema_ratio}, cpu={self.ema_cpu}, @ g_it={g_it}] cost: {time.time()-stt:.2f}s')
    
    def get_config(self):
        return {
            'dynamic_resolution_h_w': dynamic_resolution_h_w,
            'label_smooth': self.label_smooth, 'eq_loss': self.eq_loss,
            'ema_ratio':    self.ema_ratio,
            'prog_it':      self.prog_it, 'last_prog_si': self.last_prog_si, 'first_prog': self.first_prog,
        }
    
    def state_dict(self):
        m = self.vae_local
        if hasattr(m, '_orig_mod'):
            m = m._orig_mod
        state = {'config': self.get_config(), 'vae_local': m.state_dict()}
        
        if self.zero:   # TODO: fixme
            state['gpt_fsdp'] = None
            with FSDP.state_dict_type(self.gpt, StateDictType.FULL_STATE_DICT, fullstate_save_policy, fulloptstate_save_policy):
                state['gpt_fsdp'] = self.gpt.state_dict()
                if self.use_fsdp_model_ema:
                    state['gpt_ema_fsdp'] = self.gpt_ema.state_dict()
                state['gpt_fsdp_opt'] = FSDP.optim_state_dict(model=self.gpt, optim=self.gpt_opt.optimizer, optim_state_dict=self.gpt_opt.optimizer.state_dict())
            if self.student is not None:
                with FSDP.state_dict_type(self.student, StateDictType.FULL_STATE_DICT, fullstate_save_policy, fulloptstate_save_policy):
                    state['student_fsdp'] = self.student.state_dict()
                    if self.student_opt is not None:
                        state['student_fsdp_opt'] = FSDP.optim_state_dict(model=self.student, optim=self.student_opt, optim_state_dict=self.student_opt.state_dict())
            if self.same_scale_refiner is not None:
                with FSDP.state_dict_type(self.same_scale_refiner, StateDictType.FULL_STATE_DICT, fullstate_save_policy, fulloptstate_save_policy):
                    state['same_scale_refiner_fsdp'] = self.same_scale_refiner.state_dict()
                    if self.same_scale_refiner_opt is not None:
                        state['same_scale_refiner_fsdp_opt'] = FSDP.optim_state_dict(model=self.same_scale_refiner, optim=self.same_scale_refiner_opt, optim_state_dict=self.same_scale_refiner_opt.state_dict())
            if self.gpt_opt.scaler is not None:
                state['gpt_opt_scaler'] = self.gpt_opt.scaler.state_dict()
        
        else:
            if self.using_ema:  # TODO: fixme
                self.ema_load()
                state['gpt_ema_for_vis'] = {k: v.cpu() for k, v in self.gpt_wo_ddp.state_dict().items()}
                self.ema_recover()
            
            for k in ('gpt_wo_ddp', 'gpt_opt'):
                m = getattr(self, k)
                if m is not None:
                    if hasattr(m, '_orig_mod'):
                        m = m._orig_mod
                    state[k] = m.state_dict()
            if self.student_wo_ddp is not None:
                m = self.student_wo_ddp._orig_mod if hasattr(self.student_wo_ddp, '_orig_mod') else self.student_wo_ddp
                state['student_wo_ddp'] = m.state_dict()
            if self.student_opt is not None:
                state['student_opt'] = self.student_opt.state_dict()
            if self.same_scale_refiner_wo_ddp is not None:
                m = self.same_scale_refiner_wo_ddp._orig_mod if hasattr(self.same_scale_refiner_wo_ddp, '_orig_mod') else self.same_scale_refiner_wo_ddp
                state['same_scale_refiner_wo_ddp'] = m.state_dict()
            if self.same_scale_refiner_opt is not None:
                state['same_scale_refiner_opt'] = self.same_scale_refiner_opt.state_dict()
        return state
    
    def load_state_dict(self, state, strict=True, skip_vae=False):
        def _load_compatible(module, module_state, name):
            own_state = module.state_dict()
            compatible = {
                k: v for k, v in module_state.items()
                if k in own_state and tuple(own_state[k].shape) == tuple(v.shape)
            }
            skipped = sorted(k for k in module_state.keys() if k not in compatible)
            ret = module.load_state_dict(compatible, strict=False)
            if skipped:
                print(f'[VGPTTr.load_state_dict] {name} skipped incompatible:  {skipped}')
            return ret

        if self.zero:
            with FSDP.state_dict_type(self.gpt, StateDictType.FULL_STATE_DICT, fullstate_save_policy, fulloptstate_save_policy):
                self.gpt.load_state_dict(state['gpt_fsdp'])
                if self.use_fsdp_model_ema:
                    self.gpt_ema.load_state_dict(state['gpt_ema_fsdp'])
                one_group_opt_state = state['gpt_fsdp_opt']
                """
                AdamW state['gpt_fsdp_opt']:
                {
                    'state': { <para_name>: {'exp_avg': <unsharded_tensor>, 'exp_avg_sq': <unsharded_tensor>, 'step': <int>} },
                    'param_groups': [
                        {
                            'wd_sc': 1.0, 'lr_sc': 1.0, 'lr': xxx, 'betas': (0.9, 0.97), 'eps': 1e-08, 'weight_decay': 0.02,
                            'amsgrad': False, 'foreach': None, 'maximize': False, 'capturable': False, 'differentiable': False, 'fused': True,
                            'params': [<para_name> x m]
                        } x n
                    ]
                }
                one_group_opt_state['param_groups'] = self.gpt_opt.optimizer.state_dict()['param_groups']
                """
                optim_state_dict = FSDP.optim_state_dict_to_load(model=self.gpt, optim=self.gpt_opt.optimizer, optim_state_dict=one_group_opt_state)
                self.gpt_opt.optimizer.load_state_dict(optim_state_dict)

            if self.student is not None and 'student_fsdp' in state:
                with FSDP.state_dict_type(self.student, StateDictType.FULL_STATE_DICT, fullstate_save_policy, fulloptstate_save_policy):
                    self.student.load_state_dict(state['student_fsdp'])
                    if self.student_opt is not None and 'student_fsdp_opt' in state:
                        student_opt_state = FSDP.optim_state_dict_to_load(model=self.student, optim=self.student_opt, optim_state_dict=state['student_fsdp_opt'])
                        self.student_opt.load_state_dict(student_opt_state)
            if self.same_scale_refiner is not None and 'same_scale_refiner_fsdp' in state:
                with FSDP.state_dict_type(self.same_scale_refiner, StateDictType.FULL_STATE_DICT, fullstate_save_policy, fulloptstate_save_policy):
                    self.same_scale_refiner.load_state_dict(state['same_scale_refiner_fsdp'])
                    if self.same_scale_refiner_opt is not None and 'same_scale_refiner_fsdp_opt' in state:
                        refiner_opt_state = FSDP.optim_state_dict_to_load(model=self.same_scale_refiner, optim=self.same_scale_refiner_opt, optim_state_dict=state['same_scale_refiner_fsdp_opt'])
                        self.same_scale_refiner_opt.load_state_dict(refiner_opt_state)

            if self.gpt_opt.scaler is not None:
                try: self.gpt_opt.scaler.load_state_dict(state['gpt_opt_scaler'])
                except Exception as e: print(f'[fp16 load_state_dict err] {e}')
        else:
            for k in ('gpt_wo_ddp', 'gpt_opt'):
                if skip_vae and 'vae' in k: continue
                m = getattr(self, k)
                if m is not None:
                    if hasattr(m, '_orig_mod'):
                        m = m._orig_mod
                    ret = m.load_state_dict(state[k], strict=strict)
                    if ret is not None:
                        missing, unexpected = ret
                        print(f'[VGPTTr.load_state_dict] {k} missing:  {missing}')
                        print(f'[VGPTTr.load_state_dict] {k} unexpected:  {unexpected}')
            if self.student_wo_ddp is not None and 'student_wo_ddp' in state:
                ret = self.student_wo_ddp.load_state_dict(state['student_wo_ddp'], strict=False)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[VGPTTr.load_state_dict] student_wo_ddp missing:  {missing}')
                    print(f'[VGPTTr.load_state_dict] student_wo_ddp unexpected:  {unexpected}')
            if self.student_opt is not None and 'student_opt' in state:
                self.student_opt.load_state_dict(state['student_opt'])
            if self.same_scale_refiner_wo_ddp is not None and 'same_scale_refiner_wo_ddp' in state:
                ret = _load_compatible(self.same_scale_refiner_wo_ddp, state['same_scale_refiner_wo_ddp'], 'same_scale_refiner_wo_ddp')
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[VGPTTr.load_state_dict] same_scale_refiner_wo_ddp missing:  {missing}')
                    print(f'[VGPTTr.load_state_dict] same_scale_refiner_wo_ddp unexpected:  {unexpected}')
            if self.same_scale_refiner_opt is not None and 'same_scale_refiner_opt' in state:
                self.same_scale_refiner_opt.load_state_dict(state['same_scale_refiner_opt'])
            
            if self.using_ema:
                if 'gpt_ema_for_vis' in state:
                    for pi, para in self.pi_para_copy_for_parallel_ema:
                        para.copy_(state['gpt_ema_for_vis'][self.gpt_opt.names[pi]])
                    print(f'[VGPTTr.load_state_dict] gpt_ema_for_vis: load succeed')
                else:
                    print(f'[VGPTTr.load_state_dict] gpt_ema_for_vis: key NOT FOUND in state!!')
        
        config: dict = state.pop('config', None)
        self.prog_it = config.get('prog_it', 0)
        self.last_prog_si = config.get('last_prog_si', -1)
        self.first_prog = config.get('first_prog', True)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[VGPT.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err)
