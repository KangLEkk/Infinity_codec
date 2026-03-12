from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, DistributedSampler

from .proc_dataset import ProcessedJsonlDataset
from .proc_dataset_memmap import ProcessedJsonlMemmapDataset


TextCondTuple = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]


@dataclass
class Stage2DataBundle:
    dataloader: DataLoader
    sampler: DistributedSampler | None
    dataset: Any


def collate_proc_with_text_cond(batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, TextCondTuple]:
    imgs = torch.stack([b["img"] for b in batch], dim=0)

    kv_list = [b["kv"].contiguous() for b in batch]
    lens = torch.tensor([int(b.get("len", kv.shape[0])) for b, kv in zip(batch, kv_list)], dtype=torch.int32)
    max_seqlen = int(lens.max().item()) if lens.numel() else 0

    cu = torch.zeros((lens.numel() + 1,), dtype=torch.int32)
    if lens.numel() > 0:
        cu[1:] = torch.cumsum(lens, dim=0)

    kv_compact = torch.cat([kv.to(dtype=torch.float32) for kv in kv_list], dim=0).contiguous()
    return imgs, (kv_compact, lens, cu, max_seqlen)


def move_text_cond_tuple_to_device(
    text_cond_tuple: TextCondTuple,
    device: torch.device,
    kv_dtype: torch.dtype = torch.float32,
) -> TextCondTuple:
    kv_compact, lens, cu_seqlens_k, max_seqlen_k = text_cond_tuple
    return (
        kv_compact.to(device=device, dtype=kv_dtype, non_blocking=True),
        lens.to(device=device, non_blocking=True),
        cu_seqlens_k.to(device=device, non_blocking=True),
        int(max_seqlen_k),
    )


def build_stage2_dataloader(
    dataset_backend: str,
    proc_dir: str,
    res_list: List[int],
    batch_size: int,
    num_workers: int,
    seed: int = 0,
    prefetch_factor: int = 2,
    memmap_cache_size: int = 1,
    distributed: bool = False,
    drop_last: bool = True,
) -> Stage2DataBundle:
    dataset_backend = dataset_backend.lower()
    if dataset_backend == "proc":
        dataset = ProcessedJsonlDataset(proc_dir=proc_dir, res_list=res_list, seed=seed)
    elif dataset_backend in {"proc_memmap", "memmap"}:
        dataset = ProcessedJsonlMemmapDataset(
            proc_dir=proc_dir,
            res_list=res_list,
            seed=seed,
            cache_size=memmap_cache_size,
        )
    else:
        raise ValueError(f"Unsupported dataset_backend={dataset_backend!r}; expected 'proc' or 'proc_memmap'.")

    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=drop_last)

    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=collate_proc_with_text_cond,
        sampler=sampler,
        shuffle=(sampler is None),
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor

    dataloader = DataLoader(**kwargs)
    return Stage2DataBundle(dataloader=dataloader, sampler=sampler, dataset=dataset)
