# -*- coding: utf-8 -*-
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


def _pil_resize_shorter_side(im: Image.Image, target: int) -> Image.Image:
    w, h = im.size
    scale = target / min(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    return im.resize((nw, nh), resample=Image.BICUBIC)

def _pil_center_crop(im: Image.Image, size: int) -> Image.Image:
    w, h = im.size
    left = max(0, (w - size) // 2)
    top = max(0, (h - size) // 2)
    return im.crop((left, top, left + size, top + size))

def load_image_to_tensor(img_path: str, size: int) -> torch.Tensor:
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        im = _pil_resize_shorter_side(im, size)
        im = _pil_center_crop(im, size)
        arr = np.asarray(im).astype(np.float32) / 255.0
        ten = torch.from_numpy(arr).permute(2, 0, 1)  # [3,H,W]
        ten = ten * 2.0 - 1.0
        return ten


def load_condition_to_tensor(cond_path: str, size: int) -> torch.Tensor:
    path = str(cond_path)
    if path.lower().endswith(".npy"):
        arr = np.load(path)
    else:
        with Image.open(path) as im:
            im = _pil_resize_shorter_side(im, size)
            im = _pil_center_crop(im, size)
            arr = np.asarray(im)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.float32)
    if arr.size == 0:
        raise ValueError(f"Empty condition map: {cond_path}")
    mx = float(arr.max())
    if mx > 1.0:
        arr = arr / (65535.0 if mx > 255.0 else 255.0)
    arr = np.clip(arr, 0.0, 1.0)
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


class ShardEmbeddingReader:
    def __init__(self, proc_dir: str):
        self.proc_dir = Path(proc_dir)
        self.shards_dir = self.proc_dir / "shards"
        if not self.shards_dir.exists():
            raise FileNotFoundError(f"Missing shards dir: {self.shards_dir}")
        self._cache: Dict[int, Dict[str, Any]] = {}

    def load_shard(self, shard_id: int) -> Dict[str, Any]:
        if shard_id in self._cache:
            return self._cache[shard_id]
        p = self.shards_dir / f"shard_{shard_id:06d}.pt"
        pack = torch.load(p, map_location="cpu")
        self._cache[shard_id] = pack
        return pack

    def get(self, shard_id: int, idx: int) -> Tuple[torch.Tensor, int]:
        pack = self.load_shard(shard_id)
        offsets = pack["offsets"]
        lens = pack["lens"]
        kv = pack["kv_compact"]
        st = int(offsets[idx].item())
        ed = int(offsets[idx + 1].item())
        le = int(lens[idx].item())
        return kv[st:ed], le


class ProcessedJsonlDataset(Dataset):
    """
    Loads from preprocessed folder:
      samples.jsonl + samples_offsets.pt + shards/*
    This avoids loading millions of json lines into memory.
    """
    def __init__(
        self,
        proc_dir: str,
        res_list: List[int],
        seed: int = 0,
        condition_path_key: str = "",
        condition_root: str = "",
    ):
        super().__init__()
        self.proc_dir = Path(proc_dir)
        self.samples_path = self.proc_dir / "samples.jsonl"
        self.offsets_path = self.proc_dir / "samples_offsets.pt"
        if not self.samples_path.exists():
            raise FileNotFoundError(f"Missing {self.samples_path}")
        if not self.offsets_path.exists():
            raise FileNotFoundError(f"Missing {self.offsets_path}")

        self.res_list = list(res_list)
        self.rng = random.Random(seed)
        self.reader = ShardEmbeddingReader(proc_dir)
        self.condition_path_key = str(condition_path_key or "")
        self.condition_root = Path(condition_root) if condition_root else self.proc_dir

        self.offsets = torch.load(self.offsets_path, map_location="cpu").long()
        self._fh = None  # lazy open per worker

    def __len__(self):
        return int(self.offsets.numel())

    def _ensure_fh(self):
        if self._fh is None:
            self._fh = open(self.samples_path, "rb")

    def _read_rec(self, i: int) -> Dict[str, Any]:
        self._ensure_fh()
        off = int(self.offsets[i].item())
        self._fh.seek(off)
        line = self._fh.readline()
        return json.loads(line.decode("utf-8"))

    def __getitem__(self, idx: int):
        rec = self._read_rec(idx)
        img_path = rec["path"]
        shard_id = int(rec["shard"])
        local_idx = int(rec["idx"])

        res = self.res_list[0] if len(self.res_list) == 1 else self.rng.choice(self.res_list)
        img = load_image_to_tensor(img_path, res)

        kv, le = self.reader.get(shard_id, local_idx)
        out = {"img": img, "kv": kv, "len": le}
        if self.condition_path_key:
            cond_path = rec.get(self.condition_path_key)
            if not cond_path:
                raise KeyError(f"Record has no condition_path_key={self.condition_path_key!r}: {rec.keys()}")
            cond_path = Path(cond_path)
            if not cond_path.is_absolute():
                cond_path = self.condition_root / cond_path
            out["condition"] = load_condition_to_tensor(str(cond_path), res)
        return out


def collate_text_cond_tuple(batch_kv_lens: List[Tuple[torch.Tensor, int]], device: torch.device):
    lens = torch.tensor([le for (_, le) in batch_kv_lens], dtype=torch.int32, device=device)
    max_seqlen_k = int(lens.max().item()) if lens.numel() else 0
    cu = torch.zeros((lens.shape[0] + 1,), device=device, dtype=torch.int32)
    cu[1:] = torch.cumsum(lens, dim=0)
    # kv = torch.cat([kv.to(device=device, dtype=torch.float16) for (kv, _) in batch_kv_lens], dim=0).contiguous()
    kv = torch.cat([kv.to(device=device, dtype=torch.float32) for (kv, _) in batch_kv_lens], dim=0).contiguous()

    return kv, lens, cu, max_seqlen_k
