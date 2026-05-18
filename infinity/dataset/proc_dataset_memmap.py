# -*- coding: utf-8 -*-
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

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
        ten = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [3,H,W]
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


class MemmapShardEmbeddingReader:
    """
    Load shard meta via torch.load (small), kv via numpy.memmap (no full RAM load).
    Provides same get(shard_id, idx)->(kv[L,C], L) API as your old reader.
    """
    def __init__(self, proc_dir: str, cache_size: int = 1):
        self.proc_dir = Path(proc_dir)
        self.shards_dir = self.proc_dir / "shards"
        if not self.shards_dir.exists():
            raise FileNotFoundError(f"Missing shards dir: {self.shards_dir}")

        self._meta_cache: Dict[int, Dict[str, Any]] = {}
        self._mm_cache: "OrderedDict[int, np.memmap]" = OrderedDict()
        self.cache_size = int(max(1, cache_size))

    def load_meta(self, shard_id: int) -> Dict[str, Any]:
        if shard_id in self._meta_cache:
            return self._meta_cache[shard_id]
        p = self.shards_dir / f"shard_{shard_id:06d}.meta.pt"
        pack = torch.load(p, map_location="cpu")
        self._meta_cache[shard_id] = pack
        return pack

    def _open_memmap(self, shard_id: int) -> np.memmap:
        meta = self.load_meta(shard_id)
        rows = int(meta["rows"])
        hidden = int(meta["hidden_size"])
        dtype = str(meta.get("dtype", "fp16"))
        kv_bin = str(meta["kv_bin"])
        p = self.shards_dir / kv_bin

        np_dtype = np.float16 if dtype == "fp16" else np.float32
        mm = np.memmap(str(p), dtype=np_dtype, mode="r", shape=(rows, hidden))
        return mm

    def _get_mm(self, shard_id: int) -> np.memmap:
        if shard_id in self._mm_cache:
            self._mm_cache.move_to_end(shard_id)
            return self._mm_cache[shard_id]

        mm = self._open_memmap(shard_id)
        self._mm_cache[shard_id] = mm
        self._mm_cache.move_to_end(shard_id)

        # LRU eviction
        while len(self._mm_cache) > self.cache_size:
            old_sid, old_mm = self._mm_cache.popitem(last=False)
            # best-effort close: delete reference (OS will release mapping when gc)
            del old_mm
        return mm

    def get(self, shard_id: int, idx: int) -> Tuple[torch.Tensor, int]:
        meta = self.load_meta(shard_id)
        offsets: torch.Tensor = meta["offsets"]
        lens: torch.Tensor = meta["lens"]

        st = int(offsets[idx].item())
        ed = int(offsets[idx + 1].item())
        le = int(lens[idx].item())

        mm = self._get_mm(shard_id)
        sl = mm[st:ed]  # [L,C] view
        kv = torch.from_numpy(np.asarray(sl))  # float16/float32 CPU tensor
        return kv, le


class ProcessedJsonlMemmapDataset(Dataset):
    """
    Same external contract as your old ProcessedJsonlDataset,
    but uses memmap reader to avoid CPU OOM.
    """
    def __init__(
        self,
        proc_dir: str,
        res_list: List[int],
        seed: int = 0,
        cache_size: int = 1,
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
        self.condition_path_key = str(condition_path_key or "")
        self.condition_root = Path(condition_root) if condition_root else self.proc_dir

        self.reader = MemmapShardEmbeddingReader(proc_dir, cache_size=cache_size)
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


def collate_infinity_text_cond(batch: List[Dict[str, Any]]):
    """
    Returns:
      img_B3HW: [B,3,H,W]
      text_cond_tuple: (kv_compact, lens, cu_seqlens_k, max_seqlen_k)

    lens:        IntTensor [B]  (each sample token length)
    cu_seqlens:  IntTensor [B+1] prefix sum, cu[0]=0, cu[i+1]=sum_{j<=i} lens[j]
    max_seqlen:  int
    """
    imgs = torch.stack([b["img"] for b in batch], dim=0)

    kv_list = [b["kv"] for b in batch]  # each [Li, 2048] (cpu fp16/fp32)
    # 建议：lens 直接用 kv.shape[0]，避免和 b["len"] 不一致时出 bug
    lens = torch.tensor([int(kv.shape[0]) for kv in kv_list], dtype=torch.int32)

    max_seqlen = int(lens.max().item()) if lens.numel() else 0

    cu = torch.zeros((lens.numel() + 1,), dtype=torch.int32)
    cu[1:] = torch.cumsum(lens, dim=0)

    kv_compact = torch.cat([kv.contiguous() for kv in kv_list], dim=0).contiguous()
    return imgs, (kv_compact, lens, cu, max_seqlen)
