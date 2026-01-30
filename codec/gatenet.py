# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class GateNet(nn.Module):
    """
    Predict spatial importance logits for a given scale.
    Input:  [B,C,H,W]
    Output: [B,1,H,W] (logits)
    """
    def __init__(self, in_ch: int, hidden: int = 128, depth: int = 3):
        super().__init__()
        depth = max(1, int(depth))
        layers = []
        ch = in_ch
        for _ in range(depth):
            layers.append(nn.Conv2d(ch, hidden, 3, padding=1))
            layers.append(nn.SiLU(inplace=True))
            ch = hidden
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Conv2d(ch, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def topk_mask_from_scores(scores_BHW: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    """
    scores_BHW: [B,H,W], larger means keep
    returns: bool [B,H,W] with exactly topk per sample
    """
    B, H, W = scores_BHW.shape
    L = H * W
    k = max(1, min(L, int(round(L * float(keep_ratio)))))
    flat = scores_BHW.reshape(B, L)
    _, idx = torch.topk(flat, k=k, dim=1, largest=True, sorted=False)
    mask = torch.zeros((B, L), device=scores_BHW.device, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask.reshape(B, H, W)
