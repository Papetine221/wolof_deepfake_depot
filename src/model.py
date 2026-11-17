#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/model.py — Architecture Light CNN (LCNN) à 5 blocs Conv2d(->2c) -> MFM(c) -> MaxPool2d,
suivis de AdaptiveAvgPool2d(1), Flatten, Linear(·, 128) -> ReLU -> Dropout(0.5) -> Linear(128, 2).

MFM (Max-Feature-Map) : maximum élément par élément entre les deux moitiés des canaux.
La sortie d'un MFM a exactement la moitié des canaux de la convolution qui précède,
et les nombres de canaux se raccordent d'un bloc à l'autre.
La même architecture accepte les 4 représentations spectrales d'entrée (B, 1, F, T).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MFM(nn.Module):
    """Max-Feature-Map : divise les 2c canaux en deux moitiés de c canaux et prend le max."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, 2*c, F, T)
        c = x.size(1) // 2
        x1, x2 = torch.split(x, c, dim=1)
        return torch.max(x1, x2)


class LCNNBlock(nn.Module):
    """Un bloc LCNN : Conv2d(in_c, 2*out_c, 3x3, pad=1) -> BatchNorm2d(2*out_c) -> MFM(out_c) -> MaxPool2d(2, 2)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.mfm = MFM()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.mfm(x)
        x = self.pool(x)
        return x


class LCNN(nn.Module):
    """
    Architecture LCNN à 5 blocs pour la détection de deepfakes audio :
      - Bloc 1 : 1  -> Conv2d(1, 32)  -> MFM(16) -> MaxPool2d(2, 2)
      - Bloc 2 : 16 -> Conv2d(16, 48) -> MFM(24) -> MaxPool2d(2, 2)
      - Bloc 3 : 24 -> Conv2d(24, 64) -> MFM(32) -> MaxPool2d(2, 2)
      - Bloc 4 : 32 -> Conv2d(32, 64) -> MFM(32) -> MaxPool2d(2, 2)
      - Bloc 5 : 32 -> Conv2d(32, 64) -> MFM(32) -> MaxPool2d(2, 2)
      - AdaptiveAvgPool2d(1) -> Flatten -> Linear(32, 128) -> ReLU -> Dropout(0.5) -> Linear(128, 2)
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5) -> None:
        super().__init__()
        channels = [1, 16, 24, 32, 32, 32]
        blocks = []
        for i in range(5):
            blocks.append(LCNNBlock(channels[i], channels[i + 1]))
        self.features = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits

