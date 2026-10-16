from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional


class DoubleConv(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    def __init__(self, input_channels: int = 3, output_channels: int = 1, base: int = 32) -> None:
        super().__init__()
        self.enc1 = DoubleConv(input_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.bridge = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.output = nn.Conv2d(base, output_channels, 1)

    @staticmethod
    def _concat(skip: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-2:] != skip.shape[-2:]:
            value = functional.interpolate(
                value,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return torch.cat([skip, value], dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        first = self.enc1(inputs)
        second = self.enc2(self.pool(first))
        third = self.enc3(self.pool(second))
        value = self.bridge(self.pool(third))
        value = self.dec3(self._concat(third, self.up3(value)))
        value = self.dec2(self._concat(second, self.up2(value)))
        value = self.dec1(self._concat(first, self.up1(value)))
        return self.output(value)
