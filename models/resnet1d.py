"""
models/resnet1d.py
------------------
1D adaptation of ResNet-18 (He et al., 2016) for ECG classification.
All 2D convolutions replaced with 1D equivalents.
Residual blocks use identity shortcuts where dimensions match,
and 1×1 projection shortcuts otherwise.
"""

import torch
import torch.nn as nn
from typing import Optional, Type


class BasicBlock1D(nn.Module):
    """Standard residual block with two 3-sample convolutions."""

    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu2(out + identity)
        return out


class ResNet1D(nn.Module):
    """
    ResNet-18 adapted for 1D time-series.

    Parameters
    ----------
    in_channels  : number of input channels (leads)
    num_classes  : output dimension
    layers       : residual blocks per stage; [2, 2, 2, 2] = ResNet-18
    base_filters : channels in stage 1 (doubles each stage)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        layers: list = [2, 2, 2, 2],
        base_filters: int = 64,
    ) -> None:
        super().__init__()
        self.in_planes = base_filters

        # Stem
        self.conv1 = nn.Conv1d(in_channels, base_filters,
                               kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1   = nn.BatchNorm1d(base_filters)
        self.relu  = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual stages
        self.layer1 = self._make_layer(base_filters * 1, layers[0], stride=1)
        self.layer2 = self._make_layer(base_filters * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_filters * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_filters * 8, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc      = nn.Linear(base_filters * 8, num_classes)

        # Weight initialisation
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_planes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )
        layers = [BasicBlock1D(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(BasicBlock1D(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).squeeze(-1)
        return self.fc(x)

    def get_cam_target_layer(self) -> nn.Module:
        """Return last residual conv for Grad-CAM."""
        return self.layer4[-1].conv2


if __name__ == "__main__":
    model = ResNet1D(in_channels=1, num_classes=4)
    x = torch.randn(8, 1, 3000)
    out = model(x)
    print("ResNet1D output shape:", out.shape)   # (8, 4)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")
