import torch
import torch.nn as nn
import torch.nn.functional as F

class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 3, stride, 1,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.relu(x)

class RealTimeCharCNN(nn.Module):
    def __init__(self, num_classes=36):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.ds1 = DSConv(32, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.ds2 = DSConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.ds3 = DSConv(128, 256)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(self.ds1(x))
        x = self.pool2(self.ds2(x))
        x = self.ds3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)