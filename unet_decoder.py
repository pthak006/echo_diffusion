import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        # Time embedding
        time_emb = self.time_mlp(t)
        time_emb = time_emb[(...,) + (None,) * 2]  # Extend last 2 dimensions
        x = x + time_emb
        return x


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, time_emb_dim, bilinear=True):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2, t):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Time embedding
        time_emb = self.time_mlp(t)
        time_emb = time_emb[(...,) + (None,) * 2]  # Extend last 2 dimensions
        x1 = x1 + time_emb

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class UNetDecoder(nn.Module):
    def __init__(self, n_channels, time_emb_dim=32, bilinear=False):
        super(UNetDecoder, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128, time_emb_dim)
        self.down2 = Down(128, 256, time_emb_dim)
        self.down3 = Down(256, 512, time_emb_dim)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, time_emb_dim)
        self.up1 = Up(1024, 512 // factor, time_emb_dim, bilinear)
        self.up2 = Up(512, 256 // factor, time_emb_dim, bilinear)
        self.up3 = Up(256, 128 // factor, time_emb_dim, bilinear)
        self.up4 = Up(128, 64, time_emb_dim, bilinear)
        self.outc = OutConv(64, n_channels)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x4 = self.down3(x3, t)
        x5 = self.down4(x4, t)
        x = self.up1(x5, x4, t)
        x = self.up2(x, x3, t)
        x = self.up3(x, x2, t)
        x = self.up4(x, x1, t)
        logits = self.outc(x)
        return logits
