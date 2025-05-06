import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(True),
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel)
        )

    def forward(self, x):
        return F.relu(x + self.block(x), True)

class Stage2Generator(nn.Module):
    def __init__(self, z_dim=100, text_dim=1024):
        super().__init__()
        # 1) encode Stage-I image (64×64) → feature map (512 channels, 16×16)
        self.encoder = nn.Sequential(
            nn.Conv2d(3,   128, 3, 1, 1),  
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, 4, 2, 1),  # 32×32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, 4, 2, 1),  # 16×16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )
        # 2) project text embedding → (batch, 512)
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.ReLU(True)
        )
        # 3) fuse image features + text
        self.joint_conv = nn.Sequential(
            nn.Conv2d(512 + 512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        # 4) a few residual blocks
        self.residuals = nn.Sequential(
            *[ResidualBlock(512) for _ in range(4)]
        )
        # 5) upsample from 16→256 through four 2× steps
        self.upsamples = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 16→32
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 32→64
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128,  64, 4, 2, 1),  # 64→128
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # <<< NEW: add one more to go 128→256
            nn.ConvTranspose2d( 64,  32, 4, 2, 1),  # 128→256
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        self.output = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, stage1_img, text_emb):
        """
        stage1_img: (B,3,64,64)
        text_emb:   (B,1024)
        → returns (B,3,256,256)
        """
        h = self.encoder(stage1_img)        # → (B,512,16,16)
        t = self.text_proj(text_emb)        # → (B,512)
        t = t.view(t.size(0), t.size(1), 1, 1).repeat(1, 1, h.size(2), h.size(3))
        h = torch.cat([h, t], dim=1)        # → (B,1024,16,16)
        h = self.joint_conv(h)              # → (B,512,16,16)
        h = self.residuals(h)               # → (B,512,16,16)
        h = self.upsamples(h)               # → (B,32,256,256)
        return self.output(h)               # → (B,3,256,256)
