import torch
import torch.nn as nn

class Stage1Generator(nn.Module):
    def __init__(self, z_dim=100, text_dim=256, img_size=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim + text_dim, 128 * 8 * 4 * 4),
            nn.BatchNorm1d(128 * 8 * 4 * 4),
            nn.ReLU(True)
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),  # 8x8
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4, 2, 1),    # 64x64
            nn.Tanh()
        )

    def forward(self, z, text_embedding):
        x = torch.cat((z, text_embedding), 1)
        x = self.fc(x).view(-1, 1024, 4, 4)
        return self.deconv(x)
