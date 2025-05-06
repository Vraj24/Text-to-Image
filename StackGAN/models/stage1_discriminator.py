import torch
import torch.nn as nn

class Stage1Discriminator(nn.Module):
    def __init__(self, text_dim=256):
        super().__init__()
        self.img_net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),   # 32x32
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), # 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1), # 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        self.text_proj = nn.Linear(text_dim, 512)
        self.out = nn.Conv2d(1024, 1, 4)

    def forward(self, img, text_embedding):
        x_img = self.img_net(img)
        text = self.text_proj(text_embedding).unsqueeze(2).unsqueeze(3)
        text = text.repeat(1, 1, 4, 4)
        x = torch.cat((x_img, text), 1)
        return self.out(x).view(-1)
