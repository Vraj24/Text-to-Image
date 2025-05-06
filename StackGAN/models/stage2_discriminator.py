import torch
import torch.nn as nn
import torch.nn.functional as F

class Stage2Discriminator(nn.Module):
    def __init__(self, text_dim=1024):
        super().__init__()
        # downsample 256→16
        self.img_net = nn.Sequential(
            nn.Conv2d(3,   64, 4, 2, 1),   # 256→128
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, 4, 2, 1),   # 128→64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128,256,4,2,1),      # 64→32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256,512,4,2,1),      # 32→16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )
        # project text
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.LeakyReLU(0.2, True)
        )
        # fuse + final conv
        self.joint = nn.Sequential(
            nn.Conv2d(512 + 512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, 4)          # global 1×1 → real/fake
        )

    def forward(self, img, text_emb):
        h = self.img_net(img)               # → (B,512,16,16)
        t = self.text_proj(text_emb)        # → (B,512)
        t = t.view(t.size(0), t.size(1), 1, 1).repeat(1,1,h.size(2),h.size(3))
        x = torch.cat([h, t], dim=1)        # → (B,1024,16,16)
        out = self.joint(x)                 # → (B,1,13,13) (or similar)
        return out.view(-1)                 # flatten → (B,)
