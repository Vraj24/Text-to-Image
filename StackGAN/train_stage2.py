import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from utils.data_loader import CUBDataset
from models.stage1_generator import Stage1Generator
from models.stage2_generator import Stage2Generator
from models.stage2_discriminator import Stage2Discriminator
from losses.gan_loss import generator_loss, discriminator_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

big_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

def main():
    z_dim, text_dim      = 100, 1024
    batch_size, lr       = 32, 2e-4
    epochs, save_every   = 100, 5
    checkpoint_path = "output22/checkpoint.pth"

    dataset = CUBDataset(
        img_root='data/CUB_200_2011/images',
        embedding_root='data/cub_text_embeddings',
        split='train'
    )
    dataset.transform = big_transform
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    stage1 = Stage1Generator(z_dim, text_dim).to(device)
    stage1_ckpt = torch.load('output/netG_epoch_99.pth', map_location=device)
    stage1.load_state_dict(stage1_ckpt)
    stage1.eval()

    netG2 = Stage2Generator(z_dim, text_dim).to(device)
    netD2 = Stage2Discriminator(text_dim).to(device)

    optG = torch.optim.Adam(netG2.parameters(), lr=lr, betas=(0.5,0.999))
    optD = torch.optim.Adam(netD2.parameters(), lr=lr, betas=(0.5,0.999))


    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        try:
            netG2.load_state_dict(checkpoint["netG2"])
            netD2.load_state_dict(checkpoint["netD2"])
        except Exception as e:
            print("Failed to fully load model. Loading partial weights:", str(e))
            netG2.load_state_dict({k: v for k, v in checkpoint["netG2"].items() if k in netG2.state_dict()}, strict=False)
            netD2.load_state_dict({k: v for k, v in checkpoint["netD2"].items() if k in netD2.state_dict()}, strict=False)

        optG.load_state_dict(checkpoint["optG"])
        optD.load_state_dict(checkpoint["optD"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        for i, (real_imgs, text_embs) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            text_embs = text_embs.to(device)
            z = torch.randn(real_imgs.size(0), z_dim).to(device)

            with torch.no_grad():
                fake_imgs_stage1 = stage1(z, text_embs)

            # === DEBUG Save Stage-I input ===
            if i == 0 and epoch % save_every == 0:
                save_image(fake_imgs_stage1[:16], f"output22/debug_stage1_input_epoch_{epoch}.png", nrow=4, normalize=True)

            # === DEBUG Print statistics ===
            if i == 0 and epoch % save_every == 0:
                print(f"Epoch {epoch} | Stage-I Mean: {fake_imgs_stage1.mean():.4f}, Std: {fake_imgs_stage1.std():.4f}")

            # === Train Discriminator ===
            netD2.zero_grad()
            fake_imgs_stage2 = netG2(fake_imgs_stage1, text_embs).detach()
            real_preds = netD2(real_imgs, text_embs)
            fake_preds = netD2(fake_imgs_stage2, text_embs)
            d_loss = discriminator_loss(real_preds, fake_preds)
            d_loss.backward()
            optD.step()

            # === Train Generator ===
            netG2.zero_grad()
            fake_imgs_stage2 = netG2(fake_imgs_stage1, text_embs)
            fake_preds = netD2(fake_imgs_stage2, text_embs)
            g_loss = generator_loss(fake_preds)
            g_loss.backward()
            optG.step()

            if i % 10 == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                      f"D2 Loss: {d_loss.item():.4f} | G2 Loss: {g_loss.item():.4f}")

                # Optional: GPU usage print (uncomment if needed)
                # os.system("nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader")

        # === Save Outputs and Checkpoints ===
        if epoch % save_every == 0 or epoch == epochs - 1:
            os.makedirs("output22", exist_ok=True)
            save_image(fake_imgs_stage2[:16], f"output22/fake_epoch_{epoch}.png", nrow=4, normalize=True)
            torch.save(netG2.state_dict(), f"output22/netG2_epoch_{epoch}.pth")
            torch.save(netD2.state_dict(), f"output22/netD2_epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "netG2": netG2.state_dict(),
                "netD2": netD2.state_dict(),
                "optG": optG.state_dict(),
                "optD": optD.state_dict()
            }, checkpoint_path)

if __name__ == '__main__':
    main()
