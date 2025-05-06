# # train_stage1.py

# from utils.data_loader import CUBDataset
# from torch.utils.data import DataLoader

# def main():
#     dataset = CUBDataset(
#     img_root='data/cub_dataset/CUB_200_2011/CUB_200_2011/images',
#     embedding_root='data/cub_text_embeddings',
#     split='train'
#     )

#     img, emb = dataset[0]
#     print("Image shape:", img.shape)
#     print("Text embedding shape:", emb.shape)



#     # dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

#     # for images, text_embeddings in dataloader:
#     #     print("Image batch shape:", images.shape)
#     #     print("Text embedding shape:", text_embeddings.shape)
#     #     break

# if __name__ == '__main__':
#     main()



# -----------------------------------------------------------

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from utils.data_loader import CUBDataset
from models.stage1_generator import Stage1Generator
from models.stage1_discriminator import Stage1Discriminator
from losses.gan_loss import generator_loss, discriminator_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)


def main():
    # === Hyperparameters ===
    z_dim = 100
    text_dim = 1024
    batch_size = 64
    lr = 0.0002
    epochs = 100
    save_every = 5

    # === Data ===
    dataset = CUBDataset(
        img_root='data/CUB_200_2011/images',
        embedding_root='data/cub_text_embeddings',
        split='train'
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # === Models ===
    netG = Stage1Generator(z_dim=z_dim, text_dim=text_dim).to(device)
    netD = Stage1Discriminator(text_dim=text_dim).to(device)

    # === Optimizers ===
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for i, (real_imgs, text_embs) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            text_embs = text_embs.to(device)

            batch_size = real_imgs.size(0)
            z = torch.randn(batch_size, z_dim).to(device)

            # === Train Discriminator ===
            netD.zero_grad()
            fake_imgs = netG(z, text_embs).detach()
            real_preds = netD(real_imgs, text_embs)
            fake_preds = netD(fake_imgs, text_embs)
            d_loss = discriminator_loss(real_preds, fake_preds)
            d_loss.backward()
            optimizerD.step()

            # === Train Generator ===
            netG.zero_grad()
            fake_imgs = netG(z, text_embs)
            fake_preds = netD(fake_imgs, text_embs)
            g_loss = generator_loss(fake_preds)
            g_loss.backward()
            optimizerG.step()

            if i % 20 == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                      f"D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

        # === Save sample images
        if epoch % save_every == 0 or epoch == epochs - 1:
            os.makedirs("output", exist_ok=True)
            save_image(fake_imgs[:16], f"output/fake_epoch_{epoch}.png", nrow=4, normalize=True)

            # Save checkpoints
            torch.save(netG.state_dict(), f"output/netG_epoch_{epoch}.pth")
            torch.save(netD.state_dict(), f"output/netD_epoch_{epoch}.pth")

if __name__ == '__main__':
    main()
