# import os
# from PIL import Image
# from torch.utils.data import Dataset
# from torchvision import transforms


# class CUBDataset(Dataset):
#     def __init__(self, image_root, caption_root, transform=None):
#         self.image_root = image_root
#         self.caption_root = caption_root
#         self.transform = transform if transform else transforms.Compose([
#             transforms.Resize((512, 512)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5], [0.5])  # Normalization used by Stable Diffusion
#         ])

#         self.samples = []
#         for subdir, _, files in os.walk(image_root):
#             for file in files:
#                 if file.endswith('.jpg'):
#                     rel_img_path = os.path.relpath(os.path.join(subdir, file), image_root)
#                     rel_txt_path = rel_img_path.replace('.jpg', '.txt')
#                     caption_path = os.path.join(caption_root, rel_txt_path)

#                     if os.path.exists(caption_path):
#                         self.samples.append((rel_img_path, rel_txt_path))
                        
#                         if len(self.samples) >= 20:
#                             break


#     def __len__(self):
#         return len(self.samples)

#     # def __getitem__(self, idx):
#     #     rel_img_path, rel_txt_path = self.samples[idx]
#     #     full_img_path = os.path.join(self.image_root, rel_img_path)
#     #     full_txt_path = os.path.join(self.caption_root, rel_txt_path)

#     #     image = Image.open(full_img_path).convert("RGB")
#     #     image = self.transform(image)

#     #     with open(full_txt_path, 'r') as f:
#     #         caption = f.read().strip()

#     #     return {
#     #         "image": image,
#     #         "caption": caption
#     #     }

#     def __getitem__(self, idx):
#         image = Image.open(full_img_path).convert("RGB")
#         image = self.transform(image)

#         with open(full_txt_path, 'r') as f:
#             caption = f.read().strip()

#         # Add this if you're returning tensors directly (for custom training loop)
#         if torch.backends.mps.is_available():
#             image = image.to("mps")

#         return {"image": image, "txt": caption}



import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class CUBDataset(Dataset):
    def __init__(self, image_root, captions_root, transform=None):
        self.image_root = image_root
        self.captions_root = captions_root

        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),  # Converts to [C, H, W]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])


        self.samples = []
        for subdir, _, files in os.walk(image_root):
            for file in files:
                if file.endswith('.jpg'):
                    rel_img_path = os.path.relpath(os.path.join(subdir, file), image_root)
                    rel_txt_path = rel_img_path.replace('.jpg', '.txt')
                    caption_path = os.path.join(captions_root, rel_txt_path)

                    if os.path.exists(caption_path):
                        self.samples.append((rel_img_path, rel_txt_path))
                        if len(self.samples) >= 20:
                            break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_img_path, rel_txt_path = self.samples[idx]
        full_img_path = os.path.join(self.image_root, rel_img_path)
        full_txt_path = os.path.join(self.captions_root, rel_txt_path)

        image = Image.open(full_img_path).convert("RGB")
        print("[DEBUG] Image shape",image.shape)
        image = self.transform(image)
        image = image.permute(2, 0, 1) if image.shape[0] != 3 else image

        print(f"[DEBUG] Image tensor shape: {image.shape} from file: {rel_img_path}")

        if image.shape[0] != 3:
            raise RuntimeError(f"Expected image with 3 channels, but got {image.shape}")

        with open(full_txt_path, 'r') as f:
            caption = f.read().strip()

        return {"image": image, "txt": caption}
