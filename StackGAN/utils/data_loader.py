import os
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CUBDataset(Dataset):
    def __init__(self, img_root, embedding_root, split='train'):
        self.img_root = img_root
        self.split = split
        self.emb_root = os.path.join(embedding_root, split)

        self.transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3)
        ])

        # Load filenames
        with open(os.path.join(self.emb_root, 'filenames.pickle'), 'rb') as f:
            self.filenames = pickle.load(f)

        # Load embeddings (.pickle): shape should be (num_images, 10, embedding_dim)
        with open(os.path.join(self.emb_root, 'char-CNN-RNN-embeddings.pickle'), 'rb') as f:
            self.embeddings = pickle.load(f, encoding='latin1')

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        fname = self.filenames[index]
        img_path = os.path.join(self.img_root, fname + ".jpg")
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        emb = self.embeddings[index]  # shape: (10, embedding_dim)
        emb_idx = np.random.randint(0, 10)
        text_emb = emb[emb_idx]

        return image, text_emb
 