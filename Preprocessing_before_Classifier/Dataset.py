from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class MyDataset(Dataset):
    def __init__(self, image_paths, texts, transform=None):
        self.image_paths = image_paths
        self.texts = texts
        self.transform = transform

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        text = self.texts[idx]
        if self.transform:
            image = self.transform(image)
        return image, text