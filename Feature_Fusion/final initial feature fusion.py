import pickle
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BertTokenizer, BertModel
# from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet50
from torch import nn
import pandas as pd
import numpy as np
import os

class ImageEncoder(nn.Module):
    def __init__(self, model_type='resnet'):
        super(ImageEncoder, self).__init__()
        if model_type == 'resnet':
            # self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.model = resnet50(pretrained = True)
            # 1024 dimensions
            self.adjustment_layer = nn.Linear(1000, 1024)

    def forward(self, images):
        img_embeddings = self.model(images)
        img_embeddings = self.adjustment_layer(img_embeddings)
        return img_embeddings

class TextEncoder(nn.Module):
    def __init__(self, model_type='bert'):
        super(TextEncoder, self).__init__()
        if model_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased')
            # 1024 dimensions
            self.adjustment_layer = nn.Linear(768, 1024)

    def forward(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = self.model(**inputs)
        text_embeddings = outputs.last_hidden_state.mean(dim=1)
        text_embeddings = self.adjustment_layer(text_embeddings)
        return text_embeddings

class CrossModalFusion(nn.Module):
    def __init__(self, text_encoder, image_encoder, fusion_output_size=1024):
        super(CrossModalFusion, self).__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.cross_attention = nn.MultiheadAttention(embed_dim=fusion_output_size, num_heads=8)
        # adjust linear layer to match 1024 dimensions
        self.fusion = nn.Linear(fusion_output_size * 2, fusion_output_size)

    def forward(self, text_embeddings, img_embeddings):
        # Batch size 1 for multihead attention
        text_embeddings = text_embeddings.unsqueeze(0)
        img_embeddings = img_embeddings.unsqueeze(0)
        attn_output, _ = self.cross_attention(text_embeddings, img_embeddings, img_embeddings)
        fused_embeddings = torch.cat((attn_output.squeeze(0), img_embeddings.squeeze(0)), dim=1)
        fused_embeddings = self.fusion(fused_embeddings)
        return fused_embeddings


# Define the dataset of pickle
class NewsDataset(Dataset):
    def __init__(self, pickle_file, transform=None):
        with open(pickle_file, 'rb') as f:
            self.data = pickle.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data['label'])

    def __getitem__(self, idx):
        image_id = self.data['image_id'][idx]
        original_post = self.data['original_post'][idx]
        if not isinstance(original_post, str):
            original_post = ""
        image_data = self.data['image'][idx]
        if isinstance(image_data, str):
            image = Image.open(image_data).convert('RGB')
            if self.transform:
                image = self.transform(image)
        else:
            image = torch.tensor(image_data) if not isinstance(image_data, torch.Tensor) else image_data
        return original_post, image, image_id

# class NewsDataset(Dataset):
#     def __init__(self, csv_file, transform=None):
#         self.data = pd.read_csv(csv_file)
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         original_post = self.data['original_post'][idx]
#         image_id = self.data['image_id'][idx]
#         image_data = self.data['image'][idx]
#         if isinstance(image_data, str):
#             image = Image.open(image_data).convert('RGB')
#             if self.transform:
#                 image = self.transform(image)
#         else:
#             image = torch.tensor(image_data) if not isinstance(image_data, torch.Tensor) else image_data
#         return original_post, image, image_id

# class get_image_id(Dataset):
#     def __init__(self, pickle_file, transform=None):
#         with open(pickle_file, 'rb') as f:
#             self.data = pickle.load(f)
#         self.transform = transform

#     def __len__(self):
#         return len(self.data['label'])

#     def __getitem__(self, idx):
#         image_id = self.data['image_id'][idx]
#         original_post = self.data['original_post'][idx]
#         if not isinstance(original_post, str):
#             original_post = ""
#         image_data = self.data['image'][idx]
#         if isinstance(image_data, str):
#             image = Image.open(image_data).convert('RGB')
#             if self.transform:
#                 image = self.transform(image)
#         else:
#             image = torch.tensor(image_data) if not isinstance(image_data, torch.Tensor) else image_data
#         return image_id


# Preprocessing the images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize the encoders
text_encoder = TextEncoder(model_type='bert')
image_encoder = ImageEncoder(model_type='resnet')
fusion_module = CrossModalFusion(text_encoder, image_encoder)
# original_post, image, image_id = NewsDataset('"E:\\GALIP dataset\\datasets_pickle1\\train.pkl"', transform=transform)
dataset = NewsDataset("E:\GALIP dataset\\newdata10000\export_df_test.pkl", transform=transform)

data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Feature extraction and save as CSV file
# def extract_features_and_save(data_loader, fusion_module, csv_file):
#     with torch.no_grad():
#         for idx, (original_post, images, image_id) in enumerate(data_loader):
#             try:
#                 print(f"Processing batch {idx+1}/{len(data_loader)}...")
#                 if isinstance(original_post, str):
#                     original_post = [original_post]  # Ensure it is a list
#                 text_features = text_encoder(original_post)
#                 image_features = image_encoder(images)
#                 fused_features = fusion_module(text_features, image_features)

#                 # Convert image_id to a tensor and match its shape with fused_features
#                 image_id_tensor = torch.tensor([image_id], dtype=torch.float32).unsqueeze(0)

#                 # Concatenate the image_id tensor to the fused features
#                 fused_features_with_id = torch.cat((fused_features, image_id_tensor), dim=1).cpu().numpy()

#                 # Save to CSV
#                 if idx == 0:
#                     mode = 'w'
#                 else:
#                     mode = 'a'
#                 pd.DataFrame(fused_features_with_id).to_csv(csv_file, mode=mode, header=not os.path.exists(csv_file), index=False)

#             except Exception as e:
#                 print(f"Error processing batch {idx+1}: {e}")

def extract_features_and_save(data_loader, fusion_module, csv_file):
    with torch.no_grad():
        all_features = []  # Store all features and IDs in a list before saving to CSV
        for idx, (original_post, images, image_id) in enumerate(data_loader):
            print(f"Processing batch {idx+1}/{len(data_loader)}...")
            if isinstance(original_post, str):
                original_post = [original_post]  # Ensure it is a list
            text_features = text_encoder(original_post)
            image_features = image_encoder(images)
            fused_features = fusion_module(text_features, image_features).cpu().numpy()

            # Append image_id as a string, ensure it's in a format that can be written to CSV
            fused_features_with_id = np.append(fused_features, image_id)

            # Append the processed features with image ID to the list
            all_features.append(fused_features_with_id)

        # Convert all features to a DataFrame and save to CSV after processing all batches
        features_df = pd.DataFrame(all_features)
        features_df.to_csv(csv_file, index=False, header=True)

# 检查路径是否存在，如果不存在则创建
output_dir = 'E:\GALIP dataset\\newdata10000'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

csv_file = os.path.join(output_dir, 'features_newtest.csv')
extract_features_and_save(data_loader, fusion_module, csv_file=csv_file)
