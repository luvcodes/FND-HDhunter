import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from transformers import BertTokenizer, BertModel
from PIL import Image
import os

# 定义图像编码器
class ImageEncoder(nn.Module):
    def __init__(self, model_type='resnet'):
        super(ImageEncoder, self).__init__()
        if model_type == 'resnet':
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.adjustment_layer = nn.Linear(1000, 1024)  # Adjust to 1024 dimensions

    def forward(self, images):
        img_embeddings = self.model(images)
        img_embeddings = self.adjustment_layer(img_embeddings)
        return img_embeddings

# 定义文本编码器
class TextEncoder(nn.Module):
    def __init__(self, model_type='bert'):
        super(TextEncoder, self).__init__()
        if model_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased')
            self.adjustment_layer = nn.Linear(768, 1024)  # Adjust to 1024 dimensions

    def forward(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = self.model(**inputs)
        text_embeddings = outputs.last_hidden_state.mean(dim=1)
        text_embeddings = self.adjustment_layer(text_embeddings)
        return text_embeddings

# 定义交叉模态融合模块
class CrossModalFusion(nn.Module):
    def __init__(self, text_encoder, image_encoder, fusion_output_size=1024):
        super(CrossModalFusion, self).__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.cross_attention = nn.MultiheadAttention(embed_dim=fusion_output_size, num_heads=8)
        self.fusion = nn.Linear(fusion_output_size * 2, 1536)

    def forward(self, text_embeddings, img_embeddings):
        # print("Before unsqueeze - Text embeddings shape:", text_embeddings.shape)
        # print("Before unsqueeze - Image embeddings shape:", img_embeddings.shape)
        # 确保图像和文本的嵌入只有三个维度
        if text_embeddings.dim() == 2:
            text_embeddings = text_embeddings.unsqueeze(0)
        if img_embeddings.dim() == 2:
            img_embeddings = img_embeddings.unsqueeze(0)
        # print("After unsqueeze - Text embeddings shape:", text_embeddings.shape)
        # print("After unsqueeze - Image embeddings shape:", img_embeddings.shape)
        attn_output, _ = self.cross_attention(text_embeddings, img_embeddings, img_embeddings)
        fused_embeddings = torch.cat((attn_output.squeeze(0), img_embeddings.squeeze(0)), dim=1)
        fused_embeddings = self.fusion(fused_embeddings)
        return fused_embeddings


# 定义数据集
class ImageDataset(Dataset):
    def __init__(self, folder_path, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.folder_path = folder_path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_id = self.data.iloc[idx, -1]  # 假设image_id位于最后一列
        image_path = os.path.join(self.folder_path, f"{image_id}.jpg")
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        features = torch.tensor(self.data.iloc[idx, :-1].astype(float).values, dtype=torch.float32)
        return image, features

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 初始化数据集和加载器
image_folder_path = "C:\\Users\\ryanw\\OneDrive\\Desktop\\samples_test1"
csv_file = "C:\\Users\\ryanw\\OneDrive\\Desktop\\FND-HDhunter\\TextImageFusion\\csvFilesCollection\\final_features_fusion.csv"
dataset = ImageDataset(image_folder_path, csv_file, transform=transform)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# 初始化编码器和融合模块
text_encoder = TextEncoder(model_type='bert')
image_encoder = ImageEncoder(model_type='resnet')
fusion_module = CrossModalFusion(text_encoder, image_encoder)

# 特征提取和保存函数
def extract_features_and_save(data_loader, image_encoder, fusion_module, csv_output):
    with torch.no_grad():
        fused_features = []
        for idx, (images, features) in enumerate(data_loader):
            print(f"Processing image {idx+1}/{len(data_loader)}...")
            images = images.float()
            image_feature = image_encoder(images).cpu()
            # 确保在传递给融合模块前，特征只被增加到正确的维度
            if features.dim() == 1:
                features = features.unsqueeze(0)
            # print("Features shape for attention:", features.shape)
            # print("Image feature shape for attention:", image_feature.shape)
            fused_feature = fusion_module(features, image_feature)
            fused_features.append(fused_feature.squeeze(0).numpy())

        # 保存融合特征到CSV文件
        fused_features_array = np.vstack(fused_features)
        pd.DataFrame(fused_features_array).to_csv(csv_output, index=False)


csv_output = "C:\\Users\\ryanw\\OneDrive\\Desktop\\FND-HDhunter\\TextImageFusion\\csvFilesCollection\\final01.csv"
extract_features_and_save(data_loader, image_encoder, fusion_module, csv_output)
