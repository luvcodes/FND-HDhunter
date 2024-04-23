import pickle
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BertTokenizer, BertModel
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn
import pandas as pd
import numpy as np
import os
import glob

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

class CrossModalFusion(nn.Module):
    def __init__(self, text_encoder, image_encoder, fusion_output_size=1024):
        super(CrossModalFusion, self).__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        # 注意，这里的embed_dim仍然是fusion_output_size，因为你可能仍然希望在注意力机制前保持这个维度
        self.cross_attention = nn.MultiheadAttention(embed_dim=fusion_output_size, num_heads=8)
        # 修改这里的self.fusion，将输出维度改为1536
        # fusion_output_size * 2 = 2048 -> 1536
        self.fusion = nn.Linear(fusion_output_size * 2, 1536)  

    def forward(self, text_embeddings, img_embeddings):
        # Increase the batch dimensions
        text_embeddings = text_embeddings.unsqueeze(0)  
        # Increase the batch dimensions
        img_embeddings = img_embeddings.unsqueeze(0)    
        attn_output, _ = self.cross_attention(text_embeddings, img_embeddings, img_embeddings)
        # 合并注意力输出和图像嵌入
        fused_embeddings = torch.cat((attn_output.squeeze(0), img_embeddings.squeeze(0)), dim=1)
        # 应用线性变换
        fused_embeddings = self.fusion(fused_embeddings)
        return fused_embeddings


# 设置文件夹和文件路径
image_folder_path = "C:\\Users\\ryanw\\OneDrive\\Desktop\\samples_test1"
csv_text_features = "TextImageFusion\\feature_fusion.csv"
csv_output = "C:\\Users\\ryanw\\OneDrive\\Desktop\\FND-HDhunter\\feature_fusion2.csv"

class ImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.image_paths = glob.glob(os.path.join(folder_path, '*.jpg'))  # 获取所有jpg图片路径
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 初始化编码器和数据集
text_encoder = TextEncoder(model_type='bert')  # 请确保已经定义并可用
image_encoder = ImageEncoder(model_type='resnet')  # 请确保已经定义并可用
fusion_module = CrossModalFusion(text_encoder, image_encoder)  # 请确保已经定义并可用

dataset = ImageDataset(image_folder_path, transform=transform)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# 读取文本特征CSV文件
text_features = pd.read_csv(csv_text_features).values

# 特征提取和保存为CSV
def extract_features_and_save(data_loader, text_features, image_encoder, fusion_module, csv_output):
    with torch.no_grad():
        fused_features = []
        for idx, images in enumerate(data_loader):  # images 已经是 batch，所以直接传入模型
            print(f"Processing image {idx+1}/{len(data_loader)}...")
            # 转换图像张量到float32
            images = images.float()
            image_feature = image_encoder(images).cpu()
            
            # 从numpy数组转换文本特征张量，并转换为float32
            text_feature = torch.tensor(text_features[idx], dtype=torch.float32).unsqueeze(0).cpu()
            
            # 融合特征，并确保输出是numpy数组
            fused_feature = fusion_module(text_feature, image_feature).cpu().numpy()
            fused_features.append(fused_feature.squeeze(0))  # 移除batch维度
            
        # 保存融合特征到CSV文件
        fused_features_array = np.vstack(fused_features)
        pd.DataFrame(fused_features_array).to_csv(csv_output, index=False)

# 运行特征提取和保存
extract_features_and_save(data_loader, text_features, image_encoder, fusion_module, csv_output)