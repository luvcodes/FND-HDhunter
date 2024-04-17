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

class ImageEncoder(nn.Module):
    def __init__(self, model_type='resnet'):
        super(ImageEncoder, self).__init__()
        if model_type == 'resnet':
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.adjustment_layer = nn.Linear(1000, 1024)  # 1024 dimensions

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
            self.adjustment_layer = nn.Linear(768, 1024)  # 1024 dimensions

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
        self.fusion = nn.Linear(fusion_output_size * 2, fusion_output_size)  # adjust linear layer to match 1024 dimensions

    def forward(self, text_embeddings, img_embeddings):
        text_embeddings = text_embeddings.unsqueeze(0)  # Batch size 1 for multihead attention
        img_embeddings = img_embeddings.unsqueeze(0)
        attn_output, _ = self.cross_attention(text_embeddings, img_embeddings, img_embeddings)
        fused_embeddings = torch.cat((attn_output.squeeze(0), img_embeddings.squeeze(0)), dim=1)
        fused_embeddings = self.fusion(fused_embeddings)
        return fused_embeddings


# 数据集定义
class NewsDataset(Dataset):
    def __init__(self, pickle_file, transform=None):
        with open(pickle_file, 'rb') as f:
            self.data = pickle.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data['label'])

    def __getitem__(self, idx):
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
        return original_post, image

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 初始化编码器和数据集
text_encoder = TextEncoder(model_type='bert')
image_encoder = ImageEncoder(model_type='resnet')
fusion_module = CrossModalFusion(text_encoder, image_encoder)
dataset = NewsDataset('FND-HDhunter\\TextImageFusion\\datasets\\datasets_pickle1\\train.pkl', transform=transform)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# 特征提取和保存为CSV
def extract_features_and_save(data_loader, fusion_module, csv_file):
    features = []
    with torch.no_grad():
        for idx, (original_post, images) in enumerate(data_loader):
            try:
                # 打印当前批次处理信息
                print(f"Processing batch {idx+1}/{len(data_loader)}...")
                if isinstance(original_post, str):
                    original_post = [original_post]  # 确保为列表
                text_features = text_encoder(original_post)
                image_features = image_encoder(images)
                fused_features = fusion_module(text_features, image_features).cpu().numpy()
                features.append(fused_features)

                # 每处理一个批次就实时保存，不再设置间隔
                partial_features_array = np.vstack(features)
                mode = 'a' if os.path.exists(csv_file) else 'w'
                pd.DataFrame(partial_features_array).to_csv(csv_file, mode=mode, header=not os.path.exists(csv_file), index=False)
                features = []  # 清空已保存的特征以释放内存
            except Exception as e:
                print(f"Error processing batch {idx+1}: {e}")


# 检查路径是否存在，如果不存在则创建
output_dir = 'C:/Users/ryanw/OneDrive/Desktop/Project'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

csv_file = os.path.join(output_dir, 'features3.csv')
extract_features_and_save(data_loader, fusion_module, csv_file=csv_file)
