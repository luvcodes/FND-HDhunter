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
import csv
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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

def sentiment_analysis(csv_path):
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")
    analyzer = SentimentIntensityAnalyzer()

    text_id = {}
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # Skip the header if there is one
        for row in reader:
            original_post, image_id = row[0], row[1]
            text_id[image_id] = original_post
    
    s_label = {}
    index =1
    for image_id, text in text_id.items():
        translated_text = translator(text, max_length=512)[0]['translation_text']
        sentiment_scores = analyzer.polarity_scores(translated_text)
        s_label[image_id] = sentiment_scores  # Store the entire dictionary of scores
        print("5315/",index)
        index += 1
    
    return s_label

def sentiment_analysis_cp(csv_path1, csv_path2, csv_path3):
    analyzer = SentimentIntensityAnalyzer()
    text_id = {}
    # Loop over each CSV file
    for csv_path in [csv_path1, csv_path2, csv_path3]:
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)  # Skip the header if there is one
            for row in reader:
                original_post, domain, image_id, label = row
                text_id[image_id] = original_post

    s_label = {}
    index = 1
    # Process each text entry in the dictionary
    for image_id, text in text_id.items():
        sentiment_scores = analyzer.polarity_scores(text)
        s_label[image_id] = sentiment_scores  # Store the entire dictionary of scores
        print("Processing {}/{}".format(index, len(text_id)))
        index += 1

    return s_label

def write_results_to_csv(results, output_file):
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Determine headers from the first item (assuming all items have the same structure)
        if results:
            first_key = next(iter(results))
            headers = ['Image ID'] + list(results[first_key].keys())  # Dynamic headers based on the score keys
            writer.writerow(headers)
        
        for image_id, scores in results.items():
            row = [image_id] + list(scores.values())
            writer.writerow(row)

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 初始化数据集和加载器
# 这个是Trent生成的GALIP图像文件夹路径
image_folder_path = "C:\\Users\\ryanw\\OneDrive\\Desktop\\samples_test1"
# 这个是已经融合了image_id的CSV文件路径
csv_file = "C:\\Users\\Klein\\OneDrive\\文档\\capstone\\csvFilesCollection\\final_features_fusion.csv"
#text_file = "C:\\Users\\Klein\\OneDrive\\文档\\capstone\\csvFilesCollection\\feature_with_image_id.csv"
text_file1 = "C:\\Users\\Klein\\OneDrive\\文档\\capstone\\newdata_test.csv"
text_file2 = "C:\\Users\\Klein\\OneDrive\\文档\\capstone\\newdata_train.csv"
text_file3 = "C:\\Users\\Klein\\OneDrive\\文档\\capstone\\newdata_validate.csv"

#id_s_label = sentiment_analysis(text_file)
id_s_label = sentiment_analysis_cp(text_file1,text_file2,text_file3)
#write_results_to_csv(id_s_label, "C:\\Users\\Klein\\OneDrive\\文档\\capstone\\csvFilesCollection\\output_sentiment_analysis.csv")
write_results_to_csv(id_s_label, "C:\\Users\\Klein\\OneDrive\\文档\\capstone\\csvFilesCollection\\newdata_sentiment_analysis.csv")
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

# -----------------------------------
# 这里是为了最后将image_id合并到最终的feature fusion CSV文件中。运行时要先注释下面的代码，需要重新生成CSV再打开注释！！！
# 加载由特征融合模块生成的CSV文件
# fused_features_df = pd.read_csv('TextImageFusion\\csvFilesCollection\\final01.csv')

# # 加载包含image_id的CSV文件
# image_id_df = pd.read_csv('TextImageFusion\\csvFilesCollection\\feature_with_image_id.csv')

# # 确保两个DataFrame行数一致
# if len(fused_features_df) == len(image_id_df):
#     # 添加image_id列到特征DataFrame
#     fused_features_df['image_id'] = image_id_df['image_id']
#     # 保存新的CSV文件
#     fused_features_df.to_csv('C:\\Users\\ryanw\\OneDrive\\Desktop\\FND-HDhunter\\TextImageFusion\\csvFilesCollection\\final02_with_image_id.csv', index=False)
#     print("New CSV file with image_id has been saved successfully.")
# else:
#     print("Error: The number of rows in final01.csv does not match feature_with_image_id.csv")
