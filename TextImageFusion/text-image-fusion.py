import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader


# 数据集和数据加载部分代码保持不变

# 假设我们有一个包含文本和图像路径的数据集
class TextImageDataset(Dataset):
    def __init__(self, texts, image_paths, labels):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.texts = texts
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # 文本处理
        text = self.texts[idx]

        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        # 图像处理
        image = Image.open(self.image_paths[idx]).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = transform(image)

        label = torch.tensor(self.labels[idx])

        return input_ids, attention_mask, image, label


class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        self.image_model = resnet50(pretrained=True)

        # 保存原始 fc 层的 in_features
        image_features_dim = self.image_model.fc.in_features

        # 替换全连接层为 Identity
        self.image_model.fc = nn.Identity()

        # 根据你的具体情况设置融合后的特征维度
        fusion_dim = self.text_model.config.hidden_size + image_features_dim

        # 定义其他层
        self.fusion_layer = nn.Linear(fusion_dim, 512)
        # 假设是一个二分类问题
        self.output_layer = nn.Linear(512, 1)

    def forward(self, input_ids, attention_mask, images):
        text_features = self.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        image_features = self.image_model(images)
        # 特征融合
        fusion = torch.cat((text_features, image_features), dim=1)
        fusion = self.fusion_layer(fusion)  # Additional operation layers
        fusion = torch.relu(fusion)
        output = self.output_layer(fusion)  # output layer

        return output


class FeatureExtractionModel(nn.Module):
    def __init__(self):
        super(FeatureExtractionModel, self).__init__()

        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        self.image_model = resnet50(pretrained=True)

        # 将图像模型的最后一个全连接层替换为Identity，以便直接获得图像特征
        self.image_model.fc = nn.Identity()

    def forward(self, input_ids, attention_mask, images):
        # 获取文本特征
        text_features = self.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        # 获取图像特征
        image_features = self.image_model(images)
        return text_features, image_features


# 创建模型实例
model = FeatureExtractionModel()

# 数据集准备
texts = ['This is a spider man', 'This is actually a spider man']
image_paths = ['cube.jpeg', 'Spiderman1.png']
labels = [0, 1]  # 假定：0表示假，1表示真

# 数据加载
dataset = TextImageDataset(texts, image_paths, labels)
dataloader = DataLoader(dataset, batch_size=2)

# 推理过程
model.eval()
with torch.no_grad():
    for input_ids, attention_mask, images, labels in dataloader:
        text_features, image_features = model(input_ids, attention_mask, images)
        print("Text Features:", text_features)
        print("Image Features:", image_features)





