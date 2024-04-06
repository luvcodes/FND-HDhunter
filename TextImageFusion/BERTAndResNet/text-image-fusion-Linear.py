import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import BertTokenizer, BertModel
from PIL import Image


# 文本编码器
class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, text):
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        outputs = self.bert(**inputs)
        text_embeddings = outputs.last_hidden_state.mean(dim=1)
        return text_embeddings


# 图像编码器
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 768)

    def forward(self, images):
        img_embeddings = self.model(images)
        return img_embeddings


# 融合模块
class FusionModule(nn.Module):
    def __init__(self, text_encoder, image_encoder):
        super(FusionModule, self).__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.fusion = nn.Linear(1536, 768)

    def forward(self, text, images):
        text_embeddings = self.text_encoder(text)
        img_embeddings = self.image_encoder(images)

        fused_embeddings = torch.cat((text_embeddings, img_embeddings), dim=1)

        fused_embeddings = self.fusion(fused_embeddings)
        return fused_embeddings


# 图像预处理
def preprocess_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(image)
    img_tensor = img_tensor.unsqueeze(0)  # 增加批次维度
    return img_tensor


# 分类器模块
class Classifier(nn.Module):
    def __init__(self, fusion_output_size):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(fusion_output_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2)
        )

    def forward(self, fusion_output):
        logits = self.classifier(fusion_output)
        return logits


# 使用示例
text_encoder = TextEncoder()
image_encoder = ImageEncoder()
fusion_module = FusionModule(text_encoder, image_encoder)
classifier = Classifier(fusion_output_size=768)

text = "A spider man is swinging"
image_path = "../Spiderman2.jpeg"
images = preprocess_image(image_path)

fused_embeddings = fusion_module(text, images)
logits = classifier(fused_embeddings)
predicted = torch.argmax(logits, dim=1)

if predicted == 0:
    print("The text and image are predicted as real.")
else:
    print("The text and image are predicted as fake.")
