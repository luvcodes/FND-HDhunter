import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights, vgg16, VGG16_Weights


# 修改后的文本编码器，支持BERT和RoBERTa
class TextEncoder(nn.Module):
    def __init__(self, model_type='bert'):
        super(TextEncoder, self).__init__()
        if model_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased')
        elif model_type == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.model = RobertaModel.from_pretrained('roberta-base')

    def forward(self, text):
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        outputs = self.model(**inputs)
        text_embeddings = outputs.last_hidden_state.mean(dim=1)
        return text_embeddings


# 修改后的图像编码器，支持ResNet50和VGG
class ImageEncoder(nn.Module):
    def __init__(self, model_type='resnet'):
        super(ImageEncoder, self).__init__()
        if model_type == 'resnet':
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        elif model_type == 'vgg':
            self.model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            # 修改VGG的分类器以匹配输出维度
            self.model.classifier[6] = nn.Linear(4096, 768)
        # 对于ResNet，我们使用fc层来调整维度；对于VGG，我们已经修改了分类器的最后一层
        self.adjustment_layer = nn.Linear(1000, 768) if model_type == 'resnet' else None

    def forward(self, images):
        img_embeddings = self.model(images)
        # 如果是ResNet，我们需要调整维度
        if self.adjustment_layer is not None:
            img_embeddings = self.adjustment_layer(img_embeddings)
        return img_embeddings


# 融合模块
class FusionModule(nn.Module):
    def __init__(self, text_encoder, image_encoder):
        super(FusionModule, self).__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        # 维持原有的融合策略
        self.fusion = nn.Linear(1536, 768)

    def forward(self, text, images):
        text_embeddings = self.text_encoder(text)
        img_embeddings = self.image_encoder(images)
        fused_embeddings = torch.cat((text_embeddings, img_embeddings), dim=1)
        fused_embeddings = self.fusion(fused_embeddings)
        return fused_embeddings


# 图像预处理函数，维持不变
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


# 如何使用BERT + ResNet和RoBERTa + VGG两种组合
def run_example(model_type='bert_resnet', text="This is a text", image_path="image.png"):
    text_model_type, image_model_type = model_type.split('_')
    text_encoder = TextEncoder(model_type=text_model_type)
    image_encoder = ImageEncoder(model_type=image_model_type)
    fusion_module = FusionModule(text_encoder, image_encoder)
    classifier = Classifier(fusion_output_size=768)

    images = preprocess_image(image_path)
    fused_embeddings = fusion_module(text, images)
    logits = classifier(fused_embeddings)
    probabilities = torch.softmax(logits, dim=1).detach().numpy()

    print(f"Model: {model_type.upper()}")
    print(f"Text: {text}")
    print(f"Image Path: {image_path}")
    print(f"Predicted probabilities: Real: {probabilities[0][0]:.4f}, Fake: {probabilities[0][1]:.4f}")
    predicted = torch.argmax(logits, dim=1)
    if predicted == 0:
        print("Prediction: The text and image are predicted as real.\n")
    else:
        print("Prediction: The text and image are predicted as fake.\n")


# 示例运行
text = "This is a spiderman"
image_path = "../Spiderman2.jpeg"
run_example(model_type='bert_resnet', text=text, image_path=image_path)
run_example(model_type='roberta_vgg', text=text, image_path=image_path)
