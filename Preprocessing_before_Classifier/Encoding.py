import torch
import clip


def encode_data(model, images, texts, device):
    with torch.no_grad():
        image_features = model.encode_image(images).float()
        text_features = model.encode_text(clip.tokenize(texts).to(device)).float()
    return image_features, text_features

