import torch

def fuse_features(image_features, text_features):
    # Simple concatenation
    fused_features = torch.cat((image_features, text_features), dim=1)
    return fused_features

