from torchvision.transforms import Compose, Resize, Normalize, ToTensor

def get_clip_transforms():
    return Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])

