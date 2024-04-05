import torch
from torch.utils.data import DataLoader
from Dataset import MyDataset
from Transforms import get_clip_transforms
from Encoding import encode_data
from Fusion import fuse_features
import clip
import pandas as pd


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    dataset = MyDataset(
        image_paths=["C:\\Users\\13322\\OneDrive\\文档\\Captone5703\\weibo dataset\\Army officer.jpg"],
        texts=[
            "Boston Marathon Bomb victim is actually Nick Vogt, former US Army Officer who lost his legs in Kandahar Afghanistan with the 1st Stryker Brig 25th Infantry Division in Nov 2011. There is more going on here than the media will tell!"],
        transform=get_clip_transforms()
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    all_fused_features = []  # Initialize a list to store fused features of all batches
    all_image_features = []
    all_text_features = []

    for images, texts in dataloader:
        images, texts = images.to(device), texts
        image_features, text_features = encode_data(model, images, texts, device)
        fused_features = fuse_features(image_features, text_features)

        # Accumulate fused features
        all_fused_features.append(fused_features.detach().cpu())
        all_image_features.append(image_features.detach().cpu())
        all_text_features.append(text_features.detach().cpu())

    # Concatenate all fused features into a single tensor
    all_fused_features_tensor = torch.cat(all_fused_features, dim=0)
    all_image_features_tensor = torch.cat(all_image_features, dim=0)
    all_text_features_tensor = torch.cat(all_text_features, dim=0)
    print(all_image_features_tensor.shape)
    print(all_text_features_tensor.shape)
    print(all_fused_features_tensor.shape)

    # Now save this tensor
    pd.DataFrame(all_image_features_tensor.numpy()).to_csv("image_features.csv", index=False)
    pd.DataFrame(all_text_features_tensor.numpy()).to_csv("text_features.csv", index=False)
    pd.DataFrame(all_fused_features_tensor.numpy()).to_csv("fused_features.csv", index=False)
    torch.save(all_fused_features_tensor, 'fused_features.pt')


if __name__ == "__main__":
    main()
