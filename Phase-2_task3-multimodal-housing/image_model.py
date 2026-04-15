import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import os

# Image Dataset
class HouseImageDataset(Dataset):
    def __init__(self, df, image_dir, transform):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, f"{row['id']}.jpg")
        
        # Fallback: use a blank image if file doesn't exist
        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
        else:
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        
        return self.transform(image), row["id"]

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                         std=[0.229, 0.224, 0.225])
])

# ResNet18 as Feature Extractor
# Remove final classification layer — we only want the 512-d feature vector
resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Identity()
resnet.eval()

def extract_features(df, image_dir, batch_size=32):
    dataset = HouseImageDataset(df, image_dir, transform)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_features = []
    all_ids = []

    with torch.no_grad():
        for images, ids in loader:
            features = resnet(images).numpy()
            all_features.append(features)
            all_ids.extend(ids.numpy())

    features_array = np.vstack(all_features)
    feat_df = pd.DataFrame(features_array, columns=[f"img_feat_{i}" for i in range(512)])
    feat_df["id"] = all_ids
    return feat_df

if __name__ == "__main__":
    df = pd.read_csv("data/kc_house_data.csv")
    print("Extracting image features...")
    feat_df = extract_features(df, image_dir="data/images/")
    feat_df.to_csv("data/image_features.csv", index=False)
    print("Saved: data/image_features.csv")