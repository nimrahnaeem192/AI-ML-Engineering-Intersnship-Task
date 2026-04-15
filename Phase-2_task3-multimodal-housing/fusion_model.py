import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Load & Merge
tabular = pd.read_csv("data/kc_house_data.csv")
img_feats = pd.read_csv("data/image_features.csv")

df = tabular.merge(img_feats, on="id", how="inner")

# Drop columns not useful for prediction
drop_cols = ["id", "date", "price", "zipcode", "lat", "long"]
X = df.drop(columns=drop_cols)
y = df["price"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# PyTorch Dataset
class HouseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(HouseDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader  = DataLoader(HouseDataset(X_test, y_test),  batch_size=64)

# Fusion MLP
# Input = tabular features + 512 image features
input_dim = X_train.shape[1]

class FusionNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

model = FusionNet(input_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Train
for epoch in range(30):
    model.train()
    for X_batch, y_batch in train_loader:
        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/30 — Loss: {loss.item():.2f}")

# Evaluate
model.eval()
preds, actuals = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds.extend(model(X_batch).squeeze().numpy())
        actuals.extend(y_batch.squeeze().numpy())

mae  = mean_absolute_error(actuals, preds)
rmse = np.sqrt(mean_squared_error(actuals, preds))
print(f"\nMAE:  ${mae:,.0f}")
print(f"RMSE: ${rmse:,.0f}")

torch.save(model.state_dict(), "fusion_model.pth")
print("Model saved: fusion_model.pth")