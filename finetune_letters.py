import os
import random
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from model import RealTimeCharCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available())
print("Device:", device)

random.seed(42)
torch.manual_seed(42)

# 36-class mapping: 0-9, A-Z
classes = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]

# A-Z 在 36 類模型中的 index 是 10~35
letter_to_full_index = {chr(ord('A') + i): 10 + i for i in range(26)}

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.RandomAffine(
        degrees=10,
        translate=(0.08, 0.08),
        scale=(0.9, 1.1)
    ),
    transforms.ToTensor()
])

# 讀你自己的 az_dataset
base_dataset = datasets.ImageFolder("az_dataset", transform=transform)
print("ImageFolder classes:", base_dataset.classes)

# 每個字母取固定數量
samples_per_class = 1000
class_indices = defaultdict(list)

# 用 samples，不要真的讀圖
for idx, (path, label) in enumerate(base_dataset.samples):
    class_indices[label].append(idx)

selected_indices = []

for label in sorted(class_indices.keys()):
    indices = class_indices[label][:]
    random.shuffle(indices)
    selected_indices.extend(indices[:samples_per_class])

random.shuffle(selected_indices)

dataset = Subset(base_dataset, selected_indices)
print("Total selected:", len(selected_indices))

loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

# 載入原本 EMNIST 訓練好的 36 類模型
model = RealTimeCharCNN(num_classes=36).to(device)
print("Model device:", next(model.parameters()).device)

model.load_state_dict(
    torch.load("weights/emnist36_model.pt", map_location=device, weights_only=True)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 3

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)

        # ImageFolder labels: 0~25 (A~Z)
        # 映射成 36 類模型裡的 10~35
        mapped_y = []
        for idx in y.tolist():
            letter = base_dataset.classes[idx]
            mapped_y.append(letter_to_full_index[letter])

        y = torch.tensor(mapped_y, dtype=torch.long).to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

os.makedirs("weights", exist_ok=True)
torch.save(model.state_dict(), "weights/emnist36_letters_ft.pt")
print("Saved: weights/emnist36_letters_ft.pt")