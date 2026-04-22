import os
from collections import defaultdict

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from model import RealTimeCharCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomAffine(
        degrees=10,
        translate=(0.08, 0.08),
        scale=(0.9, 1.1)
    ),
    transforms.ToTensor()
])

# 0-9 + A-Z
target_classes = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]

TRAIN_PER_CLASS = 5000


class FilteredEMNISTSplit(Dataset):
    def __init__(self, split: str = "train"):
        """
        split:
            - "train": use first 5000 samples of each class
            - "test":  use remaining samples of each class
        """
        assert split in {"train", "test"}

        self.base = torchvision.datasets.EMNIST(
            root="./data",
            split="balanced",
            train=True,   # 直接用同一個大集合，自己切
            transform=transform,
            download=True
        )

        base_classes = self.base.classes

        # 只保留 0-9 + A-Z
        self.keep_indices = [i for i, c in enumerate(base_classes) if c in target_classes]

        # 把原始 label 映射到 0~35
        self.label_map = {}
        new_idx = 0
        for old_idx, c in enumerate(base_classes):
            if c in target_classes:
                self.label_map[old_idx] = new_idx
                new_idx += 1

        # 依 class 收集樣本索引
        class_to_indices = defaultdict(list)

        for i in range(len(self.base)):
            _, y = self.base[i]
            if y in self.keep_indices:
                class_to_indices[y].append(i)

        # 依每個 class 切前 5000 當 train，其餘當 test
        self.samples = []

        for old_label in sorted(class_to_indices.keys()):
            indices = class_to_indices[old_label]

            if split == "train":
                selected = indices[:TRAIN_PER_CLASS]
            else:
                selected = indices[TRAIN_PER_CLASS:]

            self.samples.extend(selected)

        self.split = split

        # 統計資訊
        print(f"[{split}] total samples: {len(self.samples)}")

        for old_label in sorted(class_to_indices.keys()):
            cls_name = base_classes[old_label]
            total_count = len(class_to_indices[old_label])
            train_count = min(TRAIN_PER_CLASS, total_count)
            test_count = max(0, total_count - TRAIN_PER_CLASS)

            if split == "train":
                print(f"[{split}] {cls_name}: {train_count}")
            else:
                print(f"[{split}] {cls_name}: {test_count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        real_idx = self.samples[idx]
        x, y = self.base[real_idx]
        y = self.label_map[y]
        return x, y


train_dataset = FilteredEMNISTSplit(split="train")
test_dataset = FilteredEMNISTSplit(split="test")

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

model = RealTimeCharCNN(num_classes=36).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = correct / total if total > 0 else 0.0
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Test Acc: {acc:.4f}")

os.makedirs("weights", exist_ok=True)
torch.save(model.state_dict(), "weights/emnist36_model_5000split.pt")
print("Model saved: weights/emnist36_model_5000split.pt")