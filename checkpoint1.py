import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F


print("Running:", os.path.abspath(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Load dataset (keep it simple for checkpoint)
transform = transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, transform=transform, download=True
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2) Define model (paper-aligned but simplified LeNet-5)
# - pad 28x28 -> 32x32 (paper uses 32x32 input)
# - use tanh + average pooling
# - keep classic 6-16-120-84 numbers
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)      # C1: 1->6
        self.pool = nn.AvgPool2d(2, 2)       # S2/S4: avg pooling (simplified)
        self.conv2 = nn.Conv2d(6, 16, 5)     # C3: 6->16 (full connections, simplified)
        self.conv3 = nn.Conv2d(16, 120, 5)   # C5: 16->120 (5x5 -> 1x1)
        self.fc = nn.Linear(120, 84)         # F6: 120->84
        self.out = nn.Linear(84, 10)         # 10 classes

    def forward(self, x):
        x = F.pad(x, (2, 2, 2, 2))           # 28x28 -> 32x32 (paper setting)
        x = torch.tanh(self.conv1(x))        # tanh activation (paper-style)
        x = self.pool(x)                    # 28->14
        x = torch.tanh(self.conv2(x))        # 14->10
        x = self.pool(x)                    # 10->5
        x = torch.tanh(self.conv3(x))        # 5->1, output (N,120,1,1)
        x = x.view(x.size(0), -1)           # (N,120)
        x = torch.tanh(self.fc(x))          # (N,84)
        return self.out(x)                  # logits (N,10)

model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 3) Training loop (keep short for checkpoint)
for epoch in range(3):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    avg_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# 4) Evaluation
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        pred = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

accuracy = 100 * correct / total
print("Test Accuracy:", accuracy)