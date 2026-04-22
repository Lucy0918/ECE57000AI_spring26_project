import cv2
import torch
import os
import random
from model import RealTimeCharCNN

classes = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]

# A-Z → 10~35
letter_to_full_index = {chr(ord('A') + i): i + 10 for i in range(26)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RealTimeCharCNN(num_classes=36)
model.load_state_dict(
    torch.load("weights/emnist36_letters_ft.pt", map_location=device, weights_only=True)
)
model.to(device)
model.eval()

root = "az_dataset"

correct = 0
total = 0

print("\n===== Random 1 Sample per Letter =====")

for letter in sorted(os.listdir(root)):
    folder = os.path.join(root, letter)

    if not os.path.isdir(folder):
        continue

    if letter not in letter_to_full_index:
        continue

    files = os.listdir(folder)
    if len(files) == 0:
        continue

    # 👉 隨機選一張
    file = random.choice(files)
    path = os.path.join(folder, file)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    # ⚠️ 跟訓練一致
    img = cv2.resize(img, (28, 28))
    img = img.astype("float32") / 255.0
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img)
        prob = torch.softmax(out, dim=1)
        conf, pred = torch.max(prob, dim=1)

    pred_char = classes[pred.item()]
    true_idx = letter_to_full_index[letter]

    is_correct = (pred.item() == true_idx)

    if is_correct:
        correct += 1
    total += 1

    print(f"{letter} → Pred: {pred_char} | Conf: {conf.item():.3f} | {'✅' if is_correct else '❌'}")

print("\n===== Summary =====")
print(f"Accuracy: {correct}/{total} = {correct/total:.2%}")