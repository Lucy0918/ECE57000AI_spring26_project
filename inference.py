import torch
from model import RealTimeCharCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RealTimeCharCNN(num_classes=36)

model.load_state_dict(
    torch.load("weights/emnist36_letters_ft.pt", map_location=device, weights_only=True)
)

model.to(device)
model.eval()

def predict(img, threshold=0.2):
    with torch.no_grad():
        out = model(img)
        prob = torch.softmax(out, dim=1)
        conf, pred = torch.max(prob, dim=1)

        if conf.item() < threshold:
            return None, conf.item()

        return pred.item(), conf.item()