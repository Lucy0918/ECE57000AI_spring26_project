print("checkpoint2.py started")
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# Paper-style LeNet-5 for digit recognition
# Based on: Gradient-Based Learning Applied to Document Recognition
# Notes:
# 1) This implementation follows the original architecture closely:
#    - 32x32 input
#    - tanh-like activation (scaled tanh)
#    - trainable subsampling layers (average-pooling style)
#    - partial connectivity in C3
#    - F6 = 84
#    - output = Euclidean RBF-style penalties over 10 classes
# 2) This code uses the public MNIST data pipeline available in torchvision.
#    The original paper used the modified NIST / MNIST construction and
#    describes extra details such as deformation training and a specialized
#    optimizer variant.
# ============================================================


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# Scaled tanh from the paper
# y = 1.7159 * tanh(2x/3)
# ----------------------------
class ScaledTanh(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 1.7159 * torch.tanh((2.0 / 3.0) * x)


# ----------------------------
# Input normalization to 32x32
# MNIST images are 28x28 grayscale.
# The paper describes size-normalized digits centered in 28x28,
# sometimes extended to 32x32 with background pixels.
# We pad 28x28 -> 32x32 and map [0,1] to approximately [-0.1, 1.175].
# white background -> -0.1, black foreground -> 1.175
# ----------------------------
class PaperInputTransform:
    """
    Closer to the MNIST construction described in the paper:
    1) convert to tensor in [0,1]
    2) resize the foreground to fit inside a 20x20 box while preserving aspect ratio
    3) paste into a 28x28 field using the foreground center-of-mass for centering
    4) optionally extend to 32x32 with background pixels
    5) map white -> -0.1, black -> 1.175
    """
    def __init__(self, extend_to_32: bool = True):
        self.extend_to_32 = extend_to_32

    @staticmethod
    def _compute_bbox(x: torch.Tensor):
        ys, xs = torch.where(x > 0)
        if len(xs) == 0:
            return 0, 27, 0, 27
        return ys.min().item(), ys.max().item(), xs.min().item(), xs.max().item()

    @staticmethod
    def _center_of_mass(x: torch.Tensor):
        h, w = x.shape
        total = x.sum()
        if total.item() <= 1e-8:
            return (h - 1) / 2.0, (w - 1) / 2.0
        yy = torch.arange(h, dtype=x.dtype).view(h, 1)
        xx = torch.arange(w, dtype=x.dtype).view(1, w)
        cy = (x * yy).sum() / total
        cx = (x * xx).sum() / total
        return cy.item(), cx.item()

    def __call__(self, img):
        x = transforms.ToTensor()(img).squeeze(0)  # [28, 28]

        # Tight crop around foreground
        y0, y1, x0, x1 = self._compute_bbox(x)
        crop = x[y0:y1 + 1, x0:x1 + 1].unsqueeze(0).unsqueeze(0)  # [1,1,h,w]

        ch, cw = crop.shape[-2:]
        scale = min(20.0 / ch, 20.0 / cw)
        new_h = max(1, int(round(ch * scale)))
        new_w = max(1, int(round(cw * scale)))

        resized = F.interpolate(crop, size=(new_h, new_w), mode="bilinear", align_corners=False)
        resized = resized.squeeze(0).squeeze(0)

        # Put in 28x28 field first
        canvas28 = torch.zeros((28, 28), dtype=resized.dtype)
        top = (28 - new_h) // 2
        left = (28 - new_w) // 2
        canvas28[top:top + new_h, left:left + new_w] = resized

        # Recenter using center of mass
        cy, cx = self._center_of_mass(canvas28)
        target_c = 13.5
        shift_y = int(round(target_c - cy))
        shift_x = int(round(target_c - cx))

        shifted = torch.zeros_like(canvas28)
        src_y0 = max(0, -shift_y)
        src_y1 = min(28, 28 - shift_y)
        dst_y0 = max(0, shift_y)
        dst_y1 = min(28, 28 + shift_y)
        src_x0 = max(0, -shift_x)
        src_x1 = min(28, 28 - shift_x)
        dst_x0 = max(0, shift_x)
        dst_x1 = min(28, 28 + shift_x)
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = canvas28[src_y0:src_y1, src_x0:src_x1]

        if self.extend_to_32:
            shifted = F.pad(shifted.unsqueeze(0), (2, 2, 2, 2), value=0.0).squeeze(0)

        shifted = shifted.unsqueeze(0)
        shifted = shifted * (1.175 - (-0.1)) + (-0.1)
        return shifted


# ----------------------------
# Trainable subsampling layer
# Each map has one trainable scale and one trainable bias.
# For each channel:
# out = scaled_tanh(alpha_c * avgpool2x2(x_c) + beta_c)
# ----------------------------
class TrainableSubsampling(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))
        self.act = ScaledTanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = x * self.alpha.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
        return self.act(x)


# ----------------------------
# C3 partial connectivity
# Table 1 in the paper:
# first 6 maps: contiguous subsets of 3 S2 maps
# next 6 maps: contiguous subsets of 4 S2 maps
# next 3 maps: selected discontinuous subsets of 4 S2 maps
# last map: all 6 S2 maps
# ----------------------------
C3_CONNECTIONS: List[List[int]] = [
    [0, 1, 2],
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 0],
    [5, 0, 1],
    [0, 1, 2, 3],
    [1, 2, 3, 4],
    [2, 3, 4, 5],
    [3, 4, 5, 0],
    [4, 5, 0, 1],
    [5, 0, 1, 2],
    [0, 1, 3, 4],
    [1, 2, 4, 5],
    [0, 2, 3, 5],
    [0, 1, 2, 3, 4, 5],
]


class C3PartialConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList()
        self.act = ScaledTanh()
        for conn in C3_CONNECTIONS:
            self.convs.append(nn.Conv2d(len(conn), 1, kernel_size=5, stride=1, padding=0, bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for conn, conv in zip(C3_CONNECTIONS, self.convs):
            x_sub = x[:, conn, :, :]
            y = conv(x_sub)
            outs.append(y)
        y = torch.cat(outs, dim=1)
        return self.act(y)


# ----------------------------
# RBF output layer with fixed class prototypes
# The paper used 84-dimensional target vectors, originally designed as
# stylized bitmaps for the full ASCII set. For digit-only recognition,
# we provide fixed ±1 prototypes for 10 classes.
# Output is a penalty: squared Euclidean distance to each prototype.
# Smaller penalty = better class.
# ----------------------------
class RBFLayer(nn.Module):
    def __init__(self, num_classes: int = 10, feat_dim: int = 84):
        super().__init__()
        prototypes = self._make_digit_prototypes(num_classes, feat_dim)
        self.register_buffer("prototypes", prototypes)

    @staticmethod
    def _make_digit_prototypes(num_classes: int, feat_dim: int) -> torch.Tensor:
        # Fixed deterministic ±1 codes
        # Not the original ASCII bitmap codes, but still faithful to the paper's
        # idea of using fixed distributed ±1 targets with Euclidean penalties.
        codes = []
        for cls in range(num_classes):
            bits = []
            val = cls + 1
            for i in range(feat_dim):
                bit = ((val * 1103515245 + 12345 + i * 2654435761) >> 3) & 1
                bits.append(1.0 if bit else -1.0)
            codes.append(bits)
        return torch.tensor(codes, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 84], prototypes: [10, 84]
        # return penalties [B, 10]
        diff = x.unsqueeze(1) - self.prototypes.unsqueeze(0)
        penalties = (diff * diff).sum(dim=-1)
        return penalties


class LeNet5Paper(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = ScaledTanh()

        # C1: 1x32x32 -> 6x28x28
        self.c1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0, bias=True)

        # S2: 6x28x28 -> 6x14x14
        self.s2 = TrainableSubsampling(6)

        # C3: 6x14x14 -> 16x10x10 (partial connectivity)
        self.c3 = C3PartialConv()

        # S4: 16x10x10 -> 16x5x5
        self.s4 = TrainableSubsampling(16)

        # C5: 16x5x5 -> 120x1x1
        self.c5 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0, bias=True)

        # F6: 120 -> 84
        self.f6 = nn.Linear(120, 84)

        # Output RBF penalties: 84 -> 10
        self.rbf = RBFLayer(num_classes=10, feat_dim=84)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                fan_in = m.weight.data[0].numel() if isinstance(m, nn.Conv2d) else m.weight.size(1)
                bound = 2.4 / fan_in
                nn.init.uniform_(m.weight, -bound, bound)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.c1(x))
        x = self.s2(x)
        x = self.c3(x)
        x = self.s4(x)
        x = self.act(self.c5(x))
        x = x.view(x.size(0), -1)
        x = self.act(self.f6(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f6 = self.forward_features(x)
        penalties = self.rbf(f6)
        return penalties


# ----------------------------
# Losses
# ----------------------------
def mse_rbf_loss(penalties: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Equation (8)-style simplification for classification:
    # minimize correct-class penalty only
    return penalties.gather(1, targets.unsqueeze(1)).mean()


def map_rbf_loss(penalties: torch.Tensor, targets: torch.Tensor, j: float = 1.0) -> torch.Tensor:
    # Equation (9)-style discriminative loss
    # penalties are y_i, smaller is better
    y_true = penalties.gather(1, targets.unsqueeze(1)).squeeze(1)
    denom = torch.exp(torch.tensor(-j, device=penalties.device)) + torch.exp(-penalties).sum(dim=1)
    loss = y_true + torch.log(denom)
    return loss.mean()


# ----------------------------
# Training / evaluation
# ----------------------------
@dataclass
class TrainConfig:
    batch_size: int = 1  # closer to stochastic updates described in the paper
    epochs: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_map_loss: bool = True
    mu: float = 0.02  # safety term mentioned in the paper
    hessian_samples: int = 5


def get_dataloaders(batch_size: int = 1) -> Tuple[DataLoader, DataLoader]:
    transform = PaperInputTransform(extend_to_32=True)

    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, test_loader


def estimate_diag_hessian(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    use_map_loss: bool,
    num_samples: int = 500,
) -> List[torch.Tensor]:
    """
    A practical approximation of the diagonal Hessian.
    We accumulate the diagonal Gauss-Newton-style proxy using squared gradients.
    This is not a perfect reproduction of the appendix derivation,
    but it is much closer in spirit than plain SGD.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    diag = [torch.zeros_like(p, device=device) for p in params]

    model.train()
    seen = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        penalties = model(x)
        loss = map_rbf_loss(penalties, y) if use_map_loss else mse_rbf_loss(penalties, y)
        grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False, allow_unused=False)
        for d, g in zip(diag, grads):
            d.add_(g.detach() * g.detach())
        seen += x.size(0)
        if seen >= num_samples:
            break

    scale = max(seen, 1)
    diag = [d / scale for d in diag]
    return diag


def sdlm_step(
    model: nn.Module,
    loss: torch.Tensor,
    diag_hessian: List[torch.Tensor],
    global_lr: float,
    mu: float,
) -> None:
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False, allow_unused=False)
    with torch.no_grad():
        for p, g, h in zip(params, grads, diag_hessian):
            denom = mu + h
            p.addcdiv_(g, denom, value=-global_lr)


def get_global_lr(epoch: int) -> float:
    # Paper schedule for 20 passes:
    # 0.0005 for first 2
    # 0.0002 for next 3
    # 0.0001 for next 3
    # 0.00005 for next 4
    # 0.00001 thereafter
    if epoch < 2:
        return 5e-4
    if epoch < 5:
        return 2e-4
    if epoch < 8:
        return 1e-4
    if epoch < 12:
        return 5e-5
    return 1e-5


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            penalties = model(x)
            loss = map_rbf_loss(penalties, y)
            preds = penalties.argmin(dim=1)

            total_loss += loss.item() * x.size(0)
            total += x.size(0)
            correct += (preds == y).sum().item()

    return total_loss / total, correct / total


def train() -> None:
    print("train() entered") 
    set_seed(42)
    cfg = TrainConfig()
    device = cfg.device
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_loader, test_loader = get_dataloaders(cfg.batch_size)
    model = LeNet5Paper().to(device)
    print("Model device:", next(model.parameters()).device)

    for epoch in range(cfg.epochs):
        lr = get_global_lr(epoch)

        # Re-estimate the diagonal Hessian proxy before each pass, on 500 samples,
        # mirroring the paper's training description.
        diag_hessian = estimate_diag_hessian(
            model=model,
            loader=train_loader,
            device=device,
            use_map_loss=cfg.use_map_loss,
            num_samples=cfg.hessian_samples,
        )

        model.train()
        running_loss = 0.0
        seen = 0
        correct = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            penalties = model(x)
            loss = map_rbf_loss(penalties, y) if cfg.use_map_loss else mse_rbf_loss(penalties, y)
            sdlm_step(model, loss, diag_hessian, global_lr=lr, mu=cfg.mu)

            preds = penalties.argmin(dim=1)
            running_loss += loss.item() * x.size(0)
            seen += x.size(0)
            correct += (preds == y).sum().item()

        train_loss = running_loss / seen
        train_acc = correct / seen
        test_loss, test_acc = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch + 1:02d}/{cfg.epochs} | "
            f"lr={lr:.5f} | mu={cfg.mu:.3f} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

    torch.save(model.state_dict(), "lenet5_paper_style_mnist.pt")
    print("Saved model to lenet5_paper_style_mnist.pt")


if __name__ == "__main__":
    train()
