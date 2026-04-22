from torchvision.datasets import EMNIST

train_dataset = EMNIST(
    root='./data',
    split='byclass',   
    train=True,
    download=True
)

test_dataset = EMNIST(
    root='./data',
    split='byclass',
    train=False,
    download=True
)