import torch
from torchvision.transforms import v2

Height, Width = 128, 128

transforms = v2.Compose([
    v2.RandomRotation(20),
    v2.RandomHorizontalFlip(0.5),
    v2.GaussianBlur(3, 1.5)
])