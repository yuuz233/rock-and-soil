from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import torch
import torch.nn as nn

from torch.optim import Adam
from torch.nn import CrossEntropyLoss

# Initialize transform for data augmentation in the next step
initial_transform = transforms.ToTensor()
dataset_path = 'Soil Data'

initial_transform = transforms.ToTensor()
dataset = datasets.ImageFolder(root=dataset_path, transform=initial_transform)

# Data augmentation
augmentation_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
])

dataset.transform = augmentation_transform
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


#model
class SandNet(nn.Module):
    def __init__(self):
        super(SandNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 5)  # 5 classes: Clay, Loam, Sand, Sandy Loam, Silt


    def forward(self, x):
        x = nn.MaxPool2d(2)(nn.ReLU()(self.conv1(x)))
        x = nn.MaxPool2d(2)(nn.ReLU()(self.conv2(x)))
        x = nn.MaxPool2d(2)(nn.ReLU()(self.conv3(x)))
        x = x.view(-1, 64 * 14 * 14)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

model = SandNet()
print('model created')

optimizer = Adam(model.parameters(), lr=0.001)
criterion = CrossEntropyLoss()

# Training loop
for epoch in range(30):  # Loop over the dataset multiple times
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print('training done')

# Function to evaluate accuracy
def evaluate(loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Evaluate model
train_accuracy = evaluate(train_loader)
test_accuracy = evaluate(test_loader)
print(f"Training accuracy: {train_accuracy}%")
print(f"Testing accuracy: {test_accuracy}%")
