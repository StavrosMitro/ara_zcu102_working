import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# -------------------------------
# 1. Custom Dataset for image list files
# -------------------------------
class CIFAR10FromList(Dataset):
    def __init__(self, list_file, transform=None):
        """
        Args:
            list_file (str): Path to the .list file with lines: "label path"
            transform: torchvision transforms to apply to images
        """
        self.samples = []
        with open(list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(' ', 1)  # split only first space
                if len(parts) != 2:
                    print(f"Skipping malformed line: {line}")
                    continue
                label, path = parts
                self.samples.append((int(label), path))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label, path = self.samples[idx]
        # Load image using PIL
        image = Image.open(path).convert('RGB')  # ensure RGB
        if self.transform:
            image = self.transform(image)
        return image, label

# -------------------------------
# 2. Data transforms
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),          # CIFAR images are 32x32; ensure size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Paths to your list files (adjust if needed)
train_list = './cifar10/images.list'
val_list = './cifar10/image_val.list'

# Create datasets
trainset = CIFAR10FromList(train_list, transform=transform)
testset = CIFAR10FromList(val_list, transform=transform)

# DataLoaders
trainloader = DataLoader(trainset, batch_size=400, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=400, shuffle=False, num_workers=2)

# Class names (for reference)
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# -------------------------------
# 3. Model: single conv layer + global average pooling
# -------------------------------
class SingleConvClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(3, num_classes, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv(x)               # (batch, 10, 32, 32)
        x = self.global_pool(x)         # (batch, 10, 1, 1)
        x = x.view(x.size(0), -1)       # (batch, 10)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SingleConvClassifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# 4. Training loop
# -------------------------------
num_epochs = 20
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    model.train()

    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if i % 200 == 199:
            print(f'Epoch {epoch+1}, Batch {i+1}: loss = {running_loss/200:.3f}')
            running_loss = 0.0

    epoch_acc = 100 * correct / total
    print(f'Epoch {epoch+1} finished, Training Accuracy: {epoch_acc:.2f}%')

# -------------------------------
# 5. Evaluation on validation set
# -------------------------------
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on validation images: {100 * correct / total:.2f}%')