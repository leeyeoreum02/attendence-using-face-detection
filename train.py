from PIL import Image

import torch
import torch.optim as optim
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchinfo import summary
from custom_dataset import FaceDataset
from custom_dataset import split_dataset

from torchvision import transforms
from models import FaceDetector


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

dataset = FaceDataset('data', 'data/labels.csv', transform)
train_indices, test_indices = split_dataset(len(dataset), 0.2)
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler, num_workers=0)
test_loader = DataLoader(dataset, batch_size=8, sampler=test_sampler, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FaceDetector(num_classes=2).to(device)
summary(model, input_size=(1, 3, 213, 160))

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

num_epochs = 20

model.train()
for epoch in range(num_epochs):
    for i, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad()

        images = images.to(device)
        targets = targets.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            outputs = outputs > 0.5
            acc = (outputs == targets).float().mean()
            print(f'epoch: {epoch}, step: {i}, loss: {loss.item():.5f}, acc: {acc.item():.5f}')

with torch.no_grad():
    model.eval()
    for images, targets in test_loader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        outputs = outputs > 0.5
        acc = (outputs == targets).float().mean()

print(acc.cpu().numpy())

PATH = 'face_detection.pth'
torch.save(model.state_dict(), PATH)

img = Image.open('captured_2021_5_25_16_30_27.jpg').convert('RGB')
img = transforms.ToTensor()(img)
img = transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )(img)
img = img.to(device)
target = model(img[None, ...])
target = target > 0.5
print(target)