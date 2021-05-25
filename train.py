from PIL import Image

import torch
import torch.optim as optim
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from custom_dataset import FaceDataset

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
data_loader = DataLoader(dataset, batch_size=32, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FaceDetector(num_classes=2).to(device)
summary(model, input_size=(1, 3, 213, 160))

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

num_epochs = 2

model.train()
for epoch in range(num_epochs):
    for i, (images, targets) in enumerate(data_loader):
        optimizer.zero_grad()

        images = images.to(device)
        targets = targets.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            outputs = outputs > 0.5
            print(f'{epoch}: {loss.item():.5f}')

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