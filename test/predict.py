from PIL import Image
from models import FaceDetector

import torch
import torch.optim as optim
from torchvision import transforms


PATH = 'face_detection.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FaceDetector(num_classes=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
img = Image.open('captured_2021_5_26_12_14_33.jpg').convert('RGB')
img = transforms.ToTensor()(img)
img = transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )(img)
img = img.to(device)
target = model(img[None, ...])
target = target > 0.5
print(target)