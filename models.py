from torch import nn, Tensor
from torchvision.models import resnet50


class FaceDetector(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.resnet = resnet50(pretrained=False)
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.resnet(x)
        x = self.classifier(x)

        return x