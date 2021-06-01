from torch import nn, Tensor
from torchvision.models import resnet50
from efficientnet_pytorch import EfficientNet


class Efficientnet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.EfficientNet = EfficientNet.from_pretrained('efficientnet-b7')
        # (_fc): Linear(in_features=2560, out_features=1000, bias=True)
        self.out_features = self.EfficientNet._fc.out_features
        self.classifier = nn.Linear(self.out_features, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.EfficientNet(x)
        x = self.classifier(x)
        return x

class Resnet50(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.resnet = resnet50(pretrained=False)
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.resnet50(x)
        x = self.classifier(x)
        return x


# model =  EfficientNet.from_pretrained('efficientnet-b7')
# print(model)