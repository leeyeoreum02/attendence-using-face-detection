from torch import nn, Tensor
from torchvision.models import resnet50
from efficientnet_pytorch import EfficientNet


class FaceDetector(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        # self.resnet = resnet50(pretrained=False)
        self.EfficientNet = EfficientNet.from_pretrained('efficientnet-b7')
        # (_fc): Linear(in_features=2560, out_features=1000, bias=True)
        self.out_features = self.EfficientNet._fc.out_features
        self.classifier = nn.Linear(self.out_features, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        # x = self.resnet(x)
        x = self.EfficientNet(x)
        x = self.classifier(x)
        return x

model =  EfficientNet.from_pretrained('efficientnet-b7')
print(model)