import torch
from torch import nn
from torchvision import models
from torchvision.models import VGG16_Weights


class VGG16Model(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.num_classes = num_classes

        self.model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        for param in self.model.parameters():
            param.requires_grad = False

        n_inputs = self.model.classifier[6].in_features

        self.model.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
