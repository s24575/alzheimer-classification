import torch
from torch import nn
from torchvision import models, transforms
from torchvision.models import VGG16_Weights


class VGG16Model(nn.Module):
    """
    Standard VGG16 model pretrained on ImageNet dataset with a modified
    classifier to match the number of classes on output.

    Args:
        num_classes (int): The number of output classes for the classifier.
    """

    def __init__(self, num_classes: int):
        super().__init__()

        self.num_classes = num_classes

        self.model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        for param in self.model.parameters():
            param.requires_grad = False

        n_inputs = self.model.classifier[0].in_features

        self.model.classifier = nn.Sequential(
            nn.Linear(n_inputs, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.model(x)

    @staticmethod
    def get_train_transform() -> transforms.Compose:
        """
        Get the transformation to be applied to training images.

        Returns:
            transforms.Compose: The transformation pipeline for training images.
        """
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )

    @staticmethod
    def get_test_transform() -> transforms.Compose:
        """
        Get the transformation to be applied to testing images.

        Returns:
            transforms.Compose: The transformation pipeline for testing images.
        """
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )
