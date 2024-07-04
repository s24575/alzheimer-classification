import torch
from torch import nn
from torchvision import transforms


class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) for image classification.

    Args:
        num_classes (int): The number of output classes.
    """

    def __init__(self, num_classes: int):
        super(SimpleCNN, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 30 * 30, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 30 * 30)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

    @staticmethod
    def get_train_transform() -> transforms.Compose:
        """
        Get the transformation to be applied to training images.

        Returns:
            transforms.Compose: The transformation pipeline for training images.
        """
        return transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.Grayscale(num_output_channels=1),
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
                transforms.Resize((128, 128)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ]
        )
