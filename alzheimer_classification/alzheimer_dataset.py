import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms


class DatasetFromSubset(Dataset):
    """
    A custom Dataset class that applies a transformation to a subset of a dataset.

    Args:
        subset (torch.utils.data.Subset): A subset of a dataset.
        transform (transforms.Compose, optional): A transform applied to images.

    Methods:
        __getitem__(index): Retrieves the item at the specified index, applying the transform if provided.
        __len__(): Returns the length of the subset.
    """

    def __init__(self, subset: Subset, transform: transforms.Compose | None = None):
        indices = torch.arange(3)
        self.subset = Subset(subset, indices)
        self.transform = transform

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self) -> int:
        return len(self.subset)


class AlzheimerDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for handling data loading for an Alzheimer's disease classification task.

    Args:
        root_dir (str): The root directory where the train and test data are stored.
        batch_size (int, optional): The batch size for the dataloaders. Defaults to 64.
        train_transform (transforms.Compose, optional): Transformations to apply to the training data.
        test_transform (transforms.Compose, optional): Transformations to apply to the validation/test data.
    """

    def __init__(
        self,
        root_dir: str,
        batch_size: int = 64,
        train_transform: transforms.Compose | None = None,
        test_transform: transforms.Compose | None = None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size

        self.train_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None

        self.train_transform = train_transform
        self.test_transform = test_transform

    def setup(self, stage: str = None) -> None:
        """
        Sets up the datasets for training, validation, and testing.

        Args:
            stage (str, optional): Stage of setup (fit or test). Defaults to None.
        """
        train_dir = os.path.join(self.root_dir, "train")
        test_dir = os.path.join(self.root_dir, "test")

        full_train_dataset = datasets.ImageFolder(train_dir)
        full_test_dataset = datasets.ImageFolder(test_dir)

        train_size = int(0.8 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_train_dataset, [train_size, val_size]
        )
        test_dataset = Subset(full_test_dataset, range(len(full_test_dataset)))

        self.train_dataset = DatasetFromSubset(
            train_dataset, transform=self.train_transform
        )
        self.val_dataset = DatasetFromSubset(val_dataset, transform=self.test_transform)
        self.test_dataset = DatasetFromSubset(
            test_dataset, transform=self.test_transform
        )

    def train_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader for the training dataset.
        """
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=3
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader for the validation dataset.
        """
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=3
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the test dataset.

        Returns:
            DataLoader: DataLoader for the test dataset.
        """
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=3
        )
