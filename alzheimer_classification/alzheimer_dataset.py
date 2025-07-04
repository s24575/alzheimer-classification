import os

import kagglehub
import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset
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
        self.subset = subset
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
        batch_size (int, optional): The batch size for the dataloaders. Defaults to 64.
        train_transform (transforms.Compose, optional): Transformations to apply to the training data.
        test_transform (transforms.Compose, optional): Transformations to apply to the validation/test data.
    """

    def __init__(
        self,
        batch_size: int,
        train_transform: transforms.Compose | None = None,
        test_transform: transforms.Compose | None = None,
    ):
        super().__init__()
        self.root_dir: str | None = None
        self.batch_size = batch_size

        self.train_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None

        self.train_transform = train_transform
        self.test_transform = test_transform

    def prepare_data(self):
        self.root_dir = kagglehub.dataset_download(
            "uraninjo/augmented-alzheimer-mri-dataset"
        )
        print("Path to dataset files:", self.root_dir)

    def setup(self, stage: str = None) -> None:
        """
        Sets up the datasets for training, validation, and testing.

        Args:
            stage (str, optional): Stage of setup (fit or test). Defaults to None.
        """
        augmented_dir = os.path.join(self.root_dir, "AugmentedAlzheimerDataset")
        full_dataset = datasets.ImageFolder(augmented_dir)

        targets = [label for _, label in full_dataset.samples]

        # First split: train_val (80%) and test (20%)
        splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_val_idx, test_idx = next(splitter1.split(np.zeros(len(targets)), targets))

        # Create test set
        test_dataset = Subset(full_dataset, test_idx)

        # Now split train_val into train (80% of 80%) and val (20% of 80%)
        train_val_targets = [targets[i] for i in train_val_idx]
        splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(
            splitter2.split(np.zeros(len(train_val_targets)), train_val_targets)
        )

        train_dataset = Subset(full_dataset, [train_val_idx[i] for i in train_idx])
        val_dataset = Subset(full_dataset, [train_val_idx[i] for i in val_idx])

        # Apply transforms
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
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader for the validation dataset.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the test dataset.

        Returns:
            DataLoader: DataLoader for the test dataset.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
        )
