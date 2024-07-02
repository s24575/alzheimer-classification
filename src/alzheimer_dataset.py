import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


class AlzheimerDataModule(pl.LightningDataModule):
    def __init__(self, root_dir: str, batch_size: int = 64):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.train_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None

    def setup(self, stage: str = None) -> None:
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                # transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ]
        )

        train_dir = os.path.join(self.root_dir, "train")
        test_dir = os.path.join(self.root_dir, "test")

        full_train_dataset = datasets.ImageFolder(train_dir, transform=transform)

        self.train_dataset, self.test_dataset = random_split(
            full_train_dataset, [0.9, 0.1]
        )

        self.val_dataset = datasets.ImageFolder(test_dir, transform=transform)

        self.train_dataset, _ = random_split(
            self.train_dataset, [5, len(self.train_dataset) - 5]
        )
        self.val_dataset, _ = random_split(
            self.val_dataset, [5, len(self.val_dataset) - 5]
        )
        self.test_dataset, _ = random_split(
            self.test_dataset, [5, len(self.test_dataset) - 5]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
