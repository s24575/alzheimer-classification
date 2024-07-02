import argparse
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from alzheimer_dataset import AlzheimerDataModule
from models.cnn import CNNModel
from models.vgg16 import VGG16Model
from utils.definitions import ROOT_DIR
from utils.enums import ModelName


def train_model(model: pl.LightningModule, model_name: ModelName) -> None:
    dataset_dir = os.path.join(ROOT_DIR, "dataset")
    data_module = AlzheimerDataModule(dataset_dir, batch_size=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Setup logger
    lightning_logs_dir = os.path.join(ROOT_DIR, "lightning_logs")
    logger = TensorBoardLogger(lightning_logs_dir, name=model_name)

    # Setup trainer
    trainer = pl.Trainer(max_epochs=10, logger=logger, log_every_n_steps=10)

    # Train the model
    trainer.fit(model, data_module)

    # Test the model
    trainer.test(model, data_module)


def get_model(model_name: ModelName) -> pl.LightningModule:
    num_classes = 4

    print("Loading the model...")

    if model_name == ModelName.VGG16:
        return VGG16Model(num_classes=num_classes)
    elif model_name == ModelName.CNN:
        return CNNModel(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model_name",
        choices=list(ModelName),
        required=True,
        help="Name of the model",
    )
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        required=True,
        help="Filename to save the trained model",
    )

    args = parser.parse_args()

    saved_models_dir = os.path.join(ROOT_DIR, "saved_models")
    if not os.path.exists(saved_models_dir):
        raise ValueError(f"Could not find directory: {saved_models_dir}")

    model = get_model(args.model_name)

    train_model(model, args.model_name)

    filename: str = args.filename
    if not filename.endswith(".pth"):
        filename += ".pth"

    save_path = os.path.join(saved_models_dir, args.filename)
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    main()
