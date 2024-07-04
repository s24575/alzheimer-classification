import argparse
import os

import pytorch_lightning as pl
import torch
from alzheimer_dataset import AlzheimerDataModule
from models.generic_model import GenericModel
from models.model_utils import ModelName, get_model
from pytorch_lightning.loggers import TensorBoardLogger
from utils.definitions import ROOT_DIR


def train_model(model: GenericModel, model_name: ModelName) -> None:
    """
    Function to train the given PyTorch Lightning model using a specified data module.

    Args:
        model (GenericModel): The PyTorch Lightning model to train.
        model_name (ModelName): Enum value specifying the model name.
    """
    train_transform = model.model.get_train_transform()
    test_transform = model.model.get_test_transform()

    dataset_dir = os.path.join(ROOT_DIR, "dataset")
    data_module = AlzheimerDataModule(
        dataset_dir,
        batch_size=4,
        train_transform=train_transform,
        test_transform=test_transform,
    )
    data_module.setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Setup logger
    lightning_logs_dir = os.path.join(ROOT_DIR, "lightning_logs")
    logger = TensorBoardLogger(lightning_logs_dir, name=model_name)

    # Setup trainer
    trainer = pl.Trainer(max_epochs=10, logger=logger, log_every_n_steps=1)

    # Train the model
    trainer.fit(model, data_module)

    # Test the model
    trainer.test(model, data_module)


def train() -> None:
    """
    Main function to parse command-line arguments and initiate model training.
    """
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
    train()
