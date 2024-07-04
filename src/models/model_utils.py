from enum import StrEnum

import torch

from models.generic_model import GenericModel
from models.simple_cnn import SimpleCNN
from models.vgg16 import VGG16Model


class ModelName(StrEnum):
    """
    Enum for supported model names.
    """

    VGG16 = "vgg16"
    CNN = "cnn"


def get_model(model_name: ModelName) -> GenericModel:
    """
    Create a model instance based on the provided model name.

    Args:
        model_name (ModelName): The name of the model to create.

    Returns:
        GenericModel: An instance of the requested model wrapped in GenericModel.

    Raises:
        ValueError: If an unsupported model name is provided.
    """
    num_classes = 4

    print("Loading the model...")

    if model_name == ModelName.VGG16:
        model = VGG16Model(num_classes=num_classes)
    elif model_name == ModelName.CNN:
        model = SimpleCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return GenericModel(model)


def load_model(model_name: ModelName, model_path: str) -> GenericModel:
    """
    Load a model from a file and prepare it for evaluation.

    Args:
        model_name (ModelName): The name of the model to load.
        model_path (str): The path to the model file.

    Returns:
        GenericModel: The loaded model ready for evaluation.

    Raises:
        RuntimeError: If the model file cannot be loaded.
    """
    model = get_model(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model
