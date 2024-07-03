import torch

from models.generic_model import GenericModel
from models.simple_cnn import SimpleCNN
from models.vgg16 import VGG16Model
from utils.enums import ModelName


def get_model(model_name: ModelName) -> GenericModel:
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
    model = get_model(model_name)

    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    return model
