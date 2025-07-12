from typing import Tuple

import mlflow.pytorch
import torch
from PIL import Image
from torchvision import transforms

from alzheimer_classification.models import VGG16Model
from alzheimer_classification.utils.definitions import CLASS_NAMES


def preprocess_image(image_path: str, transform: transforms.Compose) -> torch.Tensor:
    """
    Preprocesses an image for prediction by applying the specified transformation.

    Args:
        image_path (str): Path to the image file.
        transform (transforms.Compose): Transformations to apply to the image.

    Returns:
        torch.Tensor: Preprocessed image tensor ready for prediction.
    """
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def predict_image(model: torch.nn.Module, image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Predicts class probabilities from the input tensor using the given model.

    Args:
        model (torch.nn.Module): The pre-trained PyTorch model.
        image_tensor (torch.Tensor): Preprocessed image tensor.

    Returns:
        torch.Tensor: Predicted class probabilities.
    """
    with torch.no_grad():
        output = model(image_tensor)
        return torch.nn.functional.softmax(output, dim=1)


def predict_image_from_path(image_path: str, model_name: str) -> Tuple[str, float]:
    """
    Predicts the class and confidence for a given image path using an MLflow model.
    """
    model = mlflow.pytorch.load_model(f"models:/{model_name}/latest")

    test_transform = VGG16Model.get_test_transform()

    image_tensor = preprocess_image(image_path, test_transform)
    predictions = predict_image(model, image_tensor)

    predicted_class = torch.argmax(predictions, dim=1).item()
    confidence = torch.max(predictions).item()

    predicted_class_name = CLASS_NAMES[predicted_class]
    return predicted_class_name, confidence
