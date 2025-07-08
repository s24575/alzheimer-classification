import os
from typing import Tuple

import mlflow.pytorch
import torch
from models import VGG16Model
from PIL import Image
from torchvision import transforms
from utils.definitions import CLASS_NAMES


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


def load_model_from_mlflow(
    model_name: str, model_version: str = "latest"
) -> torch.nn.Module:
    """
    Loads a PyTorch model from MLflow Model Registry.
    """
    mlflow.set_tracking_uri("http://localhost:5000")
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pytorch.load_model(model_uri)
    return model


def predict_image_from_path(image_path: str, model_name: str) -> Tuple[str, float]:
    """
    Predicts the class and confidence for a given image path using an MLflow model.
    """
    # Load model from MLflow
    model = load_model_from_mlflow(model_name=model_name)

    test_transform = VGG16Model.get_test_transform()

    # Preprocess and predict
    image_tensor = preprocess_image(image_path, test_transform)
    predictions = predict_image(model, image_tensor)

    predicted_class = torch.argmax(predictions, dim=1).item()
    confidence = torch.max(predictions).item()

    predicted_class_name = CLASS_NAMES[predicted_class]
    return predicted_class_name, confidence


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--image_path", type=str, required=True, help="Path to image"
    )
    parser.add_argument("-m", "--model_name", type=str, help="MLflow model name")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    pred_class, conf = predict_image_from_path(
        args.image_path, model_name=args.model_name
    )
    print(f"Predicted {pred_class} with confidence {conf:.2f}")
