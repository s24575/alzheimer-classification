import argparse
import os

import torch
from models.model_utils import ModelName, load_model
from PIL import Image
from torchvision import transforms
from utils.definitions import CLASS_NAMES, SAVED_MODELS_DIR


def preprocess_image(image_path: str, transform: transforms.Compose) -> torch.Tensor:
    """
    Preprocesses an image for prediction by applying the specified transformation.

    Args:
        image_path (str): Path to the image file.
        transform (transforms.Compose): Transformations to apply to the image.

    Returns:
        torch.Tensor: Preprocessed image tensor ready for prediction.
    """
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)


def predict_image(model: torch.nn.Module, image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Predicts the class probabilities for an image tensor using the specified model.

    Args:
        model (torch.nn.Module): The pre-trained PyTorch model.
        image_tensor (torch.Tensor): Preprocessed image tensor.

    Returns:
        torch.Tensor: Predicted class probabilities.
    """
    with torch.no_grad():
        output = model(image_tensor)
    return torch.nn.functional.softmax(output, dim=1)


def predict() -> None:
    """
    Main function to parse command-line arguments and perform image prediction.
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
        help="Filename of the trained model file",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        required=True,
        help="Path to the image file to predict",
    )

    args = parser.parse_args()

    filename: str = args.filename
    if not filename.endswith(".pth"):
        filename += ".pth"

    model_path = str(os.path.join(SAVED_MODELS_DIR, filename))

    # Check if paths exist
    if not os.path.exists(model_path):
        raise ValueError(f"Could not find model file: {model_path}")
    if not os.path.exists(args.image_path):
        raise ValueError(f"Could not find image file: {args.image_path}")

    # Load the model
    try:
        model = load_model(args.model_name, model_path)
    except Exception as e:
        print(f"Error occurred while loading the model: {e}")
        return
    test_transform = model.model.get_test_transform()

    # Preprocess the image
    image_tensor = preprocess_image(args.image_path, test_transform)

    # Predict the class
    predictions = predict_image(model, image_tensor)

    # Get the predicted class and confidence
    predicted_class = torch.argmax(predictions, dim=1).item()
    confidence = torch.max(predictions).item()

    predicted_class_name = CLASS_NAMES[predicted_class]

    print(f"Predicted {predicted_class_name} with confidence: {confidence:.2f}")


if __name__ == "__main__":
    predict()
