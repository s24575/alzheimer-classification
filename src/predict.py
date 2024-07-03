import argparse
import os

import torch
from PIL import Image
from torchvision import transforms

from models.model_utils import load_model
from utils.definitions import SAVED_MODELS_DIR
from utils.enums import ModelName


def preprocess_image(image_path: str) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ]
    )
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)


def predict_image(model: torch.nn.Module, image_tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        output = model(image_tensor)
    return torch.nn.functional.softmax(output, dim=1)


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

    model_path = str(os.path.join(SAVED_MODELS_DIR, args.filename))

    # Check if paths exist
    if not os.path.exists(model_path):
        raise ValueError(f"Could not find model file: {model_path}")
    if not os.path.exists(args.image_path):
        raise ValueError(f"Could not find image file: {args.image_path}")

    # Load the model
    model = load_model(args.model_name, model_path)

    # Preprocess the image
    image_tensor = preprocess_image(args.image_path)

    # Predict the class
    predictions = predict_image(model, image_tensor)

    # Get the predicted class and confidence
    predicted_class = torch.argmax(predictions, dim=1).item()
    confidence = torch.max(predictions).item()

    class_names = [
        "MildDemented",
        "ModerateDemented",
        "NonDemented",
        "VeryMildDemented",
    ]
    predicted_class_name = class_names[predicted_class]

    print(f"Predicted {predicted_class_name} with confidence: {confidence:.2f}")


if __name__ == "__main__":
    main()
