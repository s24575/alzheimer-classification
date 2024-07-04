# Stage of Alzheimer's classification.

The purpose of this project is to detect the stage of Alzheimer's based on MRI images.

### Dataset

The models are designed to be trained on [Alzheimer's Dataset ( 4 class of Images)](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images) from Kaggle.

### Requirements
* Python 3.11+
* Poetry

### Training a Model

To train a model, run:

    python train.py -m vgg16 -f model_filename
        -m/--model_name: Name of the model to train (vgg16 or cnn).
        -f/--filename: Filename to save the trained model.

### Predicting with a Model

To predict using a trained model, run:

    python predict.py -m vgg16 -f model_filename -i path_to_image
        -m/--model_name: Name of the model to use for prediction (vgg16 or cnn).
        -f/--filename: Filename of the trained model.
        -i/--image_path: Path to the image file to predict.
