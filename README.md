# Stage of Alzheimer's classification

The purpose of this project is to detect the stage of Alzheimer's based on MRI images.

[//]: # (# Dataset)

[//]: # ()
[//]: # (The models are designed to be trained on [Alzheimer's Dataset &#40; 4 class of Images&#41;]&#40;https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images&#41; from Kaggle.)

[//]: # (It has already been downloaded and is available in the dataset/ directory.)

[//]: # ()
[//]: # (<p align="center">)

[//]: # (  <img src="assets/train_dataset.png" alt="Train Dataset" width="49%" />)

[//]: # (  <img src="assets/test_dataset.png" alt="Test Dataset" width="49%" />)

[//]: # (</p>)

# Usage

### Requirements
* Python 3.10+
* Poetry

### Installing requirements
```bash
poetry install
```

### Training a model

    python train.py -m vgg16 -f model_filename
        -m/--model_name: Name of the model to train (vgg16 or cnn).
        -f/--filename: Filename to save the trained model.

### Predicting with a model

    python predict.py -m vgg16 -f model_filename -i ../dataset/test/ModerateDemented/27.jpg
        -m/--model_name: Name of the model to use for prediction (vgg16 or cnn).
        -f/--filename: Filename of the trained model.
        -i/--image_path: Path to the image file to predict.

### Tensorboard
To view all metrics collected during model training, run:
```
tensorboard --logdir ../lightning_logs
```

# Development
## Pre-commits
Install pre-commits
https://pre-commit.com/#installation

If you are using VS-code install the extension https://marketplace.visualstudio.com/items?itemName=MarkLarah.pre-commit-vscode

To make a dry-run of the pre-commits, to see if your code passes, run:
```
pre-commit run --all-files
```


## Adding python packages
Dependencies are handled by `poetry` framework, to add new a new dependency, run:
```
poetry add <package_name>
```
