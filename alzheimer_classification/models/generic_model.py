import pytorch_lightning as pl
import torch
import torchmetrics
from torchmetrics import AUROC


class GenericModel(pl.LightningModule):
    """
    A generic model class for PyTorch Lightning that wraps a given model and
    includes various metrics for training, validation, and testing phases.

    Args:
        model (torch.nn.Module): The neural network model to be wrapped.
    """

    def __init__(self, model: torch.nn.Module):
        super(GenericModel, self).__init__()

        num_classes = model.num_classes

        self.model = model

        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

        self.train_precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes
        )
        self.val_precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes
        )
        self.test_precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes
        )

        self.train_recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes
        )
        self.val_recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes
        )
        self.test_recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes
        )

        self.train_macro_f1 = torchmetrics.F1Score(
            num_classes=num_classes, task="multiclass", average="macro"
        )
        self.val_macro_f1 = torchmetrics.F1Score(
            num_classes=num_classes, task="multiclass", average="macro"
        )
        self.test_macro_f1 = torchmetrics.F1Score(
            num_classes=num_classes, task="multiclass", average="macro"
        )

        self.train_auroc = AUROC(task="multiclass", num_classes=num_classes)
        self.val_auroc = AUROC(task="multiclass", num_classes=num_classes)
        self.test_auroc = AUROC(task="multiclass", num_classes=num_classes)

        self.val_confusion_matrix = torchmetrics.classification.MulticlassConfusionMatrix(
            num_classes=num_classes
        )
        self.test_confusion_matrix = torchmetrics.classification.MulticlassConfusionMatrix(
            num_classes=num_classes
        )

        self.val_outputs = None
        self.val_labels = None

        self.test_outputs = None
        self.test_labels = None

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer for the model.

        Returns:
            torch.optim.Optimizer: The Adam optimizer with a learning rate of 0.001.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Define the training step.

        Args:
            batch (tuple): A batch of data, containing images and labels.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The computed loss.
        """
        images, labels = batch

        outputs = self(images)
        loss = self.criterion(outputs, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True)

        outputs = torch.nn.functional.softmax(outputs, dim=1)

        self.train_acc(outputs, labels)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)

        self.train_precision(outputs, labels)
        self.log("train_precision", self.train_precision, on_step=False, on_epoch=True)

        self.train_recall(outputs, labels)
        self.log("train_recall", self.train_recall, on_step=False, on_epoch=True)

        self.train_macro_f1(outputs, labels)
        self.log("train_macro_f1", self.train_macro_f1, on_step=False, on_epoch=True)

        self.train_auroc(outputs, labels)
        self.log("train_auroc", self.train_auroc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Define the validation step.

        Args:
            batch (tuple): A batch of data, containing images and labels.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The computed loss.
        """
        images, labels = batch

        outputs = self(images)
        loss = self.criterion(outputs, labels)

        self.log("val_loss", loss, on_step=True, on_epoch=True)

        outputs = torch.nn.functional.softmax(outputs, dim=1)

        self.val_acc(outputs, labels)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)

        self.val_precision(outputs, labels)
        self.log("val_precision", self.val_precision, on_step=False, on_epoch=True)

        self.val_recall(outputs, labels)
        self.log("val_recall", self.val_recall, on_step=False, on_epoch=True)

        self.val_macro_f1(outputs, labels)
        self.log("val_macro_f1", self.val_macro_f1, on_step=False, on_epoch=True)

        self.val_auroc(outputs, labels)
        self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True)

        self.val_outputs = outputs
        self.val_labels = labels

        return loss

    def on_validation_epoch_end(self) -> None:
        """
        Perform actions at the end of the validation epoch, such as updating and logging the confusion matrix.
        """
        self.val_confusion_matrix.update(self.val_outputs, self.val_labels)
        fig, ax = self.val_confusion_matrix.plot()
        self.logger.experiment.add_figure("val_confusion_matrix", fig, self.current_epoch)

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Define the test step.

        Args:
            batch (tuple): A batch of data, containing images and labels.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The computed loss.
        """
        images, labels = batch

        outputs = self(images)
        loss = self.criterion(outputs, labels)

        self.log("test_loss", loss, on_step=True, on_epoch=True)

        outputs = torch.nn.functional.softmax(outputs, dim=1)

        self.test_acc(outputs, labels)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)

        self.test_precision(outputs, labels)
        self.log("test_precision", self.test_precision, on_step=False, on_epoch=True)

        self.test_recall(outputs, labels)
        self.log("test_recall", self.test_recall, on_step=False, on_epoch=True)

        self.test_macro_f1(outputs, labels)
        self.log("test_macro_f1", self.test_macro_f1, on_step=False, on_epoch=True)

        self.test_auroc(outputs, labels)
        self.log("test_auroc", self.test_auroc, on_step=False, on_epoch=True)

        self.test_outputs = outputs
        self.test_labels = labels

        return loss

    def on_test_epoch_end(self) -> None:
        """
        Perform actions at the end of the test epoch, such as updating and logging the confusion matrix.
        """
        self.test_confusion_matrix.update(self.test_outputs, self.test_labels)
        fig, ax = self.test_confusion_matrix.plot()
        self.logger.experiment.add_figure("test_confusion_matrix", fig, self.current_epoch)
