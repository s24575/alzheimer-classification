import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics.classification import AUROC, MulticlassConfusionMatrix


class GenericModel(pl.LightningModule):
    """
    A generic model class for PyTorch Lightning that wraps a given model and
    includes various metrics for training, validation, and testing phases.

    Args:
        model (torch.nn.Module): The neural network model to be wrapped.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        num_classes = model.num_classes

        self.criterion = torch.nn.CrossEntropyLoss()

        # Metrics
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
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_macro_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_macro_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )

        self.train_auroc = AUROC(task="multiclass", num_classes=num_classes)
        self.val_auroc = AUROC(task="multiclass", num_classes=num_classes)
        self.test_auroc = AUROC(task="multiclass", num_classes=num_classes)

        self.val_confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)
        self.test_confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)

        # For accumulating confusion matrix inputs
        self.val_preds: list[torch.Tensor] = []
        self.val_targets: list[torch.Tensor] = []
        self.test_preds: list[torch.Tensor] = []
        self.test_targets: list[torch.Tensor] = []

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer for the model.

        Returns:
            torch.optim.Optimizer: The Adam optimizer with a learning rate of 0.001.
        """
        return torch.optim.Adam(self.parameters(), lr=0.001)

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
        probs = F.softmax(outputs, dim=1)

        self.train_acc(probs, labels)
        self.train_precision(probs, labels)
        self.train_recall(probs, labels)
        self.train_macro_f1(probs, labels)
        self.train_auroc(probs, labels)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_epoch=True, prog_bar=True)
        self.log("train_precision", self.train_precision, on_epoch=True)
        self.log("train_recall", self.train_recall, on_epoch=True)
        self.log("train_macro_f1", self.train_macro_f1, on_epoch=True)
        self.log("train_auroc", self.train_auroc, on_epoch=True)

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
        probs = F.softmax(outputs, dim=1)

        self.val_acc(probs, labels)
        self.val_precision(probs, labels)
        self.val_recall(probs, labels)
        self.val_macro_f1(probs, labels)
        self.val_auroc(probs, labels)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        self.log("val_precision", self.val_precision, on_epoch=True)
        self.log("val_recall", self.val_recall, on_epoch=True)
        self.log("val_macro_f1", self.val_macro_f1, on_epoch=True)
        self.log("val_auroc", self.val_auroc, on_epoch=True)

        self.val_preds.append(torch.argmax(probs, dim=1).detach().cpu())
        self.val_targets.append(labels.detach().cpu())

        return loss

    def on_validation_epoch_end(self) -> None:
        """
        Perform actions at the end of the validation epoch.
        """
        preds = torch.cat(self.val_preds)
        targets = torch.cat(self.val_targets)

        self.val_confusion_matrix = self.val_confusion_matrix.to(preds.device)
        self.val_confusion_matrix.update(preds, targets)
        fig, _ = self.val_confusion_matrix.plot()
        self.logger.experiment.log_figure(self.logger.run_id, fig, "val_roc_curve.png")
        self.val_preds.clear()
        self.val_targets.clear()

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
        probs = F.softmax(outputs, dim=1)

        self.test_acc(probs, labels)
        self.test_precision(probs, labels)
        self.test_recall(probs, labels)
        self.test_macro_f1(probs, labels)
        self.test_auroc(probs, labels)

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc, on_epoch=True, prog_bar=True)
        self.log("test_precision", self.test_precision, on_epoch=True)
        self.log("test_recall", self.test_recall, on_epoch=True)
        self.log("test_macro_f1", self.test_macro_f1, on_epoch=True)
        self.log("test_auroc", self.test_auroc, on_epoch=True)

        self.test_preds.append(torch.argmax(probs, dim=1).detach().cpu())
        self.test_targets.append(labels.detach().cpu())

        return loss

    def on_test_epoch_end(self) -> None:
        """
        Perform actions at the end of the test epoch.
        """
        preds = torch.cat(self.test_preds)
        targets = torch.cat(self.test_targets)

        self.test_confusion_matrix = self.test_confusion_matrix.to(preds.device)
        self.test_confusion_matrix.update(preds, targets)
        fig, _ = self.test_confusion_matrix.plot()
        self.logger.experiment.log_figure(self.logger.run_id, fig, "test_roc_curve.png")
        self.test_preds.clear()
        self.test_targets.clear()
