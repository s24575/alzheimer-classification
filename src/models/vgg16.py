import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn
from torchmetrics.classification import MulticlassConfusionMatrix
from torchvision import models
from torchvision.models import VGG16_Weights


class VGG16Model(pl.LightningModule):
    def __init__(self, num_classes: int):
        super(VGG16Model, self).__init__()

        self.model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier[6] = nn.Linear(
            self.model.classifier[6].in_features, num_classes
        )
        self.model.classifier[6].weight.requires_grad = True
        self.model.classifier[6].bias.requires_grad = True

        self.criterion = nn.CrossEntropyLoss()

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

        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)

        self.val_outputs = None
        self.val_labels = None

        self.save_hyperparameters()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
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

        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        inputs, labels = batch

        outputs = self.forward(inputs.float())
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

        self.val_outputs = outputs
        self.val_labels = labels

        return loss

    def on_validation_epoch_end(self) -> None:
        self.confusion_matrix.update(self.val_outputs, self.val_labels)
        fig, ax = self.confusion_matrix.plot(add_text=False)
        self.logger.experiment.add_figure("Confusion matrix", fig, self.current_epoch)

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        inputs, labels = batch

        outputs = self.forward(inputs.float())
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

        return loss
