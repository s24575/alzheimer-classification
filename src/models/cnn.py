import pytorch_lightning as pl


class CNNModel(pl.LightningModule):
    def __init__(self, num_classes: int):
        super(CNNModel, self).__init__()
