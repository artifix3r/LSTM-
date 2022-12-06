import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, Tensor
from torchmetrics.classification import MulticlassConfusionMatrix
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Dict, Iterable, Callable


class LSTMClassifier(pl.LightningModule):
    def __init__(
        self,
        n_features,
        hidden_size,
        batch_size,
        num_layers,
        dropout,
        learning_rate,
        criterion,
        bidirectional,
    ):
        super(LSTMClassifier, self).__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.linear = nn.Linear(hidden_size, 20)

    def forward(self, x):
        x, (hn, cn) = self.lstm(x)
        x = self.dropout(hn[-1])
        x = self.linear(x)
        x = self.softmax(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def shared_step(self, batch):
        x, y = batch
        y_pred = self.forward(x)
        y = y.long()
        loss = self.criterion(y_pred, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss)
        return loss

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     loss = self.criterion(y_hat, y)
    #     result = pl.EvalResult()
    #     result.log("test_loss", loss)
    #     return result

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        y = y.long()
        matric = MulticlassConfusionMatrix(num_classes=20)
        confusion_matrix = matric(y_pred, y)
        loss = self.criterion(y_pred, y)
        self.log("test_loss", loss)

        df_cm = pd.DataFrame(
            confusion_matrix.numpy(), index=range(20), columns=range(20)
        )
        plt.figure(figsize=(10, 7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap="Spectral").get_figure()
        plt.close(fig_)

        self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)
        return loss


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output

        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        _ = self.model(x)
        return self._features
