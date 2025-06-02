from abc import abstractmethod, ABC
from typing import Dict, Callable, Tuple, List

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from addict import Dict as Adict
from deeptime.decomposition.deep import vampnet_loss
from tqdm import tqdm_notebook as tqdm
import numpy as np

from celerity.utils import get_logger
from celerity.layers import Lobe
from celerity.optimizers import HedgeOptimizer

logger = get_logger(__name__)


class VAMPnetEstimator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_hidden_layers: int = 1,
        hidden_layer_width: int = 10,
        output_softmax: bool = False,
        device: str = "cpu",
        lr: float = 5e-4,
        n_epochs: int = 30,
        optimizer: str = "Adam",
        score: Dict = dict(method="VAMP2", mode="regularize", epsilon=1e-6),
        scheduler: str = None,
        scheduler_kwargs: Dict = {},
    ):
        super(VAMPnetEstimator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden_layers = n_hidden_layers
        self.hidden_layer_width = hidden_layer_width
        self.output_softmax = output_softmax
        self.device = torch.device(device)
        self.lr = lr
        self.n_epochs = n_epochs
        self.score = score
        try:
            self.optimizer_class = getattr(torch.optim, optimizer)
        except AttributeError:
            logger.exception(f"Couldn't load optimizer {optimizer}", exc_info=True)
            raise

        self.t_0 = self.create_lobe()
        self.t_tau = self.t_0
        self.optimizer = self.optimizer_class(self.parameters(), lr=self.lr)

        if scheduler is not None:
            import torch.optim.lr_scheduler as sched

            try:
                scheduler_class = getattr(sched, scheduler)
                self.scheduler = scheduler_class(self.optimizer, **scheduler_kwargs)
            except AttributeError:
                logger.exception(f"Couldn't load scheduler {scheduler}", exc_info=True)
                raise
        else:
            self.scheduler = None

        self.to(self.device)

        self.step = 0
        self.dict_scores = dict(
            {
                "train": {self.score["method"]: {}, "loss": {}},
                "validate": {self.score["method"]: {}, "loss": {}},
            }
        )

    def create_lobe(self):
        return Lobe(
            self.input_dim,
            self.output_dim,
            self.n_hidden_layers,
            self.hidden_layer_width,
            self.output_softmax,
        )

    def forward(self, x):
        x_0 = self.t_0(x[0])[-1]
        x_t = self.t_tau(x[1])[-1]
        return (x_0, x_t)

    def fit(
        self,
        train_loader,
        validate_loader,
        record_interval=None,
        train_callbacks=None,
        validate_callbacks=None,
    ):
        self.optimizer.zero_grad()
        n_batches = len(train_loader)
        if record_interval is None:
            record_interval = n_batches - 1

        for epoch_ix in range(self.n_epochs):
            # for epoch_ix in tqdm(range(self.options.n_epochs), desc='Epoch', total=self.options.n_epochs):
            self.train()
            for batch_ix, batch in enumerate(train_loader):
                # for batch_ix, batch in tqdm(enumerate(train_loader), desc='Batch', total=n_batches):

                self.train_batch(batch, train_callbacks)
                if (batch_ix % record_interval == 0) and (batch_ix > 0):
                    self.eval()
                    if validate_loader is not None:
                        self.validate(validate_loader, validate_callbacks)
        if self.scheduler is not None:
            self.scheduler.step()

    def score_batch(self, x):
        x0, xt = x[0].to(self.device), x[1].to(self.device)
        output = self((x0, xt))  # calls the forward method
        loss = vampnet_loss(output[0], output[1], **self.score)
        return loss

    def train_batch(self, x, callbacks):
        self.optimizer.zero_grad()
        loss = self.score_batch(x)
        loss.backward()
        self.optimizer.step()
        loss_value = loss.item()
        self.dict_scores["train"][self.score["method"]][self.step] = -loss_value
        self.dict_scores["train"]["loss"][self.step] = loss_value
        if callbacks is not None:
            for callback in callbacks:
                callback(self.step, self.dict_scores)
        self.step += 1

    def validate(self, data_loader, callbacks):
        losses = []
        for i, batch in enumerate(data_loader):
            with torch.no_grad():
                val_loss = self.score_batch(batch)
                losses.append(val_loss)
            mean_score = -torch.mean(torch.stack(losses)).item()

        self.dict_scores["validate"][self.score["method"]][self.step] = mean_score
        self.dict_scores["validate"]["loss"][self.step] = -mean_score
        if callbacks is not None:
            for callback in callbacks:
                callback(self.step, self.dict_scores)

class HedgeVAMPNetEstimator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_hidden_layers: int,
        hidden_layer_width: int,
        loss: Callable[[np.ndarray, np.ndarray], float],
        output_softmax: bool = False,
        device: str = "cpu",
        b: float = 0.99,
        n: float = 0.01,
        s: float = 0.1,
        score_method: str = "VAMP2",
        n_epochs: int = 1,
    ):
        super().__init__()

        # Setup device
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and "cuda" in device else "cpu"
        )

        self.loss = loss
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_softmax = output_softmax

        # Setup layers
        self.n_hidden_layers = n_hidden_layers
        self.lobe = Lobe(
            input_dim,
            output_dim,
            n_hidden_layers,
            hidden_layer_width,
            output_softmax,
        ).to(self.device)

        # Other training parameters
        self.b = Parameter(torch.tensor(b), requires_grad=False).to(self.device)
        self.n = Parameter(torch.tensor(n), requires_grad=False).to(self.device)
        self.s = Parameter(torch.tensor(s), requires_grad=False).to(self.device)

        self.alpha = Parameter(
            torch.Tensor(self.n_hidden_layers).fill_(1 / (self.n_hidden_layers + 1)),
            requires_grad=False,
        ).to(self.device)

        self.optimizer = HedgeOptimizer(self)

        # Output accumulators
        self.loss_array = []
        self.alpha_array = []

        self.score_method = score_method
        self.n_epochs = n_epochs
        self.step = 0
        self.dict_scores = dict(
            {
                "train": {self.score_method: {}, "loss": {}},
                "validate": {self.score_method: {}, "loss": {}},
            }
        )

    def partial_forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.lobe(x)

    def zero_grad(self):
        for i in range(self.n_hidden_layers):
            self.lobe.hidden_layers[i].zero_grad()
            self.lobe.output_layers[i].zero_grad()

    def forward(self, x: List[torch.Tensor]) -> Tuple[List[torch.Tensor]]:
        x_0, x_tau = x[0], x[1]
        pred_0_per_layer = self.partial_forward(x_0)
        pred_tau_per_layer = self.partial_forward(x_tau)
        return (pred_0_per_layer, pred_tau_per_layer)

    def loss_per_layer(
        self, predictions_per_layer: Tuple[List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        losses_per_layer = []

        for pred_0, pred_tau in zip(*predictions_per_layer):
            loss = self.loss(pred_0, pred_tau)
            losses_per_layer.append(loss)
        return losses_per_layer

    def predict(self, x: List[torch.Tensor]) -> float:
        preds_by_layer = self.forward(x)
        loss_by_layer = self.loss_per_layer(preds_by_layer)
        loss_by_layer = torch.stack(loss_by_layer)
        average_loss = torch.sum(torch.mul(self.alpha, loss_by_layer))
        return float(average_loss)

    def transform(self, x: torch.Tensor) -> np.ndarray:
        # Assume stationarity here.
        pred_0_by_layer, _ = self.forward([x, x])
        pred_0_by_layer = torch.stack(pred_0_by_layer)
        # dims are: layers, frames, output states
        a = self.alpha.reshape(self.alpha.shape[0], 1, 1)
        ave_pred_0 = torch.sum(torch.mul(a, pred_0_by_layer), dim=0)
        ave_pred_0 = ave_pred_0.detach().cpu().numpy()
        return ave_pred_0

    def get_alphas(self) -> np.ndarray:
        if self.device.type == "cuda":
            return self.alpha.to("cpu").numpy()
        else:
            return self.alpha.numpy()


    def partial_fit(self, X: List[torch.Tensor]) -> None:
        self.optimizer.step(X)

    def score_batch(self, x):
        # Similar to predict, but return tensor
        preds_by_layer = self.forward(x)
        loss_by_layer = self.loss_per_layer(preds_by_layer)
        loss_by_layer = torch.stack(loss_by_layer)
        average_loss = torch.sum(torch.mul(self.alpha, loss_by_layer))
        return average_loss

    def fit(
        self,
        train_loader,
        validate_loader,
        record_interval=None,
        train_callbacks=None,
        validate_callbacks=None,
    ):
        n_batches = len(train_loader)
        if record_interval is None:
            record_interval = n_batches - 1

        for epoch_ix in range(self.n_epochs):
            self.train()
            for batch_ix, batch in enumerate(train_loader):
                self.train_batch(batch, train_callbacks)
                if (batch_ix % record_interval == 0) and (batch_ix > 0):
                    self.eval()
                    if validate_loader is not None:
                        self.validate(validate_loader, validate_callbacks)

    def train_batch(self, x, callbacks):
        self.partial_fit(x)
        loss = self.score_batch(x)
        loss_value = loss.item()
        self.dict_scores["train"][self.score_method][self.step] = -loss_value
        self.dict_scores["train"]["loss"][self.step] = loss_value
        if callbacks is not None:
            for callback in callbacks:
                callback(self.step, self.dict_scores)
        self.step += 1

    def validate(self, data_loader, callbacks):
        losses = []
        for i, batch in enumerate(data_loader):
            with torch.no_grad():
                val_loss = self.score_batch(batch)
                losses.append(val_loss)
        mean_score = -torch.mean(torch.stack(losses)).item()
        self.dict_scores["validate"][self.score_method][self.step] = mean_score
        self.dict_scores["validate"]["loss"][self.step] = -mean_score
        if callbacks is not None:
            for callback in callbacks:
                callback(self.step, self.dict_scores)


estimator_by_type = dict(batch=VAMPnetEstimator, online=HedgeVAMPNetEstimator)
