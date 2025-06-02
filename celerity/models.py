from typing import Dict, Callable, Tuple, List, Union, Optional

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np

from deeptime.decomposition.deep import vampnet_loss
from celerity.utils import get_logger
from celerity.layers import Lobe
from celerity.optimizers import HedgeOptimizer

logger = get_logger(__name__)


class VAMPnetEstimator(nn.Module):
    """Standard VAMPnet estimator using batch training."""
    
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
        score_method: str = "VAMP2",
        score_mode: str = "regularize",
        score_epsilon: float = 1e-6,
    ):
        super().__init__()
        
        # Model parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden_layers = n_hidden_layers
        self.hidden_layer_width = hidden_layer_width
        self.output_softmax = output_softmax
        
        # Training parameters
        self.device = torch.device(device)
        self.lr = lr
        self.n_epochs = n_epochs
        self.score_method = score_method
        self.score = dict(method=score_method, mode=score_mode, epsilon=score_epsilon)
        
        # Setup network
        self.lobe = Lobe(
            input_dim,
            output_dim,
            n_hidden_layers,
            hidden_layer_width,
            output_softmax,
        )
        
        # Setup optimizer
        try:
            optimizer_class = getattr(torch.optim, optimizer)
            self.optimizer = optimizer_class(self.parameters(), lr=self.lr)
        except AttributeError:
            logger.exception(f"Couldn't load optimizer {optimizer}", exc_info=True)
            raise

        # Move model to device
        self.to(self.device)
        
        # Training tracking
        self.step = 0
        self.dict_scores = {
            "train": {self.score_method: {}, "loss": {}},
            "validate": {self.score_method: {}, "loss": {}},
        }

    def forward(self, x: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.
        
        Args:
            x: List containing [x_0, x_tau] tensors
            
        Returns:
            Tuple of (x_0_output, x_tau_output)
        """
        x_0 = self.lobe(x[0])[-1]
        x_tau = self.lobe(x[1])[-1]
        return (x_0, x_tau)

    def score_batch(self, x: List[torch.Tensor]) -> torch.Tensor:
        """Calculate the score for a batch.
        
        Args:
            x: List containing [x_0, x_tau] tensors
            
        Returns:
            Loss tensor
        """
        x0, xt = x[0].to(self.device), x[1].to(self.device)
        output = self((x0, xt))
        loss = vampnet_loss(output[0], output[1], **self.score)
        return loss

    def train_batch(self, x: List[torch.Tensor], callbacks: Optional[List[Callable]] = None) -> None:
        """Train on a single batch.
        
        Args:
            x: List containing [x_0, x_tau] tensors
            callbacks: Optional list of callback functions
        """
        self.optimizer.zero_grad()
        loss = self.score_batch(x)
        loss.backward()
        self.optimizer.step()
        
        loss_value = loss.item()
        self.dict_scores["train"][self.score_method][self.step] = -loss_value
        self.dict_scores["train"]["loss"][self.step] = loss_value
        
        if callbacks is not None:
            for callback in callbacks:
                callback(self.step, self.dict_scores)
                
        self.step += 1

    def validate(self, data_loader, callbacks: Optional[List[Callable]] = None) -> None:
        """Validate the model on a data loader.
        
        Args:
            data_loader: DataLoader containing validation data
            callbacks: Optional list of callback functions
        """
        losses = []
        for batch in data_loader:
            with torch.no_grad():
                val_loss = self.score_batch(batch)
                losses.append(val_loss)
                
        mean_score = -torch.mean(torch.stack(losses)).item()
        self.dict_scores["validate"][self.score_method][self.step] = mean_score
        self.dict_scores["validate"]["loss"][self.step] = -mean_score
        
        if callbacks is not None:
            for callback in callbacks:
                callback(self.step, self.dict_scores)

    def fit(
        self,
        train_loader,
        validate_loader=None,
        record_interval=None,
        train_callbacks=None,
        validate_callbacks=None,
    ) -> None:
        """Fit the model to the data.
        
        Args:
            train_loader: DataLoader containing training data
            validate_loader: Optional DataLoader containing validation data
            record_interval: Interval for recording validation scores
            train_callbacks: Optional list of callback functions for training
            validate_callbacks: Optional list of callback functions for validation
        """
        n_batches = len(train_loader)
        if record_interval is None:
            record_interval = n_batches - 1

        for _ in range(self.n_epochs):
            self.train()
            for batch_ix, batch in enumerate(train_loader):
                self.train_batch(batch, train_callbacks)
                if (batch_ix % record_interval == 0) and (batch_ix > 0):
                    self.eval()
                    if validate_loader is not None:
                        self.validate(validate_loader, validate_callbacks)
                        
        if self.scheduler is not None:
            self.scheduler.step()

    def transform(self, x: torch.Tensor) -> np.ndarray:
        """Transform data using the trained model.
        
        Args:
            x: Input tensor
            
        Returns:
            Transformed data as numpy array
        """
        with torch.no_grad():
            output = self.lobe(x)[-1]
            return output.detach().cpu().numpy()


class HedgeVAMPNetEstimator(nn.Module):
    """VAMPnet estimator using online Hedge training."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_hidden_layers: int,
        hidden_layer_width: int,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        output_softmax: bool = False,
        device: str = "cpu",
        b: float = 0.99,
        n: float = 0.01,
        s: float = 0.1,
        score_method: str = "VAMP2",
        n_epochs: int = 1,
    ):
        super().__init__()
        
        # Model parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden_layers = n_hidden_layers
        self.hidden_layer_width = hidden_layer_width
        self.output_softmax = output_softmax
        
        # Training parameters
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and "cuda" in device else "cpu"
        )
        self.loss = loss
        self.score_method = score_method
        self.n_epochs = n_epochs
        
        # Setup network
        self.lobe = Lobe(
            input_dim,
            output_dim,
            n_hidden_layers,
            hidden_layer_width,
            output_softmax,
        ).to(self.device)
        
        # Hedge parameters
        self.b = Parameter(torch.tensor(b), requires_grad=False).to(self.device)
        self.n = Parameter(torch.tensor(n), requires_grad=False).to(self.device)
        self.s = Parameter(torch.tensor(s), requires_grad=False).to(self.device)
        self.alpha = Parameter(
            torch.Tensor(self.n_hidden_layers).fill_(1 / (self.n_hidden_layers + 1)),
            requires_grad=False,
        ).to(self.device)
        
        # Setup optimizer
        self.optimizer = HedgeOptimizer(self)
        
        # Training tracking
        self.step = 0
        self.dict_scores = {
            "train": {self.score_method: {}, "loss": {}},
            "validate": {self.score_method: {}, "loss": {}},
        }

    def forward(self, x: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Forward pass through the network.
        
        Args:
            x: List containing [x_0, x_tau] tensors
            
        Returns:
            Tuple of (x_0_outputs, x_tau_outputs) where each element is a list of outputs per layer
        """
        x_0, x_tau = x[0], x[1]
        pred_0_per_layer = self.lobe(x_0)
        pred_tau_per_layer = self.lobe(x_tau)
        return (pred_0_per_layer, pred_tau_per_layer)

    def score_batch(self, x: List[torch.Tensor]) -> torch.Tensor:
        """Calculate the score for a batch.
        
        Args:
            x: List containing [x_0, x_tau] tensors
            
        Returns:
            Loss tensor
        """
        preds_by_layer = self.forward(x)
        loss_by_layer = self.loss_per_layer(preds_by_layer)
        loss_by_layer = torch.stack(loss_by_layer)
        average_loss = torch.sum(torch.mul(self.alpha, loss_by_layer))
        return average_loss

    def train_batch(self, x: List[torch.Tensor], callbacks: Optional[List[Callable]] = None) -> None:
        """Train on a single batch.
        
        Args:
            x: List containing [x_0, x_tau] tensors
            callbacks: Optional list of callback functions
        """
        self.optimizer.step(x)
        loss = self.score_batch(x)
        
        loss_value = loss.item()
        self.dict_scores["train"][self.score_method][self.step] = -loss_value
        self.dict_scores["train"]["loss"][self.step] = loss_value
        
        if callbacks is not None:
            for callback in callbacks:
                callback(self.step, self.dict_scores)
                
        self.step += 1

    def validate(self, data_loader, callbacks: Optional[List[Callable]] = None) -> None:
        """Validate the model on a data loader.
        
        Args:
            data_loader: DataLoader containing validation data
            callbacks: Optional list of callback functions
        """
        losses = []
        for batch in data_loader:
            with torch.no_grad():
                val_loss = self.score_batch(batch)
                losses.append(val_loss)
                
        mean_score = -torch.mean(torch.stack(losses)).item()
        self.dict_scores["validate"][self.score_method][self.step] = mean_score
        self.dict_scores["validate"]["loss"][self.step] = -mean_score
        
        if callbacks is not None:
            for callback in callbacks:
                callback(self.step, self.dict_scores)

    def fit(
        self,
        train_loader,
        validate_loader=None,
        record_interval=None,
        train_callbacks=None,
        validate_callbacks=None,
    ) -> None:
        """Fit the model to the data.
        
        Args:
            train_loader: DataLoader containing training data
            validate_loader: Optional DataLoader containing validation data
            record_interval: Interval for recording validation scores
            train_callbacks: Optional list of callback functions for training
            validate_callbacks: Optional list of callback functions for validation
        """
        n_batches = len(train_loader)
        if record_interval is None:
            record_interval = n_batches - 1

        for _ in range(self.n_epochs):
            self.train()
            for batch_ix, batch in enumerate(train_loader):
                self.train_batch(batch, train_callbacks)
                if (batch_ix % record_interval == 0) and (batch_ix > 0):
                    self.eval()
                    if validate_loader is not None:
                        self.validate(validate_loader, validate_callbacks)

    def transform(self, x: torch.Tensor) -> np.ndarray:
        """Transform data using the trained model.
        
        Args:
            x: Input tensor
            
        Returns:
            Transformed data as numpy array
        """
        with torch.no_grad():
            pred_by_layer = self.lobe(x)
            pred_by_layer = torch.stack(pred_by_layer)
            # dims are: layers, frames, output states
            a = self.alpha.reshape(self.alpha.shape[0], 1, 1)
            ave_pred = torch.sum(torch.mul(a, pred_by_layer), dim=0)
            return ave_pred.detach().cpu().numpy()

    def loss_per_layer(
        self, predictions_per_layer: Tuple[List[torch.Tensor], List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """Calculate loss for each layer.
        
        Args:
            predictions_per_layer: Tuple of (x_0_outputs, x_tau_outputs)
            
        Returns:
            List of loss tensors per layer
        """
        losses_per_layer = []
        for pred_0, pred_tau in zip(*predictions_per_layer):
            loss = self.loss(pred_0, pred_tau)
            losses_per_layer.append(loss)
        return losses_per_layer

    def get_alphas(self) -> np.ndarray:
        """Get the alpha weights for each layer.
        
        Returns:
            Alpha weights as numpy array
        """
        if self.device.type == "cuda":
            return self.alpha.to("cpu").numpy()
        else:
            return self.alpha.numpy()

estimator_by_type = dict(batch=VAMPnetEstimator, online=HedgeVAMPNetEstimator)
