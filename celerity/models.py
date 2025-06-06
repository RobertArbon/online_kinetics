from typing import Dict, Callable, Tuple, List, Union, Optional
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np

from deeptime.decomposition.deep import vampnet_loss
from celerity.utils import get_logger
from celerity.layers import Lobe
from celerity.optimizers import HedgeOptimizer

logger = get_logger(__name__)


# ============================================================================
# Base Classes
# ============================================================================


class BaseVAMPNetModel(nn.Module, ABC):
    """Base class for VAMPNet models."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_hidden_layers: int = 1,
        hidden_layer_width: int = 10,
        output_softmax: bool = False,
        device: str = "cpu",
    ):
        super().__init__()

        # Network architecture parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden_layers = n_hidden_layers
        self.hidden_layer_width = hidden_layer_width
        self.output_softmax = output_softmax

        # Device setup
        self.device = torch.device(device)

        # Build network
        self.lobe = Lobe(
            input_dim=input_dim,
            output_dim=output_dim,
            n_hidden_layers=n_hidden_layers,
            hidden_layer_width=hidden_layer_width,
            output_softmax=output_softmax,
        )

        # Move to device
        self.to(self.device)

    @abstractmethod
    def forward(
        self, x: List[torch.Tensor]
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor]]
    ]:
        """Forward pass through the network."""
        pass

    def transform(self, x: torch.Tensor) -> np.ndarray:
        """Transform data using the trained model."""
        with torch.no_grad():
            x = x.to(self.device)
            return self._transform_impl(x)

    @abstractmethod
    def _transform_impl(self, x: torch.Tensor) -> np.ndarray:
        """Implementation of transform method."""
        pass


class BaseVAMPNetEstimator(nn.Module, ABC):
    """Base class for VAMPNet estimators."""

    def __init__(
        self,
        model: BaseVAMPNetModel,
        score_method: str = "VAMP2",
        n_epochs: int = 30,
    ):
        super().__init__()

        # Model and training parameters
        self.model = model
        self.score_method = score_method
        self.n_epochs = n_epochs

        # Training state
        self.step = 0
        self.training_scores = {
            "train": {self.score_method: {}, "loss": {}},
            "validate": {self.score_method: {}, "loss": {}},
        }

    def forward(
        self, x: List[torch.Tensor]
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor]]
    ]:
        """Forward pass through the model."""
        return self.model(x)

    @abstractmethod
    def score_batch(self, x: List[torch.Tensor]) -> torch.Tensor:
        """Calculate the score for a batch."""
        pass

    @abstractmethod
    def train_batch(
        self, x: List[torch.Tensor], callbacks: Optional[List[Callable]] = None
    ) -> None:
        """Train on a single batch."""
        pass

    def validate(self, data_loader, callbacks: Optional[List[Callable]] = None) -> None:
        """Validate the model on a data loader."""
        validation_losses = []

        for batch in data_loader:
            with torch.no_grad():
                batch_loss = self.score_batch(batch)
                validation_losses.append(batch_loss)

        mean_score = -torch.mean(torch.stack(validation_losses)).item()
        self.training_scores["validate"][self.score_method][self.step] = mean_score
        self.training_scores["validate"]["loss"][self.step] = -mean_score

        if callbacks is not None:
            for callback in callbacks:
                callback(self.step, self.training_scores)

    def fit(
        self,
        train_loader,
        validate_loader=None,
        record_interval=None,
        train_callbacks=None,
        validate_callbacks=None,
    ) -> None:
        """Fit the model to the data."""
        n_batches = len(train_loader)
        if record_interval is None:
            record_interval = n_batches - 1

        for _ in range(self.n_epochs):
            self.train()
            for batch_idx, batch in enumerate(train_loader):
                self.train_batch(batch, train_callbacks)

                if (batch_idx % record_interval == 0) and (batch_idx > 0):
                    self.eval()
                    if validate_loader is not None:
                        self.validate(validate_loader, validate_callbacks)

    def transform(self, x: torch.Tensor) -> np.ndarray:
        """Transform data using the trained model."""
        return self.model.transform(x)


# ============================================================================
# Standard VAMPNet Implementation
# ============================================================================


class StandardVAMPNetModel(BaseVAMPNetModel):
    """Standard VAMPNet model using single output per forward pass."""

    def forward(self, x: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.

        Args:
            x: List containing [x_0, x_tau] tensors

        Returns:
            Tuple of (x_0_output, x_tau_output)
        """
        x_0_output = self.lobe(x[0])[-1]  # Take final layer output
        x_tau_output = self.lobe(x[1])[-1]  # Take final layer output
        return (x_0_output, x_tau_output)

    def _transform_impl(self, x: torch.Tensor) -> np.ndarray:
        """Transform implementation for standard model."""
        output = self.lobe(x)[-1]  # Take final layer output
        return output.detach().cpu().numpy()


class StandardVAMPNetEstimator(BaseVAMPNetEstimator):
    """Standard VAMPnet estimator using batch training."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_hidden_layers: int = 1,
        hidden_layer_width: int = 10,
        output_softmax: bool = False,
        device: str = "cpu",
        learning_rate: float = 5e-4,
        n_epochs: int = 1,
        optimizer_name: str = "Adam",
        score_method: str = "VAMP2",
        score_mode: str = "regularize",
        score_epsilon: float = 1e-6,
    ):
        # Create model
        model = StandardVAMPNetModel(
            input_dim=input_dim,
            output_dim=output_dim,
            n_hidden_layers=n_hidden_layers,
            hidden_layer_width=hidden_layer_width,
            output_softmax=output_softmax,
            device=device,
        )

        super().__init__(
            model=model,
            score_method=score_method,
            n_epochs=n_epochs,
        )

        # Training parameters
        self.learning_rate = learning_rate
        self.score_config = dict(
            method=score_method, mode=score_mode, epsilon=score_epsilon
        )

        # Setup optimizer
        try:
            optimizer_class = getattr(torch.optim, optimizer_name)
            self.optimizer = optimizer_class(self.parameters(), lr=self.learning_rate)
        except AttributeError:
            logger.exception(f"Couldn't load optimizer {optimizer_name}", exc_info=True)
            raise

    def score_batch(self, x: List[torch.Tensor]) -> torch.Tensor:
        """Calculate the score for a batch.

        Args:
            x: List containing [x_0, x_tau] tensors

        Returns:
            Loss tensor
        """
        x_0 = x[0].to(self.model.device)
        x_tau = x[1].to(self.model.device)

        model_output = self.model([x_0, x_tau])
        loss = vampnet_loss(model_output[0], model_output[1], **self.score_config)
        return loss

    def train_batch(
        self, x: List[torch.Tensor], callbacks: Optional[List[Callable]] = None
    ) -> None:
        """Train on a single batch.

        Args:
            x: List containing [x_0, x_tau] tensors
            callbacks: Optional list of callback functions
        """
        self.optimizer.zero_grad()
        batch_loss = self.score_batch(x)
        batch_loss.backward()
        self.optimizer.step()

        loss_value = batch_loss.item()
        self.training_scores["train"][self.score_method][self.step] = -loss_value
        self.training_scores["train"]["loss"][self.step] = loss_value

        if callbacks is not None:
            for callback in callbacks:
                callback(self.step, self.training_scores)

        self.step += 1


# ============================================================================
# Hedge VAMPNet Implementation
# ============================================================================


class HedgeVAMPNetModel(BaseVAMPNetModel):
    """Hedge VAMPNet model using multi-layer outputs and weighted combination."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_hidden_layers: int,
        hidden_layer_width: int,
        output_softmax: bool = False,
        device: str = "cpu",
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            n_hidden_layers=n_hidden_layers,
            hidden_layer_width=hidden_layer_width,
            output_softmax=output_softmax,
            device=device,
        )

        # Layer weights (alpha) - initialized uniformly
        initial_alpha = 1.0 / (self.n_hidden_layers)
        self.layer_weights = Parameter(
            torch.full((self.n_hidden_layers,), initial_alpha),
            requires_grad=False,
        ).to(self.device)

    def forward(
        self, x: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Forward pass through the network.

        Args:
            x: List containing [x_0, x_tau] tensors

        Returns:
            Tuple of (x_0_outputs, x_tau_outputs) where each element is a list of outputs per layer
        """
        x_0, x_tau = x[0], x[1]
        x_0_outputs = self.lobe(x_0)
        x_tau_outputs = self.lobe(x_tau)
        return (x_0_outputs, x_tau_outputs)

    def _transform_impl(self, x: torch.Tensor) -> np.ndarray:
        """Transform implementation for hedge model using weighted combination."""
        layer_outputs = self.lobe(x)
        layer_outputs_tensor = torch.stack(layer_outputs)

        # Weighted combination: dims are [layers, batch, features]
        weights_reshaped = self.layer_weights.reshape(self.layer_weights.shape[0], 1, 1)
        weighted_output = torch.sum(
            torch.mul(weights_reshaped, layer_outputs_tensor), dim=0
        )

        return weighted_output.detach().cpu().numpy()

    def get_layer_weights(self) -> np.ndarray:
        """Get the current layer weights (alpha values).

        Returns:
            Layer weights as numpy array
        """
        if self.device.type == "cuda":
            return self.layer_weights.to("cpu").numpy()
        else:
            return self.layer_weights.numpy()


class HedgeVAMPNetEstimator(BaseVAMPNetEstimator):
    """VAMPnet estimator using online Hedge training."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_hidden_layers: int,
        hidden_layer_width: int,
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        output_softmax: bool = False,
        device: str = "cpu",
        hedge_beta: float = 0.99,
        hedge_eta: float = 0.01,
        hedge_gamma: float = 0.1,
        score_method: str = "VAMP2",
        n_epochs: int = 1,
    ):
        # Create model
        model = HedgeVAMPNetModel(
            input_dim=input_dim,
            output_dim=output_dim,
            n_hidden_layers=n_hidden_layers,
            hidden_layer_width=hidden_layer_width,
            output_softmax=output_softmax,
            device=device,
        )

        super().__init__(
            model=model,
            score_method=score_method,
            n_epochs=n_epochs,
        )

        # Hedge algorithm parameters
        self.hedge_beta = Parameter(torch.tensor(hedge_beta), requires_grad=False).to(
            self.model.device
        )
        self.hedge_eta = Parameter(torch.tensor(hedge_eta), requires_grad=False).to(
            self.model.device
        )
        self.hedge_gamma = Parameter(torch.tensor(hedge_gamma), requires_grad=False).to(
            self.model.device
        )

        # Training parameters
        self.loss_function = loss_function

        # Setup optimizer
        self.optimizer = HedgeOptimizer(self)

    def score_batch(self, x: List[torch.Tensor]) -> torch.Tensor:
        """Calculate the score for a batch.

        Args:
            x: List containing [x_0, x_tau] tensors

        Returns:
            Loss tensor
        """
        layer_predictions = self.model(x)
        layer_losses = self._compute_layer_losses(layer_predictions)
        layer_losses_tensor = torch.stack(layer_losses)

        # Weighted combination of losses
        weighted_loss = torch.sum(
            torch.mul(self.model.layer_weights, layer_losses_tensor)
        )
        return weighted_loss

    def train_batch(
        self, x: List[torch.Tensor], callbacks: Optional[List[Callable]] = None
    ) -> None:
        """Train on a single batch.

        Args:
            x: List containing [x_0, x_tau] tensors
            callbacks: Optional list of callback functions
        """
        self.optimizer.step(x)
        batch_loss = self.score_batch(x)

        loss_value = batch_loss.item()
        self.training_scores["train"][self.score_method][self.step] = -loss_value
        self.training_scores["train"]["loss"][self.step] = loss_value

        if callbacks is not None:
            for callback in callbacks:
                callback(self.step, self.training_scores)

        self.step += 1

    def _compute_layer_losses(
        self, layer_predictions: Tuple[List[torch.Tensor], List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """Calculate loss for each layer.

        Args:
            layer_predictions: Tuple of (x_0_outputs, x_tau_outputs)

        Returns:
            List of loss tensors per layer
        """
        layer_losses = []
        x_0_outputs, x_tau_outputs = layer_predictions

        for x_0_pred, x_tau_pred in zip(x_0_outputs, x_tau_outputs):
            layer_loss = self.loss_function(x_0_pred, x_tau_pred)
            layer_losses.append(layer_loss)

        return layer_losses

    def get_layer_weights(self) -> np.ndarray:
        """Get the current layer weights from the model.

        Returns:
            Layer weights as numpy array
        """
        return self.model.get_layer_weights()


# ============================================================================
# Legacy Compatibility
# ============================================================================

# Maintain backward compatibility with old names
VAMPnetEstimator = StandardVAMPNetEstimator
HedgeVAMPNetEstimator = HedgeVAMPNetEstimator

# Estimator registry
estimator_by_type = {"batch": StandardVAMPNetEstimator, "online": HedgeVAMPNetEstimator}
