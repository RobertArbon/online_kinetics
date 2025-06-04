from typing import List

import torch
from torch import Tensor
from torch.optim import Optimizer


class HedgeOptimizer(Optimizer):
    """
    Hedge algorithm optimizer for online learning with multiple experts (layers).

    This optimizer implements the Hedge algorithm for combining predictions from
    multiple layers in a neural network, updating both the network weights and
    the layer combination weights (alpha values).
    """

    def __init__(self, estimator):
        """
        Initialize the Hedge optimizer.

        Args:
            estimator: The HedgeVAMPNetEstimator instance containing the model
        """
        self.estimator = estimator
        self.model = estimator.model
        super().__init__(self.model.parameters(), defaults={})

    def step(self, batch_data: List[Tensor]) -> None:
        """
        Perform one optimization step using the Hedge algorithm.

        Args:
            batch_data: List containing [x_0, x_tau] tensors for the current batch
        """
        # Get predictions and losses for each layer
        layer_predictions = self.model(batch_data)
        layer_losses = self.estimator._compute_layer_losses(layer_predictions)

        # Initialize gradient accumulators for hidden layers
        hidden_weight_grads = [None] * len(layer_losses)
        hidden_bias_grads = [None] * len(layer_losses)

        with torch.no_grad():
            # Process each layer's loss and update weights
            for layer_idx in range(len(layer_losses)):
                # Compute gradients for this layer
                layer_losses[layer_idx].backward(retain_graph=True)

                # Update output layer weights for this layer
                self._update_output_layer(layer_idx)

                # Accumulate gradients for hidden layers (weighted by alpha)
                self._accumulate_hidden_gradients(
                    layer_idx, hidden_weight_grads, hidden_bias_grads
                )

                # Clear gradients for next iteration
                self.model.zero_grad()

            # Update all hidden layers with accumulated gradients
            self._update_hidden_layers(hidden_weight_grads, hidden_bias_grads)

            # Update layer weights (alpha values) using Hedge algorithm
            self._update_layer_weights(layer_losses)

    def _update_output_layer(self, layer_idx: int) -> None:
        """
        Update the output layer weights for a specific layer.

        Args:
            layer_idx: Index of the layer to update
        """
        output_layer = self.model.lobe.output_layers[layer_idx]
        layer_weight = self.model.layer_weights[layer_idx]
        learning_rate = self.model.hedge_eta

        # Update weights and biases
        output_layer.weight.data -= (
            learning_rate * layer_weight * output_layer.weight.grad.data
        )
        output_layer.bias.data -= (
            learning_rate * layer_weight * output_layer.bias.grad.data
        )

    def _accumulate_hidden_gradients(
        self, layer_idx: int, weight_grads: List, bias_grads: List
    ) -> None:
        """
        Accumulate gradients for hidden layers, weighted by layer importance.

        Args:
            layer_idx: Current layer index
            weight_grads: List to accumulate weight gradients
            bias_grads: List to accumulate bias gradients
        """
        layer_weight = self.model.layer_weights[layer_idx]

        # Accumulate gradients for all hidden layers up to current layer
        for hidden_idx in range(layer_idx + 1):
            hidden_layer = self.model.lobe.hidden_layers[hidden_idx]
            weighted_weight_grad = layer_weight * hidden_layer.weight.grad.data
            weighted_bias_grad = layer_weight * hidden_layer.bias.grad.data

            if weight_grads[hidden_idx] is None:
                weight_grads[hidden_idx] = weighted_weight_grad
                bias_grads[hidden_idx] = weighted_bias_grad
            else:
                weight_grads[hidden_idx] += weighted_weight_grad
                bias_grads[hidden_idx] += weighted_bias_grad

    def _update_hidden_layers(
        self, weight_grads: List[Tensor], bias_grads: List[Tensor]
    ) -> None:
        """
        Update all hidden layers with accumulated gradients.

        Args:
            weight_grads: Accumulated weight gradients for each hidden layer
            bias_grads: Accumulated bias gradients for each hidden layer
        """
        learning_rate = self.model.hedge_eta

        for hidden_idx in range(len(weight_grads)):
            if weight_grads[hidden_idx] is not None:
                hidden_layer = self.model.lobe.hidden_layers[hidden_idx]
                hidden_layer.weight.data -= learning_rate * weight_grads[hidden_idx]
                hidden_layer.bias.data -= learning_rate * bias_grads[hidden_idx]

    def _update_layer_weights(self, layer_losses: List[Tensor]) -> None:
        """
        Update layer combination weights using the Hedge algorithm.

        Args:
            layer_losses: List of loss values for each layer
        """
        # Update alpha values using exponential weights
        for layer_idx in range(len(layer_losses)):
            loss_value = layer_losses[layer_idx]

            # Exponential update: alpha_i *= beta^(loss_i)
            self.model.layer_weights[layer_idx] *= torch.pow(
                self.model.hedge_beta, loss_value
            )

            # Apply minimum weight constraint
            min_weight = self.model.hedge_gamma / self.model.n_hidden_layers
            self.model.layer_weights[layer_idx] = torch.max(
                self.model.layer_weights[layer_idx], min_weight
            )

        # Normalize weights to sum to 1
        total_weight = torch.sum(self.model.layer_weights)
        self.model.layer_weights.data = self.model.layer_weights / total_weight
