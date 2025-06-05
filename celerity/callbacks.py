from typing import *
import numpy as np
import torch


class LossLogger(object):
    """Callback to write out into tensorboard

    Parameters
    ----------
    tb_writer : tensorboard.Writer
        The tensorboard writer.
    data_set: str, either 'train' or 'valid'.
        If it is the training/validation set.
    """

    def __init__(self, tb_writer, data_set) -> None:
        super().__init__()
        self.writer = tb_writer
        self.data_set = data_set

    def __call__(self, step: int, dict: Dict) -> None:
        """Update the tensorboard with the recorded scores from step

        Parameters
        ----------
        step : int
            The training step number.
        dict : Dict
            The dictionary containing the information to write out.
        """
        for scoring_method, items in dict[self.data_set].items():
            if step in items.keys():
                self.writer.add_scalars(
                    scoring_method, {self.data_set: items[step]}, step
                )


class AlphaRecorder(object):
    """Callback to record alpha (expert weight) values during training for HedgeVAmpNetEstimator

    Parameters
    ----------
    estimator : HedgeVAmpNetEstimator
        The estimator to record alphas from
    """

    def __init__(self, estimator) -> None:
        super().__init__()
        self.estimator = estimator
        self.alphas = []

    def __call__(self, step: int, dict: Dict) -> None:
        """Record the current alpha values

        Parameters
        ----------
        step : int
            The training step number
        dict : Dict
            The dictionary containing training information (unused)
        """
        current_alphas = self.estimator.get_layer_weights().copy()
        self.alphas.append(current_alphas)

    def get_alphas(self) -> List[np.ndarray]:
        """Get all recorded alpha values

        Returns
        -------
        List[np.ndarray]
            List of alpha arrays recorded during training
        """
        return self.alphas


class PredictionsByLayerRecorder(object):
    """Callback to record predictions by layer during validation for HedgeVAmpNetEstimator

    Parameters
    ----------
    estimator : HedgeVAmpNetEstimator
        The estimator to record predictions from
    data_tensors : List[torch.Tensor]
        The data tensors to get predictions for
    """

    def __init__(self, estimator, data_tensors) -> None:
        super().__init__()
        self.estimator = estimator
        self.data_tensors = data_tensors
        self.predictions_by_layer = {}

        # Initialize storage for each layer
        for layer_num in range(estimator.model.n_hidden_layers):
            self.predictions_by_layer[layer_num] = []

    def __call__(self, step: int, dict: Dict) -> None:
        """Record predictions by layer for the current validation step

        Parameters
        ----------
        step : int
            The training step number
        dict : Dict
            The dictionary containing validation information (unused)
        """
        self.estimator.eval()
        with torch.no_grad():
            # Record predictions for each data tensor
            step_predictions = {
                layer_num: [] for layer_num in range(self.estimator.model.n_hidden_layers)
            }

            for x in self.data_tensors:
                y, _ = self.estimator.forward((x, x))
                for layer_num in range(self.estimator.model.n_hidden_layers):
                    result = y[layer_num].detach().cpu().numpy()
                    step_predictions[layer_num].append(result)

            # Store the predictions for this validation step
            for layer_num in range(self.estimator.model.n_hidden_layers):
                self.predictions_by_layer[layer_num].append(step_predictions[layer_num])

    def get_predictions_by_layer(self) -> Dict[int, List[np.ndarray]]:
        """Get all recorded predictions by layer

        Returns
        -------
        Dict[int, List[np.ndarray]]
            Dictionary mapping layer numbers to lists of prediction arrays
        """
        return self.predictions_by_layer
