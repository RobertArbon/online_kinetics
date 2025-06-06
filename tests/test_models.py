"""
Unit tests for models in the celerity package.

These tests focus on the functionality of the model classes, ensuring that
they initialize correctly, perform forward passes as expected, and transform
data correctly.
"""

import pytest
import torch
import numpy as np
from functools import partial
from deeptime.decomposition.deep import vampnet_loss

from celerity.models import (
    StandardVAMPNetModel,
    StandardVAMPNetEstimator,
    HedgeVAMPNetModel,
    HedgeVAMPNetEstimator,
)


class TestStandardVAMPNetModel:
    """Test suite for StandardVAMPNetModel functionality."""

    @pytest.fixture
    def model_config(self):
        """Base configuration for StandardVAMPNetModel."""
        return {
            "input_dim": 4,
            "output_dim": 2,
            "n_hidden_layers": 2,
            "hidden_layer_width": 8,
            "output_softmax": False,
            "device": "cpu",
        }

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        torch.manual_seed(42)  # For reproducibility
        batch_size = 10
        input_dim = 4
        x_0 = torch.randn(batch_size, input_dim)
        x_tau = torch.randn(batch_size, input_dim)
        return [x_0, x_tau]

    def test_model_initialization(self, model_config):
        """Test that StandardVAMPNetModel initializes correctly."""
        model = StandardVAMPNetModel(**model_config)

        # Check lobe structure
        assert len(model.lobe.hidden_layers) == model_config["n_hidden_layers"]
        assert len(model.lobe.output_layers) == model_config["n_hidden_layers"]
        
        # Check layer dimensions
        assert model.lobe.hidden_layers[0].in_features == model_config["input_dim"]
        assert model.lobe.output_layers[0].out_features == model_config["output_dim"]

    def test_forward_pass(self, model_config, sample_data):
        """Test forward pass through the model."""
        model = StandardVAMPNetModel(**model_config)
        x_0, x_tau = sample_data
        
        # Perform forward pass
        outputs = model(sample_data)
        
        # Check output structure
        assert isinstance(outputs, tuple)
        assert len(outputs) == 2
        
        # Check output shapes
        assert outputs[0].shape == (x_0.shape[0], model_config["output_dim"])
        assert outputs[1].shape == (x_tau.shape[0], model_config["output_dim"])

    def test_transform(self, model_config, sample_data):
        """Test transform method."""
        model = StandardVAMPNetModel(**model_config)
        x_0 = sample_data[0]
        
        # Transform data
        transformed = model.transform(x_0)
        
        # Check output type and shape
        assert isinstance(transformed, np.ndarray)
        assert transformed.shape == (x_0.shape[0], model_config["output_dim"])

    def test_output_softmax(self, model_config, sample_data):
        """Test model with output_softmax=True."""
        model_config["output_softmax"] = True
        model = StandardVAMPNetModel(**model_config)
        
        # Perform forward pass
        outputs = model(sample_data)
        
        # Check that outputs sum to 1 along dim 1 (softmax property)
        assert torch.allclose(torch.sum(outputs[0], dim=1), torch.ones(outputs[0].shape[0]))
        assert torch.allclose(torch.sum(outputs[1], dim=1), torch.ones(outputs[1].shape[0]))


class TestHedgeVAMPNetModel:
    """Test suite for HedgeVAMPNetModel functionality."""

    @pytest.fixture
    def model_config(self):
        """Base configuration for HedgeVAMPNetModel."""
        return {
            "input_dim": 4,
            "output_dim": 2,
            "n_hidden_layers": 2,
            "hidden_layer_width": 8,
            "output_softmax": False,
            "device": "cpu",
        }

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        torch.manual_seed(42)  # For reproducibility
        batch_size = 10
        input_dim = 4
        x_0 = torch.randn(batch_size, input_dim)
        x_tau = torch.randn(batch_size, input_dim)
        return [x_0, x_tau]

    def test_model_initialization(self, model_config):
        """Test that HedgeVAMPNetModel initializes correctly."""
        model = HedgeVAMPNetModel(**model_config)
        
        # Check layer weights initialization
        assert model.layer_weights.shape[0] == model_config["n_hidden_layers"]
        expected_weight = 1.0 / model_config["n_hidden_layers"]
        # Check that all weights are equal to the expected value
        assert torch.allclose(model.layer_weights.data, torch.full_like(model.layer_weights.data, expected_weight), rtol=1e-5)
        
        # Check that layer weights sum to 1
        assert abs(torch.sum(model.layer_weights).item() - 1.0) < 1e-6

    def test_forward_pass(self, model_config, sample_data):
        """Test forward pass through the model."""
        model = HedgeVAMPNetModel(**model_config)
        x_0, x_tau = sample_data
        
        # Perform forward pass
        outputs = model(sample_data)
        
        # Check output structure
        assert isinstance(outputs, tuple)
        assert len(outputs) == 2
        assert isinstance(outputs[0], list)
        assert isinstance(outputs[1], list)
        
        # Check output shapes
        for layer_output in outputs[0]:
            assert layer_output.shape == (x_0.shape[0], model_config["output_dim"])
        for layer_output in outputs[1]:
            assert layer_output.shape == (x_tau.shape[0], model_config["output_dim"])
        
        # Check number of layer outputs
        assert len(outputs[0]) == model_config["n_hidden_layers"]
        assert len(outputs[1]) == model_config["n_hidden_layers"]

    def test_transform(self, model_config, sample_data):
        """Test transform method with weighted combination."""
        model = HedgeVAMPNetModel(**model_config)
        x_0 = sample_data[0]
        
        # Transform data
        transformed = model.transform(x_0)
        
        # Check output type and shape
        assert isinstance(transformed, np.ndarray)
        assert transformed.shape == (x_0.shape[0], model_config["output_dim"])
        
        # Verify that transform uses weighted combination
        # First get individual layer outputs
        with torch.no_grad():
            layer_outputs = model.lobe(x_0)
            
            # Manually compute weighted combination
            layer_outputs_tensor = torch.stack(layer_outputs)
            weights_reshaped = model.layer_weights.reshape(model.layer_weights.shape[0], 1, 1)
            expected_output = torch.sum(torch.mul(weights_reshaped, layer_outputs_tensor), dim=0)
            
            # Compare with transform output
            assert np.allclose(transformed, expected_output.numpy())

    def test_get_layer_weights(self, model_config):
        """Test get_layer_weights method."""
        model = HedgeVAMPNetModel(**model_config)
        
        # Get layer weights
        weights = model.get_layer_weights()
        
        # Check output type and shape
        assert isinstance(weights, np.ndarray)
        assert weights.shape == (model_config["n_hidden_layers"],)
        
        # Check that weights match the model's layer_weights
        assert np.allclose(weights, model.layer_weights.numpy())

    def test_output_softmax(self, model_config, sample_data):
        """Test model with output_softmax=True."""
        model_config["output_softmax"] = True
        model = HedgeVAMPNetModel(**model_config)
        
        # Perform forward pass
        outputs = model(sample_data)
        
        # Check that all layer outputs sum to 1 along dim 1 (softmax property)
        x_0_outputs, x_tau_outputs = outputs
        
        for layer_output in x_0_outputs:
            assert torch.allclose(torch.sum(layer_output, dim=1), torch.ones(layer_output.shape[0]))
        
        for layer_output in x_tau_outputs:
            assert torch.allclose(torch.sum(layer_output, dim=1), torch.ones(layer_output.shape[0]))


class TestStandardVAMPNetEstimator:
    """Test suite for StandardVAMPNetEstimator functionality."""

    @pytest.fixture
    def estimator_config(self):
        """Base configuration for StandardVAMPNetEstimator."""
        return {
            "input_dim": 4,
            "output_dim": 2,
            "n_hidden_layers": 2,
            "hidden_layer_width": 8,
            "output_softmax": False,
            "device": "cpu",
            "learning_rate": 5e-4,
            "n_epochs": 1,
            "optimizer_name": "Adam",
            "score_method": "VAMP2",
            "score_mode": "regularize",
            "score_epsilon": 1e-6,
        }

    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch for testing."""
        torch.manual_seed(42)  # For reproducibility
        batch_size = 10
        input_dim = 4
        x_0 = torch.randn(batch_size, input_dim)
        x_tau = torch.randn(batch_size, input_dim)
        return [x_0, x_tau]


    def test_score_batch(self, estimator_config, sample_batch):
        """Test score_batch method."""
        estimator = StandardVAMPNetEstimator(**estimator_config)
        
        # Calculate score
        score = estimator.score_batch(sample_batch)
        
        # Check output type
        assert isinstance(score, torch.Tensor)
        assert score.shape == ()  # Scalar tensor
        
        # Score should be negative (loss)
        assert score.item() < 0

    def test_train_batch(self, estimator_config, sample_batch):
        """Test train_batch method."""
        estimator = StandardVAMPNetEstimator(**estimator_config)
        
        # Store initial parameters
        initial_params = {}
        for name, param in estimator.model.named_parameters():
            if param.requires_grad:
                initial_params[name] = param.data.clone()
        
        # Train on batch
        estimator.train_batch(sample_batch)
        
        # Check that step was incremented
        assert estimator.step == 1
        
        # Check that parameters changed
        params_changed = False
        for name, param in estimator.model.named_parameters():
            if param.requires_grad and name in initial_params:
                if not torch.equal(param.data, initial_params[name]):
                    params_changed = True
                    break
        
        assert params_changed, "Parameters should change after training"

    def test_transform(self, estimator_config, sample_batch):
        """Test transform method."""
        estimator = StandardVAMPNetEstimator(**estimator_config)
        x_0 = sample_batch[0]
        
        # Transform data
        transformed = estimator.transform(x_0)
        
        # Check output type and shape
        assert isinstance(transformed, np.ndarray)
        assert transformed.shape == (x_0.shape[0], estimator_config["output_dim"])
        
        # Should match model's transform method
        assert np.array_equal(transformed, estimator.model.transform(x_0))


class TestHedgeVAMPNetEstimator:
    """Test suite for HedgeVAMPNetEstimator functionality."""

    @pytest.fixture
    def estimator_config(self):
        """Base configuration for HedgeVAMPNetEstimator."""
        return {
            "input_dim": 4,
            "output_dim": 2,
            "n_hidden_layers": 2,
            "hidden_layer_width": 8,
            "loss_function": partial(
                vampnet_loss, method="VAMP2", mode="regularize", epsilon=1e-6
            ),
            "output_softmax": False,
            "device": "cpu",
            "hedge_beta": 0.98,
            "hedge_eta": 0.01,
            "hedge_gamma": 0.1,
            "score_method": "VAMP2",
            "n_epochs": 1,
        }

    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch for testing."""
        torch.manual_seed(42)  # For reproducibility
        batch_size = 10
        input_dim = 4
        x_0 = torch.randn(batch_size, input_dim)
        x_tau = torch.randn(batch_size, input_dim)
        return [x_0, x_tau]

    def test_score_batch(self, estimator_config, sample_batch):
        """Test score_batch method."""
        estimator = HedgeVAMPNetEstimator(**estimator_config)
        
        # Calculate score
        score = estimator.score_batch(sample_batch)
        
        # Check output type
        assert isinstance(score, torch.Tensor)
        assert score.shape == ()  # Scalar tensor

    def test_compute_layer_losses(self, estimator_config, sample_batch):
        """Test _compute_layer_losses method."""
        estimator = HedgeVAMPNetEstimator(**estimator_config)
        
        # Get layer predictions
        layer_predictions = estimator.model(sample_batch)
        
        # Compute layer losses
        layer_losses = estimator._compute_layer_losses(layer_predictions)
        
        # Check output structure
        assert isinstance(layer_losses, list)
        assert len(layer_losses) == estimator_config["n_hidden_layers"]
        
        # Check that each loss is a tensor
        for loss in layer_losses:
            assert isinstance(loss, torch.Tensor)
            assert loss.shape == ()  # Scalar tensor

    def test_train_batch(self, estimator_config, sample_batch):
        """Test train_batch method."""
        estimator = HedgeVAMPNetEstimator(**estimator_config)
        
        # Store initial parameters
        initial_params = {}
        for name, param in estimator.model.named_parameters():
            if param.requires_grad:
                initial_params[name] = param.data.clone()
        
        initial_layer_weights = estimator.model.layer_weights.data.clone()
        
        # Train on batch
        estimator.train_batch(sample_batch)
        
        # Check that step was incremented
        assert estimator.step == 1
        
        # Check that parameters changed
        params_changed = False
        for name, param in estimator.model.named_parameters():
            if param.requires_grad and name in initial_params:
                if not torch.equal(param.data, initial_params[name]):
                    params_changed = True
                    break
        
        assert params_changed, "Parameters should change after training"
        
        # Check that layer weights changed
        assert not torch.equal(estimator.model.layer_weights.data, initial_layer_weights)
        
        # Check that layer weights still sum to 1
        assert abs(torch.sum(estimator.model.layer_weights).item() - 1.0) < 1e-6

    def test_get_layer_weights(self, estimator_config):
        """Test get_layer_weights method."""
        estimator = HedgeVAMPNetEstimator(**estimator_config)
        
        # Get layer weights
        weights = estimator.get_layer_weights()
        
        # Check output type and shape
        assert isinstance(weights, np.ndarray)
        assert weights.shape == (estimator_config["n_hidden_layers"],)
        
        # Check that weights match the model's layer_weights
        assert np.array_equal(weights, estimator.model.get_layer_weights())
