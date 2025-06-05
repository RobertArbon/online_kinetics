"""
Unit tests for layers in the celerity package.

These tests focus on the functionality of the Lobe class, ensuring that
it initializes correctly and performs forward passes as expected.
"""

import pytest
import torch
import torch.nn as nn

from celerity.layers import Lobe


class TestLobe:
    """Test suite for Lobe functionality."""

    @pytest.fixture
    def lobe_config(self):
        """Base configuration for Lobe."""
        return {
            "input_dim": 4,
            "output_dim": 2,
            "n_hidden_layers": 2,
            "hidden_layer_width": 8,
            "output_softmax": False,
        }

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        torch.manual_seed(42)  # For reproducibility
        batch_size = 10
        input_dim = 4
        return torch.randn(batch_size, input_dim)

    def test_lobe_initialization(self, lobe_config):
        """Test that Lobe initializes correctly."""
        lobe = Lobe(**lobe_config)
        
        # Check lobe structure
        assert isinstance(lobe.hidden_layers, nn.ModuleList)
        assert isinstance(lobe.output_layers, nn.ModuleList)
        assert len(lobe.hidden_layers) == lobe_config["n_hidden_layers"]
        assert len(lobe.output_layers) == lobe_config["n_hidden_layers"]
        
        # Check layer dimensions
        for i, hidden_layer in enumerate(lobe.hidden_layers):
            if i == 0:
                assert hidden_layer.in_features == lobe_config["input_dim"]
            else:
                assert hidden_layer.in_features == lobe_config["hidden_layer_width"]
            assert hidden_layer.out_features == lobe_config["hidden_layer_width"]
        
        for output_layer in lobe.output_layers:
            assert output_layer.in_features == lobe_config["hidden_layer_width"]
            assert output_layer.out_features == lobe_config["output_dim"]
        
        # Check output_softmax flag
        assert lobe.output_softmax == lobe_config["output_softmax"]

    def test_forward_pass(self, lobe_config, sample_data):
        """Test forward pass through the lobe."""
        lobe = Lobe(**lobe_config)
        
        # Perform forward pass
        outputs = lobe(sample_data)
        
        # Check output structure
        assert isinstance(outputs, list)
        assert len(outputs) == lobe_config["n_hidden_layers"]
        
        # Check output shapes
        for output in outputs:
            assert output.shape == (sample_data.shape[0], lobe_config["output_dim"])

    def test_output_softmax(self, lobe_config, sample_data):
        """Test lobe with output_softmax=True."""
        lobe_config["output_softmax"] = True
        lobe = Lobe(**lobe_config)
        
        # Perform forward pass
        outputs = lobe(sample_data)
        
        # Check that outputs sum to 1 along dim 1 (softmax property)
        for output in outputs:
            assert torch.allclose(torch.sum(output, dim=1), torch.ones(output.shape[0]))

    def test_single_layer(self, lobe_config, sample_data):
        """Test lobe with a single hidden layer."""
        lobe_config["n_hidden_layers"] = 1
        lobe = Lobe(**lobe_config)
        
        # Check lobe structure
        assert len(lobe.hidden_layers) == 1
        assert len(lobe.output_layers) == 1
        
        # Perform forward pass
        outputs = lobe(sample_data)
        
        # Check output structure
        assert isinstance(outputs, list)
        assert len(outputs) == 1
        
        # Check output shape
        assert outputs[0].shape == (sample_data.shape[0], lobe_config["output_dim"])

    def test_multiple_layers(self, lobe_config, sample_data):
        """Test lobe with multiple hidden layers."""
        lobe_config["n_hidden_layers"] = 3
        lobe = Lobe(**lobe_config)
        
        # Check lobe structure
        assert len(lobe.hidden_layers) == 3
        assert len(lobe.output_layers) == 3
        
        # Perform forward pass
        outputs = lobe(sample_data)
        
        # Check output structure
        assert isinstance(outputs, list)
        assert len(outputs) == 3
        
        # Check output shapes
        for output in outputs:
            assert output.shape == (sample_data.shape[0], lobe_config["output_dim"])

    def test_activation_function(self, lobe_config, sample_data):
        """Test that ELU activation is applied in the forward pass."""
        lobe = Lobe(**lobe_config)
        
        # Create a lobe with a large negative bias to force negative values
        # This will help verify that ELU is being applied
        for hidden_layer in lobe.hidden_layers:
            hidden_layer.bias.data.fill_(-10.0)  # Large negative bias
        
        # Perform forward pass
        outputs = lobe(sample_data)
        
        # Check that outputs are not all negative
        # ELU allows negative values but transforms them, so we can't check for positivity
        # Instead, we'll check that the outputs are different from a linear layer without activation
        
        # Create a linear layer with the same weights but no activation
        linear_layer = nn.Linear(lobe_config["input_dim"], lobe_config["output_dim"])
        linear_layer.weight.data = lobe.output_layers[0].weight.data @ lobe.hidden_layers[0].weight.data
        linear_layer.bias.data = lobe.output_layers[0].bias.data + lobe.output_layers[0].weight.data @ lobe.hidden_layers[0].bias.data
        
        # Apply linear layer
        linear_output = linear_layer(sample_data)
        
        # Check that outputs are different (indicating activation was applied)
        assert not torch.allclose(outputs[0], linear_output)
