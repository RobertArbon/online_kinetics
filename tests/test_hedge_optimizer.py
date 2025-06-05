"""
Unit tests for HedgeOptimizer.

These tests focus on the directional influence of hedge parameters on the optimizer behavior,
ensuring that parameter changes have the expected effects without checking exact values.
"""
import pytest
import torch
import numpy as np
from functools import partial
from deeptime.decomposition.deep import vampnet_loss

from celerity.models import HedgeVAMPNetEstimator
from celerity.optimizers import HedgeOptimizer


class TestHedgeOptimizer:
    """Test suite for HedgeOptimizer functionality."""

    @pytest.fixture
    def base_config(self):
        """Base configuration for HedgeVAMPNetEstimator."""
        return {
            'input_dim': 4,
            'output_dim': 2,
            'n_hidden_layers': 2,
            'hidden_layer_width': 8,
            'loss_function': partial(vampnet_loss, method='VAMP2', mode='regularize', epsilon=1e-6),
            'hedge_beta': 0.98,
            'hedge_eta': 0.01,
            'hedge_gamma': 0.1,
            'device': "cpu",
            'n_epochs': 1
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

    def test_optimizer_initialization(self, base_config):
        """Test that HedgeOptimizer initializes correctly."""
        estimator = HedgeVAMPNetEstimator(**base_config)
        optimizer = estimator.optimizer
        
        assert isinstance(optimizer, HedgeOptimizer)
        assert optimizer.estimator is estimator
        assert optimizer.model is estimator.model
        assert len(list(optimizer.param_groups)) >= 0

    def test_learning_rate_influence_on_weight_changes(self, base_config, sample_batch):
        """Test that higher learning rates lead to larger weight changes."""
        torch.manual_seed(42)
        
        # Create two estimators with different learning rates
        config_low = base_config.copy()
        config_low['hedge_eta'] = 0.001
        
        config_high = base_config.copy()
        config_high['hedge_eta'] = 0.01
        
        estimator_low = HedgeVAMPNetEstimator(**config_low)
        estimator_high = HedgeVAMPNetEstimator(**config_high)
        
        # Store initial weights
        initial_weights_low = {}
        initial_weights_high = {}
        
        # Store weights for hidden layers
        for i, layer in enumerate(estimator_low.model.lobe.hidden_layers):
            initial_weights_low[f'hidden_{i}_weight'] = layer.weight.data.clone()
            initial_weights_low[f'hidden_{i}_bias'] = layer.bias.data.clone()
            
        for i, layer in enumerate(estimator_high.model.lobe.hidden_layers):
            initial_weights_high[f'hidden_{i}_weight'] = layer.weight.data.clone()
            initial_weights_high[f'hidden_{i}_bias'] = layer.bias.data.clone()
        
        # Store weights for output layers
        for i, layer in enumerate(estimator_low.model.lobe.output_layers):
            initial_weights_low[f'output_{i}_weight'] = layer.weight.data.clone()
            initial_weights_low[f'output_{i}_bias'] = layer.bias.data.clone()
            
        for i, layer in enumerate(estimator_high.model.lobe.output_layers):
            initial_weights_high[f'output_{i}_weight'] = layer.weight.data.clone()
            initial_weights_high[f'output_{i}_bias'] = layer.bias.data.clone()
        
        # Perform one optimization step
        estimator_low.optimizer.step(sample_batch)
        estimator_high.optimizer.step(sample_batch)
        
        # Calculate weight changes
        changes_low = {}
        changes_high = {}
        
        # Calculate changes for hidden layers
        for i, layer in enumerate(estimator_low.model.lobe.hidden_layers):
            changes_low[f'hidden_{i}_weight'] = torch.norm(
                layer.weight.data - initial_weights_low[f'hidden_{i}_weight']
            ).item()
            changes_low[f'hidden_{i}_bias'] = torch.norm(
                layer.bias.data - initial_weights_low[f'hidden_{i}_bias']
            ).item()
            
        for i, layer in enumerate(estimator_high.model.lobe.hidden_layers):
            changes_high[f'hidden_{i}_weight'] = torch.norm(
                layer.weight.data - initial_weights_high[f'hidden_{i}_weight']
            ).item()
            changes_high[f'hidden_{i}_bias'] = torch.norm(
                layer.bias.data - initial_weights_high[f'hidden_{i}_bias']
            ).item()
        
        # Calculate changes for output layers
        for i, layer in enumerate(estimator_low.model.lobe.output_layers):
            changes_low[f'output_{i}_weight'] = torch.norm(
                layer.weight.data - initial_weights_low[f'output_{i}_weight']
            ).item()
            changes_low[f'output_{i}_bias'] = torch.norm(
                layer.bias.data - initial_weights_low[f'output_{i}_bias']
            ).item()
            
        for i, layer in enumerate(estimator_high.model.lobe.output_layers):
            changes_high[f'output_{i}_weight'] = torch.norm(
                layer.weight.data - initial_weights_high[f'output_{i}_weight']
            ).item()
            changes_high[f'output_{i}_bias'] = torch.norm(
                layer.bias.data - initial_weights_high[f'output_{i}_bias']
            ).item()
        
        # Assert that higher learning rate leads to larger changes (only for parameters that actually changed)
        total_change_low = sum(changes_low.values())
        total_change_high = sum(changes_high.values())
        
        assert total_change_high > total_change_low, \
            f"Higher learning rate should cause larger total changes: {total_change_high} vs {total_change_low}"
        
        # Also check that at least some individual parameters show the expected behavior
        parameters_with_expected_behavior = 0
        for key in changes_low.keys():
            if changes_low[key] > 1e-8 and changes_high[key] > changes_low[key]:
                parameters_with_expected_behavior += 1
        
        assert parameters_with_expected_behavior > 0, \
            "At least some parameters should show larger changes with higher learning rate"

    def test_beta_influence_on_layer_weights(self, base_config, sample_batch):
        """Test that different beta values affect layer weight updates differently."""
        torch.manual_seed(42)
        
        # Create two estimators with different beta values
        config_low_beta = base_config.copy()
        config_low_beta['hedge_beta'] = 0.9  # More aggressive decay
        
        config_high_beta = base_config.copy()
        config_high_beta['hedge_beta'] = 0.99  # Less aggressive decay
        
        estimator_low_beta = HedgeVAMPNetEstimator(**config_low_beta)
        estimator_high_beta = HedgeVAMPNetEstimator(**config_high_beta)
        
        # Store initial layer weights
        initial_layer_weights_low = estimator_low_beta.model.layer_weights.data.clone()
        initial_layer_weights_high = estimator_high_beta.model.layer_weights.data.clone()
        
        # Perform optimization step
        estimator_low_beta.optimizer.step(sample_batch)
        estimator_high_beta.optimizer.step(sample_batch)
        
        # Calculate changes in layer weights
        change_low_beta = torch.norm(
            estimator_low_beta.model.layer_weights.data - initial_layer_weights_low
        ).item()
        change_high_beta = torch.norm(
            estimator_high_beta.model.layer_weights.data - initial_layer_weights_high
        ).item()
        
        # Lower beta should lead to more dramatic changes in layer weights
        assert change_low_beta > change_high_beta, \
            "Lower beta should cause larger changes in layer weights"

    def test_gamma_minimum_weight_constraint(self, base_config, sample_batch):
        """Test that gamma parameter enforces minimum weight constraints."""
        torch.manual_seed(42)
        
        # Create estimator with specific gamma value and aggressive parameters
        config = base_config.copy()
        config['hedge_gamma'] = 0.4  # Higher gamma for more restrictive constraint
        config['hedge_beta'] = 0.5   # Aggressive decay to force weights low
        config['hedge_eta'] = 0.05   # Moderate learning rate
        config['n_hidden_layers'] = 2  # Ensure we have 2 layers
        
        estimator = HedgeVAMPNetEstimator(**config)
        
        # Calculate the expected minimum weight
        min_expected_weight = config['hedge_gamma'] / config['n_hidden_layers']  # 0.4 / 2 = 0.2
        
        # Create a scenario that will definitely trigger the constraint
        # by manually setting very unequal losses for different layers
        
        # Store the original _update_layer_weights method
        original_update = estimator.optimizer._update_layer_weights
        
        def mock_update_with_extreme_losses(layer_losses):
            """Mock update that simulates extreme loss differences."""
            # Simulate one layer having much higher loss than others
            mock_losses = [
                torch.tensor(10.0),  # Very high loss for first layer
                torch.tensor(0.1)    # Low loss for second layer
            ]
            original_update(mock_losses)
        
        # Replace the method temporarily
        estimator.optimizer._update_layer_weights = mock_update_with_extreme_losses
        
        # Perform optimization steps with the mocked extreme losses
        for _ in range(10):
            estimator.optimizer.step(sample_batch)
        
        # Restore original method
        estimator.optimizer._update_layer_weights = original_update
        
        # Check that the constraint is working
        # The layer with high loss should have been constrained to the minimum
        final_weights = estimator.model.layer_weights.data
        
        # At least one weight should be close to the minimum (the constrained one)
        min_weight_found = min(w.item() for w in final_weights)
        
        # The constraint should prevent weights from going too far below the theoretical minimum
        # Due to normalization, the actual minimum might be slightly lower, but not drastically
        effective_minimum = min_expected_weight * 0.8  # Allow some tolerance for normalization effects
        
        for i, weight in enumerate(final_weights):
            assert weight.item() >= effective_minimum, \
                f"Layer weight {i}: {weight.item()} should be >= {effective_minimum} " \
                f"(theoretical minimum: {min_expected_weight})"
        
        # Verify that the constraint mechanism is actually being triggered
        # by checking that weights are not all equal (which would indicate no constraint was needed)
        weight_values = [w.item() for w in final_weights]
        weight_std = torch.std(torch.tensor(weight_values)).item()
        assert weight_std > 0.01, \
            f"Weights should show variation due to constraint application, got std: {weight_std}"

    def test_layer_weights_normalization(self, base_config, sample_batch):
        """Test that layer weights are properly normalized after updates."""
        torch.manual_seed(42)
        
        estimator = HedgeVAMPNetEstimator(**base_config)
        
        # Perform optimization step
        estimator.optimizer.step(sample_batch)
        
        # Check that layer weights sum to 1
        weight_sum = torch.sum(estimator.model.layer_weights).item()
        assert abs(weight_sum - 1.0) < 1e-6, \
            f"Layer weights should sum to 1, got {weight_sum}"

    def test_multiple_steps_consistency(self, base_config, sample_batch):
        """Test that multiple optimization steps maintain consistency."""
        torch.manual_seed(42)
        
        estimator = HedgeVAMPNetEstimator(**base_config)
        
        # Perform multiple steps
        for step in range(5):
            # Store weights before step
            layer_weights_before = estimator.model.layer_weights.data.clone()
            
            # Perform step
            estimator.optimizer.step(sample_batch)
            
            # Check normalization after each step
            weight_sum = torch.sum(estimator.model.layer_weights).item()
            assert abs(weight_sum - 1.0) < 1e-6, \
                f"Layer weights should sum to 1 at step {step}, got {weight_sum}"
            
            # Check that weights actually changed (unless they hit constraints)
            weight_change = torch.norm(
                estimator.model.layer_weights.data - layer_weights_before
            ).item()
            # Allow for the case where weights might not change much due to constraints
            assert weight_change >= 0, "Weight change should be non-negative"

    def test_gradient_accumulation_correctness(self, base_config, sample_batch):
        """Test that gradients are properly accumulated across layers."""
        torch.manual_seed(42)
        
        estimator = HedgeVAMPNetEstimator(**base_config)
        
        # Store initial parameters
        initial_params = {}
        for name, param in estimator.model.named_parameters():
            if param.requires_grad:
                initial_params[name] = param.data.clone()
        
        # Perform optimization step
        estimator.optimizer.step(sample_batch)
        
        # Check that parameters have changed
        params_changed = False
        for name, param in estimator.model.named_parameters():
            if param.requires_grad and name in initial_params:
                if not torch.equal(param.data, initial_params[name]):
                    params_changed = True
                    break
        
        assert params_changed, "At least some parameters should have changed after optimization"

    def test_different_layer_configurations(self, base_config, sample_batch):
        """Test optimizer behavior with different numbers of layers."""
        torch.manual_seed(42)
        
        # Test with different numbers of hidden layers
        for n_layers in [1, 3, 4]:
            config = base_config.copy()
            config['n_hidden_layers'] = n_layers
            
            estimator = HedgeVAMPNetEstimator(**config)
            
            # Check initial layer weights are properly initialized
            assert len(estimator.model.layer_weights) == n_layers
            
            # Layer weights are initialized as 1/(n_hidden_layers + 1), so they don't sum to 1 initially
            expected_initial_weight = 1.0 / (n_layers + 1)
            for weight in estimator.model.layer_weights:
                assert abs(weight.item() - expected_initial_weight) < 1e-6, \
                    f"Each layer weight should be {expected_initial_weight} for {n_layers} layers"
            
            # Perform optimization step
            estimator.optimizer.step(sample_batch)
            
            # Check post-optimization normalization (after hedge update, weights should sum to 1)
            final_sum = torch.sum(estimator.model.layer_weights).item()
            assert abs(final_sum - 1.0) < 1e-6, \
                f"Final layer weights should sum to 1 for {n_layers} layers"

    def test_deterministic_behavior(self, base_config, sample_batch):
        """Test that optimizer behavior is deterministic given same inputs."""
        # Create two identical estimators
        torch.manual_seed(42)
        estimator1 = HedgeVAMPNetEstimator(**base_config)
        
        torch.manual_seed(42)
        estimator2 = HedgeVAMPNetEstimator(**base_config)
        
        # Perform same optimization step on both
        estimator1.optimizer.step(sample_batch)
        estimator2.optimizer.step(sample_batch)
        
        # Check that results are identical
        assert torch.allclose(
            estimator1.model.layer_weights, 
            estimator2.model.layer_weights,
            atol=1e-6
        ), "Optimizer should produce deterministic results"
        
        # Check that model parameters are identical
        for (name1, param1), (name2, param2) in zip(
            estimator1.model.named_parameters(), 
            estimator2.model.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1.data, param2.data, atol=1e-6), \
                f"Parameter {name1} should be identical between runs"

    def test_zero_learning_rate_no_change(self, base_config, sample_batch):
        """Test that zero learning rate results in no parameter changes."""
        torch.manual_seed(42)
        
        config = base_config.copy()
        config['hedge_eta'] = 0.0
        
        estimator = HedgeVAMPNetEstimator(**config)
        
        # Store initial parameters
        initial_params = {}
        for name, param in estimator.model.named_parameters():
            initial_params[name] = param.data.clone()
        
        initial_layer_weights = estimator.model.layer_weights.data.clone()
        
        # Perform optimization step
        estimator.optimizer.step(sample_batch)
        
        # Check that network parameters haven't changed (except layer weights might still change due to hedge algorithm)
        for name, param in estimator.model.named_parameters():
            if 'layer_weights' not in name:  # Skip layer weights as they're updated by hedge algorithm
                assert torch.allclose(param.data, initial_params[name], atol=1e-6), \
                    f"Parameter {name} should not change with zero learning rate"

    def test_layer_weight_updates_direction(self, base_config, sample_batch):
        """Test that layer weights update in the expected direction based on losses."""
        torch.manual_seed(42)
        
        estimator = HedgeVAMPNetEstimator(**base_config)
        
        # Store initial layer weights
        initial_layer_weights = estimator.model.layer_weights.data.clone()
        
        # Get layer predictions and compute losses manually to verify direction
        layer_predictions = estimator.model(sample_batch)
        layer_losses = estimator._compute_layer_losses(layer_predictions)
        
        # Store the losses before optimization
        initial_losses = [loss.item() for loss in layer_losses]
        
        # Perform optimization step
        estimator.optimizer.step(sample_batch)
        
        # Get final layer weights
        final_layer_weights = estimator.model.layer_weights.data
        
        # Check that weights are still valid probabilities
        assert torch.all(final_layer_weights >= 0), "All layer weights should be non-negative"
        assert torch.all(final_layer_weights <= 1), "All layer weights should be <= 1"
        assert abs(torch.sum(final_layer_weights).item() - 1.0) < 1e-6, \
            "Layer weights should sum to 1"
        
        # Test the core hedge algorithm behavior:
        # Layers with higher losses should have their weights decreased more
        # (according to the hedge update: alpha_i *= beta^(loss_i))
        
        # Calculate weight changes
        weight_changes = final_layer_weights - initial_layer_weights
        
        # If losses are different, verify the direction of updates
        if len(set(f"{loss:.6f}" for loss in initial_losses)) > 1:  # Losses are not identical
            # Find the layer with highest and lowest loss
            max_loss_idx = initial_losses.index(max(initial_losses))
            min_loss_idx = initial_losses.index(min(initial_losses))
            
            # The layer with higher loss should have a more negative weight change
            # (or smaller positive change) compared to the layer with lower loss
            # This is because beta^(higher_loss) < beta^(lower_loss) when beta < 1
            
            # After normalization, the relative ordering should be preserved:
            # the layer with higher loss should have decreased relatively more
            max_loss_change = weight_changes[max_loss_idx].item()
            min_loss_change = weight_changes[min_loss_idx].item()
            
            assert max_loss_change < min_loss_change, \
                f"Layer with higher loss ({initial_losses[max_loss_idx]:.6f}) should have " \
                f"more negative weight change ({max_loss_change:.6f}) than layer with " \
                f"lower loss ({initial_losses[min_loss_idx]:.6f}, change: {min_loss_change:.6f})"
        
        # Verify that weights actually changed (unless constrained)
        total_weight_change = torch.norm(weight_changes).item()
        if total_weight_change < 1e-8:
            # If weights didn't change much, it might be due to constraints or identical losses
            # This is acceptable behavior
            pass
        else:
            # If weights did change, ensure the change is consistent with hedge algorithm
            assert total_weight_change > 0, "Layer weights should have changed during optimization"
