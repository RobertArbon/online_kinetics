# HedgeOptimizer Unit Tests

This document describes the comprehensive unit test suite for the `HedgeOptimizer` class in `test_hedge_optimizer.py`.

## Overview

The test suite focuses on verifying the **directional influence** of hedge parameters on optimizer behavior, ensuring that parameter changes have the expected effects without checking exact numerical values. This approach makes the tests robust and deterministic while still validating the core functionality.

## Test Categories

### 1. Initialization Tests
- **`test_optimizer_initialization`**: Verifies that the HedgeOptimizer initializes correctly with proper references to the estimator and model.

### 2. Learning Rate Effects
- **`test_learning_rate_influence_on_weight_changes`**: Tests that higher learning rates (`hedge_eta`) lead to larger weight changes in both hidden and output layers.
- **`test_zero_learning_rate_no_change`**: Verifies that zero learning rate prevents network parameter updates (while layer weights may still change due to the hedge algorithm).

### 3. Hedge Algorithm Parameters
- **`test_beta_influence_on_layer_weights`**: Tests that different beta values affect layer weight updates differently (lower beta = more aggressive decay).
- **`test_gamma_minimum_weight_constraint`**: Verifies that the gamma parameter enforces minimum weight constraints on layer weights.

### 4. Layer Weight Management
- **`test_layer_weights_normalization`**: Ensures layer weights are properly normalized to sum to 1 after updates.
- **`test_layer_weight_updates_direction`**: Validates that layer weights remain valid probabilities and sum to 1.

### 5. Consistency and Robustness
- **`test_multiple_steps_consistency`**: Tests that multiple optimization steps maintain consistency in weight normalization.
- **`test_deterministic_behavior`**: Verifies that the optimizer produces identical results given the same inputs and random seeds.
- **`test_gradient_accumulation_correctness`**: Ensures gradients are properly accumulated across layers and parameters actually change.

### 6. Architecture Flexibility
- **`test_different_layer_configurations`**: Tests optimizer behavior with different numbers of hidden layers (1, 3, 4).

## Key Design Principles

### Directional Testing
Instead of checking exact numerical values, tests verify:
- Higher learning rates → larger parameter changes
- Lower beta values → more dramatic layer weight changes
- Gamma parameter → enforces minimum weight constraints
- Multiple steps → maintain normalization properties

### Deterministic Behavior
All tests use fixed random seeds (`torch.manual_seed(42)`) to ensure reproducible results.

### Robustness
Tests handle edge cases like:
- Zero learning rates
- Parameters that might not change due to zero gradients
- Different network architectures
- Multiple optimization steps

## Test Configuration

The tests use a standardized configuration:
- Input dimension: 4
- Output dimension: 2
- Hidden layers: 2 (configurable in specific tests)
- Hidden layer width: 8
- Batch size: 10
- Loss function: VAMP2 with regularization

## Running the Tests

```bash
# Run only HedgeOptimizer tests
python -m pytest tests/test_hedge_optimizer.py -v

# Run all tests
python -m pytest tests/ -v
```

## Expected Behavior

All tests should pass, demonstrating that:
1. The HedgeOptimizer correctly implements the hedge algorithm
2. Parameter updates follow expected directional patterns
3. Layer weights are properly managed and normalized
4. The optimizer behaves deterministically and consistently
5. The implementation works across different network architectures

## Integration with Existing Tests

These unit tests complement the existing integration tests in `test_integration.py`, which test the full training pipeline with callbacks and data loaders. Together, they provide comprehensive coverage of the HedgeOptimizer functionality.
