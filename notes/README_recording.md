# Recording Alpha Variables and Predictions by Layer for HedgeVAmpNetEstimator

This document explains how to use the new callback functionality to record alpha variables during training and predictions by layer during validation for the `HedgeVAmpNetEstimator`.

## Overview

The implementation provides two new callbacks:

1. **`AlphaRecorder`**: Records the expert weights (alpha values) during training
2. **`PredictionsByLayerRecorder`**: Records predictions from each hidden layer during validation

## Quick Start

### Basic Usage

```python
from celerity.models import HedgeVAMPNetEstimator
from celerity.callbacks import AlphaRecorder, PredictionsByLayerRecorder

# Initialize your estimator
est = HedgeVAMPNetEstimator(**config)

# Initialize callbacks
alpha_recorder = AlphaRecorder(est)
predictions_recorder = PredictionsByLayerRecorder(est, data_tensors)

# Manual training loop with recording
for i, x in enumerate(loader_train):
    est.partial_fit(x)
    
    if i % record_interval == 0:
        # Record alphas during training
        alpha_recorder(i, {})
        
        # Record predictions during validation
        est.eval()
        predictions_recorder(i, {})

# Get recorded data
recorded_alphas = alpha_recorder.get_alphas()
recorded_predictions = predictions_recorder.get_predictions_by_layer()
```

### Using with fit() method

```python
# Use callbacks with the built-in fit method
est.fit(
    train_loader=loader_train,
    validate_loader=loader_val,
    record_interval=10,
    train_callbacks=[alpha_recorder],
    validate_callbacks=[predictions_recorder]
)
```

## Callbacks Documentation

### AlphaRecorder

Records the alpha (expert weight) values during training.

**Parameters:**
- `estimator`: The HedgeVAmpNetEstimator instance to record from

**Methods:**
- `__call__(step, dict)`: Records current alpha values
- `get_alphas()`: Returns list of recorded alpha arrays

**Example output:**
```python
alphas = alpha_recorder.get_alphas()
# alphas[0] = [0.500, 0.500]  # Initial weights
# alphas[1] = [0.495, 0.505]  # After some training
# alphas[n] = [0.396, 0.604]  # Final weights
```

### PredictionsByLayerRecorder

Records predictions from each hidden layer during validation.

**Parameters:**
- `estimator`: The HedgeVAmpNetEstimator instance to record from
- `data_tensors`: List of torch.Tensor objects to get predictions for

**Methods:**
- `__call__(step, dict)`: Records predictions for current validation step
- `get_predictions_by_layer()`: Returns dictionary mapping layer numbers to prediction lists

**Example output:**
```python
predictions = predictions_recorder.get_predictions_by_layer()
# predictions[0] = [step0_preds, step1_preds, ...]  # Layer 0 predictions
# predictions[1] = [step0_preds, step1_preds, ...]  # Layer 1 predictions
# Each step_preds is a list of numpy arrays (one per data tensor)
```

## Data Structure

### Alpha Recordings
- **Type**: `List[np.ndarray]`
- **Shape**: Each array has shape `(n_hidden_layers,)`
- **Content**: Expert weights at each recording step

### Predictions by Layer
- **Type**: `Dict[int, List[List[np.ndarray]]]`
- **Structure**: 
  ```
  {
    layer_num: [
      [data_tensor_0_preds, data_tensor_1_preds, ...],  # Step 0
      [data_tensor_0_preds, data_tensor_1_preds, ...],  # Step 1
      ...
    ]
  }
  ```
- **Content**: Predictions from each layer at each validation step

## Example Scripts

### 1. `example_hedge_vampnet_recording.py`
Complete example showing both manual training loop and fit() method usage.

### 2. `analyze_recorded_data.py`
Analysis script to examine the recorded data structure and statistics.

### 3. `visualize_recorded_data.py`
Visualization script (requires matplotlib) to plot training progress and alpha evolution.

## Running the Examples

```bash
# Activate the conda environment
conda activate online

# Run the recording example
python example_hedge_vampnet_recording.py

# Analyze the recorded data
python analyze_recorded_data.py

# Visualize (if matplotlib is available)
python visualize_recorded_data.py
```

## Output Files

The example scripts save the following files:

- `hedge_alphas.pkl`: Recorded alpha values
- `hedge_predictions_by_layer.pkl`: Recorded predictions by layer
- `hedge_training_scores.pkl`: Training and validation scores

## Key Features

1. **Flexible Recording**: Works with both manual training loops and the `fit()` method
2. **Efficient Storage**: Only records at specified intervals to manage memory usage
3. **Layer-wise Predictions**: Captures predictions from each hidden layer separately
4. **Expert Weight Evolution**: Tracks how the hedging algorithm adjusts expert weights over time
5. **Easy Analysis**: Provides utilities to analyze and visualize the recorded data

## Use Cases

- **Algorithm Analysis**: Understanding how expert weights evolve during training
- **Layer Comparison**: Comparing predictions from different layers
- **Convergence Studies**: Analyzing training dynamics and convergence behavior
- **Model Debugging**: Identifying issues with specific layers or experts
- **Research**: Studying the behavior of hedged learning algorithms

## Integration with Existing Code

The callbacks are designed to integrate seamlessly with existing HedgeVAmpNetEstimator code:

1. **No changes required** to the core estimator
2. **Optional callbacks** - can be used or omitted as needed
3. **Backward compatible** - existing code continues to work unchanged
4. **Minimal overhead** - callbacks only execute when explicitly called

## Performance Considerations

- Recording predictions can be memory-intensive for large datasets
- Use appropriate `record_interval` to balance detail vs. memory usage
- Consider recording only specific layers if memory is limited
- The callbacks add minimal computational overhead when used appropriately
