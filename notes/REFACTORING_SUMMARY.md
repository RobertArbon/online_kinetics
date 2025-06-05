# VAMPNet Refactoring Summary

## Overview

This refactoring separates the 'model' parts from the 'estimator' parts in the VAMPNet implementation, creating a cleaner, more maintainable architecture. The refactoring emphasizes code clarity, consistency between implementations, and follows object-oriented design principles.

## Key Changes

### 1. Separation of Concerns

**Before**: Monolithic classes that mixed model architecture with training logic
**After**: Clear separation between:
- **Models**: Handle network architecture and forward passes
- **Estimators**: Handle training, optimization, and scoring logic

### 2. Hierarchical Class Structure

```
BaseVAMPNetModel (Abstract)
├── StandardVAMPNetModel
└── HedgeVAMPNetModel

BaseVAMPNetEstimator (Abstract)
├── StandardVAMPNetEstimator
└── HedgeVAMPNetEstimator
```

### 3. Consistent Naming and Interface

#### Variable Renaming for Clarity:
- `lr` → `learning_rate`
- `n` → `hedge_eta` (learning rate for Hedge algorithm)
- `b` → `hedge_beta` (exponential decay factor)
- `s` → `hedge_gamma` (minimum weight threshold)
- `alpha` → `layer_weights` (weights for combining layer outputs)
- `dict_scores` → `training_scores`
- `optimizer` → `optimizer_name` (in constructor parameters)

#### Method Consistency:
- Both estimators now have identical method signatures
- Consistent parameter names across implementations
- Unified callback handling and validation logic

### 4. Logical Grouping of Parameters

#### Network Architecture Parameters:
```python
input_dim, output_dim, n_hidden_layers, hidden_layer_width, output_softmax
```

#### Training Parameters:
```python
learning_rate, n_epochs, score_method, optimizer_name
```

#### Hedge-Specific Parameters:
```python
hedge_beta, hedge_eta, hedge_gamma
```

#### Device and Scoring Parameters:
```python
device, score_mode, score_epsilon
```

## Class Descriptions

### Base Classes

#### `BaseVAMPNetModel`
- Abstract base class for all VAMPNet models
- Handles common network setup and device management
- Defines interface for `forward()` and `transform()` methods
- Manages the underlying `Lobe` network architecture

#### `BaseVAMPNetEstimator`
- Abstract base class for all VAMPNet estimators
- Handles common training logic (validation, fitting, callbacks)
- Defines interface for `score_batch()` and `train_batch()` methods
- Manages training state and score tracking

### Concrete Implementations

#### `StandardVAMPNetModel`
- Implements standard VAMPNet forward pass
- Returns single output per time point (final layer only)
- Simple transform implementation using final layer

#### `StandardVAMPNetEstimator`
- Uses traditional batch training with PyTorch optimizers
- Single loss calculation per batch
- Compatible with Adam, SGD, and other standard optimizers

#### `HedgeVAMPNetModel`
- Implements multi-layer output forward pass
- Returns all layer outputs for expert combination
- Weighted transform using layer importance weights
- Manages Hedge algorithm parameters (beta, eta, gamma)

#### `HedgeVAMPNetEstimator`
- Uses online Hedge algorithm for training
- Combines losses from multiple layers using learned weights
- Integrates with custom `HedgeOptimizer`

### HedgeOptimizer Improvements

#### Enhanced Structure:
- Clear separation of optimization steps
- Descriptive method names and documentation
- Proper gradient accumulation and weight updates

#### Key Methods:
- `_update_output_layer()`: Updates individual output layer weights
- `_accumulate_hidden_gradients()`: Accumulates weighted gradients
- `_update_hidden_layers()`: Applies accumulated gradients
- `_update_layer_weights()`: Updates layer combination weights

## Benefits of Refactoring

### 1. **Code Clarity**
- Clear separation between model architecture and training logic
- Consistent naming conventions across implementations
- Well-documented methods and parameters

### 2. **Maintainability**
- Easier to modify model architecture without affecting training logic
- Shared base classes reduce code duplication
- Consistent interfaces make debugging easier

### 3. **Extensibility**
- Easy to add new model variants by extending base classes
- Training logic can be reused across different model architectures
- Clear extension points for new optimization algorithms

### 4. **Testability**
- Models can be tested independently of training logic
- Estimators can be tested with mock models
- Clear interfaces make unit testing straightforward

### 5. **Readability**
- Similar structure between Standard and Hedge implementations
- Easy to compare and contrast the two approaches
- Logical parameter grouping makes configuration clearer

## Backward Compatibility

The refactoring maintains backward compatibility through:
- Legacy class aliases (`VAMPnetEstimator = StandardVAMPNetEstimator`)
- Preserved `estimator_by_type` dictionary
- Same public API for existing code

## Usage Examples

### Standard VAMPNet
```python
estimator = StandardVAMPNetEstimator(
    input_dim=10,
    output_dim=5,
    learning_rate=1e-3,
    n_epochs=50
)
```

### Hedge VAMPNet
```python
estimator = HedgeVAMPNetEstimator(
    input_dim=10,
    output_dim=5,
    n_hidden_layers=3,
    loss_function=my_loss_fn,
    hedge_beta=0.95,
    hedge_eta=0.01
)
```

Both estimators now have identical training interfaces:
```python
estimator.fit(train_loader, validate_loader)
transformed_data = estimator.transform(input_data)
