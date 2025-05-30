"""
Script to visualize the recorded alpha variables and predictions by layer
from the HedgeVAmpNetEstimator training.
"""

import pickle as pk
import numpy as np
import matplotlib.pyplot as plt


def load_and_visualize():
    """Load and visualize the recorded training data"""
    
    # Load the recorded data
    try:
        alphas = pk.load(open('hedge_alphas.pkl', 'rb'))
        predictions_by_layer = pk.load(open('hedge_predictions_by_layer.pkl', 'rb'))
        test_times, test_scores, train_scores = pk.load(open('hedge_training_scores.pkl', 'rb'))
        
        print("Successfully loaded recorded data!")
        print(f"Alpha recordings: {len(alphas)} snapshots")
        print(f"Predictions by layer: {len(predictions_by_layer)} layers")
        print(f"Training scores: {len(train_scores)} recordings")
        
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please run example_hedge_vampnet_recording.py first to generate the data files.")
        return
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Training and validation scores
    ax = axes[0]
    record_interval = 10
    steps = np.arange(len(test_scores)) * record_interval
    
    ax.plot(steps, train_scores, label='Training', linewidth=1)
    ax.plot(steps, test_scores, label='Validation', linewidth=1)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('VAMP Score')
    ax.legend()
    ax.set_title('Training Progress')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Alpha evolution (expert weights)
    ax = axes[1]
    alphas_array = np.array(alphas)
    
    for i in range(alphas_array.shape[1]):
        ax.plot(steps, alphas_array[:, i], label=f'Expert {i+1}', linewidth=2)
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Expert Weight (Alpha)')
    ax.set_title('Expert Weight Evolution')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hedge_vampnet_training_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print some statistics
    print("\n" + "="*50)
    print("TRAINING STATISTICS")
    print("="*50)
    
    print(f"Final alpha values: {alphas[-1]}")
    print(f"Initial alpha values: {alphas[0]}")
    print(f"Alpha change: {alphas[-1] - alphas[0]}")
    
    print(f"\nFinal training score: {train_scores[-1]:.4f}")
    print(f"Final validation score: {test_scores[-1]:.4f}")
    print(f"Best validation score: {max(test_scores):.4f}")
    
    # Analyze predictions by layer structure
    print(f"\nPREDICTIONS BY LAYER STRUCTURE")
    print("="*50)
    for layer_num in predictions_by_layer:
        layer_data = predictions_by_layer[layer_num]
        print(f"Layer {layer_num}:")
        print(f"  - Number of validation recordings: {len(layer_data)}")
        if len(layer_data) > 0:
            print(f"  - Number of data tensors per recording: {len(layer_data[0])}")
            if len(layer_data[0]) > 0:
                print(f"  - Shape of each prediction: {layer_data[0][0].shape}")


def demonstrate_prediction_access():
    """Demonstrate how to access specific predictions from the recorded data"""
    
    try:
        predictions_by_layer = pk.load(open('hedge_predictions_by_layer.pkl', 'rb'))
    except FileNotFoundError:
        print("Please run example_hedge_vampnet_recording.py first to generate the data files.")
        return
    
    print("\n" + "="*50)
    print("ACCESSING PREDICTIONS BY LAYER")
    print("="*50)
    
    # Example: Get predictions from layer 0 at the final validation step
    final_step = len(predictions_by_layer[0]) - 1
    layer_0_final_predictions = predictions_by_layer[0][final_step]
    
    print(f"Final validation step: {final_step}")
    print(f"Layer 0 predictions at final step:")
    print(f"  - Number of data tensors: {len(layer_0_final_predictions)}")
    
    for i, pred in enumerate(layer_0_final_predictions):
        print(f"  - Data tensor {i} shape: {pred.shape}")
        print(f"  - Data tensor {i} first 5 predictions: {pred[:5]}")
        
    # Example: Compare predictions between layers at the same time step
    print(f"\nComparing layers at final validation step:")
    for layer_num in predictions_by_layer:
        layer_predictions = predictions_by_layer[layer_num][final_step]
        # Take first data tensor for comparison
        first_tensor_pred = layer_predictions[0]
        print(f"Layer {layer_num} - first tensor mean prediction: {np.mean(first_tensor_pred, axis=0)}")


if __name__ == "__main__":
    load_and_visualize()
    demonstrate_prediction_access()
