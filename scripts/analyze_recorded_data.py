"""
Script to analyze the recorded alpha variables and predictions by layer
from the HedgeVAmpNetEstimator training (without matplotlib dependency).
"""

import pickle as pk
import numpy as np


def load_and_analyze():
    """Load and analyze the recorded training data"""
    
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
    
    # Show alpha evolution over time
    print(f"\nALPHA EVOLUTION")
    print("="*50)
    alphas_array = np.array(alphas)
    print(f"Alpha array shape: {alphas_array.shape}")
    
    # Show first few and last few alpha values
    print("First 5 alpha recordings:")
    for i in range(min(5, len(alphas))):
        print(f"  Step {i*10}: {alphas[i]}")
    
    print("Last 5 alpha recordings:")
    for i in range(max(0, len(alphas)-5), len(alphas)):
        print(f"  Step {i*10}: {alphas[i]}")
    
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
                print(f"  - First prediction sample: {layer_data[0][0][:3, :3]}")  # Show 3x3 sample


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
        print(f"  - Data tensor {i} first 3 predictions: {pred[:3]}")
        
    # Example: Compare predictions between layers at the same time step
    print(f"\nComparing layers at final validation step:")
    for layer_num in predictions_by_layer:
        layer_predictions = predictions_by_layer[layer_num][final_step]
        # Take first data tensor for comparison
        first_tensor_pred = layer_predictions[0]
        print(f"Layer {layer_num} - first tensor mean prediction: {np.mean(first_tensor_pred, axis=0)}")


def show_alpha_statistics():
    """Show detailed statistics about alpha evolution"""
    
    try:
        alphas = pk.load(open('hedge_alphas.pkl', 'rb'))
    except FileNotFoundError:
        print("Please run example_hedge_vampnet_recording.py first to generate the data files.")
        return
    
    print("\n" + "="*50)
    print("DETAILED ALPHA STATISTICS")
    print("="*50)
    
    alphas_array = np.array(alphas)
    
    print(f"Number of experts (layers): {alphas_array.shape[1]}")
    print(f"Number of recordings: {alphas_array.shape[0]}")
    
    for expert_idx in range(alphas_array.shape[1]):
        expert_alphas = alphas_array[:, expert_idx]
        print(f"\nExpert {expert_idx}:")
        print(f"  Initial weight: {expert_alphas[0]:.6f}")
        print(f"  Final weight: {expert_alphas[-1]:.6f}")
        print(f"  Change: {expert_alphas[-1] - expert_alphas[0]:+.6f}")
        print(f"  Min weight: {np.min(expert_alphas):.6f}")
        print(f"  Max weight: {np.max(expert_alphas):.6f}")
        print(f"  Mean weight: {np.mean(expert_alphas):.6f}")
        print(f"  Std weight: {np.std(expert_alphas):.6f}")


if __name__ == "__main__":
    load_and_analyze()
    demonstrate_prediction_access()
    show_alpha_statistics()
