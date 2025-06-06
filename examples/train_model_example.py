"""
Example script demonstrating how to use the celerity training API.
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from celerity.api import train_model
from celerity.callbacks import AlphaRecorder, PredictionsByLayerRecorder


def create_example_data(output_path, n_samples=1000, n_features=10):
    """Create example data for training."""
    # Create a temporary directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Create two sets of random data
    data1 = np.random.random((n_samples, n_features)).astype(np.float32)
    data2 = np.random.random((n_samples, n_features)).astype(np.float32)
    
    # Save as .npz file
    np.savez(os.path.join(output_path, "example_data.npz"), data1=data1, data2=data2)
    
    return os.path.join(output_path, "example_data.npz")


def train_standard_model(data_path):
    """Train a standard VAMPNet model."""
    print("Training standard VAMPNet model...")
    
    # Define model configuration
    model_config = {
        'input_dim': 10,
        'output_dim': 5,
        'n_hidden_layers': 2,
        'hidden_layer_width': 50,
        'output_softmax': True,
        'device': 'cpu'  # Change to 'cuda' if available
    }
    
    # Define training configuration
    training_config = {
        'n_epochs': 5,
        'batch_size': 100,
        'lagtime': 1,
        'learning_rate': 0.001,
        'optimizer_name': 'Adam',
        'score_method': 'VAMP2',
        'record_interval': 10
    }
    
    # Train the model
    estimator = train_model(
        data_path=data_path,
        model_config=model_config,
        training_config=training_config,
        use_hedging=False,
        validation_split=0.3
    )
    
    print("Standard model training complete.")
    print(f"Final training score: {max(estimator.training_scores['train']['VAMP2'].values()):.4f}")
    
    return estimator


def train_hedge_model(data_path):
    """Train a hedge VAMPNet model."""
    print("Training hedge VAMPNet model...")
    
    # Define model configuration
    model_config = {
        'input_dim': 10,
        'output_dim': 5,
        'n_hidden_layers': 5,  # More layers for hedging
        'hidden_layer_width': 50,
        'output_softmax': True,
        'device': 'cpu'  # Change to 'cuda' if available
    }
    
    # Define training configuration
    training_config = {
        'n_epochs': 5,
        'batch_size': 100,
        'lagtime': 1,
        'hedge_beta': 0.99,
        'hedge_eta': 0.01,
        'hedge_gamma': 0.1,
        'score_method': 'VAMP2',
        'record_interval': 10
    }
    
    # Train the model
    estimator = train_model(
        data_path=data_path,
        model_config=model_config,
        training_config=training_config,
        use_hedging=True,
        validation_split=0.3
    )
    
    print("Hedge model training complete.")
    print(f"Final training score: {max(estimator.training_scores['train']['VAMP2'].values()):.4f}")
    
    # Get layer weights
    layer_weights = estimator.get_layer_weights()
    print("Final layer weights:")
    for i, weight in enumerate(layer_weights):
        print(f"  Layer {i+1}: {weight:.4f}")
    
    # Plot layer weights evolution
    alpha_recorder = None
    for callback in estimator.training_scores.get('callbacks', []):
        if isinstance(callback, AlphaRecorder):
            alpha_recorder = callback
            break
    
    if alpha_recorder:
        alphas = alpha_recorder.get_alphas()
        plt.figure(figsize=(10, 6))
        for i in range(len(alphas[0])):
            plt.plot([alpha[i] for alpha in alphas], label=f'Layer {i+1}')
        plt.xlabel('Training Step')
        plt.ylabel('Layer Weight')
        plt.title('Evolution of Layer Weights During Training')
        plt.legend()
        plt.savefig('layer_weights_evolution.png')
        print("Layer weights evolution plot saved as 'layer_weights_evolution.png'")
    
    return estimator


def main():
    """Main function to run the example."""
    # Create example data
    data_dir = "example_data"
    data_path = create_example_data(data_dir)
    print(f"Example data created at: {data_path}")
    
    # Train standard model
    standard_estimator = train_standard_model(data_path)
    
    # Train hedge model
    hedge_estimator = train_hedge_model(data_path)
    
    print("\nExample complete!")


if __name__ == "__main__":
    main()
