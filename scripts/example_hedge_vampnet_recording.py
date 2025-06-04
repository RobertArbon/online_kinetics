"""
Example script demonstrating how to record alpha variables during training
and predictions_by_layer during validation for HedgeVAmpNetEstimator.

This script is based on the hedgevampnets.ipynb notebook example.
"""

from pathlib import Path
from functools import partial
import pickle as pk
import numpy as np
import torch
from torch.utils.data import DataLoader
from addict import Dict as Adict
from deeptime.util.types import to_dataset
from deeptime.decomposition.deep import vampnet_loss

from celerity.models import HedgeVAMPNetEstimator
from celerity.callbacks import AlphaRecorder, PredictionsByLayerRecorder


def main():
    # Load data (using test data for this example)
    with np.load(Path('tests/data/alanine-dipeptide-3x250ns-backbone-dihedrals.npz')) as fh:
        dihedral = [fh[f"arr_{i}"] for i in range(3)]
    data = dihedral

    # Configuration
    lag_time = 1
    validation_split = 0.3
    batch_size = 1000
    record_interval = 10

    # VAMPNet estimator config
    nn_config = Adict(
        input_dim=data[0].shape[1], 
        output_dim=6,
        n_hidden_layers=2,
        hidden_layer_width=100, 
        loss_function=partial(vampnet_loss, method='VAMP2', mode='regularize', epsilon=1e-6), 
        hedge_beta=0.98,  # beta, expert update rate
        hedge_eta=3*(5e-3),  # eta, learning rate
        hedge_gamma=0.1,  # minimum expert weight
        device="cuda" if torch.cuda.is_available() else "cpu",
        n_epochs=1  # Set to 1 for online learning
    )

    # Prepare data
    dataset = to_dataset(data=data, lagtime=lag_time)
    n_val = int(len(dataset) * validation_split)
    train_data, val_data = torch.utils.data.random_split(dataset, [len(dataset) - n_val, n_val])

    loader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(val_data, batch_size=len(val_data), shuffle=False)
    data_tensors = [torch.Tensor(x) for x in data]

    # Initialize estimator
    est = HedgeVAMPNetEstimator(**nn_config)

    # Initialize callbacks
    alpha_recorder = AlphaRecorder(est)
    predictions_recorder = PredictionsByLayerRecorder(est, data_tensors)

    # Manual training loop with recording (similar to notebook example)
    test_scores = []
    train_scores = []
    test_times = []

    print("Starting training with recording...")
    
    for i, x in enumerate(loader_train):
        est.train()
        est.train_batch(x)

        if i % record_interval == 0:
            print(f"Step {i}/{len(loader_train)} ({i/len(loader_train)*100:.1f}%)")
            
            # Record alphas during training
            alpha_recorder(i, {})
            
            # Record training score
            with torch.no_grad():
                train_loss = est.score_batch(x)
                train_scores.append(-train_loss.item())
            
            # Record validation score and predictions by layer
            est.eval()
            tmp = []
            for val in loader_val:
                with torch.no_grad():
                    val_loss = est.score_batch(val)
                    tmp.append(-val_loss.item())
            test_scores.append(np.mean(tmp))
            test_times.append((i+1) * batch_size)
            
            # Record predictions by layer during validation
            predictions_recorder(i, {})

    print("Training completed!")

    # Get recorded data
    recorded_alphas = alpha_recorder.get_alphas()
    recorded_predictions_by_layer = predictions_recorder.get_predictions_by_layer()

    print(f"Recorded {len(recorded_alphas)} alpha snapshots")
    print(f"Recorded predictions for {len(recorded_predictions_by_layer)} layers")
    
    # Save results (similar to notebook)
    pk.dump(file=open('hedge_alphas.pkl', 'wb'), obj=recorded_alphas)
    pk.dump(file=open('hedge_predictions_by_layer.pkl', 'wb'), obj=recorded_predictions_by_layer)
    pk.dump(file=open('hedge_training_scores.pkl', 'wb'), obj=(test_times, test_scores, train_scores))

    print("Results saved to:")
    print("- hedge_alphas.pkl")
    print("- hedge_predictions_by_layer.pkl") 
    print("- hedge_training_scores.pkl")

    # Print some statistics
    print(f"\nFinal alpha values: {recorded_alphas[-1]}")
    print(f"Alpha evolution shape: {np.array(recorded_alphas).shape}")
    for layer_num in recorded_predictions_by_layer:
        print(f"Layer {layer_num} predictions shape: {len(recorded_predictions_by_layer[layer_num])} recordings")


def example_with_fit_method():
    """
    Example showing how to use the callbacks with the fit() method
    """
    # Load data
    with np.load(Path('tests/data/alanine-dipeptide-3x250ns-backbone-dihedrals.npz')) as fh:
        dihedral = [fh[f"arr_{i}"] for i in range(3)]
    data = dihedral

    # Configuration
    lag_time = 1
    validation_split = 0.3
    batch_size = 1000

    nn_config = Adict(
        input_dim=data[0].shape[1], 
        output_dim=6,
        n_hidden_layers=2,
        hidden_layer_width=100, 
        loss_function=partial(vampnet_loss, method='VAMP2', mode='regularize', epsilon=1e-6), 
        hedge_beta=0.98,
        hedge_eta=3*(5e-3),
        hedge_gamma=0.1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        n_epochs=1  # Multiple epochs for this example
    )

    # Prepare data
    dataset = to_dataset(data=data, lagtime=lag_time)
    n_val = int(len(dataset) * validation_split)
    train_data, val_data = torch.utils.data.random_split(dataset, [len(dataset) - n_val, n_val])

    loader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(val_data, batch_size=len(val_data), shuffle=False)
    data_tensors = [torch.Tensor(x) for x in data]

    # Initialize estimator
    est = HedgeVAMPNetEstimator(**nn_config)

    # Initialize callbacks
    alpha_recorder = AlphaRecorder(est)
    predictions_recorder = PredictionsByLayerRecorder(est, data_tensors)

    print("Training with fit() method and callbacks...")
    
    # Use the fit method with callbacks
    est.fit(
        train_loader=loader_train,
        validate_loader=loader_val,
        record_interval=10,
        train_callbacks=[alpha_recorder],
        validate_callbacks=[predictions_recorder]
    )

    print("Training completed!")

    # Get recorded data
    recorded_alphas = alpha_recorder.get_alphas()
    recorded_predictions_by_layer = predictions_recorder.get_predictions_by_layer()

    print(f"Recorded {len(recorded_alphas)} alpha snapshots")
    print(f"Recorded predictions for {len(recorded_predictions_by_layer)} layers")


if __name__ == "__main__":
    print("Running manual training loop example...")
    main()
    
    # print("\n" + "="*50 + "\n")
    
    # print("Running fit() method example...")
    # example_with_fit_method()
