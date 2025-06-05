"""
Integration tests for model estimation.
"""
import pytest
import numpy as np
import torch
from pathlib import Path
from functools import partial
from typing import Dict, List
from torch.utils.data import DataLoader
from addict import Dict as Adict
from deeptime.util.types import to_dataset
from deeptime.decomposition.deep import vampnet_loss
from torch.utils.tensorboard import SummaryWriter

from celerity.models import HedgeVAMPNetEstimator, StandardVAMPNetEstimator
from celerity.callbacks import AlphaRecorder, PredictionsByLayerRecorder, LossLogger


@pytest.fixture()
def data():
    """Load test data."""
    with np.load(Path('tests/data/alanine-dipeptide-3x250ns-backbone-dihedrals.npz')) as fh:
        dihedral = [fh[f"arr_{i}"] for i in range(3)]
    dihedral = [x[:100] for x in dihedral]
    return dihedral


@pytest.fixture()
def loaders(data):
    """Create data loaders for training and validation."""
    lag_time = 1
    validation_split = 0.3
    batch_size = 10
    
    # Prepare data
    dataset = to_dataset(data=data, lagtime=lag_time)
    n_val = int(len(dataset) * validation_split)
    train_data, val_data = torch.utils.data.random_split(dataset, [len(dataset) - n_val, n_val])

    loader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(val_data, batch_size=len(val_data), shuffle=False)
    
    return loader_train, loader_val


@pytest.fixture()
def hedge_config(data):
    """Configuration for HedgeVAMPNetEstimator."""
    return Adict(
        input_dim=data[0].shape[1], 
        output_dim=6,
        n_hidden_layers=2,
        hidden_layer_width=100, 
        loss_function=partial(vampnet_loss, method='VAMP2', mode='regularize', epsilon=1e-6), 
        hedge_beta=0.98,
        hedge_eta=3*(5e-3),
        hedge_gamma=0.1,
        device="cpu",
        n_epochs=1
    )


@pytest.fixture()
def vamp_config(data):
    """Configuration for VAMPNetEstimator."""
    return Adict(
        input_dim=data[0].shape[1], 
        output_dim=6,
        n_hidden_layers=2,
        hidden_layer_width=100,
        learning_rate=5e-4,
        score_method='VAMP2',
        score_mode='regularize',
        score_epsilon=1e-6,
        device="cpu",
        n_epochs=1
    )


def test_hedge_vampnet_no_callbacks(hedge_config, loaders):
    """
    Test HedgeVAMPNetEstimator without callbacks.
    """
    # Initialize estimator
    est = HedgeVAMPNetEstimator(**hedge_config)
    
    # Get loaders
    loader_train, loader_val = loaders
    
    # Train for a few batches
    for i, batch in enumerate(loader_train):
        est.train_batch(batch)
        if i >= 2:  # Just train for a few batches for testing
            break
    
    # Validate
    est.validate(loader_val)
    
    # Check that we have some training scores
    assert len(est.training_scores['train']['VAMP2']) > 0
    assert len(est.training_scores['validate']['VAMP2']) > 0


def test_hedge_vampnet_with_alpha_recorder(hedge_config, loaders, data):
    """
    Test HedgeVAMPNetEstimator with AlphaRecorder callback.
    """
    # Initialize estimator
    est = HedgeVAMPNetEstimator(**hedge_config)
    
    # Initialize callback
    alpha_recorder = AlphaRecorder(est)
    
    # Get loaders
    loader_train, loader_val = loaders
    
    # Train for a few batches with callback
    for i, batch in enumerate(loader_train):
        est.train_batch(batch, callbacks=[alpha_recorder])
        if i >= 2:  # Just train for a few batches for testing
            break
    
    # Validate
    est.validate(loader_val)
    
    # Check that alphas were recorded
    alphas = alpha_recorder.get_alphas()
    assert len(alphas) > 0
    assert len(alphas[0]) == hedge_config.n_hidden_layers


def test_hedge_vampnet_with_predictions_recorder(hedge_config, loaders, data):
    """
    Test HedgeVAMPNetEstimator with PredictionsByLayerRecorder callback.
    """
    # Initialize estimator
    est = HedgeVAMPNetEstimator(**hedge_config)
    
    # Prepare data tensors for recorder
    data_tensors = [torch.Tensor(x) for x in data]
    
    # Initialize callback
    predictions_recorder = PredictionsByLayerRecorder(est, data_tensors)
    
    # Get loaders
    loader_train, loader_val = loaders
    
    # Train for a few batches
    for i, batch in enumerate(loader_train):
        est.train_batch(batch)
        if i >= 2:  # Just train for a few batches for testing
            break
    
    # Validate with callback
    est.validate(loader_val, callbacks=[predictions_recorder])
    
    # Check that predictions were recorded
    predictions = predictions_recorder.get_predictions_by_layer()
    assert len(predictions) == hedge_config.n_hidden_layers
    for layer_num in range(hedge_config.n_hidden_layers):
        assert len(predictions[layer_num]) > 0


def test_hedge_vampnet_with_all_callbacks(hedge_config, loaders, data):
    """
    Test HedgeVAMPNetEstimator with all callbacks.
    """
    # Initialize estimator
    est = HedgeVAMPNetEstimator(**hedge_config)
    
    # Prepare data tensors for recorder
    data_tensors = [torch.Tensor(x) for x in data]
    
    # Initialize callbacks
    alpha_recorder = AlphaRecorder(est)
    predictions_recorder = PredictionsByLayerRecorder(est, data_tensors)
    
    # Get loaders
    loader_train, loader_val = loaders
    
    # Train with fit method and callbacks
    est.fit(
        train_loader=loader_train,
        validate_loader=loader_val,
        record_interval=1,  # Record at every step for testing
        train_callbacks=[alpha_recorder],
        validate_callbacks=[predictions_recorder]
    )
    
    # Check that alphas were recorded
    alphas = alpha_recorder.get_alphas()
    assert len(alphas) > 0
    
    # Check that predictions were recorded
    predictions = predictions_recorder.get_predictions_by_layer()
    assert len(predictions) == hedge_config.n_hidden_layers


def test_vampnet_no_callbacks(vamp_config, loaders):
    """
    Test VAMPNetEstimator without callbacks.
    """
    # Initialize estimator
    est = StandardVAMPNetEstimator(**vamp_config)
    
    # Get loaders
    loader_train, loader_val = loaders
    
    # Train for a few batches
    for i, batch in enumerate(loader_train):
        est.train_batch(batch)
        if i >= 2:  # Just train for a few batches for testing
            break
    
    # Validate
    est.validate(loader_val)
    
    # Check that we have some training scores
    assert len(est.training_scores['train']['VAMP2']) > 0
    assert len(est.training_scores['validate']['VAMP2']) > 0


def test_vampnet_with_loss_logger(vamp_config, loaders, tmp_path):
    """
    Test VAMPNetEstimator with LossLogger callback.
    """
    # Initialize estimator
    est = StandardVAMPNetEstimator(**vamp_config)
    
    # Create a temporary tensorboard writer
    tb_log_dir = tmp_path / "tensorboard_logs"
    tb_writer = SummaryWriter(log_dir=tb_log_dir)
    
    # Initialize callbacks
    train_logger = LossLogger(tb_writer, 'train')
    val_logger = LossLogger(tb_writer, 'validate')
    
    # Get loaders
    loader_train, loader_val = loaders
    
    # Train with fit method and callbacks
    est.fit(
        train_loader=loader_train,
        validate_loader=loader_val,
        record_interval=100,  # Record at every step for testing
        train_callbacks=[train_logger],
        validate_callbacks=[val_logger]
    )
    
    # Check that tensorboard log directory was created
    assert tb_log_dir.exists()
    
    # Clean up
    tb_writer.close()
