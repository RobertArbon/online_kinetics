"""
Tests for the celerity API.
"""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from celerity.api import (
    load_data_from_path,
    validate_data_format,
    create_data_loaders,
    setup_default_callbacks,
    create_estimator,
    train_model
)
from celerity.models import StandardVAMPNetEstimator, HedgeVAMPNetEstimator


class TestDataLoading(unittest.TestCase):
    """Test data loading functions."""
    
    def setUp(self):
        """Set up test data."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Create test data
        self.data1 = np.random.random((100, 10)).astype(np.float32)
        self.data2 = np.random.random((150, 10)).astype(np.float32)
        
        # Save as .npy files
        np.save(self.temp_path / "data1.npy", self.data1)
        np.save(self.temp_path / "data2.npy", self.data2)
        
        # Save as .npz file
        np.savez(self.temp_path / "combined.npz", data1=self.data1, data2=self.data2)
    
    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
    
    def test_load_npy_file(self):
        """Test loading a single .npy file."""
        data = load_data_from_path(self.temp_path / "data1.npy")
        self.assertEqual(len(data), 1)
        np.testing.assert_array_equal(data[0], self.data1)
    
    def test_load_npz_file(self):
        """Test loading a .npz file."""
        data = load_data_from_path(self.temp_path / "combined.npz")
        self.assertEqual(len(data), 2)
        # Order might vary, so check that both arrays are present
        self.assertTrue(any(np.array_equal(arr, self.data1) for arr in data))
        self.assertTrue(any(np.array_equal(arr, self.data2) for arr in data))
    
    def test_load_directory(self):
        """Test loading a directory of .npy files."""
        data = load_data_from_path(self.temp_path)
        # Should load 3 arrays (2 from .npy files, 2 from .npz file, but we count them as 1)
        self.assertEqual(len(data), 4)
    
    def test_validate_data_format(self):
        """Test data format validation."""
        # Valid data
        data = [self.data1, self.data2]
        self.assertTrue(validate_data_format(data))
        
        # Invalid data - different feature dimensions
        invalid_data = [self.data1, np.random.random((100, 5))]
        with self.assertRaises(ValueError):
            validate_data_format(invalid_data)


class TestDataLoaders(unittest.TestCase):
    """Test data loader creation."""
    
    def setUp(self):
        """Set up test data."""
        self.data = [
            np.random.random((100, 10)).astype(np.float32),
            np.random.random((150, 10)).astype(np.float32)
        ]
    
    def test_create_data_loaders(self):
        """Test creating data loaders."""
        loader_train, loader_val, val_tensors = create_data_loaders(
            self.data, lagtime=1, batch_size=10, validation_split=0.2
        )
        
        # Check that loaders have correct sizes
        total_samples = 100 + 150 - 2  # -2 for lagtime=1
        val_samples = int(total_samples * 0.2)
        train_samples = total_samples - val_samples
        
        # Calculate expected number of batches
        expected_train_batches = (train_samples + 10 - 1) // 10  # Ceiling division
        expected_val_batches = (val_samples + 10 - 1) // 10  # Ceiling division
        
        self.assertEqual(len(loader_train), expected_train_batches)
        self.assertEqual(len(loader_val), expected_val_batches)
        
        # Check that validation tensors are created
        self.assertGreater(len(val_tensors), 0)


class TestEstimatorCreation(unittest.TestCase):
    """Test estimator creation."""
    
    def test_create_standard_estimator(self):
        """Test creating a standard estimator."""
        model_config = {
            'n_hidden_layers': 2,
            'hidden_layer_width': 20,
            'output_softmax': True,
            'device': 'cpu'
        }
        training_config = {
            'n_epochs': 2,
            'learning_rate': 0.001,
            'optimizer_name': 'Adam',
            'score_method': 'VAMP2'
        }
        
        estimator = create_estimator(
            input_dim=10,
            output_dim=5,
            use_hedging=False,
            model_config=model_config,
            training_config=training_config
        )
        
        self.assertIsInstance(estimator, StandardVAMPNetEstimator)
        self.assertEqual(estimator.model.input_dim, 10)
        self.assertEqual(estimator.model.output_dim, 5)
        self.assertEqual(estimator.model.n_hidden_layers, 2)
        self.assertEqual(estimator.model.hidden_layer_width, 20)
        self.assertEqual(estimator.model.output_softmax, True)
        self.assertEqual(estimator.n_epochs, 2)
        self.assertEqual(estimator.score_method, 'VAMP2')
    
    def test_create_hedge_estimator(self):
        """Test creating a hedge estimator."""
        model_config = {
            'n_hidden_layers': 3,
            'hidden_layer_width': 30,
            'output_softmax': False,
            'device': 'cpu'
        }
        training_config = {
            'n_epochs': 3,
            'hedge_beta': 0.95,
            'hedge_eta': 0.02,
            'hedge_gamma': 0.2,
            'score_method': 'VAMP2'
        }
        
        estimator = create_estimator(
            input_dim=15,
            output_dim=8,
            use_hedging=True,
            model_config=model_config,
            training_config=training_config
        )
        
        self.assertIsInstance(estimator, HedgeVAMPNetEstimator)
        self.assertEqual(estimator.model.input_dim, 15)
        self.assertEqual(estimator.model.output_dim, 8)
        self.assertEqual(estimator.model.n_hidden_layers, 3)
        self.assertEqual(estimator.model.hidden_layer_width, 30)
        self.assertEqual(estimator.model.output_softmax, False)
        self.assertEqual(estimator.n_epochs, 3)
        self.assertEqual(estimator.score_method, 'VAMP2')


class TestCallbackSetup(unittest.TestCase):
    """Test callback setup."""
    
    def setUp(self):
        """Set up test estimators."""
        # Create a standard estimator
        self.standard_estimator = StandardVAMPNetEstimator(
            input_dim=10,
            output_dim=5,
            n_hidden_layers=2,
            hidden_layer_width=20
        )
        
        # Create a hedge estimator
        self.hedge_estimator = HedgeVAMPNetEstimator(
            input_dim=10,
            output_dim=5,
            n_hidden_layers=2,
            hidden_layer_width=20,
            loss_function=lambda x, y: torch.mean(x - y)
        )
        
        # Create validation tensors
        self.val_tensors = [torch.randn(10, 10)]
    
    def test_standard_callbacks(self):
        """Test setting up callbacks for standard training."""
        train_callbacks, val_callbacks = setup_default_callbacks(
            self.standard_estimator, use_hedging=False, validation_tensors=self.val_tensors
        )
        
        # Standard training should have no default callbacks
        self.assertEqual(len(train_callbacks), 0)
        self.assertEqual(len(val_callbacks), 0)
    
    def test_hedge_callbacks(self):
        """Test setting up callbacks for hedge training."""
        train_callbacks, val_callbacks = setup_default_callbacks(
            self.hedge_estimator, use_hedging=True, validation_tensors=self.val_tensors
        )
        
        # Hedge training should have AlphaRecorder and PredictionsByLayerRecorder
        self.assertEqual(len(train_callbacks), 1)
        self.assertEqual(len(val_callbacks), 1)
        
        from celerity.callbacks import AlphaRecorder, PredictionsByLayerRecorder
        self.assertIsInstance(train_callbacks[0], AlphaRecorder)
        self.assertIsInstance(val_callbacks[0], PredictionsByLayerRecorder)


# Integration test for the full training API
class TestTrainModel(unittest.TestCase):
    """Test the train_model function."""
    
    def setUp(self):
        """Set up test data."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Create test data
        self.data1 = np.random.random((50, 10)).astype(np.float32)
        self.data2 = np.random.random((50, 10)).astype(np.float32)
        
        # Save as .npz file
        np.savez(self.temp_path / "test_data.npz", data1=self.data1, data2=self.data2)
    
    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
    
    def test_train_standard_model(self):
        """Test training a standard model."""
        model_config = {
            'input_dim': 10,
            'output_dim': 5,
            'n_hidden_layers': 1,
            'hidden_layer_width': 10
        }
        training_config = {
            'n_epochs': 1,
            'batch_size': 10,
            'lagtime': 1
        }
        
        estimator = train_model(
            data_path=self.temp_path / "test_data.npz",
            model_config=model_config,
            training_config=training_config,
            use_hedging=False,
            validation_split=0.2
        )
        
        self.assertIsInstance(estimator, StandardVAMPNetEstimator)
        # Check that training has occurred (step > 0)
        self.assertGreater(estimator.step, 0)
    
    def test_train_hedge_model(self):
        """Test training a hedge model."""
        model_config = {
            'input_dim': 10,
            'output_dim': 5,
            'n_hidden_layers': 2,
            'hidden_layer_width': 10
        }
        training_config = {
            'n_epochs': 1,
            'batch_size': 10,
            'lagtime': 1
        }
        
        estimator = train_model(
            data_path=self.temp_path / "test_data.npz",
            model_config=model_config,
            training_config=training_config,
            use_hedging=True,
            validation_split=0.2
        )
        
        self.assertIsInstance(estimator, HedgeVAMPNetEstimator)
        # Check that training has occurred (step > 0)
        self.assertGreater(estimator.step, 0)


if __name__ == '__main__':
    unittest.main()
