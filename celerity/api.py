"""
Functional API for training models in celerity.
"""

from typing import Dict, Callable, Tuple, List, Union, Optional
from pathlib import Path
import os
import glob

import numpy as np
import torch
from torch.utils.data import DataLoader

from deeptime.util.types import to_dataset
from deeptime.decomposition.deep import vampnet_loss

from celerity.models import StandardVAMPNetEstimator, HedgeVAMPNetEstimator
from celerity.callbacks import AlphaRecorder, PredictionsByLayerRecorder
from celerity.utils import get_logger

logger = get_logger(__name__)


def load_data_from_path(data_path: Union[str, Path]) -> List[np.ndarray]:
    """Load data from .npy, .npz files or directory of numpy arrays.

    Parameters
    ----------
    data_path : Union[str, Path]
        Path to a .npy file, .npz file, or directory containing .npy/.npz files

    Returns
    -------
    List[np.ndarray]
        List of numpy arrays containing the loaded data

    Raises
    ------
    FileNotFoundError
        If the specified path does not exist
    ValueError
        If the path is not a valid file or directory, or if no valid files are found
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    data = []
    
    # Case 1: Single .npy file
    if data_path.is_file() and data_path.suffix == '.npy':
        array = np.load(data_path)
        if isinstance(array, np.ndarray):
            data.append(array)
        else:
            raise ValueError(f"Expected numpy array from {data_path}, got {type(array)}")
    
    # Case 2: Single .npz file
    elif data_path.is_file() and data_path.suffix == '.npz':
        with np.load(data_path) as npz_file:
            # Extract all arrays from the .npz file
            for key in npz_file.files:
                data.append(npz_file[key])
    
    # Case 3: Directory of .npy/.npz files
    elif data_path.is_dir():
        # Get all .npy files
        npy_files = sorted(glob.glob(os.path.join(data_path, "*.npy")))
        for npy_file in npy_files:
            array = np.load(npy_file)
            data.append(array)
        
        # Get all .npz files
        npz_files = sorted(glob.glob(os.path.join(data_path, "*.npz")))
        for npz_file in npz_files:
            with np.load(npz_file) as npz:
                for key in npz.files:
                    data.append(npz[key])
    
    else:
        raise ValueError(f"Invalid data path: {data_path}. Must be a .npy file, .npz file, or directory.")
    
    if not data:
        raise ValueError(f"No valid numpy arrays found at {data_path}")
    
    return data


def validate_data_format(data: List[np.ndarray]) -> bool:
    """Validate data format and dimensions.

    Parameters
    ----------
    data : List[np.ndarray]
        List of numpy arrays to validate

    Returns
    -------
    bool
        True if data is valid, False otherwise

    Raises
    ------
    ValueError
        If data format is invalid
    """
    if not isinstance(data, list):
        raise ValueError(f"Data must be a list of numpy arrays, got {type(data)}")
    
    if not all(isinstance(arr, np.ndarray) for arr in data):
        raise ValueError("All elements in data must be numpy arrays")
    
    # Check that all arrays have the same shape[1:] (feature dimensions)
    if len(data) > 1:
        first_shape = data[0].shape[1:]
        for i, arr in enumerate(data[1:], 1):
            if arr.shape[1:] != first_shape:
                raise ValueError(
                    f"All arrays must have the same feature dimensions. "
                    f"Array 0 has shape {data[0].shape}, but array {i} has shape {arr.shape}"
                )
    
    return True


def create_data_loaders(
    data: List[np.ndarray], 
    lagtime: int, 
    batch_size: int, 
    validation_split: float
) -> Tuple[DataLoader, DataLoader, List[torch.Tensor]]:
    """Create training and validation data loaders.

    Parameters
    ----------
    data : List[np.ndarray]
        List of numpy arrays containing the data
    lagtime : int
        Lag time for time-lagged data
    batch_size : int
        Batch size for training
    validation_split : float
        Fraction of data to use for validation (0.0 to 1.0)

    Returns
    -------
    Tuple[DataLoader, DataLoader, List[torch.Tensor]]
        Training data loader, validation data loader, and validation tensors
    """
    # Convert data to torch dataset
    dataset = to_dataset(data=data, lagtime=lagtime)
    
    # Split into training and validation sets
    n_val = int(len(dataset) * validation_split)
    n_train = len(dataset) - n_val
    
    if n_val <= 0:
        raise ValueError(f"Validation split {validation_split} results in empty validation set")
    
    train_data, val_data = torch.utils.data.random_split(dataset, [n_train, n_val])
    
    # Create data loaders
    loader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    # Extract validation tensors for callbacks
    val_tensors = []
    for batch in loader_val:
        val_tensors.append(batch[0])  # First tensor in the batch
    
    return loader_train, loader_val, val_tensors


def setup_default_callbacks(
    estimator,
    use_hedging: bool,
    validation_tensors: List[torch.Tensor]
) -> Tuple[List[Callable], List[Callable]]:
    """Setup default callbacks based on training type.

    Parameters
    ----------
    estimator : Union[StandardVAMPNetEstimator, HedgeVAMPNetEstimator]
        The estimator being trained
    use_hedging : bool
        Whether hedging is being used
    validation_tensors : List[torch.Tensor]
        Validation data tensors for layer output recording

    Returns
    -------
    Tuple[List[Callable], List[Callable]]
        Training callbacks and validation callbacks
    """
    train_callbacks = []
    val_callbacks = []
    
    if use_hedging:
        # Add layer weights recorder for hedge training
        alpha_recorder = AlphaRecorder(estimator)
        train_callbacks.append(alpha_recorder)
        
        # Add predictions by layer recorder for validation data
        if validation_tensors:
            layer_outputs_recorder = PredictionsByLayerRecorder(estimator, validation_tensors)
            val_callbacks.append(layer_outputs_recorder)
    
    return train_callbacks, val_callbacks


def create_estimator(
    input_dim: int,
    output_dim: int,
    use_hedging: bool,
    model_config: Dict,
    training_config: Dict
) -> Union[StandardVAMPNetEstimator, HedgeVAMPNetEstimator]:
    """Factory function to create appropriate estimator.

    Parameters
    ----------
    input_dim : int
        Input dimension of the model
    output_dim : int
        Output dimension of the model
    use_hedging : bool
        Whether to use hedging
    model_config : Dict
        Model configuration parameters
    training_config : Dict
        Training configuration parameters

    Returns
    -------
    Union[StandardVAMPNetEstimator, HedgeVAMPNetEstimator]
        The created estimator
    """
    # Extract common parameters with defaults
    n_hidden_layers = model_config.get('n_hidden_layers', 1)
    hidden_layer_width = model_config.get('hidden_layer_width', 10)
    output_softmax = model_config.get('output_softmax', False)
    device = model_config.get('device', 'cpu')
    n_epochs = training_config.get('n_epochs', 1)
    score_method = training_config.get('score_method', 'VAMP2')
    
    if use_hedging:
        # Extract hedge-specific parameters with defaults
        hedge_beta = training_config.get('hedge_beta', 0.99)
        hedge_eta = training_config.get('hedge_eta', 0.01)
        hedge_gamma = training_config.get('hedge_gamma', 0.1)
        
        # Create loss function
        score_mode = training_config.get('score_mode', 'regularize')
        score_epsilon = training_config.get('score_epsilon', 1e-6)
        loss_function = lambda x_0, x_tau: vampnet_loss(
            x_0, x_tau, 
            method=score_method, 
            mode=score_mode, 
            epsilon=score_epsilon
        )
        
        # Create hedge estimator
        estimator = HedgeVAMPNetEstimator(
            input_dim=input_dim,
            output_dim=output_dim,
            n_hidden_layers=n_hidden_layers,
            hidden_layer_width=hidden_layer_width,
            loss_function=loss_function,
            output_softmax=output_softmax,
            device=device,
            hedge_beta=hedge_beta,
            hedge_eta=hedge_eta,
            hedge_gamma=hedge_gamma,
            score_method=score_method,
            n_epochs=n_epochs
        )
    else:
        # Extract standard-specific parameters with defaults
        learning_rate = training_config.get('learning_rate', 5e-4)
        optimizer_name = training_config.get('optimizer_name', 'Adam')
        score_mode = training_config.get('score_mode', 'regularize')
        score_epsilon = training_config.get('score_epsilon', 1e-6)
        
        # Create standard estimator
        estimator = StandardVAMPNetEstimator(
            input_dim=input_dim,
            output_dim=output_dim,
            n_hidden_layers=n_hidden_layers,
            hidden_layer_width=hidden_layer_width,
            output_softmax=output_softmax,
            device=device,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            optimizer_name=optimizer_name,
            score_method=score_method,
            score_mode=score_mode,
            score_epsilon=score_epsilon
        )
    
    return estimator


def train_model(
    data_path: Union[str, Path],
    model_config: Dict,
    training_config: Dict,
    use_hedging: bool = False,
    validation_split: float = 0.3,
    callbacks: Optional[Dict[str, List[Callable]]] = None
) -> Union[StandardVAMPNetEstimator, HedgeVAMPNetEstimator]:
    """Train a model using the specified configuration.

    Parameters
    ----------
    data_path : Union[str, Path]
        Path to data file (.npy, .npz) or directory of numpy arrays
    model_config : Dict
        Model configuration parameters:
        - input_dim (int): Input dimension
        - output_dim (int): Output dimension
        - n_hidden_layers (int, optional): Number of hidden layers. Default: 1
        - hidden_layer_width (int, optional): Width of hidden layers. Default: 10
        - output_softmax (bool, optional): Whether to apply softmax to output. Default: False
        - device (str, optional): Device to use. Default: 'cpu'
    training_config : Dict
        Training configuration parameters:
        - n_epochs (int, optional): Number of epochs. Default: 1
        - batch_size (int, optional): Batch size. Default: 100
        - lagtime (int, optional): Lag time. Default: 1
        - record_interval (int, optional): Interval for recording. Default: None
        - score_method (str, optional): Scoring method. Default: 'VAMP2'
        - score_mode (str, optional): Scoring mode. Default: 'regularize'
        - score_epsilon (float, optional): Epsilon for scoring. Default: 1e-6
        
        For standard training:
        - learning_rate (float, optional): Learning rate. Default: 5e-4
        - optimizer_name (str, optional): Optimizer name. Default: 'Adam'
        
        For hedge training:
        - hedge_beta (float, optional): Beta parameter. Default: 0.99
        - hedge_eta (float, optional): Eta parameter. Default: 0.01
        - hedge_gamma (float, optional): Gamma parameter. Default: 0.1
    use_hedging : bool, optional
        Whether to use hedging. Default: False
    validation_split : float, optional
        Fraction of data to use for validation. Default: 0.3
    callbacks : Optional[Dict[str, List[Callable]]], optional
        Custom callbacks to use. Keys should be 'train' and 'validate'.
        If None, default callbacks will be used. Default: None

    Returns
    -------
    Union[StandardVAMPNetEstimator, HedgeVAMPNetEstimator]
        The trained estimator
    """
    # Validate required parameters
    if 'input_dim' not in model_config:
        raise ValueError("model_config must contain 'input_dim'")
    if 'output_dim' not in model_config:
        raise ValueError("model_config must contain 'output_dim'")
    
    # Load and validate data
    logger.info(f"Loading data from {data_path}")
    data = load_data_from_path(data_path)
    validate_data_format(data)
    
    # Extract training parameters with defaults
    batch_size = training_config.get('batch_size', 100)
    lagtime = training_config.get('lagtime', 1)
    record_interval = training_config.get('record_interval', None)
    
    # Create data loaders
    logger.info(f"Creating data loaders with validation split {validation_split}")
    loader_train, loader_val, val_tensors = create_data_loaders(
        data, lagtime, batch_size, validation_split
    )
    
    # Create estimator
    logger.info(f"Creating {'hedge' if use_hedging else 'standard'} estimator")
    estimator = create_estimator(
        input_dim=model_config['input_dim'],
        output_dim=model_config['output_dim'],
        use_hedging=use_hedging,
        model_config=model_config,
        training_config=training_config
    )
    
    # Setup callbacks
    if callbacks is None:
        logger.info("Setting up default callbacks")
        train_callbacks, val_callbacks = setup_default_callbacks(
            estimator, use_hedging, val_tensors
        )
    else:
        logger.info("Using provided callbacks")
        train_callbacks = callbacks.get('train', [])
        val_callbacks = callbacks.get('validate', [])
    
    # Train the model
    logger.info("Starting model training")
    estimator.fit(
        train_loader=loader_train,
        validate_loader=loader_val,
        record_interval=record_interval,
        train_callbacks=train_callbacks,
        validate_callbacks=val_callbacks
    )
    
    logger.info("Model training complete")
    return estimator
