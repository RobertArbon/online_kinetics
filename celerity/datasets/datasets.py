"""
This module provides access to datasets used in testing and examples.
"""
from typing import List
from pathlib import Path
from functools import partial

import mdtraj
import mdshare
import numpy
import numpy as np

from celerity.utils import get_logger

logger = get_logger(__name__)


def download_dataset(data_dir: str, dataset: str):
    if dataset.startswith('alanine-dipeptide'):
        feature = dataset.split('-')[-1]
        f = partial(alanine_dipeptide_features, feature=feature,
                    n_trajectories=3, n_frames=-1,
                    cossin=True)
    else:
        raise ValueError(f"{dataset} is not valid")
    data_dir = Path(data_dir) / dataset
    logger.info(f"Downloading {dataset} features to {data_dir}")
    _ = f(data_dir=data_dir)


def ensure_data_dir(path: str = None) -> Path:
    if path is None:
        path = Path(__file__) / 'data'
    else:
        path = Path(path)
    return path



def alanine_dipeptide_trajectories(n_trajectories: int = 1, n_frames: int = 10, data_dir: str = None) -> List[mdtraj.Trajectory]:
    """
    Fetches mdtraj Trajectories of alanine dipeptide from `mdshare`. This function thus requires an internet connection when
    first called. Creates a `data/` subdirectory to store the trajectories, so subsequent calls will not download data.

    Parameters
    ----------
    n_trajectories: 1 - 3.  The number of trajectories to return.
    n_frames: 1 - 250k. The number of frames in each trajectory.

    Returns
    -------
    A list of length n_trajectories, with each element a mdtraj.Trajectory object of length n_frames

    """
    output_dir = ensure_data_dir(data_dir)
    files = mdshare.fetch('alanine-dipeptide*.xtc', working_directory=str(output_dir))
    pdb = mdshare.fetch('alanine-dipeptide*pdb', working_directory=str(output_dir))

    p_n_frames = n_frames if n_frames != -1 else 'all'
    logger.info(f'Fetching {n_trajectories} alanine dipeptide trajectories with '
                f'{p_n_frames} frames')
    trajs = []
    for i in range(n_trajectories):
        try:
            traj = mdtraj.load(files[i], top=pdb)
            trajs.append(traj[:n_frames])
        except (KeyError, FileNotFoundError, IndexError):
            logger.exception(f"Couldn't load alanine-dipeptide trajectory", exc_info=True)
            return trajs
    return trajs


def alanine_dipeptide_features(feature: str, n_trajectories: int = 1, n_frames: int = 10, cossin: bool = True,
                               data_dir: str = None) -> List[numpy.ndarray]:
    """
    Fetches numpy arrays of the phi/psi backbone dihedral angles of alanine dipeptide from `mdshare`.
    This function thus requires an internet connection when
    first called. Creates a `data/` subdirectory to store the trajectories, so subsequent calls will not download data.

    Parameters
    ----------
    feature: one of 'dihedrals', 'positions', 'distances' - the feature to download.
    n_trajectories: 1 - 3.  The number of trajectories to return.
    n_frames: 1 - 250k. The number of frames in each trajectory.
    cossin: whether to project the angles into sin / cos space (only applies for 'dihedrals' features).

    Returns
    -------
    A list of length n_trajectories, with each element a numpy array of arr.shape[0] n_frames

    """
    features = dict(dihedrals='alanine-dipeptide-3x250ns-backbone-dihedrals.npz',
                    positions='alanine-dipeptide-3x250ns-heavy-atom-positions.npz',
                    distances='alanine-dipeptide-3x250ns-heavy-atom-distances.npz')

    output_dir = ensure_data_dir(data_dir)
    file = mdshare.fetch(features[feature], working_directory=str(output_dir))

    p_n_frames = n_frames if n_frames != -1 else 'all'
    logger.info(f'Fetching {n_trajectories} alanine dipeptide trajectories with '
                f'{p_n_frames} frames')
    trajs = []
    for i in range(n_trajectories):
        try:
            traj = numpy.load(file)[f'arr_{i}']
            if cossin and (feature == 'dihedrals'):
                traj = np.concatenate((np.sin(traj), np.cos(traj)), axis=1)
            trajs.append(traj[:n_frames, ...])
        except (KeyError, FileNotFoundError, IndexError):
            logger.exception(f"Couldn't load alanine-dipeptide feature array", exc_info=True)
            return trajs

    if cossin and (feature == 'dihedrals'):
        np.savez(output_dir / features[feature], *trajs)
    return trajs
