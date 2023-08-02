"""
This module provides access to datasets used in testing and examples.
"""
from typing import List
from pathlib import Path

import mdtraj
import mdshare
import numpy
import numpy as np

from celerity.utils import get_logger

logger = get_logger(__name__)


def alanine_dipeptide_coordinates(n_trajectories: int = 1, n_frames: int = 10) -> List[mdtraj.Trajectory]:
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
    output_dir = Path(__file__) / 'data'
    files = mdshare.fetch('alanine-dipeptide*.xtc', working_directory=str(output_dir))
    pdb = mdshare.fetch('alanine-dipeptide*pdb', working_directory=str(output_dir))

    logger.info(f'Fetching {n_trajectories} alanine dipeptide trajectories with {n_frames}')
    trajs = []
    for i in range(n_trajectories):
        try:
            traj = mdtraj.load(files[i], top=pdb)
            trajs.append(traj[:n_frames])
        except (KeyError, FileNotFoundError, IndexError):
            logger.exception(f"Couldn't load alanine-dipeptide trajectory", exc_info=True)
            return trajs
    return trajs


def alanine_dipeptide_backbone_angles(n_trajectories: int = 1, n_frames: int = 10, cossin: bool = True) -> List[numpy.ndarray]:
    """
    Fetches numpy arrays of the phi/psi backbone dihedral angles of alanine dipeptide from `mdshare`.
    This function thus requires an internet connection when
    first called. Creates a `data/` subdirectory to store the trajectories, so subsequent calls will not download data.

    Parameters
    ----------
    n_trajectories: 1 - 3.  The number of trajectories to return.
    n_frames: 1 - 250k. The number of frames in each trajectory.
    cossin: whether to project the angles into sin / cos space.

    Returns
    -------
    A list of length n_trajectories, with each element a numpy array of arr.shape[0] n_frames

    """
    output_dir = Path(__file__) / 'data'
    file = mdshare.fetch('alanine-dipeptide-3x250ns-backbone-dihedrals.npz', working_directory=str(output_dir))

    logger.info(f'Fetching {n_trajectories} alanine dipeptide feature arrays with {n_frames}')
    trajs = []
    for i in range(n_trajectories):
        try:
            traj = numpy.load(file)[f'arr_{i}']
            if cossin:
                traj = np.concatenate((np.sin(traj), np.cos(traj)), axis=1)
            trajs.append(traj[:n_frames, ...])
        except (KeyError, FileNotFoundError, IndexError):
            logger.exception(f"Couldn't load alanine-dipeptide feature array", exc_info=True)
            return trajs
    return trajs
