"""
This module provides access to datasets used in testing and examples.
"""
from typing import List

import mdtraj
import mdshare


def alanine_dipeptide_coordinates(n_trajectories: int = 1, n_frames: int = 10) -> List[mdtraj.Trajectory]:
    """
    Fetches mdtraj Trajectories of alanine dipeptide from `mdshare`. This function thus requires an internet connection when
    first called. Creates a `data/` subdirectory to store the trajectories, so subsequent calls will not download data.
    :param n_trajectories: 1 - 3.  The number of trajectories to return.
    :param n_frames: 1 - 10k. The number of frames in each trajectory.
    :return: A list of length n_trajectories, with each element a mdtraj.Trajectory object of length n_frames
    """

