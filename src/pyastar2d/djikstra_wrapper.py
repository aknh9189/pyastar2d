import ctypes
import numpy as np
import pyastar2d.djikstra
from enum import IntEnum
from typing import Optional, Tuple


# Define array types
ndmat_f_type = np.ctypeslib.ndpointer(
    dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
ndmat_f2_type = np.ctypeslib.ndpointer(
    dtype=np.float32, ndim=2, flags="C_CONTIGUOUS")

# Define input/output types
pyastar2d.djikstra.restype = ndmat_f2_type  # Nx2 (i, j) coordinates or None
pyastar2d.djikstra.argtypes = [
    ndmat_f_type,   # weights
    ctypes.c_int,   # height
    ctypes.c_int,   # width
    ctypes.c_int,   # start index in flattened grid
    ctypes.c_int,   # goal index in flattened grid
    ctypes.c_bool,  # allow diagonal
    ctypes.c_int,   # heuristic_override
]

class Heuristic(IntEnum):
    """The supported heuristics."""

    DEFAULT = 0
    ORTHOGONAL_X = 1
    ORTHOGONAL_Y = 2

def djikstra_fill(
        weights: np.ndarray,
        start: Tuple[int, int],
        fill_radius: int,
        allow_diagonal: bool = False,
        ) -> Optional[np.ndarray]:
    """
    Run djikstra algorithm on 2d weights.

    param np.ndarray weights: A grid of weights e.g. np.ones((10, 10), dtype=np.float32)
    param Tuple[int, int] start: (i, j)
    param fill_radius: The radius to fill
    param bool allow_diagonal: Whether to allow diagonal moves
    param Heuristic heuristic_override: Override heuristic, see Heuristic(IntEnum)

    """
    assert weights.dtype == np.float32, (
        f"weights must have np.float32 data type, but has {weights.dtype}"
    )
    # Make sure all cost values are greater than zero
    if weights.min(axis=None) < 0.:
        raise ValueError("Minimum cost to move must be 1, but got %f" % (
            weights.min(axis=None)))
    # Ensure start is within bounds.
    if (start[0] < 0 or start[0] >= weights.shape[0] or
            start[1] < 0 or start[1] >= weights.shape[1]):
        raise ValueError(f"Start of {start} lies outside grid.")
    # Ensure goal is within bounds.

    height, width = weights.shape
    start_idx = np.ravel_multi_index(start, (height, width))

    djikstra_map = pyastar2d.djikstra.djikstra(
        weights.flatten(), height, width, start_idx, fill_radius, allow_diagonal
    )
    # center the start point in the map. This only changes things if we are on a boundary. This makes the map dimensions symmetric
    full_map = np.full((2 * fill_radius + 1, 2 * fill_radius + 1), np.nan, dtype=np.float32)
    #NANs are used to indicate the point is outside of the map
    full_map[fill_radius - start[0]:fill_radius - start[0] + height, fill_radius - start[1]:fill_radius - start[1] + width] = djikstra_map
    return full_map
