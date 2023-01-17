import numpy as np
import pytest

import os
import sys
import pyastar2d

def test_small():
    weights = np.array([[1, 2, 4, 4, 3],
                        [1, np.inf, 3, 3, 3],
                        [1, 1, 1, 1, 3],
                        [2, 2, 2, 1, 3],
                        [2, 2, 2, 2, 1]], dtype=np.float32)
    # Run down the diagonal.
    path = pyastar2d.djikstra_fill(weights, (0, 0), 3, allow_diagonal=False)
    expected = np.array([[0, 2, 6, 10, np.nan],
                         [1, np.inf, 7, np.nan, np.nan],
                         [2, 3, 4, np.nan, np.nan],
                         [4, np.nan, np.nan, np.nan, np.nan],
                         [np.nan, np.nan, np.nan, np.nan, np.nan]], dtype=np.float32)
    print(path == expected)
    print(path)
    print(type(path[-1,-1]), path[-1,-1] == np.nan)
    non_nan = np.logical_not(np.isnan(expected))
    assert np.all(path[non_nan] == expected[non_nan])
