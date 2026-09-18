"""This module contains tests for the CB_Resampling class."""

# Make other project modules accessible
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

# Import CB_Resampling
from src.resampling.cb_resampling import CB_Resampling

# Package imports
import numpy as np

# Minkowski distance test
x = [1,1]
y = [3,3]

euclidean = CB_Resampling._minkowski_dist(x, y)
manhattan = CB_Resampling._minkowski_dist(x,y,p=1)

assert euclidean == np.sqrt(8), f"Minkowski euclidean distance should be {np.sqrt(8)}, was {euclidean}"
assert manhattan == 4, f"Minkowski manhattan distance should be 4, was {euclidean}"

print("Minkowski distance test [x]")