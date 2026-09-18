"""Tests if all modules can be imported properly."""

import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

try:
    from src.algorithms import *
except ImportError as err:
    raise ImportError(f"{err.name} could not be imported. Path: {err.path}.")

print("Algorithms imported: [x]")

try:
    from src.problems import *
except ImportError as err:
    raise ImportError(f"{err.name} could not be imported. Path: {err.path}.")

print("Problems imported:   [x]")

try:
    from config import *
except ImportError as err:
    raise ImportError(f"{err.name} could not be imported. Path: {err.path}.")

print("Config imported:     [x]")