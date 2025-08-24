"""
Drivable Corridor Detection for Autonomous Vehicles

A PyTorch implementation of CNN-based drivable corridor detection
for autonomous vehicle perception.
"""

__version__ = "0.1.0"
__author__ = "sriram-0311"
__email__ = "ramesh.anu@northeastern.edu"

from .models import CNN, BiSeNet
from .data import BDD

__all__ = ["CNN", "BiSeNet", "BDD"]
