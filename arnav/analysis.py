import numpy as np
import h5py
import glob
import os

def load_positions_and_types(h5_path, monomer_types):
    """Load positions and monomer types for a given .h5 file."""
    with h5py.File(h5_path, 'r') as f:
        group = list(f.keys())[0]
        pos = f[group]['pos'][:]  # shape: (N, 3)
    return pos, monomer_types
