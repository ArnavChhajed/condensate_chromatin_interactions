import h5py
import numpy as np

with h5py.File("C:/Users/arnav/Personal - Arnav Chhajed/Northwestern/genome_organization/examples/arnav/test_output/block_0-0.h5", "r") as f:
    group_name = list(f.keys())[0]  # usually '0'
    monomer_types = f[group_name]['monomer_types'][:]

np.save("monomer_types.npy", monomer_types)