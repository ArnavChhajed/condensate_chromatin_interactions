import numpy as np
monomer_types = np.load("C:/Users/arnav/Personal - Arnav Chhajed/Northwestern/genome_organization/examples/arnav/test_output/monomer_types.npy")
print("Shape:", monomer_types.shape)
print("Unique values:", np.unique(monomer_types))