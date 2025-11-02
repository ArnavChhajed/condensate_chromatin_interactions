import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob

# Path where your .h5 files are
input_dir = r"C:/Users/arnav/Personal - Arnav Chhajed/Northwestern/genome_organization/examples/arnav/test_output_condensate"

# Expand all matching block files
h5_files = sorted(glob.glob(os.path.join(input_dir, "blocks_*.h5")))

print(f"Found {len(h5_files)} files")

# Parameters
block_size = 10
block_labels = ["A", "B", "C"] * 4   # 12 blocks, if you ran with 120 beads

# Storage for average block distances
all_distances = []

for h5_path in h5_files:
    with h5py.File(h5_path, "r") as f:
        # Pick a random frame key (e.g. "0", "10", "20"...)
        first_frame = list(f.keys())[0]
        pos = f[first_frame]["pos"][:]  # (N,3)

    n_beads = pos.shape[0]
    n_blocks = n_beads // block_size

    # Compute block COMs
    coms = []
    for i in range(n_blocks):
        start, end = i*block_size, (i+1)*block_size
        coms.append(np.mean(pos[start:end], axis=0))
    coms = np.array(coms)

    # Distance matrix between blocks
    dist_matrix = np.linalg.norm(coms[:,None,:] - coms[None,:,:], axis=-1)
    all_distances.append(dist_matrix)

# Average across all files
avg_distances = np.mean(all_distances, axis=0)

# Plot
plt.figure(figsize=(6,5))
plt.imshow(avg_distances, cmap="viridis")
plt.colorbar(label="Avg Distance")
plt.title("Average Block-to-Block Distances (with condensates)")
plt.xticks(range(n_blocks), [block_labels[i % len(block_labels)] for i in range(n_blocks)])
plt.yticks(range(n_blocks), [block_labels[i % len(block_labels)] for i in range(n_blocks)])
plt.tight_layout()
plt.show()
