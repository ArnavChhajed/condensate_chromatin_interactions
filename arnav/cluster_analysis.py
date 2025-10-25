import numpy as np
import h5py
import glob
import os
import matplotlib.pyplot as plt

def load_positions(h5_path):
    """Load positions from a .h5 file."""
    with h5py.File(h5_path, "r") as f:
        group = list(f.keys())[0]
        pos = f[group]["pos"][:]  # shape: (N, 3)
    return pos

def compute_block_coms(positions, monomer_types, block_size=10):
    """
    Compute center of mass (COM) for each block.
    Blocks are formed as 10*A, 10*B, 10*C repeated 3 times.
    """
    block_coms = []
    block_labels = []
    N = len(monomer_types)

    for i in range(0, N, block_size):
        block_types = monomer_types[i:i+block_size]
        block_pos = positions[i:i+block_size]
        com = block_pos.mean(axis=0)
        label = block_types[0]  # assume block is pure A/B/C
        block_coms.append(com)
        block_labels.append(label)

    return np.array(block_coms), np.array(block_labels)

def compute_avg_distances(block_coms, block_labels):
    """
    Compute average distances between block COMs, grouped by type pairs.
    """
    pairs = {}
    for i in range(len(block_coms)):
        for j in range(i+1, len(block_coms)):
            d = np.linalg.norm(block_coms[i] - block_coms[j])
            pair = tuple(sorted([block_labels[i], block_labels[j]]))  # e.g., (0,1)
            if pair not in pairs:
                pairs[pair] = []
            pairs[pair].append(d)

    avg_distances = {pair: np.mean(dlist) for pair, dlist in pairs.items()}
    return avg_distances

def plot_avg_distances(avg_distances, outpath):
    """Plot average distances between block types."""
    labels = [f"{a}-{b}" for (a, b) in avg_distances.keys()]
    values = list(avg_distances.values())

    plt.bar(labels, values, color="skyblue")
    plt.ylabel("Average Distance")
    plt.title("Block COM Distances (Averaged)")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_time_series(all_dists, outpath):
    """Plot how distances change over frames (time series)."""
    plt.figure(figsize=(8, 5))
    for pair, values in all_dists.items():
        label = f"{pair[0]}-{pair[1]}"
        plt.plot(values, label=label, marker="o", markersize=3, linewidth=1)

    plt.xlabel("Frame (simulation timestep)")
    plt.ylabel("Average Distance")
    plt.title("Block COM Distances Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# === MAIN ===
if __name__ == "__main__":
    # Load monomer types
    monomer_types = np.load(
        "C:/Users/arnav/Personal - Arnav Chhajed/Northwestern/genome_organization/examples/arnav/test_output/monomer_types.npy"
    )

    # Find all simulation outputs
    filepaths = sorted(
        glob.glob("C:/Users/arnav/Personal - Arnav Chhajed/Northwestern/genome_organization/examples/arnav/test_output/blocks_*.h5")
    )

    # Aggregate distances across frames
    all_dists = {}

    for path in filepaths:
        pos = load_positions(path)
        block_coms, block_labels = compute_block_coms(pos, monomer_types, block_size=10)
        avg_distances = compute_avg_distances(block_coms, block_labels)

        for k, v in avg_distances.items():
            if k not in all_dists:
                all_dists[k] = []
            all_dists[k].append(v)

    # Compute mean over all frames
    final_dists = {k: np.mean(vlist) for k, vlist in all_dists.items()}

    # Save plots + results
    os.makedirs("analysis_results", exist_ok=True)
    plot_avg_distances(final_dists, "analysis_results/block_distances_avg.png")
    plot_time_series(all_dists, "analysis_results/block_distances_time.png")

    with open("analysis_results/block_distances.txt", "w") as f:
        for pair, d in final_dists.items():
            f.write(f"{pair}: {d:.3f}\n")

    print("Analysis complete. Results saved in analysis_results/")
