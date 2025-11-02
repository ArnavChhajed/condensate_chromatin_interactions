import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

def load_positions(folder):
    """Load particle positions from HDF5 output."""
    files = [f for f in os.listdir(folder) if f.endswith(".h5")]
    assert len(files) > 0, "No HDF5 found!"
    path = os.path.join(folder, files[0])
    with h5py.File(path, "r") as f:
        return np.array(f["positions"]), np.array(f["types"])

def mask_chromatin(positions, types):
    """Remove condensates (-1 type)."""
    mask = types != -1
    return positions[:, mask, :], types[mask]

def block_COM_analysis(positions, types, block_size=10, pattern="ABC"):
    """
    Divide genome into blocks of size 10 with repeating A,B,C labels.
    Compute center of mass (COM) per block and distances between blocks.
    """
    n_frames, n_particles, _ = positions.shape
    n_blocks = n_particles // block_size
    labels = [pattern[i % len(pattern)] for i in range(n_blocks)]

    avg_distances = {("A","A"):[],("A","B"):[],("A","C"):[],("B","B"):[],("B","C"):[],("C","C"):[]}

    for frame in range(n_frames):
        frame_pos = positions[frame]
        coms = []
        for i in range(n_blocks):
            block = frame_pos[i*block_size:(i+1)*block_size]
            coms.append(block.mean(axis=0))
        coms = np.array(coms)

        # pairwise distances
        for i in range(n_blocks):
            for j in range(i+1, n_blocks):
                d = np.linalg.norm(coms[i]-coms[j])
                pair = (labels[i], labels[j])
                pair = tuple(sorted(pair))  # ("A","B"), not ("B","A")
                if pair in avg_distances:
                    avg_distances[pair].append(d)

    # Average across frames
    for k in avg_distances:
        avg_distances[k] = np.mean(avg_distances[k])

    return avg_distances

def plot_distances(avg_distances, out_file):
    """Plot average COM distances."""
    keys = list(avg_distances.keys())
    vals = [avg_distances[k] for k in keys]
    plt.figure(figsize=(6,4))
    plt.bar(keys, vals)
    plt.ylabel("Average Distance")
    plt.title("Block COM Distances")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()

if __name__ == "__main__":
    base = "loop_outputs"
    out_dir = "analysis_results"
    os.makedirs(out_dir, exist_ok=True)

    for cond in sorted(os.listdir(base)):
        folder = os.path.join(base, cond)
        if not os.path.isdir(folder):
            continue

        positions, types = load_positions(folder)
        positions, types = mask_chromatin(positions, types)
        avg_distances = block_COM_analysis(positions, types)

        # Save plot + table
        plot_distances(avg_distances, os.path.join(out_dir, f"{cond}_block_distances.png"))
        with open(os.path.join(out_dir, f"{cond}_block_distances.txt"), "w") as f:
            for k,v in avg_distances.items():
                f.write(f"{k}: {v:.3f}\n")
