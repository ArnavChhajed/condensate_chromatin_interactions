import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from collections import defaultdict

# === Fixed block definitions (mentor specified) ===
def define_blocks():
    return {
        "A": [[0,10], [30,40], [60,70], [90,100]],
        "B": [[10,20], [40,50], [70,80], [100,110]],
        "C": [[20,30], [50,60], [80,90], [110,120]],
    }

def compute_com(positions, block_ranges):
    """Compute COM for each block given position array [N,3]."""
    return np.array([np.mean(positions[start:end], axis=0) for start, end in block_ranges])

def load_positions_from_blocks(folder):
    """Load all positions from blocks_*.h5 files in folder."""
    all_pos = []
    files = sorted([f for f in os.listdir(folder) if f.startswith("blocks_") and f.endswith(".h5")])
    if not files:
        raise FileNotFoundError(f"No block files found in {folder}")
    for fname in files:
        with h5py.File(os.path.join(folder, fname), "r") as h5:
            # Expect exactly one top-level group (e.g. "0", "1", "2"...)
            groups = list(h5.keys())
            if len(groups) != 1:
                raise KeyError(f"Expected one group in {fname}, found {groups}")
            grp = groups[0]
            if "pos" not in h5[grp]:
                raise KeyError(f"No 'pos' dataset found under group {grp} in {fname}")
            all_pos.append(h5[f"{grp}/pos"][:])  # shape: [frames, N, 3]
    return np.concatenate(all_pos, axis=0)  # [all_frames, N, 3]

def analyze_dataset(folder, out_dir, dataset_name):
    os.makedirs(out_dir, exist_ok=True)

    # Load trajectory
    positions = load_positions_from_blocks(folder)  # [frames, N, 3]

    # Keep only chromatin beads (first 120)
    if positions.shape[1] > 120:
        print(f"Dataset has {positions.shape[1]} beads — trimming to first 120 (chromatin only).")
        positions = positions[:, :120, :]

    blocks = define_blocks()

    # Collect type-to-type distances per frame
    pair_dists = defaultdict(list)
    for frame in range(positions.shape[0]):
        frame_pos = positions[frame]
        coms = {t: compute_com(frame_pos, ranges) for t, ranges in blocks.items()}

        for t1 in ["A", "B", "C"]:
            for t2 in ["A", "B", "C"]:
                dists = []
                for com1 in coms[t1]:
                    for com2 in coms[t2]:
                        if t1 == t2 and np.allclose(com1, com2):
                            continue
                        dists.append(np.linalg.norm(com1 - com2))
                if dists:
                    pair_dists[(t1, t2)].append(np.mean(dists))

    # Average across frames
    avg_dists = {k: np.mean(v) for k,v in pair_dists.items()}

    # Save text summary
    txt_path = os.path.join(out_dir, f"{dataset_name}_distances.txt")
    with open(txt_path, "w") as f:
        f.write("=== Average block COM distances ===\n")
        for k,v in avg_dists.items():
            f.write(f"{k[0]}–{k[1]}: {v:.3f}\n")

    # Build 3×3 matrix
    dist_matrix = np.zeros((3,3))
    type_map = {"A":0, "B":1, "C":2}
    for (t1,t2),val in avg_dists.items():
        dist_matrix[type_map[t1], type_map[t2]] = val

    # Heatmap
    plt.figure(figsize=(6,5))
    plt.imshow(dist_matrix, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Average COM distance")
    plt.xticks([0,1,2], ["A","B","C"])
    plt.yticks([0,1,2], ["A","B","C"])
    plt.title(f"Block COM Distances – {dataset_name}")
    plt.savefig(os.path.join(out_dir, f"{dataset_name}_heatmap.png"), dpi=300)
    plt.close()

    # Bar chart
    labels = [f"{a}–{b}" for a in ["A","B","C"] for b in ["A","B","C"]]
    values = [avg_dists.get((a,b),0) for a in ["A","B","C"] for b in ["A","B","C"]]
    plt.figure(figsize=(8,5))
    plt.bar(labels, values)
    plt.ylabel("Avg COM Distance")
    plt.title(f"Type-to-Type COM Distances – {dataset_name}")
    plt.savefig(os.path.join(out_dir, f"{dataset_name}_bar.png"), dpi=300)
    plt.close()

    print(f"Analysis complete. Results are saved in {out_dir}")


if __name__ == "__main__":
    # === USER INPUTS HERE ===
    folder = "C:/Users/arnav/Personal - Arnav Chhajed/Northwestern/genome_organization/examples/arnav/test_output_chr_1"
    dataset_name = "chr_analysis"
    out_dir = os.path.join("cluster_dist_analysis", dataset_name)

    analyze_dataset(folder, out_dir, dataset_name)
