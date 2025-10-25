import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Parameters
# -------------------------------
BASELINE_DIR = r"C:/Users/arnav/Personal - Arnav Chhajed/Northwestern/genome_organization/examples/arnav/test_output"
CONDENSATE_DIR = r"C:/Users/arnav/Personal - Arnav Chhajed/Northwestern/genome_organization/examples/arnav/test_output_condensate"
FILE_PATTERN = "blocks_*.h5"
BLOCK_LEN = 10   # 10 beads per block
OUT_DIR = r"C:/Users/arnav/Personal - Arnav Chhajed/Northwestern/genome_organization/examples/arnav/cluster_dist_analysis"

os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------
# Helper functions
# -------------------------------
def block_coms_for_frame(pos, monomer_types, block_len=10):
    """
    Compute center of mass (COM) for each block of monomers.
    Ignores condensates (monomer_types = -1).
    """
    # Pad monomer_types if needed (condensates present)
    if len(monomer_types) < pos.shape[0]:
        pad_len = pos.shape[0] - len(monomer_types)
        monomer_types = np.concatenate([monomer_types, -1 * np.ones(pad_len, dtype=int)])
    elif len(monomer_types) > pos.shape[0]:
        monomer_types = monomer_types[:pos.shape[0]]

    # Keep only chromatin beads
    chromatin_mask = monomer_types != -1
    pos_chrom = pos[chromatin_mask]
    types_chrom = monomer_types[chromatin_mask]

    n_monomers = pos_chrom.shape[0]
    n_blocks = n_monomers // block_len

    coms, labels, block_types = [], [], []
    for i in range(n_blocks):
        block_start = i * block_len
        block_end = (i + 1) * block_len
        block_pos = pos_chrom[block_start:block_end]
        block_com = block_pos.mean(axis=0)
        coms.append(block_com)

        block_type = types_chrom[block_start]
        block_types.append(block_type)
        labels.append(f"Block{i+1} (type={block_type})")

    return np.array(coms), labels, block_types


def average_block_distance_matrix(data_dir, monomer_types, file_pattern, block_len=10):
    files = sorted(glob.glob(os.path.join(data_dir, file_pattern)))
    print(f"Found {len(files)} files in {data_dir}")

    all_mats = []
    for fpath in files:
        with h5py.File(fpath, "r") as f:
            for frame_key in f.keys():
                pos = f[frame_key]["pos"][:]
                coms, labels, block_types = block_coms_for_frame(pos, monomer_types, block_len=block_len)
                dmat = np.linalg.norm(coms[:, None, :] - coms[None, :, :], axis=-1)
                all_mats.append(dmat)

    all_mats = np.array(all_mats)
    avg_mat = all_mats.mean(axis=0)
    return avg_mat, labels, block_types


def plot_matrix(mat, labels, title, outpath=None):
    plt.figure(figsize=(8, 7))
    im = plt.imshow(mat, cmap="viridis")
    plt.colorbar(im, label="Avg Distance")
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.title(title)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=150)
    plt.close()


def plot_typewise_barplot(mat, block_types, title, outpath=None):
    unique_types = sorted(set(block_types))
    results = {}
    for t1 in unique_types:
        for t2 in unique_types:
            mask1 = [i for i, t in enumerate(block_types) if t == t1]
            mask2 = [i for i, t in enumerate(block_types) if t == t2]
            if mask1 and mask2:
                vals = [mat[i,j] for i in mask1 for j in mask2 if i != j]
                results[f"{t1}-{t2}"] = np.mean(vals)

    plt.figure(figsize=(6,4))
    plt.bar(results.keys(), results.values())
    plt.ylabel("Avg Distance")
    plt.title(title)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=150)
    plt.close()


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    # Load chromatin monomer types
    monomer_types = np.load(
        r"C:/Users/arnav/Personal - Arnav Chhajed/Northwestern/genome_organization/examples/arnav/test_output/monomer_types.npy"
    ).flatten()
    print(np.sum(monomer_types==2))
    #print("Monomer types shape (raw):", monomer_types.shape)

    # Baseline
    base_mat, labels, block_types = average_block_distance_matrix(BASELINE_DIR, monomer_types, FILE_PATTERN, BLOCK_LEN)
    plot_matrix(base_mat, labels, "Baseline Block-to-Block Distances",
                outpath=os.path.join(OUT_DIR, "baseline_block_distances.png"))
    plot_typewise_barplot(base_mat, block_types, "Baseline Typewise Distances",
                outpath=os.path.join(OUT_DIR, "baseline_typewise_distances.png"))
    np.savetxt(os.path.join(OUT_DIR, "baseline_avg_distances.txt"), base_mat, fmt="%.3f")

    # Condensates
    cond_mat, labels, block_types = average_block_distance_matrix(CONDENSATE_DIR, monomer_types, FILE_PATTERN, BLOCK_LEN)
    plot_matrix(cond_mat, labels, "Condensate Block-to-Block Distances",
                outpath=os.path.join(OUT_DIR, "condensate_block_distances.png"))
    plot_typewise_barplot(cond_mat, block_types, "Condensate Typewise Distances",
                outpath=os.path.join(OUT_DIR, "condensate_typewise_distances.png"))
    np.savetxt(os.path.join(OUT_DIR, "condensate_avg_distances.txt"), cond_mat, fmt="%.3f")

    # Difference
    diff_mat = cond_mat - base_mat
    plot_matrix(diff_mat, labels, "Difference (Condensate - Baseline)",
                outpath=os.path.join(OUT_DIR, "diff_block_distances.png"))
    plot_typewise_barplot(diff_mat, block_types, "Difference Typewise Distances",
                outpath=os.path.join(OUT_DIR, "diff_typewise_distances.png"))
    np.savetxt(os.path.join(OUT_DIR, "diff_avg_distances.txt"), diff_mat, fmt="%.3f")

    print(f"Done. All results saved in {OUT_DIR}")
