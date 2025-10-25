import numpy as np
import matplotlib.pyplot as plt
import itertools
import glob
import h5py
from matplotlib.backends.backend_pdf import PdfPages

def load_positions_and_types(h5_path, monomer_types):
    """Load positions and monomer types for a given .h5 file."""
    with h5py.File(h5_path, 'r') as f:
        group = list(f.keys())[0]
        pos = f[group]['pos'][:]  # shape: (N, 3)
    return pos, monomer_types

def compute_pairwise_distances(pos, types, type_pair, num_bins=50, max_dist=20):
    """Compute distance histogram for type1-type2 pairs."""
    i_type, j_type = type_pair
    idx_i = np.where(types == i_type)[0]
    idx_j = np.where(types == j_type)[0]

    distances = []
    for i in idx_i:
        for j in idx_j:
            if i == j:
                continue  # skip self
            if i_type == j_type and i > j:
                continue  # avoid double counting for same type
            d = np.linalg.norm(pos[i] - pos[j])
            distances.append(d)

    hist, bins = np.histogram(distances, bins=num_bins, range=(0, max_dist))
    return bins[:-1], hist

def plot_pairwise_distribution(bins, avg_hist, type_pair):
    label = f"Type {type_pair[0]}-{type_pair[1]}"
    plt.plot(bins, avg_hist, label=label)
    plt.xlabel("Distance")
    plt.ylabel("Number of pairs")
    plt.title(f"Spatial Distribution: {label}")
    plt.legend()
    plt.tight_layout()

# ---- Main script ----
if __name__ == "__main__":
    monomer_types = np.load("C:/Users/arnav/Personal - Arnav Chhajed/Northwestern/genome_organization/examples/arnav/test_output/monomer_types.npy")
    monomer_types = monomer_types.flatten()
    filepaths = sorted(glob.glob("C:/Users/arnav/Personal - Arnav Chhajed/Northwestern/genome_organization/examples/arnav/test_output/blocks_*.h5"))

    # Automatically get all unique type combinations (with repetition)
    unique_types = np.unique(monomer_types)
    type_pairs = list(itertools.combinations_with_replacement(unique_types, 2)) 
    all_distributions = {}

    for pair in type_pairs:
        total_hist = None
        for path in filepaths:
            pos, types = load_positions_and_types(path, monomer_types)
            bins, hist = compute_pairwise_distances(pos, types, pair)

            if total_hist is None:
                total_hist = hist
            else:
                total_hist += hist

        avg_hist = total_hist / len(filepaths)
        all_distributions[pair] = (bins, avg_hist)

    # Save all plots to PDF
    from matplotlib.backends.backend_pdf import PdfPages

with PdfPages("spatial_distributions.pdf") as pdf:
    for pair in type_pairs:
        bins, avg_hist = all_distributions[pair]
        plt.plot(bins, avg_hist, label=f"Type {pair[0]}-{pair[1]}")
        plt.xlabel("Distance")
        plt.ylabel("Number of pairs")
        plt.title(f"Spatial Distribution: Type {pair[0]}-{pair[1]}")
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.clf()

all_distributions = {}
for pair in type_pairs:
    # do your averaging as before
    all_distributions[pair] = (bins, avg_hist)




