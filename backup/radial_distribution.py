import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py
import os

def load_positions_and_types(h5_path, monomer_types):
    """Load positions and monomer types for a given .h5 file."""
    with h5py.File(h5_path, 'r') as f:
        group = list(f.keys())[0]
        pos = f[group]['pos'][:]  # shape: (N, 3)
    return pos, monomer_types

def compute_radial_distribution(pos, types, num_bins=50, max_radius=20):
    """Return histogram of distance from origin for each monomer type."""
    r = np.linalg.norm(pos, axis=1)  # Distance of each monomer from origin
    histograms = {}
    for t in np.unique(types):
        mask = types == t  # Boolean mask for type t
        hist, bins = np.histogram(r[mask], bins=num_bins, range=(0, max_radius))
        histograms[t] = (bins[:-1], hist)
    return histograms

def plot_radial_distributions(avg_hists):
    for t, (bins, avg_hist) in avg_hists.items():
        plt.plot(bins, avg_hist, label=f"Type {t}")
    plt.xlabel("Distance from origin")
    plt.ylabel("Number of particles")
    plt.title("Average Radial Distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---- Main script ----
if __name__ == "__main__":
    monomer_types = np.load(
        "C:/Users/arnav/Personal - Arnav Chhajed/Northwestern/genome_organization/examples/arnav/test_output/monomer_types.npy"
    )
    monomer_types = monomer_types.flatten()  # Flatten from (2, 50) to (100,)

    filepaths = sorted(
        glob.glob("C:/Users/arnav/Personal - Arnav Chhajed/Northwestern/genome_organization/examples/arnav/test_output/blocks_*.h5")
    )

    summed_histograms = {}
    for path in filepaths:
        pos, types = load_positions_and_types(path, monomer_types)
        hists = compute_radial_distribution(pos, types)

        for t, (bins, hist) in hists.items():
            if t not in summed_histograms:
                summed_histograms[t] = hist
            else:
                summed_histograms[t] += hist

    # Average histograms across timepoints
    for t in summed_histograms:
        summed_histograms[t] = (bins, summed_histograms[t] / len(filepaths))

    plot_radial_distributions(summed_histograms)

