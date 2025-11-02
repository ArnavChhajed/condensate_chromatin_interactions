import numpy as np
import h5py
import glob
import os
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ----------------------------------------------------
# Function: load_positions_and_types
# Loads positions (x, y, z) and monomer types for all particles from an h5 file.
# ----------------------------------------------------
def load_positions_and_types(h5_path, monomer_types):
    """
    Parameters:
        h5_path (str) - path to the .h5 file containing simulation frame data
        monomer_types (np.ndarray) - 1D array of particle types for all N particles

    Returns:
        pos (np.ndarray) - shape (N, 3), particle positions
        types (np.ndarray) - shape (N,), particle types
    """
    with h5py.File(h5_path, 'r') as f:
        group = list(f.keys())[0]           # typically "particles"
        pos = f[group]['pos'][:]            # read all positions
    return pos, monomer_types

# ----------------------------------------------------
# Function: compute_purity
# For each particle, find k nearest neighbors and calculate fraction
# of neighbors that have the same type.
# ----------------------------------------------------
def compute_purity(pos, types, k):
    """
    Parameters:
        pos (np.ndarray) - shape (N, 3), positions of all particles
        types (np.ndarray) - shape (N,), particle types
        k (int) - number of neighbors to consider

    Returns:
        purity_per_type (dict) - {type: average purity for that type in this frame}
    """
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(pos)
    distances, indices = nbrs.kneighbors(pos)

    # Ignore the first neighbor (distance=0 to itself)
    neighbor_indices = indices[:, 1:]

    N = len(types)
    purities = np.zeros(N)

    for i in range(N):
        neighbor_types = types[neighbor_indices[i]]
        same_type_count = np.sum(neighbor_types == types[i])
        purities[i] = same_type_count / k

    # Compute average purity per type
    purity_per_type = {}
    for t in np.unique(types):
        mask = types == t
        purity_per_type[t] = purities[mask].mean()

    return purity_per_type

# ----------------------------------------------------
# Function: process_all_frames
# Loops over all simulation frames (.h5 files) and computes purity per type.
# ----------------------------------------------------
def process_all_frames(h5_files, monomer_types, k):
    """
    Returns:
        purity_over_time (dict) - {type: [purity_frame0, purity_frame1, ...]}
    """
    purity_over_time = {t: [] for t in np.unique(monomer_types)}

    for h5_path in h5_files:
        pos, types = load_positions_and_types(h5_path, monomer_types)
        purity_per_type = compute_purity(pos, types, k)

        for t in purity_over_time:
            purity_over_time[t].append(purity_per_type[t])

    return purity_over_time

# ----------------------------------------------------
# Function: make_pdf_plots
# Creates a PDF file with one bar chart per frame + one line plot at the end.
# ----------------------------------------------------
def make_pdf_plots(purity_over_time, k, output_pdf):
    """
    Parameters:
        purity_over_time (dict) - {type: list of purities per frame}
        k (int) - k value used
        output_pdf (str) - path to save the PDF
    """
    types_sorted = sorted(purity_over_time.keys())
    num_frames = len(next(iter(purity_over_time.values())))

    with PdfPages(output_pdf) as pdf:
        # Per-frame bar plots
        for frame_idx in range(num_frames):
            plt.figure(figsize=(6, 4))
            purities = [purity_over_time[t][frame_idx] for t in types_sorted]
            plt.bar(range(len(types_sorted)), purities, tick_label=[f"Type {t}" for t in types_sorted])
            plt.ylim(0, 1)
            plt.ylabel("Average Purity")
            plt.title(f"Frame {frame_idx} - k={k}")
            pdf.savefig()
            plt.close()

        # Purity over time plot
        plt.figure(figsize=(6, 4))
        for t in types_sorted:
            plt.plot(range(num_frames), purity_over_time[t], label=f"Type {t}")
        plt.xlabel("Frame")
        plt.ylabel("Average Purity")
        plt.title(f"Purity Over Time - k={k}")
        plt.ylim(0, 1)
        plt.legend()
        pdf.savefig()
        plt.close()

# ----------------------------------------------------
# Main script
# ----------------------------------------------------
if __name__ == "__main__":
    # Load monomer types
    monomer_types = np.load("C:/Users/arnav/Personal - Arnav Chhajed/Northwestern/genome_organization/examples/arnav/test_output/monomer_types.npy").flatten()

    # Get list of .h5 files (frames)
    h5_files = sorted(glob.glob("C:/Users/arnav/Personal - Arnav Chhajed/Northwestern/genome_organization/examples/arnav/test_output/blocks_*.h5"))

    # Choose k values to analyze
    k_values = [5, 10, 20]

    for k in k_values:
        purity_over_time = process_all_frames(h5_files, monomer_types, k)
        make_pdf_plots(purity_over_time, k, f"purity_k{k}.pdf")

    print("PDFs created for each k value.")
