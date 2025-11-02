# condensate_chromatin_interactions
This github repository contains the simulations and analysis for modeling chromatin–condensate interactions using molecular dynamics to study how attraction strengths affect clustering, spatial organization, and genome structure. Includes code for simulations, analysis (kNN, RDF, COM), and visualizations of all five test conditions.

Chromatin is modeled as a block heteropolymer with three monomer types (A, B, C), representing different chromatin states. Condensates are treated as spherical particles that can selectively attract specific monomer types. The potential energy function includes harmonic bonds, angle bending, confinement, and type–type interaction terms that together reproduce realistic polymer behavior.

Five test cases are included:
C0: Baseline control (no strong preferences)
C1: Condensate favors Type A
C2: Chromatin self-attraction for Type A
C3: Condensate favors Types B and C
C4: Chromatin self-attraction for all types

Each simulation runs for 200 time frames (1000 timesteps each), and outputs are analyzed for:
Radial distribution functions (RDFs)
Type–type contact fractions
Condensate recruitment profiles
Clustering scores using the k-nearest neighbors algorithm


The repository is organized so anyone can reproduce the full workflow from system setup to analysis. The general order of execution is:

Generate initial monomer positions using the provided scripts in the simulations_notebooks_jupyter/ directory. These define the spatial layout of chromatin monomers (types A, B, and C) and condensates inside the simulation box.

Run the molecular dynamics simulations for each condition (C0–C5) using the corresponding input files. Each run outputs a trajectory file that stores coordinates for all particles over time.

Extract simulation frames using the extraction utilities in the simulation_data/ folder. These scripts convert raw trajectory data into frame-by-frame coordinate files that can be used for visualization or analysis.

Visualize trajectories in OVITO by opening the extracted files and applying the ovito_processing.py script in the analysis/ directory. This script color-codes monomer types, tracks condensate movement, and helps verify that the system behaves as expected before running detailed analyses.

Perform quantitative analysis using the scripts in the analysis_scripts/ folder. Each notebook corresponds to one metric—radial distributions (RDF), type–type contact fractions, condensate recruitment profiles, and clustering scores (k-nearest neighbors).

Aggregate and visualize results using the provided plotting utilities. The figures generated from these scripts reproduce the heatmaps, contact matrices, and clustering profiles presented in the paper.

By following this workflow, anyone can rebuild the full set of simulations and analyses exactly as described, from initialization through visualization to quantitative interpretation.


Collaborators/Mentors: Kaden Dimarco, Dr. Krishna Shrinivas