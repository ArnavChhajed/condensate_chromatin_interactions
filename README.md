# Chromatin-Condensate Interactions: Molecular Dynamics Simulations
# Chromatin–Condensate Interactions: Molecular Dynamics Simulations

This repository contains the molecular dynamics simulations and analysis code used to model **chromatin–condensate interactions**. The project investigates how different attraction strengths and interaction preferences affect chromatin clustering, spatial organization, and genome structure.

The repository includes the complete workflow for system initialization, molecular dynamics simulations, trajectory extraction, visualization, quantitative analysis, and figure generation across five test conditions.

## Model Overview

Chromatin is modeled as a **block heteropolymer** consisting of three monomer types:

* **Type A**
* **Type B**
* **Type C**

These monomer types represent different chromatin states. Condensates are modeled as spherical particles that can selectively attract specific chromatin types.

The simulation potential includes:

* Harmonic bond interactions
* Angular bending potentials
* Spatial confinement
* Type-specific chromatin–chromatin interactions
* Type-specific chromatin–condensate interactions

Together, these interactions provide a coarse-grained representation of polymer behavior and chromatin–condensate interactions.

## Simulation Conditions

Five test conditions are included:

| Condition | Description                                             |
| --------- | ------------------------------------------------------- |
| **C0**    | Baseline control with no strong interaction preferences |
| **C1**    | Condensate preferentially attracts Type A               |
| **C2**    | Type A chromatin has increased self-attraction          |
| **C3**    | Condensate preferentially attracts Types B and C        |
| **C4**    | All chromatin types have increased self-attraction      |

Each simulation consists of **200 frames**, with **1,000 molecular dynamics timesteps per frame**.

## Analysis

Simulation trajectories are analyzed to quantify changes in chromatin organization and condensate interactions. The primary analyses include:

### Radial Distribution Functions (RDF)

RDFs characterize the spatial distribution of different particle types relative to one another and quantify how interaction preferences influence spatial organization.

### Type–Type Contact Fractions

Contact fractions measure the frequency of interactions between different chromatin and condensate types, allowing comparison of preferential associations across simulation conditions.

### Condensate Recruitment Profiles

Recruitment profiles quantify the spatial enrichment of each chromatin type around condensates and identify which chromatin states are preferentially recruited.

### k-Nearest Neighbors (kNN) Clustering

kNN-based clustering scores quantify the degree to which chromatin monomers of the same type cluster together within the simulation box.

## Repository Workflow

The repository is organized to allow the full simulation and analysis pipeline to be reproduced from initial system setup through quantitative analysis.

### 1. Generate Initial Configurations

Use the scripts and notebooks in:

```text
simulations_notebooks_jupyter/
```

These scripts generate the initial spatial arrangement of chromatin monomers and condensates inside the simulation box, including the three chromatin types A, B, and C.

### 2. Run Molecular Dynamics Simulations

Run the molecular dynamics simulations for each condition (**C0–C4**) using the corresponding simulation input files.

Each simulation produces a trajectory containing the coordinates of all particles over time.

### 3. Extract Simulation Frames

Use the utilities in:

```text
simulation_data/
```

to extract individual simulation frames from the raw trajectory files.

The extracted coordinate files are used for visualization and downstream analysis.

### 4. Visualize Trajectories

Simulation trajectories can be inspected using **OVITO**.

The processing script in:

```text
analysis/ovito_processing.py
```

can be applied to the extracted trajectory files to:

* Color-code the different chromatin monomer types
* Track condensate movement
* Visualize chromatin organization
* Verify that simulations behave as expected

Visual inspection provides an initial check of the simulation before performing quantitative analyses.

### 5. Perform Quantitative Analysis

The analysis notebooks and scripts are located in:

```text
analysis_scripts/
```

Each notebook focuses on a specific quantitative metric, including:

* Radial distribution functions (RDFs)
* Type–type contact fractions
* Condensate recruitment profiles
* k-nearest neighbors (kNN) clustering scores

These analyses are applied across the five simulation conditions to compare how different interaction parameters influence chromatin organization.

### 6. Generate Figures and Aggregate Results

The resulting data can be aggregated and visualized using the provided plotting utilities.

These scripts generate the primary visualizations used to compare simulation conditions, including:

* Heatmaps
* Contact matrices
* Radial distribution plots
* Condensate recruitment profiles
* Clustering profiles

The resulting figures reproduce the analyses presented in the associated research work.

## Reproducibility

The repository is designed to reproduce the complete simulation and analysis workflow:

```text
Initial Configuration
        ↓
Molecular Dynamics Simulation
        ↓
Trajectory Extraction
        ↓
OVITO Visualization
        ↓
Quantitative Analysis
        ↓
Data Aggregation
        ↓
Figure Generation
```

Following this workflow allows the simulations and analyses to be reconstructed from system initialization through visualization and quantitative interpretation.

## Collaborators and Mentors

* **Kaden Dimarco**
* **Dr. Krishna Shrinivas**
