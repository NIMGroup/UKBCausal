# UKBCausal

**Deep graph causal modeling for brain-body-disease interactions in psychiatric-endocrine disorders**

This repository contains the implementation of:

> **Unveiling Brain-Body Axis Interactions in Psychiatric-Endocrine Disorder with Deep Graph Causal Neural Network**

The project builds a missingness-aware causal discovery pipeline over three types of biomedical nodes:

- **Brain nodes**: diffusion MRI-derived white-matter tract features;
- **Body/organ nodes**: physiological indicators from multiple organ systems;
- **Disease nodes**: psychiatric and endocrine disorder phenotypes.

The core idea is to first align heterogeneous biomedical features into comparable node representations, and then learn a directed acyclic graph (DAG) that describes potential brain-body-disease dependencies.


## OverView

<p align="center">
  <img src="assets/OverView.png" width="100%" alt="Overview of the UKBCausal framework">
</p>


<p align="center">
  <em>Overview of the proposed brain-body-disease causal discovery framework.</em>
</p>


UKBCausal is designed to explore potential directional relationships among brain microstructure, body-organ physiology, and psychiatric-endocrine disorders from population-scale biomedical data. The framework first constructs node-level representations for different biomedical entities, then learns a directed acyclic graph to describe their dependencies.

Specifically, organ-level physiological features are encoded by a masked autoencoder to handle missing observations and obtain compact organ embeddings. Brain white-matter tract features are compressed with PCA to build comparable brain-node representations. These brain, organ, and disease features are then integrated into a unified node-feature tensor and passed to a DAG-GNN model for causal structure learning.

The learned graph can be used to visualize and analyze brain-body-disease pathways, such as direct disease-to-brain links, organ-mediated pathways, and shared white-matter vulnerabilities across disorders.



## Repository Map

| File                    | Role in the pipeline                                         |
| ----------------------- | ------------------------------------------------------------ |
| `MaskedAE_train.py`     | Trains one masked autoencoder for each organ system.         |
| `MaskedAE_inference.py` | Uses trained MAEs to extract organ latent embeddings.        |
| `PCA_compute.py`        | Runs PCA and parallel analysis for each brain ROI CSV.       |
| `PCA_select_k.py`       | Summarizes selected PCA dimensions across ROIs.              |
| `PCA_star.py`           | Draws representative cumulative explained variance and EVR curves. |
| `main.py`               | Self-contained DAG-GNN training script.                      |
| `DAG.py`                | Modular DAG-GNN training script; depends on `utils.py` and `modules.py`. |
| `modules.py`            | Defines the DAG-GNN encoder and decoder modules.             |
| `CauscalPlot.py`        | Draws a chord diagram from the learned adjacency matrix.     |



## Installation

```bash
conda create -n ukbcausal python=3.10 -y
conda activate ukbcausal

pip install numpy pandas scikit-learn matplotlib seaborn pycirclize
pip install torch
```



## Data Organization

The data used in the paper are not included because UK Biobank data require controlled access. A recommended local structure is:

```text
data/
├── MAE-Train-Data/          # Complete organ CSVs for MAE training
├── by_body-organ/           # Organ CSVs used for latent extraction
├── Organ-Latent/            # Output organ embeddings
├── ori/                     # Brain ROI CSVs for PCA analysis
├── node.npy                 # Final node-feature tensor for DAG-GNN
├── label.json               # Labels for adjacency heatmap
├── node.json                # Labels for chord diagram
└── node_abbrev.json         # Abbreviated labels for chord diagram
```



## Quick Start

```bash
# 1. Train organ-specific MAE models
python MaskedAE_train.py

# 2. Extract organ latent embeddings
python MaskedAE_inference.py

# 3. Analyze PCA dimensions for brain ROIs
python PCA_compute.py
mkdir -p sum
python PCA_select_k.py
python PCA_star.py

# 4. Train DAG-GNN after preparing node.npy
python main.py \
  --data_dir data/node.npy \
  --data_variable_size <NUM_NODES> \
  --x_dims <FEATURE_DIM> \
  --z_dims <FEATURE_DIM> \
  --epochs 300 \
  --batch-size 100 \
  --lr 3e-3 \
  --save-folder logs

# 5. Visualize the learned graph
python CauscalPlot.py
```



## Acknowledgement

This project uses data from the UK Biobank resource. We thank the UK Biobank participants and study team for making this research possible.
