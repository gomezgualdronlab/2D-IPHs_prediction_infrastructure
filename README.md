# 2D-IPHs Prediction Infrastructure

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Computational infrastructure to reproduce the machine learning predictions presented in:

> **"Two-dimensional interaction parameter histograms as a simple and versatile nanoporous material representation for machine learning prediction of adsorption properties"**  
> T. Gercina de Vilas, F. Fajardo-Rojas, O. Mansurov, R. Devaisher, E. Toberer, D. A. Gómez-Gualdrón*
> DOI: [XXXXXXXXX](https://doi.org/) *(update with final DOI)*

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Reproducing Publication Results](#reproducing-publication-results)
- [Data Availability](#data-availability)
- [Citation](#citation)
- [Contact](#contact)

---

## Overview

Two-dimensional interaction parameter histograms (2D-IPHs) are a **simulation-free, physics-based structural representation** for nanoporous materials such as metal-organic frameworks (MOFs). A 2D-IPH encodes the distribution of pairwise Lennard-Jones interaction parameters (ε, σ) between framework atoms and a probe molecule, providing a compact fingerprint of the adsorption energy landscape without requiring Grand Canonical Monte Carlo (GCMC) simulations.

This repository provides all scripts and input files needed to:

2. Run Henry's Law (KH) and molecular loading GCMC simulations using RASPA as reference data.
3. Train convolutional neural network (CNN) and gradient-boosted tree (GBT) models on 2D-IPHs to predict adsorption properties.
4. Reproduce the general model and the gas/adsorbent-specific (SFS, scratch) models reported in the publication.

To generate the 2D-IPHs please check the *GitHub repository*: [https://github.com/JFajardoRojas/2D-IPHs_builder](https://github.com/JFajardoRojas/2D-IPHs_builder)

**Key result:** 2D-IPHs is a simple and scalable representation for ML-based adsorption-related properties prediction, enabling screening of millions of MOF structures without molecular simulations.

---

## Repository Structure

```
2D-IPHs_prediction_infrastructure/
│
├── 2D-IPHS-Results-General-model/        # Scripts and results for the general multi-gas ML model
│   ├── 2D-IPHS-General-model.py          # Main training/evaluation script for the general model
│   └── ...
│
├── 2D-IPHS-Results-KH-models/            # Scripts and results for Henry's Law coefficient models
│   └── ...
│
├── 2D-IPHS-Results-SFS-Scratch-models/   # Scripts for gas/adsorbent-specific (SFS & scratch) models
│   └── ...
│
├── KH_RASPA_simulation_example/          # Example RASPA input files for Henry's Law simulations
│   ├── simulation.input                  # RASPA simulation input file
│   └── ...
│
├── Molecular_Loading_simulation_example/ # Example RASPA input files for GCMC loading simulations
│   ├── simulation.input
│   └── ...
│
├── RASPA_Force_Fields/                   # Force field files (UFF, TraPPE, etc.) for RASPA
│   ├── force_field_mixing_rules.def
│   ├── pseudo_atoms.def
│   └── ...
│
└── README.md
```

---

## Prerequisites

### Software

| Dependency | Version | Purpose |
|---|---|---|
| Python | ≥ 3.8 | ML model training and 2D-IPH generation |
| NumPy | ≥ 1.21 | Array operations |
| Pandas | ≥ 1.3 | Data handling |
| Scikit-learn | ≥ 1.0 | Gradient boosting, preprocessing, metrics |
| TensorFlow / PyTorch | ≥ 2.0 / ≥ 1.10 | CNN model training *(check individual scripts)* |
| Matplotlib | ≥ 3.4 | Plotting |
| [RASPA2](https://github.com/iRASPA/RASPA2) | ≥ 2.0 | GCMC / KH molecular simulations |
| [pymatgen](https://pymatgen.org/) | ≥ 2022 | CIF parsing and structure handling |

### Data

- MOF crystal structures in CIF format: [MOFMinE Dataset](https://osf.io/z7nvt/overview)
- RASPA-compatible force field files (provided in `RASPA_Force_Fields/`).

---

## Installation

```bash
# Clone the repository
git clone https://github.com/gomezgualdronlab/2D-IPHs_prediction_infrastructure.git
cd 2D-IPHs_prediction_infrastructure

# Create and activate a virtual environment (recommended)
conda create -n 2diphs python=3.9
conda activate 2diphs

# Install Python dependencies
pip install numpy pandas scikit-learn matplotlib pymatgen tensorflow
# or
pip install -r requirements.txt  # if provided
```

For **RASPA2** installation, follow the official guide at https://github.com/iRASPA/RASPA2.

---

## Usage

Individual README.txt files included in each drectory for usage instructions.

---

## Citation

If you use this code or the 2D-IPH representation in your work, please cite:

```bibtex
@article{FajardoRojas2024_2DIPH,
  title   = {Two-dimensional interaction parameter histograms as a simple and versatile
             nanoporous material representation for machine learning prediction of
             adsorption properties},
  author  = {Fajardo-Rojas, Jair Fernando and Gómez-Gualdrón, Diego A. and ...},
  journal = {....},
  year    = {2026},
  volume  = {XX},
  pages   = {XXXX--XXXX},
  doi     = {....}
}
```

---

## Contact

For questions about the code or the methodology, please contact:

- **Jair Fernando Fajardo-Rojas** — jfajardorojas@mines.edu  
- **Diego A. Gómez-Gualdrón** — dgomezgualdron@mines.edu
- Gómez-Gualdrón Lab, Department of Chemical and Biological Engineering, Colorado School of Mines  
  https://gomezgualdronlab.github.io

Issues and pull requests are welcome.
