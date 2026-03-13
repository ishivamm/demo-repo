# PINNs for Data Center Cooling Optimization

> **Physics-Informed Neural Networks for modeling heat diffusion in data centers and optimizing cooling efficiency.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Physics Background](#physics-background)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [Results](#results)
8. [Streamlit Dashboard](#streamlit-dashboard)
9. [Future Improvements](#future-improvements)

---

## Problem Statement

Modern data centers house thousands of server racks that generate significant
heat. Maintaining safe operating temperatures is both an engineering challenge
and a major cost driver. This project uses **Physics-Informed Neural Networks
(PINNs)** to:

* **Model** the temperature distribution across a simplified 2-D data center.
* **Predict** hotspot locations.
* **Optimize** cooling-vent placement to reduce peak temperatures and improve
  thermal uniformity.

Unlike purely data-driven approaches, PINNs embed the governing physical laws
directly into the loss function, enabling accurate predictions even with sparse
measurement data.

---

## Physics Background

### Heat Diffusion Equation

The temperature field *T(x, y, t)* in the data center is governed by the 2-D
heat equation:

```
∂T/∂t = α ∇²T + Q(x, y)
```

| Symbol | Meaning |
|--------|---------|
| *T* | Temperature (°C) |
| *α* | Thermal diffusivity (m²/s) |
| ∇²T | Laplacian of T (second spatial derivatives) |
| *Q* | Volumetric heat source (server racks) |

For the **steady-state** case (∂T/∂t = 0):

```
α ∇²T + Q(x, y) = 0
```

### PINN Loss Function

The network is trained to minimise:

```
L = w_pde · L_PDE  +  w_bc · L_BC  +  w_data · L_Data
```

* **L_PDE** – PDE residual evaluated at interior collocation points.
* **L_BC** – Boundary condition error at cooling vents / domain edges.
* **L_Data** – Data-fit error at sensor / synthetic measurement points.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      PINN Architecture                       │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   Input (x, y)                                               │
│       │                                                      │
│       ▼                                                      │
│   ┌──────────┐                                               │
│   │ Linear(2, 64) + Tanh │                                   │
│   └──────────┘                                               │
│       │                                                      │
│       ▼                                                      │
│   ┌──────────┐                                               │
│   │ Linear(64, 64) + Tanh │                                  │
│   └──────────┘                                               │
│       │                                                      │
│       ▼                                                      │
│   ┌──────────┐                                               │
│   │ Linear(64, 64) + Tanh │                                  │
│   └──────────┘                                               │
│       │                                                      │
│       ▼                                                      │
│   ┌──────────┐                                               │
│   │ Linear(64, 1)         │  ← Temperature prediction T̂     │
│   └──────────┘                                               │
│                                                              │
│   Loss = w₁ · ‖α∇²T̂ ‖²  +  w₂ · ‖T̂_bc - T_bc‖²          │
│        + w₃ · ‖T̂_data - T_data‖²                           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

The gradients ∂T̂/∂x, ∂²T̂/∂x², etc. are computed via **automatic
differentiation** (PyTorch autograd), which is the key enabler of the
physics-informed approach.

---

## Project Structure

```
project/
│
├── data/
│   ├── __init__.py
│   └── generate.py          # Synthetic data center environment
│
├── models/
│   └── .gitkeep             # Saved model checkpoints
│
├── pinn/
│   ├── __init__.py
│   ├── model.py             # PINN architecture & loss functions
│   └── train.py             # Training loop
│
├── simulation/
│   ├── __init__.py
│   └── heat_solver.py       # Finite-difference reference solver
│
├── optimization/
│   ├── __init__.py
│   └── cooling_optimizer.py # Vent-placement optimization
│
├── visualization/
│   ├── __init__.py
│   └── plots.py             # Heat maps, hotspots, airflow, loss curves
│
├── app/
│   ├── __init__.py
│   └── dashboard.py         # Streamlit interactive dashboard
│
├── notebooks/
│   └── demo.ipynb           # Step-by-step demo notebook
│
├── main.py                  # End-to-end pipeline script
├── requirements.txt         # Python dependencies
└── README.md                # (this file)
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/ishivamm/demo-repo.git
cd demo-repo

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux / macOS
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r project/requirements.txt
```

### Requirements

| Package | Purpose |
|---------|---------|
| `torch >= 2.0` | Deep learning framework |
| `numpy >= 1.24` | Numerical computation |
| `matplotlib >= 3.7` | Plotting |
| `streamlit >= 1.28` | Interactive dashboard |
| `scipy >= 1.10` | Scientific computing utilities |

---

## Quick Start

### Run the full pipeline

```bash
cd demo-repo
python -m project.main
```

This will:

1. Generate a synthetic data center environment.
2. Train the PINN for 3,000 epochs.
3. Save visualisation plots to `project/outputs/`.
4. Run cooling-vent optimization and print suggestions.

### Launch the Streamlit dashboard

```bash
streamlit run project/app/dashboard.py
```

### Run the Jupyter notebook

```bash
cd project/notebooks
jupyter notebook demo.ipynb
```

---

## Results

After training, the pipeline produces the following outputs in `project/outputs/`:

| Output | Description |
|--------|-------------|
| `pinn_temperature.png` | PINN-predicted temperature heat map |
| `ground_truth_temperature.png` | Synthetic ground-truth heat map |
| `hotspot_detection.png` | Hotspot regions highlighted |
| `airflow.png` | Simulated cooling airflow (neg. gradient) |
| `training_loss.png` | Multi-component loss curves |
| `temperature_evolution.png` | Time-series snapshots |
| `optimization_comparison.png` | Before vs. after vent optimization |

### Example: Temperature Heat Map

The PINN learns to approximate the temperature field from the heat equation
and sparse boundary data. Server racks (white squares) generate heat;
cooling vents (cyan triangles) pull the temperature down.

### Example: Training Loss

The total loss is the weighted sum of PDE residual, boundary-condition, and
data-fit components. All three decrease together, indicating that the network
is simultaneously learning the physics and fitting the data.

### Example: Optimization

The random-search optimizer evaluates 40 candidate vent configurations and
selects the one that minimises peak temperature and thermal variance.

---

## Streamlit Dashboard

The interactive dashboard allows users to:

- **Place server racks** – set position (x, y) and heat output.
- **Place cooling vents** – set position and cooling temperature.
- **Adjust room dimensions** and ambient temperature.
- **Visualise** the resulting temperature field and hotspots in real time.
- **Run optimization** to find better vent placements.

---

## Future Improvements

| Area | Description |
|------|-------------|
| **3-D extension** | Extend from 2-D to full 3-D heat equation |
| **Time-dependent PINN** | Add a temporal input to model transient dynamics |
| **CFD coupling** | Replace gradient-based airflow with Navier–Stokes |
| **Bayesian PINN** | Uncertainty quantification on predictions |
| **DeepXDE integration** | Use the DeepXDE library for advanced PINN features |
| **Transfer learning** | Pre-train on canonical heat problems, fine-tune on site data |
| **Real sensor data** | Incorporate IoT sensor measurements from live data centers |
| **Reinforcement learning** | Use RL to dynamically adjust cooling in real time |
| **GPU acceleration** | Multi-GPU distributed training for large-scale models |

---

## License

This project is provided for educational and research purposes.

## Author

Shivam Maurya
