"""
main.py – End-to-end pipeline for the Data Center Cooling PINN project.

Usage:
    python -m project.main              # run from repository root
    python project/main.py              # alternative

Steps executed:
    1. Generate synthetic 2-D data center environment
    2. Train the Physics-Informed Neural Network
    3. Evaluate and visualise results
    4. Run cooling-vent optimisation
    5. Save all outputs to project/outputs/
"""

import os
import sys
import time
import numpy as np
import torch

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from project.data.generate import (
    DEFAULT_CONFIG,
    generate_steady_state_temperature,
    generate_time_dependent_data,
    prepare_training_data,
)
from project.pinn.model import HeatPINN
from project.pinn.train import train
from project.optimization.cooling_optimizer import (
    optimize_vent_placement,
    suggest_improvements,
)
from project.visualization.plots import (
    plot_temperature_heatmap,
    plot_hotspot_detection,
    plot_airflow,
    plot_training_loss,
    plot_temperature_evolution,
    plot_optimization_comparison,
)


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def ensure_output_dir() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def main() -> None:
    out = ensure_output_dir()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # ------------------------------------------------------------------
    # Step 1: Generate synthetic data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Generating synthetic 2-D data center environment")
    print("=" * 60)

    steady_data = generate_steady_state_temperature()
    time_data = generate_time_dependent_data(nt=20, dt=0.5)
    training_data = prepare_training_data(steady_data)
    training_data["config"] = steady_data["config"]

    print(f"  Grid size : {steady_data['T'].shape}")
    print(f"  Temp range: {steady_data['T'].min():.1f} – {steady_data['T'].max():.1f} °C")
    print(f"  Interior training points : {training_data['xy_interior'].shape[0]}")
    print(f"  Boundary training points : {training_data['xy_boundary'].shape[0]}")
    print()

    # ------------------------------------------------------------------
    # Step 2: Train the PINN
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 2: Training the PINN model")
    print("=" * 60)

    model_path = os.path.join(out, "best_pinn.pt")
    result = train(
        training_data,
        layers=[2, 64, 64, 64, 1],
        alpha=DEFAULT_CONFIG["thermal_diffusivity"],
        lr=1e-3,
        epochs=3000,
        w_pde=1.0,
        w_bc=10.0,
        w_data=1.0,
        device=device,
        save_path=model_path,
        verbose=True,
        log_every=500,
    )
    model = result["model"]
    history = result["history"]
    print()

    # ------------------------------------------------------------------
    # Step 3: Evaluate & visualise
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 3: Evaluating and saving visualisations")
    print("=" * 60)

    config = steady_data["config"]
    X, Y = steady_data["X"], steady_data["Y"]

    # PINN prediction on the full grid
    model.eval()
    xy_grid = np.stack([X.ravel(), Y.ravel()], axis=1)
    with torch.no_grad():
        T_pred = model(
            torch.tensor(xy_grid, dtype=torch.float32, device=device)
        ).cpu().numpy().reshape(X.shape)

    plot_temperature_heatmap(
        X, Y, T_pred,
        title="PINN-Predicted Temperature",
        server_racks=config["server_racks"],
        cooling_vents=config["cooling_vents"],
        save_path=os.path.join(out, "pinn_temperature.png"),
    )

    plot_temperature_heatmap(
        X, Y, steady_data["T"],
        title="Ground Truth Temperature",
        server_racks=config["server_racks"],
        cooling_vents=config["cooling_vents"],
        save_path=os.path.join(out, "ground_truth_temperature.png"),
    )

    plot_hotspot_detection(
        X, Y, T_pred,
        save_path=os.path.join(out, "hotspot_detection.png"),
    )

    plot_airflow(
        X, Y, T_pred,
        save_path=os.path.join(out, "airflow.png"),
    )

    plot_training_loss(
        history,
        save_path=os.path.join(out, "training_loss.png"),
    )

    plot_temperature_evolution(
        time_data["X"], time_data["Y"],
        time_data["T_series"], time_data["t"],
        save_path=os.path.join(out, "temperature_evolution.png"),
    )

    print(f"  Plots saved to {out}/")
    print()

    # ------------------------------------------------------------------
    # Step 4: Cooling-vent optimisation
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 4: Running cooling-vent optimisation")
    print("=" * 60)

    opt = optimize_vent_placement(n_candidates=40, seed=42)
    suggestions = suggest_improvements(opt)
    for s in suggestions:
        print(f"  → {s}")

    plot_optimization_comparison(
        X, Y, steady_data["T"], opt["best_T"],
        save_path=os.path.join(out, "optimization_comparison.png"),
    )
    print(f"\n  Optimisation plot saved.")
    print()

    print("=" * 60)
    print("Pipeline complete. All outputs in:", out)
    print("=" * 60)


if __name__ == "__main__":
    main()
