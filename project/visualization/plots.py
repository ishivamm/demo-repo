"""
Visualisation Utilities for the Data Center PINN Project.

Provides functions to plot:
    - temperature heat maps
    - hotspot detection overlays
    - training-loss curves
    - cooling airflow direction fields
    - temperature evolution over time
    - optimisation comparison plots
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI/server
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from typing import Dict, List, Optional


# ------------------------------------------------------------------ #
#  Heat maps
# ------------------------------------------------------------------ #

def plot_temperature_heatmap(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    title: str = "Temperature Distribution (°C)",
    server_racks: Optional[List[Dict]] = None,
    cooling_vents: Optional[List[Dict]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a 2-D temperature heat map with optional rack/vent markers."""
    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.pcolormesh(X, Y, T, cmap="hot", shading="auto")
    fig.colorbar(c, ax=ax, label="Temperature (°C)")

    if server_racks is not None:
        for rack in server_racks:
            ax.plot(rack["x"], rack["y"], "ws", markersize=10, label="Server Rack")
    if cooling_vents is not None:
        for vent in cooling_vents:
            ax.plot(vent["x"], vent["y"], "c^", markersize=10, label="Cooling Vent")

    # De-duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper right")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


# ------------------------------------------------------------------ #
#  Hotspot detection
# ------------------------------------------------------------------ #

def plot_hotspot_detection(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    threshold_percentile: float = 90.0,
    title: str = "Hotspot Detection",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Overlay hotspot regions on the temperature field."""
    threshold = np.percentile(T, threshold_percentile)
    hotspot_mask = T > threshold

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pcolormesh(X, Y, T, cmap="hot", shading="auto")
    ax.contour(X, Y, hotspot_mask.astype(float), levels=[0.5], colors="cyan", linewidths=2)
    ax.set_title(f"{title} (threshold={threshold:.1f} °C)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


# ------------------------------------------------------------------ #
#  Cooling airflow (synthetic gradient field)
# ------------------------------------------------------------------ #

def plot_airflow(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    title: str = "Cooling Airflow (negative temperature gradient)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot synthetic airflow as the negative temperature gradient.

    In a real scenario the airflow would come from CFD; here we
    approximate it as heat moving from hot to cold.
    """
    dy, dx = np.gradient(T, Y[:, 0], X[0, :])
    # Airflow opposes the temperature gradient
    u, v = -dx, -dy
    speed = np.sqrt(u ** 2 + v ** 2) + 1e-8
    u /= speed
    v /= speed

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pcolormesh(X, Y, T, cmap="hot", shading="auto", alpha=0.5)
    step = max(1, X.shape[0] // 16)
    ax.quiver(
        X[::step, ::step], Y[::step, ::step],
        u[::step, ::step], v[::step, ::step],
        color="blue", scale=30,
    )
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


# ------------------------------------------------------------------ #
#  Training loss curves
# ------------------------------------------------------------------ #

def plot_training_loss(
    history: Dict[str, List[float]],
    title: str = "PINN Training Loss",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot the multi-component training loss over epochs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, vals in history.items():
        ax.semilogy(vals, label=key)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


# ------------------------------------------------------------------ #
#  Temperature evolution
# ------------------------------------------------------------------ #

def plot_temperature_evolution(
    X: np.ndarray,
    Y: np.ndarray,
    T_series: List[np.ndarray],
    t: np.ndarray,
    n_snapshots: int = 4,
    title: str = "Temperature Evolution",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Show selected time snapshots of the temperature field."""
    indices = np.linspace(0, len(T_series) - 1, n_snapshots, dtype=int)
    fig, axes = plt.subplots(1, n_snapshots, figsize=(4 * n_snapshots, 4))

    vmin = min(Ti.min() for Ti in T_series)
    vmax = max(Ti.max() for Ti in T_series)
    norm = Normalize(vmin=vmin, vmax=vmax)

    for ax, idx in zip(axes, indices):
        c = ax.pcolormesh(X, Y, T_series[idx], cmap="hot", shading="auto", norm=norm)
        ax.set_title(f"t = {t[idx]:.1f} s")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

    fig.colorbar(c, ax=axes, label="Temperature (°C)", shrink=0.8)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


# ------------------------------------------------------------------ #
#  Optimisation comparison
# ------------------------------------------------------------------ #

def plot_optimization_comparison(
    X: np.ndarray,
    Y: np.ndarray,
    T_before: np.ndarray,
    T_after: np.ndarray,
    title: str = "Cooling Optimisation: Before vs After",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Side-by-side comparison of temperature before and after optimisation."""
    vmin = min(T_before.min(), T_after.min())
    vmax = max(T_before.max(), T_after.max())
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    c1 = ax1.pcolormesh(X, Y, T_before, cmap="hot", shading="auto", norm=norm)
    ax1.set_title("Before Optimisation")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")

    c2 = ax2.pcolormesh(X, Y, T_after, cmap="hot", shading="auto", norm=norm)
    ax2.set_title("After Optimisation")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")

    fig.colorbar(c2, ax=[ax1, ax2], label="Temperature (°C)", shrink=0.8)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig
