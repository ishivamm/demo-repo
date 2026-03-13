"""
Finite-Difference Heat Equation Solver.

Provides a reference implementation of the 2-D transient heat equation:

    ∂T/∂t = α ∇²T + Q(x, y)

using a Forward-Time Central-Space (FTCS) explicit scheme.  The results
serve as ground-truth validation data for the PINN model.
"""

import numpy as np
from typing import Dict, List, Optional

from project.data.generate import (
    DEFAULT_CONFIG,
    create_spatial_grid,
    compute_heat_source_field,
    apply_boundary_conditions,
)


def ftcs_solve(
    config: Optional[Dict] = None,
    nt: int = 100,
    dt: float = 0.1,
) -> Dict:
    """Run the FTCS solver and return the full space-time solution.

    Parameters
    ----------
    config : dict, optional
        Data-center configuration (defaults to ``DEFAULT_CONFIG``).
    nt : int
        Number of time steps.
    dt : float
        Time-step size (seconds).

    Returns
    -------
    dict
        x, y, X, Y, t, T_series, Q, config
    """
    if config is None:
        config = DEFAULT_CONFIG

    alpha = config["thermal_diffusivity"]
    x, y, X, Y = create_spatial_grid(
        config["room_width"],
        config["room_depth"],
        config["nx"],
        config["ny"],
    )
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # CFL stability check
    cfl = alpha * dt * (1.0 / dx ** 2 + 1.0 / dy ** 2)
    if cfl > 0.5:
        raise ValueError(
            f"CFL condition violated (cfl={cfl:.3f} > 0.5). "
            "Reduce dt or increase grid spacing."
        )

    T = np.full_like(X, config["ambient_temp"])
    Q = compute_heat_source_field(X, Y, config["server_racks"])
    Q_norm = Q / (np.max(Q) + 1e-8) * 5.0

    T_series: List[np.ndarray] = [T.copy()]
    t_values = [0.0]

    for step in range(1, nt + 1):
        T_pad = np.pad(T, 1, mode="edge")
        lap = (
            (T_pad[1:-1, 2:] - 2 * T_pad[1:-1, 1:-1] + T_pad[1:-1, :-2]) / dx ** 2
            + (T_pad[2:, 1:-1] - 2 * T_pad[1:-1, 1:-1] + T_pad[:-2, 1:-1]) / dy ** 2
        )
        T = T + dt * (alpha * lap + Q_norm)
        T = apply_boundary_conditions(T, X, Y, config["cooling_vents"])
        T_series.append(T.copy())
        t_values.append(step * dt)

    return {
        "x": x,
        "y": y,
        "X": X,
        "Y": Y,
        "t": np.array(t_values),
        "T_series": T_series,
        "Q": Q,
        "config": config,
    }


def compute_steady_state(
    config: Optional[Dict] = None,
    tol: float = 1e-5,
    max_iter: int = 50000,
    dt: float = 0.05,
) -> Dict:
    """Iterate until the temperature field converges to steady state.

    Parameters
    ----------
    config : dict, optional
        Data-center configuration.
    tol : float
        Convergence tolerance (max absolute change per step).
    max_iter : int
        Maximum number of iterations.
    dt : float
        Pseudo-time step for the iteration.

    Returns
    -------
    dict with keys: x, y, X, Y, T, Q, iterations, converged
    """
    if config is None:
        config = DEFAULT_CONFIG

    alpha = config["thermal_diffusivity"]
    x, y, X, Y = create_spatial_grid(
        config["room_width"],
        config["room_depth"],
        config["nx"],
        config["ny"],
    )
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    T = np.full_like(X, config["ambient_temp"])
    Q = compute_heat_source_field(X, Y, config["server_racks"])
    Q_norm = Q / (np.max(Q) + 1e-8) * 5.0

    converged = False
    for iteration in range(1, max_iter + 1):
        T_pad = np.pad(T, 1, mode="edge")
        lap = (
            (T_pad[1:-1, 2:] - 2 * T_pad[1:-1, 1:-1] + T_pad[1:-1, :-2]) / dx ** 2
            + (T_pad[2:, 1:-1] - 2 * T_pad[1:-1, 1:-1] + T_pad[:-2, 1:-1]) / dy ** 2
        )
        T_new = T + dt * (alpha * lap + Q_norm)
        T_new = apply_boundary_conditions(T_new, X, Y, config["cooling_vents"])

        if np.max(np.abs(T_new - T)) < tol:
            converged = True
            T = T_new
            break
        T = T_new

    return {
        "x": x,
        "y": y,
        "X": X,
        "Y": Y,
        "T": T,
        "Q": Q,
        "iterations": iteration,
        "converged": converged,
        "config": config,
    }
