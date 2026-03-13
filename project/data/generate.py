"""
Synthetic Data Generation for 2D Data Center Environment.

This module creates a simplified 2D spatial grid representing a data center
floor plan, with server racks producing heat and cooling vents acting as
boundary conditions. The generated data is used to train and evaluate the
Physics-Informed Neural Network (PINN).

Grid convention:
    - x-axis: width of the data center (meters)
    - y-axis: depth of the data center (meters)
    - Temperature values in degrees Celsius
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# Default data-center configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict = {
    "room_width": 10.0,          # meters
    "room_depth": 10.0,          # meters
    "nx": 64,                    # grid points along x
    "ny": 64,                    # grid points along y
    "ambient_temp": 22.0,        # °C
    "thermal_diffusivity": 0.01, # α in m²/s (simplified)
    "server_racks": [
        {"x": 3.0, "y": 3.0, "heat_output": 80.0},
        {"x": 3.0, "y": 7.0, "heat_output": 90.0},
        {"x": 7.0, "y": 3.0, "heat_output": 85.0},
        {"x": 7.0, "y": 7.0, "heat_output": 95.0},
    ],
    "cooling_vents": [
        {"x": 0.0, "y": 5.0, "temp": 16.0},
        {"x": 10.0, "y": 5.0, "temp": 16.0},
        {"x": 5.0, "y": 0.0, "temp": 18.0},
        {"x": 5.0, "y": 10.0, "temp": 18.0},
    ],
}


def create_spatial_grid(
    room_width: float,
    room_depth: float,
    nx: int,
    ny: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return 1-D coordinate arrays and 2-D mesh grids.

    Returns
    -------
    x : np.ndarray, shape (nx,)
    y : np.ndarray, shape (ny,)
    X : np.ndarray, shape (ny, nx)  – meshgrid
    Y : np.ndarray, shape (ny, nx)  – meshgrid
    """
    x = np.linspace(0, room_width, nx)
    y = np.linspace(0, room_depth, ny)
    X, Y = np.meshgrid(x, y)
    return x, y, X, Y


def compute_heat_source_field(
    X: np.ndarray,
    Y: np.ndarray,
    server_racks: List[Dict],
    spread: float = 1.0,
) -> np.ndarray:
    """Build a smooth heat-source field from server rack positions.

    Each rack is modelled as a 2-D Gaussian centered at ``(rack_x, rack_y)``
    with amplitude proportional to *heat_output* and spatial extent *spread*.
    """
    Q = np.zeros_like(X)
    for rack in server_racks:
        rx, ry, q = rack["x"], rack["y"], rack["heat_output"]
        Q += q * np.exp(-((X - rx) ** 2 + (Y - ry) ** 2) / (2 * spread ** 2))
    return Q


def apply_boundary_conditions(
    T: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    cooling_vents: List[Dict],
    vent_radius: float = 1.5,
) -> np.ndarray:
    """Apply cooling-vent boundary conditions as localised cold regions.

    Vents are modelled as soft Gaussian sinks that pull the temperature
    toward the vent temperature within a radius of influence.
    """
    T_out = T.copy()
    for vent in cooling_vents:
        vx, vy, vt = vent["x"], vent["y"], vent["temp"]
        mask = np.exp(-((X - vx) ** 2 + (Y - vy) ** 2) / (2 * vent_radius ** 2))
        T_out = T_out * (1 - mask) + vt * mask
    return T_out


def generate_steady_state_temperature(
    config: Optional[Dict] = None,
) -> Dict:
    """Generate a synthetic steady-state temperature field.

    The temperature is constructed analytically by superimposing heat-source
    Gaussians onto the ambient temperature and then applying cooling-vent
    boundary conditions.  This serves as *ground-truth* training data for
    the PINN.

    Returns
    -------
    dict with keys:
        x, y           – 1-D coordinate arrays
        X, Y           – 2-D meshgrids
        T              – temperature field  (ny, nx)
        Q              – heat source field  (ny, nx)
        config         – the configuration dictionary used
    """
    if config is None:
        config = DEFAULT_CONFIG

    x, y, X, Y = create_spatial_grid(
        config["room_width"],
        config["room_depth"],
        config["nx"],
        config["ny"],
    )

    # Start from ambient temperature
    T = np.full_like(X, config["ambient_temp"])

    # Add heat sources
    Q = compute_heat_source_field(X, Y, config["server_racks"])
    T = T + Q / np.max(Q) * 20.0  # scale heat contribution to ~20 °C rise

    # Apply cooling vents
    T = apply_boundary_conditions(T, X, Y, config["cooling_vents"])

    return {"x": x, "y": y, "X": X, "Y": Y, "T": T, "Q": Q, "config": config}


def generate_time_dependent_data(
    config: Optional[Dict] = None,
    nt: int = 20,
    dt: float = 0.5,
) -> Dict:
    """Generate time-dependent temperature snapshots using explicit Euler.

    This performs a simple forward-time central-space (FTCS) finite-difference
    solve of the 2-D heat equation:

        ∂T/∂t = α ∇²T + Q

    Returns
    -------
    dict with keys:
        x, y, X, Y, t  – coordinate arrays
        T_series        – list of temperature fields at each time step
        Q               – heat source field
        config          – configuration used
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
    Q_norm = Q / (np.max(Q) + 1e-8) * 5.0  # normalised source term

    T_series: List[np.ndarray] = [T.copy()]
    t_values = [0.0]

    for step in range(1, nt + 1):
        # Laplacian via central differences (zero-flux at edges via padding)
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


def prepare_training_data(
    data: Dict,
    n_interior: int = 2000,
    n_boundary: int = 500,
    seed: int = 42,
) -> Dict:
    """Sample training points from the generated data.

    Returns
    -------
    dict with keys:
        xy_interior   – (n_interior, 2)  interior collocation points
        T_interior    – (n_interior, 1)  temperature at those points
        xy_boundary   – (n_boundary, 2)  boundary collocation points
        T_boundary    – (n_boundary, 1)  temperature at boundary points
    """
    rng = np.random.RandomState(seed)

    X, Y, T = data["X"], data["Y"], data["T"]
    ny, nx = X.shape

    # --- Interior points (random) ---
    ix = rng.randint(1, nx - 1, size=n_interior)
    iy = rng.randint(1, ny - 1, size=n_interior)
    xy_int = np.stack([X[iy, ix], Y[iy, ix]], axis=1)
    T_int = T[iy, ix].reshape(-1, 1)

    # --- Boundary points (edges of the domain) ---
    edges: List[Tuple[np.ndarray, np.ndarray]] = []
    # bottom
    idx = rng.randint(0, nx, size=n_boundary // 4)
    edges.append((X[0, idx], Y[0, idx]))
    # top
    idx = rng.randint(0, nx, size=n_boundary // 4)
    edges.append((X[-1, idx], Y[-1, idx]))
    # left
    idx = rng.randint(0, ny, size=n_boundary // 4)
    edges.append((X[idx, 0], Y[idx, 0]))
    # right
    idx = rng.randint(0, ny, size=n_boundary // 4)
    edges.append((X[idx, -1], Y[idx, -1]))

    bx = np.concatenate([e[0] for e in edges])
    by = np.concatenate([e[1] for e in edges])
    xy_bnd = np.stack([bx, by], axis=1)

    # Temperature at boundary – look up from grid (nearest)
    bix = np.clip(np.searchsorted(data["x"], bx), 0, nx - 1)
    biy = np.clip(np.searchsorted(data["y"], by), 0, ny - 1)
    T_bnd = T[biy, bix].reshape(-1, 1)

    return {
        "xy_interior": xy_int,
        "T_interior": T_int,
        "xy_boundary": xy_bnd,
        "T_boundary": T_bnd,
    }
