"""
Cooling-Vent Placement Optimisation.

This module implements a simple optimisation loop that evaluates candidate
cooling-vent configurations and selects the one that best reduces hotspots
and improves thermal uniformity.

Strategy
--------
1. Generate a set of candidate vent positions (grid search or random).
2. For each configuration, solve the steady-state temperature field using
   the finite-difference solver (fast) or the trained PINN (faster at
   inference).
3. Score each configuration with an objective that penalises:
   - maximum temperature (hotspot severity)
   - temperature variance (thermal imbalance)
4. Return the best configuration found.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import copy

from project.data.generate import (
    DEFAULT_CONFIG,
    generate_steady_state_temperature,
)


def score_temperature_field(T: np.ndarray) -> Dict[str, float]:
    """Compute quality metrics for a temperature field.

    Returns
    -------
    dict with keys:
        max_temp      – peak temperature (lower is better)
        mean_temp     – average temperature
        std_temp      – standard deviation (thermal imbalance)
        hotspot_frac  – fraction of cells above a threshold
        score         – combined objective (lower is better)
    """
    threshold = np.percentile(T, 90)
    hotspot_frac = float(np.mean(T > threshold))

    return {
        "max_temp": float(np.max(T)),
        "mean_temp": float(np.mean(T)),
        "std_temp": float(np.std(T)),
        "hotspot_frac": hotspot_frac,
        "score": float(np.max(T)) + 2.0 * float(np.std(T)),
    }


def generate_candidate_vents(
    room_width: float = 10.0,
    room_depth: float = 10.0,
    n_vents: int = 4,
    n_candidates: int = 20,
    vent_temp: float = 16.0,
    seed: int = 123,
) -> List[List[Dict]]:
    """Generate random candidate cooling-vent configurations.

    Each candidate is a list of *n_vents* dicts with keys ``x``, ``y``,
    ``temp``.
    """
    rng = np.random.RandomState(seed)
    candidates: List[List[Dict]] = []

    for _ in range(n_candidates):
        vents = []
        for _ in range(n_vents):
            vx = float(rng.uniform(0, room_width))
            vy = float(rng.uniform(0, room_depth))
            vents.append({"x": vx, "y": vy, "temp": vent_temp})
        candidates.append(vents)

    return candidates


def optimize_vent_placement(
    base_config: Optional[Dict] = None,
    n_vents: int = 4,
    n_candidates: int = 30,
    vent_temp: float = 16.0,
    seed: int = 42,
) -> Dict:
    """Run a random-search optimisation over vent placements.

    Returns
    -------
    dict with keys:
        best_vents    – the optimal vent configuration found
        best_score    – the objective value
        best_metrics  – full quality metrics
        best_T        – temperature field of the best configuration
        all_scores    – list of (score, vents) for every candidate
        baseline      – metrics for the original configuration
    """
    if base_config is None:
        base_config = copy.deepcopy(DEFAULT_CONFIG)

    # Evaluate baseline
    baseline_data = generate_steady_state_temperature(base_config)
    baseline_metrics = score_temperature_field(baseline_data["T"])

    candidates = generate_candidate_vents(
        room_width=base_config["room_width"],
        room_depth=base_config["room_depth"],
        n_vents=n_vents,
        n_candidates=n_candidates,
        vent_temp=vent_temp,
        seed=seed,
    )

    all_scores: List[Tuple[float, List[Dict]]] = []
    best_score = float("inf")
    best_vents: List[Dict] = []
    best_T: Optional[np.ndarray] = None
    best_metrics: Dict = {}

    for vents in candidates:
        cfg = copy.deepcopy(base_config)
        cfg["cooling_vents"] = vents
        result = generate_steady_state_temperature(cfg)
        metrics = score_temperature_field(result["T"])
        all_scores.append((metrics["score"], vents))

        if metrics["score"] < best_score:
            best_score = metrics["score"]
            best_vents = vents
            best_T = result["T"]
            best_metrics = metrics

    all_scores.sort(key=lambda x: x[0])

    return {
        "best_vents": best_vents,
        "best_score": best_score,
        "best_metrics": best_metrics,
        "best_T": best_T,
        "all_scores": all_scores,
        "baseline": baseline_metrics,
    }


def suggest_improvements(
    opt_result: Dict,
) -> List[str]:
    """Return human-readable suggestions based on optimisation results."""
    suggestions: List[str] = []
    bl = opt_result["baseline"]
    bm = opt_result["best_metrics"]

    temp_reduction = bl["max_temp"] - bm["max_temp"]
    if temp_reduction > 0:
        suggestions.append(
            f"Peak temperature reduced by {temp_reduction:.1f} °C "
            f"(from {bl['max_temp']:.1f} to {bm['max_temp']:.1f} °C)."
        )
    else:
        suggestions.append(
            "No improvement in peak temperature — consider adding more vents "
            "or increasing their cooling capacity."
        )

    std_reduction = bl["std_temp"] - bm["std_temp"]
    if std_reduction > 0:
        suggestions.append(
            f"Thermal variance reduced by {std_reduction:.2f} °C "
            f"(σ from {bl['std_temp']:.2f} to {bm['std_temp']:.2f})."
        )

    suggestions.append(
        "Recommended vent positions: "
        + ", ".join(
            f"({vent['x']:.1f}, {vent['y']:.1f})" for vent in opt_result["best_vents"]
        )
    )

    return suggestions
