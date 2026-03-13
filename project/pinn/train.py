"""
Training Loop for the Heat-Diffusion PINN.

Provides a self-contained ``train`` function that:
    1. Converts NumPy arrays to PyTorch tensors.
    2. Runs Adam + optional L-BFGS optimisation.
    3. Logs losses per epoch.
    4. Optionally saves the best model checkpoint.
"""

import os
import time
import numpy as np
import torch
from typing import Dict, List, Optional

from project.pinn.model import HeatPINN, total_loss


def _to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float32, device=device)


def train(
    training_data: Dict,
    model: Optional[HeatPINN] = None,
    layers: Optional[List[int]] = None,
    alpha: float = 0.01,
    lr: float = 1e-3,
    epochs: int = 5000,
    w_pde: float = 1.0,
    w_bc: float = 10.0,
    w_data: float = 1.0,
    device: str = "cpu",
    save_path: Optional[str] = None,
    verbose: bool = True,
    log_every: int = 500,
) -> Dict:
    """Train the PINN model.

    Parameters
    ----------
    training_data : dict
        Output of ``prepare_training_data`` containing ``xy_interior``,
        ``T_interior``, ``xy_boundary``, ``T_boundary``.
    model : HeatPINN, optional
        Pre-initialised model.  A new one is created if *None*.
    layers : list of int, optional
        Layer sizes for a freshly created model.
    alpha : float
        Thermal diffusivity for the PDE loss.
    lr : float
        Learning rate for Adam.
    epochs : int
        Number of training epochs.
    w_pde, w_bc, w_data : float
        Loss-component weights.
    device : str
        ``'cpu'`` or ``'cuda'``.
    save_path : str, optional
        File path to save the best model.
    verbose : bool
        Whether to print progress.
    log_every : int
        Print interval (epochs).

    Returns
    -------
    dict with keys: model, history (dict of loss lists)
    """
    dev = torch.device(device)

    if model is None:
        model = HeatPINN(layers=layers).to(dev)
    else:
        model = model.to(dev)

    # Prepare tensors
    xy_int = _to_tensor(training_data["xy_interior"], dev)
    T_int = _to_tensor(training_data["T_interior"], dev)
    xy_bnd = _to_tensor(training_data["xy_boundary"], dev)
    T_bnd = _to_tensor(training_data["T_boundary"], dev)

    # Generate additional collocation points for PDE loss
    rng = np.random.RandomState(0)
    n_colloc = max(xy_int.shape[0], 3000)
    config = training_data.get("config", {})
    w_max = config.get("room_width", 10.0)
    d_max = config.get("room_depth", 10.0)
    colloc_np = np.column_stack([
        rng.uniform(0, w_max, n_colloc),
        rng.uniform(0, d_max, n_colloc),
    ])
    xy_colloc = _to_tensor(colloc_np, dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

    history: Dict[str, List[float]] = {
        "total": [],
        "pde": [],
        "bc": [],
        "data": [],
    }
    best_loss = float("inf")
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        l_total, l_pde, l_bc, l_data = total_loss(
            model, xy_colloc, xy_bnd, T_bnd, xy_int, T_int,
            alpha=alpha, w_pde=w_pde, w_bc=w_bc, w_data=w_data,
        )

        l_total.backward()
        optimizer.step()
        scheduler.step()

        # Record
        history["total"].append(l_total.item())
        history["pde"].append(l_pde.item())
        history["bc"].append(l_bc.item())
        history["data"].append(l_data.item())

        if l_total.item() < best_loss:
            best_loss = l_total.item()
            if save_path is not None:
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                torch.save(model.state_dict(), save_path)

        if verbose and epoch % log_every == 0:
            elapsed = time.time() - start
            print(
                f"Epoch {epoch:>5d}/{epochs} | "
                f"Loss {l_total.item():.4e} "
                f"(PDE {l_pde.item():.4e}, BC {l_bc.item():.4e}, "
                f"Data {l_data.item():.4e}) | "
                f"{elapsed:.1f}s"
            )

    if verbose:
        print(f"\nTraining complete. Best loss: {best_loss:.4e}")

    return {"model": model, "history": history}
