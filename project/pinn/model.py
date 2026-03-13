"""
Physics-Informed Neural Network (PINN) for 2-D Heat Diffusion.

Architecture
------------
The network takes spatial coordinates (x, y) as input and predicts the
temperature T(x, y).  Three loss components drive training:

1. **PDE loss** – enforces  ∂T/∂t = α ∇²T  at interior collocation points.
2. **Boundary loss** – enforces cooling-vent temperatures at domain edges.
3. **Data loss** – matches the network output to observed / synthetic data.

The model uses a fully-connected feed-forward architecture with *tanh*
activations, which is well-suited for representing smooth PDE solutions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple


class HeatPINN(nn.Module):
    """Fully-connected PINN for 2-D heat diffusion.

    Parameters
    ----------
    layers : list of int
        Sizes of each layer, including input (2) and output (1).
        Example: ``[2, 64, 64, 64, 1]``
    activation : nn.Module, optional
        Activation function (default ``nn.Tanh``).
    """

    def __init__(
        self,
        layers: Optional[List[int]] = None,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if layers is None:
            layers = [2, 64, 64, 64, 1]
        if activation is None:
            activation = nn.Tanh()

        net: List[nn.Module] = []
        for i in range(len(layers) - 1):
            net.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                net.append(activation)
        self.net = nn.Sequential(*net)

        # Xavier initialisation
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """Predict temperature T given coordinates ``(x, y)``."""
        return self.net(xy)


# ------------------------------------------------------------------
# Loss functions
# ------------------------------------------------------------------

def pde_loss(
    model: HeatPINN,
    xy: torch.Tensor,
    alpha: float = 0.01,
) -> torch.Tensor:
    """Compute the PDE residual loss  |α ∇²T - 0|² (steady-state).

    For the steady-state heat equation  α ∇²T + Q = 0 we approximate
    Q ≈ 0 at collocation points far from sources. The network should
    learn to satisfy the Laplacian constraint implicitly.
    """
    xy = xy.clone().requires_grad_(True)
    T = model(xy)

    # First derivatives
    grad_T = torch.autograd.grad(
        T, xy, grad_outputs=torch.ones_like(T), create_graph=True
    )[0]
    dT_dx = grad_T[:, 0:1]
    dT_dy = grad_T[:, 1:2]

    # Second derivatives
    dT_dxx = torch.autograd.grad(
        dT_dx, xy, grad_outputs=torch.ones_like(dT_dx), create_graph=True
    )[0][:, 0:1]
    dT_dyy = torch.autograd.grad(
        dT_dy, xy, grad_outputs=torch.ones_like(dT_dy), create_graph=True
    )[0][:, 1:2]

    laplacian = dT_dxx + dT_dyy
    residual = alpha * laplacian  # should ≈ 0 at collocation points
    return torch.mean(residual ** 2)


def boundary_loss(
    model: HeatPINN,
    xy_bnd: torch.Tensor,
    T_bnd: torch.Tensor,
) -> torch.Tensor:
    """MSE between predicted and prescribed boundary temperatures."""
    T_pred = model(xy_bnd)
    return torch.mean((T_pred - T_bnd) ** 2)


def data_loss(
    model: HeatPINN,
    xy_data: torch.Tensor,
    T_data: torch.Tensor,
) -> torch.Tensor:
    """MSE between predicted and observed interior temperatures."""
    T_pred = model(xy_data)
    return torch.mean((T_pred - T_data) ** 2)


def total_loss(
    model: HeatPINN,
    xy_interior: torch.Tensor,
    xy_boundary: torch.Tensor,
    T_boundary: torch.Tensor,
    xy_data: torch.Tensor,
    T_data: torch.Tensor,
    alpha: float = 0.01,
    w_pde: float = 1.0,
    w_bc: float = 10.0,
    w_data: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weighted sum of PDE, boundary, and data losses.

    Returns
    -------
    loss_total, loss_pde, loss_bc, loss_data
    """
    l_pde = pde_loss(model, xy_interior, alpha)
    l_bc = boundary_loss(model, xy_boundary, T_boundary)
    l_data = data_loss(model, xy_data, T_data)
    l_total = w_pde * l_pde + w_bc * l_bc + w_data * l_data
    return l_total, l_pde, l_bc, l_data
