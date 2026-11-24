"""Tests for ODE solvers, Graph Neural ODE, Optimal Transport CFM, and baseline models."""

import pytest
import torch

from src.models.graph_neural_ode import (
    GraphNeuralODE,
    ConnectomeODEFunc,
    rk4_step,
    dopri5_step,
    odeint_custom,
)
from src.models.flow_matching import OptimalTransportCFM
from src.models.baselines import DiscreteGRU, GCNDiscrete, LinearLaplacianDiffusion
from src.metrics.evaluators import compute_trajectory_mse, compute_crps, compute_spectral_energy_drift


def test_rk4_and_dopri5_solvers_analytical_decay() -> None:
    """Verifies that custom RK4 and DOPRI5 solvers match analytical solution for dy/dt = -alpha * y."""
    alpha = 0.5
    y0 = torch.tensor([[2.0]], dtype=torch.float32)
    dt = torch.tensor(0.05)

    def linear_decay(t, y):
        return -alpha * y

    # Single RK4 step
    y_rk4 = rk4_step(linear_decay, torch.tensor(0.0), y0, dt)
    y_exact = y0 * torch.exp(-alpha * dt)
    assert torch.allclose(y_rk4, y_exact, atol=1e-5), "RK4 step error exceeds analytical threshold"

    # Multi-step integration with DOPRI5
    t_eval = torch.linspace(0.0, 1.0, 11)
    traj_dopri5 = odeint_custom(linear_decay, y0, t_eval, method="dopri5").view(11, 1)
    traj_exact = (y0 * torch.exp(-alpha * t_eval).unsqueeze(-1)).view(11, 1)
    assert torch.allclose(traj_dopri5, traj_exact, atol=1e-3), "DOPRI5 trajectory deviates from exact decay"


def test_graph_neural_ode_forward_and_autograd() -> None:
    """Verifies GraphNeuralODE forward integration and gradient backpropagation."""
    B, N, d = 2, 16, 4
    x0 = torch.randn(B, N, d, requires_grad=True)
    L = torch.eye(N)
    t_eval = torch.tensor([0.0, 0.2, 0.5, 1.0])

    model = GraphNeuralODE(node_dim=d, hidden_dim=16, spectral_orders=2, solver="rk4")
    out = model(x0, t_eval, L)

    assert out.shape == (len(t_eval), B, N, d), f"Unexpected output shape: {out.shape}"
    assert torch.isfinite(out).all(), "ODE trajectory contains NaNs or Infs"

    # Backward gradient check
    loss = out.sum()
    loss.backward()
    assert x0.grad is not None, "Gradient was not backpropagated to x0"
    assert torch.isfinite(x0.grad).all(), "x0.grad contains non-finite values"
    assert model.ode_func.raw_alpha.grad is not None, "Parameter gradients missing"


def test_optimal_transport_cfm_loss_and_sampling() -> None:
    """Verifies OT-CFM loss computation, linear path validity, and conditional sampling."""
    B, N, d = 4, 16, 4
    x1 = torch.randn(B, N, d)
    x_obs = torch.randn(B, N, d)
    L = torch.eye(N)

    cfm = OptimalTransportCFM(node_dim=d, hidden_dim=16, time_embed_dim=8, num_layers=2)

    # Path check: psi_0 = x0, psi_1 = x1 + sigma_min * x0
    x0 = torch.randn_like(x1)
    psi_0, u_0 = cfm.sample_conditional_path(x0, x1, torch.zeros(B))
    psi_1, u_1 = cfm.sample_conditional_path(x0, x1, torch.ones(B))
    assert torch.allclose(psi_0, x0, atol=1e-4)
    expected_psi_1 = x1 + cfm.sigma_min * x0
    assert torch.allclose(psi_1, expected_psi_1, atol=1e-4)

    # Loss check
    loss = cfm.compute_loss(x1, x_obs, L)
    assert torch.isfinite(loss), "CFM loss is non-finite"
    loss.backward()

    # Sampling check
    gen = cfm.sample_trajectory(x_obs, L, steps=5)
    assert gen.shape == (B, N, d), f"Unexpected generated sample shape: {gen.shape}"
    assert torch.isfinite(gen).all(), "Generated sample contains NaNs"


def test_baseline_models_shapes() -> None:
    """Verifies input-output parity and tensor shapes for all baseline models."""
    B, T, N, d = 2, 6, 16, 4
    x_seq = torch.randn(B, T, N, d)
    x0 = x_seq[:, 0]
    t_seq = torch.linspace(0.0, 1.0, T)
    L = torch.eye(N)

    # Discrete GRU
    gru = DiscreteGRU(node_dim=d, hidden_dim=16)
    gru_out = gru(x_seq)
    assert gru_out.shape == (B, T, N, d)

    # GCN Discrete
    gcn = GCNDiscrete(node_dim=d, hidden_dim=16, spectral_orders=2)
    gcn_out = gcn(x0, T, L)
    assert gcn_out.shape == (T, B, N, d)

    # Linear Laplacian Diffusion
    diff = LinearLaplacianDiffusion()
    diff_out = diff(x0, t_seq, L)
    assert diff_out.shape == (T, B, N, d)


def test_metrics_evaluators() -> None:
    """Verifies evaluation metric functions (MSE, CRPS, Spectral Drift)."""
    y_true = torch.ones(5, 10, 2)
    y_pred = torch.zeros(5, 10, 2)
    mse = compute_trajectory_mse(y_true, y_pred)
    assert abs(mse - 1.0) < 1e-5

    ensemble = torch.stack([torch.ones(5, 10, 2) + 0.1 * i for i in range(5)], dim=0)
    crps = compute_crps(y_true, ensemble)
    assert crps >= 0.0, "CRPS must be non-negative"

    L = torch.eye(10)
    drift = compute_spectral_energy_drift(y_true, L)
    assert "energy_conservation_ratio" in drift
    assert "spectral_drift_l2" in drift
