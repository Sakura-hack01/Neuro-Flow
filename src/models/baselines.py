"""Comparative baseline models for cortical dynamics: DiscreteGRU, GCNDiscrete, and LinearLaplacianDiffusion."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_neural_ode import SpectralChebyshevConv


class DiscreteGRU(nn.Module):
    """Discrete-time Recurrent Baseline (GRU).

    Processes observations strictly sequential-step by sequential-step,
    disregarding irregular physical time intervals (delta t) and spatial connectome topology.
    """

    def __init__(self, node_dim: int = 4, hidden_dim: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Flat feature projection across regions
        self.gru = nn.GRU(
            input_size=node_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_dim, node_dim)

    def forward(
        self,
        x_seq: torch.Tensor,
        t_seq: Optional[torch.Tensor] = None,
        laplacian: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward rollout over discrete observation sequence.

        Args:
            x_seq: Tensor of shape (B, T, N, d) or (T, N, d)
            t_seq: Ignored in standard discrete GRU
            laplacian: Ignored in standard discrete GRU
        Returns:
            Forecasted states: shape (B, T, N, d)
        """
        if x_seq.dim() == 3:
            x_seq = x_seq.unsqueeze(0)  # (1, T, N, d)

        B, T, N, d = x_seq.shape
        # Flatten nodes into batch dimension to run GRU per node independently
        x_reshaped = x_seq.permute(0, 2, 1, 3).reshape(B * N, T, d)

        gru_out, _ = self.gru(x_reshaped)  # (B*N, T, hidden_dim)
        out = self.fc_out(gru_out)          # (B*N, T, d)

        out = out.reshape(B, N, T, d).permute(0, 2, 1, 3)  # (B, T, N, d)
        return out


class GCNDiscrete(nn.Module):
    """Discrete-time Graph Convolutional Network baseline with fixed-step autoregressive rollout."""

    def __init__(
        self,
        node_dim: int = 4,
        hidden_dim: int = 64,
        spectral_orders: int = 3,
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim

        self.conv1 = SpectralChebyshevConv(node_dim, hidden_dim, K=spectral_orders)
        self.conv2 = SpectralChebyshevConv(hidden_dim, hidden_dim, K=spectral_orders)
        self.fc_delta = nn.Linear(hidden_dim, node_dim)

    def step(self, x: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        """Computes discrete single step update: x_{t+1} = x_t + Delta(x_t)."""
        h = F.silu(self.conv1(x, laplacian))
        h = F.silu(self.conv2(h, laplacian))
        delta = self.fc_delta(h)
        return x + delta

    def forward(
        self,
        x0: torch.Tensor,
        num_steps: int,
        laplacian: torch.Tensor,
    ) -> torch.Tensor:
        """Autoregressively rolls out predictions over num_steps."""
        trajectory = [x0]
        curr = x0
        for _ in range(num_steps - 1):
            curr = self.step(curr, laplacian)
            trajectory.append(curr)
        return torch.stack(trajectory, dim=0)  # (T, B, N, d)


class LinearLaplacianDiffusion(nn.Module):
    """Analytical heat-kernel diffusion on symmetric normalized Graph Laplacian.

    Models continuous passive spreading without non-linear neural transformations:
        X(t) = exp(-kappa * L_sym * t) * X(0)
    Computed via the matrix exponential or spectral decomposition of L_sym.
    """

    def __init__(self, initial_kappa: float = 0.85) -> None:
        super().__init__()
        self.kappa = nn.Parameter(torch.tensor(initial_kappa, dtype=torch.float32))

    def forward(
        self,
        x0: torch.Tensor,
        t_eval: torch.Tensor,
        laplacian: torch.Tensor,
    ) -> torch.Tensor:
        """Analytical continuous-time diffusion forward rollout.

        Args:
            x0: Initial node features, shape (B, N, d) or (N, d)
            t_eval: Timestamps, shape (T,)
            laplacian: Symmetric normalized Laplacian, shape (N, N)
        Returns:
            Diffused states over time, shape (T, B, N, d)
        """
        squeeze_batch = False
        if x0.dim() == 2:
            x0 = x0.unsqueeze(0)
            squeeze_batch = True

        B, N, d = x0.shape
        T = len(t_eval)

        # Spectral eigendecomposition of L_sym: L = V * Lambda * V^T
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)  # (N,), (N, N)

        # exp(-kappa * L * t) = V * diag(exp(-kappa * lambda_i * t)) * V^T
        trajectory = []
        k = torch.clamp(self.kappa, min=0.0)

        for t in t_eval:
            decay = torch.exp(-k * eigenvalues * t)  # (N,)
            heat_kernel = eigenvectors @ torch.diag(decay) @ eigenvectors.T  # (N, N)
            xt = torch.matmul(heat_kernel, x0)  # (B, N, d)
            trajectory.append(xt)

        out = torch.stack(trajectory, dim=0)  # (T, B, N, d)
        if squeeze_batch:
            out = out.squeeze(1)

        return out
