"""Optimal Transport Conditional Flow Matching (OT-CFM) for neural trajectory generation and sensor imputation."""

from typing import Callable, Dict, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_neural_ode import SpectralChebyshevConv, odeint_custom


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal position embedding for continuous flow time t in [0, 1]."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embeds continuous scalar or batch time t into (B, embed_dim)."""
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (B, 1)

        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half_dim, dtype=torch.float32, device=t.device) / half_dim
        )
        args = t * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding


class ConnectomeVelocityField(nn.Module):
    """Time-dependent velocity field v_phi(X_t, t | X_obs, L_sym) for cortical dynamics."""

    def __init__(
        self,
        node_dim: int = 4,
        hidden_dim: int = 64,
        time_embed_dim: int = 32,
        num_layers: int = 3,
        spectral_orders: int = 3,
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim

        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Condition projection (concatenating state and conditioning observation)
        in_channels = node_dim * 2 + 1  # [X_t, X_obs, mask]
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        # Spectral Graph Convolutions
        self.layers = nn.ModuleList([
            SpectralChebyshevConv(hidden_dim, hidden_dim, K=spectral_orders)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Final velocity projection back to node_dim
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_obs: torch.Tensor,
        laplacian: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluates velocity v_phi(x_t, t | x_obs).

        Args:
            x_t: State at flow time t, shape (B, N, d)
            t: Scalar or (B,) tensor in [0, 1]
            x_obs: Conditioning observation, shape (B, N, d)
            laplacian: Graph Laplacian, shape (N, N)
            mask: Optional observation mask, shape (B, N, 1) or (N, 1)
        Returns:
            Predicted vector field velocity, shape (B, N, d)
        """
        batch_size, num_nodes, _ = x_t.shape
        device = x_t.device

        if mask is None:
            mask = torch.ones((batch_size, num_nodes, 1), device=device, dtype=x_t.dtype)
        elif mask.dim() == 2:
            mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        elif mask.dim() == 1:
            mask = mask.view(1, num_nodes, 1).expand(batch_size, -1, -1)

        # Broadcast time
        if t.dim() == 0 or (t.dim() == 1 and len(t) == 1):
            t = t.expand(batch_size)
        t_emb = self.time_proj(self.time_embed(t)).unsqueeze(1)  # (B, 1, hidden_dim)

        # Concatenate features
        combined_in = torch.cat([x_t, x_obs, mask], dim=-1)  # (B, N, 2*d + 1)
        h = self.input_proj(combined_in) + t_emb

        for conv, norm in zip(self.layers, self.norms):
            residual = h
            h = F.silu(conv(h, laplacian))
            h = norm(h + residual)

        velocity = self.out_proj(h)
        return velocity


class OptimalTransportCFM(nn.Module):
    """Optimal Transport Conditional Flow Matching (OT-CFM) model."""

    def __init__(
        self,
        node_dim: int = 4,
        hidden_dim: int = 64,
        time_embed_dim: int = 32,
        num_layers: int = 3,
        spectral_orders: int = 3,
        sigma_min: float = 1e-3,
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.sigma_min = sigma_min

        self.velocity_field = ConnectomeVelocityField(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            time_embed_dim=time_embed_dim,
            num_layers=num_layers,
            spectral_orders=spectral_orders,
        )

    def sample_conditional_path(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes OT probability path psi_t(x0, x1) and target velocity u_t(x1 | x0).

        psi_t(x0, x1) = (1 - (1 - sigma_min) * t) * x0 + t * x1
        u_t(x1 | x0) = x1 - (1 - sigma_min) * x0
        """
        # Ensure t matches batch broadcasting
        while t.dim() < x0.dim():
            t = t.unsqueeze(-1)

        psi_t = (1.0 - (1.0 - self.sigma_min) * t) * x0 + t * x1
        u_t = x1 - (1.0 - self.sigma_min) * x0
        return psi_t, u_t

    def compute_loss(
        self,
        x1: torch.Tensor,
        x_obs: torch.Tensor,
        laplacian: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes OT-CFM regression loss: E || v_phi(psi_t, t | x_obs) - u_t ||^2."""
        batch_size = x1.shape[0]
        device = x1.device

        # Sample flow time t ~ Uniform(0, 1) and prior noise x0 ~ N(0, I)
        t = torch.rand(batch_size, device=device, dtype=x1.dtype)
        x0 = torch.randn_like(x1)

        psi_t, u_t = self.sample_conditional_path(x0, x1, t)
        v_pred = self.velocity_field(psi_t, t, x_obs, laplacian, mask=mask)

        loss = F.mse_loss(v_pred, u_t)
        return loss

    @torch.no_grad()
    def sample_trajectory(
        self,
        x_obs: torch.Tensor,
        laplacian: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        steps: int = 50,
        solver: str = "rk4",
    ) -> torch.Tensor:
        """Generates trajectory from prior noise x0 ~ N(0, I) by integrating velocity field from t=0 to t=1."""
        batch_size = x_obs.shape[0]
        device = x_obs.device

        x0 = torch.randn_like(x_obs)
        t_span = torch.linspace(0.0, 1.0, steps + 1, device=device)

        def flow_vector_field(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return self.velocity_field(x, t, x_obs, laplacian, mask=mask)

        # Integrate from t=0 to t=1
        trajectory = odeint_custom(flow_vector_field, x0, t_span, method=solver, substeps=1)
        x1_generated = trajectory[-1]
        return x1_generated

    @torch.no_grad()
    def impute_missing_channels(
        self,
        x_partial: torch.Tensor,
        mask: torch.Tensor,
        laplacian: torch.Tensor,
        steps: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Imputes missing cortical regions or channels and quantifies uncertainty via ensemble sampling."""
        # Multi-sample generation for uncertainty calibration
        ensemble_size = 5
        samples = []

        for _ in range(ensemble_size):
            gen = self.sample_trajectory(x_obs=x_partial, laplacian=laplacian, mask=mask, steps=steps)
            # Enforce exact observed channels
            reconstructed = mask * x_partial + (1.0 - mask) * gen
            samples.append(reconstructed)

        samples_stack = torch.stack(samples, dim=0)  # (K, B, N, d)
        mean_prediction = torch.mean(samples_stack, dim=0)
        uncertainty_std = torch.std(samples_stack, dim=0)

        return mean_prediction, uncertainty_std
