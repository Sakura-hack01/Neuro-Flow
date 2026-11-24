"""Continuous-Time Graph Neural ODE with adaptive Dormand-Prince (DOPRI5) and RK4 solvers."""

from typing import Callable, Dict, List, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Pure PyTorch Numerical ODE Solvers (RK4 and DOPRI5)
# ==============================================================================

def rk4_step(
    func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    t: torch.Tensor,
    y: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    """Performs a single 4th-order Runge-Kutta integration step."""
    k1 = func(t, y)
    k2 = func(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = func(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = func(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def dopri5_step(
    func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    t: torch.Tensor,
    y: torch.Tensor,
    dt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dormand-Prince 5(4) Butcher tableau embedded step returning (y_5th, error_estimate, k7)."""
    # Butcher tableau coefficients
    c2, c3, c4, c5 = 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0
    a21 = 1.0 / 5.0
    a31, a32 = 3.0 / 40.0, 9.0 / 40.0
    a41, a42, a43 = 44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0
    a51, a52, a53, a54 = 19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0
    a61, a62, a63, a64, a65 = 9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0

    # 5th-order weights
    b1, b3, b4, b5, b6 = 35.0 / 384.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0

    # 4th-order weights difference (error coefficients: b_i - b_star_i)
    e1 = 71.0 / 57600.0
    e3 = -71.0 / 16695.0
    e4 = 71.0 / 1920.0
    e5 = -17253.0 / 339200.0
    e6 = 22.0 / 525.0
    e7 = -1.0 / 40.0

    k1 = func(t, y)
    k2 = func(t + c2 * dt, y + dt * (a21 * k1))
    k3 = func(t + c3 * dt, y + dt * (a31 * k1 + a32 * k2))
    k4 = func(t + c4 * dt, y + dt * (a41 * k1 + a42 * k2 + a43 * k3))
    k5 = func(t + c5 * dt, y + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
    k6 = func(t + dt, y + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5))

    y_next = y + dt * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)
    k7 = func(t + dt, y_next)  # FSAL property

    # Error estimate vector
    error = dt * (e1 * k1 + e3 * k3 + e4 * k4 + e5 * k5 + e6 * k6 + e7 * k7)
    return y_next, error, k7


def integrate_adaptive_dopri5(
    func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    y0: torch.Tensor,
    t_span: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    max_step_size: float = 0.2,
    min_step_size: float = 1e-5,
) -> torch.Tensor:
    """Integrates ODE over target evaluation timestamps with adaptive step size control."""
    device = y0.device
    num_eval_points = len(t_span)
    trajectory = [y0]

    current_y = y0
    current_t = t_span[0]
    dt = torch.tensor(min(0.05, float(t_span[1] - t_span[0])), device=device, dtype=y0.dtype)

    for i in range(1, num_eval_points):
        target_t = t_span[i]

        while current_t < target_t - 1e-7:
            # Clip dt to reach target_t exactly
            step_dt = torch.min(dt, target_t - current_t)
            if step_dt < min_step_size:
                step_dt = target_t - current_t

            y_candidate, error, _ = dopri5_step(func, current_t, current_y, step_dt)

            # Error tolerance scale
            scale = atol + rtol * torch.maximum(torch.abs(current_y), torch.abs(y_candidate))
            norm_error = torch.sqrt(torch.mean((error / scale) ** 2))

            if norm_error <= 1.0 or step_dt <= min_step_size:
                # Accept step
                current_y = y_candidate
                current_t = current_t + step_dt

                # Adapt next step size
                factor = 0.9 * (1.0 / (norm_error + 1e-8)) ** 0.2
                factor = torch.clamp(factor, 0.2, 5.0)
                dt = torch.clamp(step_dt * factor, min_step_size, max_step_size)
            else:
                # Reject step and decrease step size
                factor = torch.clamp(0.9 * (1.0 / (norm_error + 1e-8)) ** 0.25, 0.1, 0.5)
                dt = torch.clamp(step_dt * factor, min_step_size, max_step_size)

        trajectory.append(current_y)

    return torch.stack(trajectory, dim=0)


def odeint_custom(
    func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    y0: torch.Tensor,
    t_eval: torch.Tensor,
    method: str = "rk4",
    substeps: int = 4,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> torch.Tensor:
    """Evaluates ODE across specified timestamps using RK4 or adaptive DOPRI5."""
    if method.lower() == "dopri5":
        return integrate_adaptive_dopri5(func, y0, t_eval, rtol=rtol, atol=atol)

    # Fixed-step RK4 with substepping between timestamps
    trajectory = [y0]
    current_y = y0

    for i in range(len(t_eval) - 1):
        t_start = t_eval[i]
        t_end = t_eval[i + 1]
        dt = (t_end - t_start) / float(substeps)

        for step in range(substeps):
            t_curr = t_start + step * dt
            current_y = rk4_step(func, t_curr, current_y, dt)

        trajectory.append(current_y)

    return torch.stack(trajectory, dim=0)


# ==============================================================================
# Spectral Message Passing Vector Field & Graph Neural ODE
# ==============================================================================

class SpectralChebyshevConv(nn.Module):
    """K-order Spectral Graph Chebyshev Convolution on normalized Laplacian."""

    def __init__(self, in_features: int, out_features: int, K: int = 3) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = K
        self.weight = nn.Parameter(torch.empty(K, in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        """Forward spectral convolution.

        Args:
            x: Node features of shape (..., N, in_features)
            laplacian: Symmetric normalized Laplacian of shape (N, N)
        Returns:
            Output features of shape (..., N, out_features)
        """
        # Scaled Laplacian for Chebyshev recursion: L_tilde = L - I (eigenvalues in [-1, 1])
        N = laplacian.size(-1)
        identity = torch.eye(N, device=laplacian.device, dtype=laplacian.dtype)
        l_tilde = laplacian - identity

        t_0 = x
        out = torch.matmul(t_0, self.weight[0])

        if self.K > 1:
            t_1 = torch.matmul(l_tilde, x)
            out = out + torch.matmul(t_1, self.weight[1])

            for k in range(2, self.K):
                t_k = 2.0 * torch.matmul(l_tilde, t_1) - t_0
                out = out + torch.matmul(t_k, self.weight[k])
                t_0, t_1 = t_1, t_k

        return out + self.bias


class ConnectomeODEFunc(nn.Module):
    """Vector field: dX/dt = -alpha * X - kappa * L_sym * X + f_theta(X, L_sym, u(t))."""

    def __init__(
        self,
        node_dim: int = 4,
        hidden_dim: int = 32,
        spectral_orders: int = 3,
        initial_alpha: float = 0.15,
        initial_kappa: float = 0.85,
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim

        # Learnable positive leak and coupling parameters
        self.raw_alpha = nn.Parameter(torch.tensor(math.log(math.exp(initial_alpha) - 1.0)))
        self.raw_kappa = nn.Parameter(torch.tensor(math.log(math.exp(initial_kappa) - 1.0)))

        # Spectral message passing network f_theta
        self.spec_conv1 = SpectralChebyshevConv(node_dim, hidden_dim, K=spectral_orders)
        self.spec_conv2 = SpectralChebyshevConv(hidden_dim, node_dim, K=spectral_orders)

        # Non-linear point-wise MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )
        self.layer_norm = nn.LayerNorm(node_dim)

    @property
    def alpha(self) -> torch.Tensor:
        """Enforces strictly positive intrinsic leak rate alpha > 0."""
        return F.softplus(self.raw_alpha) + 1e-4

    @property
    def kappa(self) -> torch.Tensor:
        """Enforces non-negative connectome coupling kappa >= 0."""
        return F.softplus(self.raw_kappa)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        laplacian: torch.Tensor,
        perturbation_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Computes continuous-time derivative dX/dt."""
        # 1. Intrinsic linear damping
        leak_term = -self.alpha * x

        # 2. Inter-regional structural diffusion via L_sym: -kappa * (L_sym @ X)
        coupling_term = -self.kappa * torch.matmul(laplacian, x)

        # 3. Non-linear spectral interaction f_theta(X, L_sym)
        h = F.silu(self.spec_conv1(x, laplacian))
        f_theta = self.spec_conv2(h, laplacian)

        # 4. External perturbation u(t) if applied
        u_t = 0.0
        if perturbation_fn is not None:
            u_t = perturbation_fn(t)

        # Time modulation
        t_feat = t.reshape(-1, 1).float()
        time_bias = self.time_mlp(t_feat).unsqueeze(-2)

        dxdt = leak_term + coupling_term + f_theta + u_t + time_bias
        return dxdt


class GraphNeuralODE(nn.Module):
    """End-to-end continuous-time Graph Neural ODE model."""

    def __init__(
        self,
        node_dim: int = 4,
        hidden_dim: int = 32,
        spectral_orders: int = 3,
        solver: str = "rk4",
        substeps: int = 4,
        rtol: float = 1e-4,
        atol: float = 1e-5,
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.solver = solver
        self.substeps = substeps
        self.rtol = rtol
        self.atol = atol

        self.ode_func = ConnectomeODEFunc(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            spectral_orders=spectral_orders,
        )

    def forward(
        self,
        x0: torch.Tensor,
        t_eval: torch.Tensor,
        laplacian: torch.Tensor,
        perturbation_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Integrates trajectory starting from initial state x0.

        Args:
            x0: Initial state tensor of shape (B, N, d) or (N, d)
            t_eval: 1D tensor of irregular timestamps [t0, t1, ..., tT]
            laplacian: Symmetric normalized graph Laplacian (N, N)
            perturbation_fn: Optional callable u(t)
        Returns:
            Integrated state trajectory of shape (len(t_eval), B, N, d)
        """
        squeeze_batch = False
        if x0.dim() == 2:
            x0 = x0.unsqueeze(0)
            squeeze_batch = True

        def vector_field(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return self.ode_func(t, y, laplacian, perturbation_fn)

        traj = odeint_custom(
            vector_field,
            x0,
            t_eval,
            method=self.solver,
            substeps=self.substeps,
            rtol=self.rtol,
            atol=self.atol,
        )

        if squeeze_batch:
            traj = traj.squeeze(1)

        return traj
