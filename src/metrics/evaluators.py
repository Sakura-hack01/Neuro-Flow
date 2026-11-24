"""Evaluation metrics: MSE, Continuous Ranked Probability Score (CRPS), Kuramoto order parameter, and spectral energy drift."""

from typing import Dict, Optional, Tuple, Union
import time
import numpy as np
import torch


def compute_trajectory_mse(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
) -> float:
    """Computes Mean Squared Error across time, nodes, and feature dimensions."""
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)
    return float(torch.mean((y_true - y_pred) ** 2).item())


def compute_crps(
    y_true: torch.Tensor,
    ensemble_pred: torch.Tensor,
) -> float:
    """Computes empirical Continuous Ranked Probability Score (CRPS).

    Args:
        y_true: Ground truth tensor of shape (..., D)
        ensemble_pred: Monte Carlo ensemble predictions of shape (M, ..., D)
    Returns:
        Scalar CRPS score (lower is better calibrated).
    """
    # M = ensemble size
    M = ensemble_pred.shape[0]

    # Term 1: E|Y - X|
    diff_true = torch.abs(ensemble_pred - y_true.unsqueeze(0))
    term1 = torch.mean(diff_true)

    # Term 2: 0.5 * E|X - X'|
    # Vectorized pairwise differences across ensemble samples
    diff_ensemble = torch.abs(ensemble_pred.unsqueeze(1) - ensemble_pred.unsqueeze(0))
    term2 = 0.5 * torch.mean(diff_ensemble)

    crps = term1 - term2
    return float(crps.item())


def compute_kuramoto_order_parameter(
    phases: Union[torch.Tensor, np.ndarray],
) -> Tuple[np.ndarray, float]:
    """Computes the Kuramoto phase synchronization order parameter R(t).

    R(t) = | (1/N) * sum_{j=1}^N exp(i * theta_j(t)) | in [0, 1].

    Args:
        phases: Tensor or array of shape (T, N) containing phase angles theta in radians.
    Returns:
        r_t: Time series of synchronization order parameter R(t) in [0, 1].
        r_mean: Mean order parameter over time.
    """
    if isinstance(phases, torch.Tensor):
        phases_np = phases.detach().cpu().numpy()
    else:
        phases_np = np.asarray(phases)

    # Complex phase vectors
    complex_phases = np.exp(1j * phases_np)
    # Average across nodes (axis=-1)
    mean_complex = np.mean(complex_phases, axis=-1)
    r_t = np.abs(mean_complex)
    r_mean = float(np.mean(r_t))

    return r_t, r_mean


def compute_spectral_energy_drift(
    trajectory: torch.Tensor,
    laplacian: torch.Tensor,
) -> Dict[str, float]:
    """Quantifies spectral energy distribution and drift across Laplacian eigenmodes.

    For state X(t), spectral energy at mode k is E_k(t) = |v_k^T X(t)|^2.
    Computes initial vs terminal spectral distribution divergence and energy loss.
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)  # (N,), (N, N)

    if trajectory.dim() == 4:  # (T, B, N, d) -> average over batch
        x = trajectory[:, 0]    # (T, N, d)
    else:
        x = trajectory         # (T, N, d)

    T, N, d = x.shape
    # Project node states into spectral domain: X_spec(t) = V^T @ X(t)
    x_spec = torch.matmul(eigenvectors.T, x)  # (T, N, d)
    spectral_power = torch.sum(x_spec ** 2, dim=-1)  # (T, N)

    e_initial = spectral_power[0]
    e_terminal = spectral_power[-1]

    # Total energy conservation ratio
    total_energy_init = torch.sum(e_initial).item() + 1e-8
    total_energy_term = torch.sum(e_terminal).item()
    energy_conservation_ratio = total_energy_term / total_energy_init

    # Spectral drift (Jensen-Shannon / normalized Euclidean distance between energy distributions)
    p_init = e_initial / torch.sum(e_initial)
    p_term = e_terminal / (torch.sum(e_terminal) + 1e-8)
    drift = float(torch.norm(p_init - p_term, p=2).item())

    return {
        "energy_conservation_ratio": energy_conservation_ratio,
        "spectral_drift_l2": drift,
    }


def measure_inference_latency(
    model: torch.nn.Module,
    sample_inputs: Tuple,
    device: str = "cpu",
    warmup_runs: int = 5,
    benchmark_runs: int = 25,
) -> Dict[str, float]:
    """Benchmarks exact inference latency in milliseconds."""
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(warmup_runs):
            _ = model(*sample_inputs)

        # Timed benchmark
        latencies = []
        for _ in range(benchmark_runs):
            t_start = time.perf_counter()
            _ = model(*sample_inputs)
            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1000.0)  # ms

    latencies_arr = np.array(latencies)
    return {
        "mean_latency_ms": float(np.mean(latencies_arr)),
        "std_latency_ms": float(np.std(latencies_arr)),
        "p95_latency_ms": float(np.percentile(latencies_arr, 95)),
    }
