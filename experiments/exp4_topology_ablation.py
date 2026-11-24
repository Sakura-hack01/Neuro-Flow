"""Experiment 4: Graph Topology Ablation: Anatomical Connectome vs Erdos-Renyi vs Disconnected Graph."""

from typing import Dict
import json
import os
import numpy as np
import torch

from src.env.connectome_graph import ConnectomeGraph, WilsonCowanSimulator
from src.models.graph_neural_ode import GraphNeuralODE
from src.metrics.evaluators import compute_trajectory_mse, compute_kuramoto_order_parameter, compute_spectral_energy_drift


def build_erdos_renyi_laplacian(num_nodes: int = 68, density: float = 0.18, seed: int = 42) -> np.ndarray:
    """Builds an Erdos-Renyi random graph with matching edge density and computes its Laplacian."""
    rng = np.random.default_rng(seed)
    w = (rng.random((num_nodes, num_nodes)) < density).astype(np.float32)
    w = 0.5 * (w + w.T)
    np.fill_diagonal(w, 0.0)

    # Ensure connectedness
    for i in range(num_nodes):
        if np.sum(w[i]) == 0.0:
            target = (i + 1) % num_nodes
            w[i, target] = 1.0
            w[target, i] = 1.0

    deg = np.sum(w, axis=1)
    with np.errstate(divide="ignore"):
        inv_sqrt_d = 1.0 / np.sqrt(deg)
    inv_sqrt_d[np.isinf(inv_sqrt_d)] = 0.0
    d_mat = np.diag(inv_sqrt_d)
    norm_adj = d_mat @ w @ d_mat
    lap = np.eye(num_nodes, dtype=np.float32) - norm_adj
    return lap.astype(np.float32)


def run_experiment_4(output_dir: str = "results", quick_run: bool = False) -> Dict:
    """Ablates network topology to measure dynamical fidelity and predictive accuracy."""
    os.makedirs(output_dir, exist_ok=True)
    device = "cpu"
    num_nodes = 68

    # 1. Structural Connectome
    true_connectome = ConnectomeGraph(num_regions=num_nodes, seed=42)
    lap_true = torch.from_numpy(true_connectome.laplacian).to(device)

    # 2. Erdos-Renyi random graph
    lap_er_np = build_erdos_renyi_laplacian(num_nodes=num_nodes, density=0.18, seed=42)
    lap_er = torch.from_numpy(lap_er_np).to(device)

    # 3. Disconnected graph (no cross-region edges)
    lap_disc = torch.eye(num_nodes, device=device, dtype=torch.float32)

    topologies = {
        "Anatomical_Connectome": lap_true,
        "Erdos_Renyi_Random": lap_er,
        "Disconnected_Independent": lap_disc,
    }

    # Generate ground-truth cortical dynamics on true connectome
    dataset = WilsonCowanSimulator.load_or_generate_dataset(
        num_samples=8 if not quick_run else 2,
        num_regions=num_nodes,
        t_span=(0.0, 3.0),
        dt=0.05,
        seed=42,
    )
    trajectories = dataset["trajectories"].to(device)
    times = dataset["times"].to(device)
    x0 = trajectories[:, 0]
    target = trajectories.permute(1, 0, 2, 3)  # (T, B, N, d)

    gno = GraphNeuralODE(node_dim=4, hidden_dim=32, spectral_orders=3).to(device)
    gno.eval()

    ablation_metrics = {}

    for name, lap in topologies.items():
        with torch.no_grad():
            pred = gno(x0, times, lap)
            mse = compute_trajectory_mse(target, pred)

            # Kuramoto phase order parameter from predicted phases (channel 2)
            pred_phases = pred[:, 0, :, 2].cpu().numpy()  # (T, N)
            _, r_mean = compute_kuramoto_order_parameter(pred_phases)

            # Spectral energy drift
            drift = compute_spectral_energy_drift(pred, lap)

        ablation_metrics[name] = {
            "trajectory_mse": mse,
            "kuramoto_sync_r": r_mean,
            "energy_conservation_ratio": drift["energy_conservation_ratio"],
            "spectral_drift_l2": drift["spectral_drift_l2"],
        }

    results = {
        "experiment": "exp4_topology_ablation",
        "topologies": ablation_metrics,
    }

    out_file = os.path.join(output_dir, "exp4_topology_ablation.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[Experiment 4] Completed. Topology ablation saved to {out_file}")
    return results


if __name__ == "__main__":
    run_experiment_4()
