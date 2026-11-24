"""Experiment 2: 2D Bifurcation Phase Diagram across Connectome Coupling (kappa) vs Stimulation Amplitude (I_stim)."""

from typing import Dict
import json
import os
import numpy as np

from src.env.connectome_graph import ConnectomeGraph, WilsonCowanSimulator
from src.metrics.evaluators import compute_kuramoto_order_parameter


def run_experiment_2(output_dir: str = "results", grid_size: int = 12) -> Dict:
    """Computes a 2D phase diagram of Kuramoto phase synchronization order parameter R."""
    os.makedirs(output_dir, exist_ok=True)
    connectome = ConnectomeGraph(num_regions=68, seed=42)

    kappas = np.linspace(0.0, 2.0, grid_size).tolist()
    stim_amps = np.linspace(0.0, 2.5, grid_size).tolist()

    r_matrix = np.zeros((grid_size, grid_size), dtype=np.float32)
    regimes_matrix = [["" for _ in range(grid_size)] for _ in range(grid_size)]

    print(f"[Experiment 2] Running 2D phase sweep ({grid_size}x{grid_size} grid)...")

    for i, kappa in enumerate(kappas):
        for j, amp in enumerate(stim_amps):
            sim = WilsonCowanSimulator(
                connectome=connectome,
                coupling_kappa=float(kappa),
                p_base=1.1,
            )
            perturbation = {
                "t_start": 1.0,
                "t_end": 4.0,
                "amplitude": float(amp),
                "frequency": 10.0,
                "target_subnetwork": 3,
            }
            # Short simulation for equilibrium phase extraction
            out = sim.simulate(t_span=(0.0, 4.0), dt=0.05, perturbation=perturbation, seed=42 + i * 10 + j)
            phases = out["phase"]  # (T, N)

            # Evaluate steady-state Kuramoto order parameter over second half
            half_idx = len(phases) // 2
            _, r_mean = compute_kuramoto_order_parameter(phases[half_idx:])
            r_matrix[i, j] = r_mean

            # Classify dynamic regime
            if r_mean < 0.30:
                regime = "Incoherent_Asynchronous"
            elif r_mean < 0.72:
                regime = "Metastable_PartiallySynchronized"
            else:
                regime = "Globally_PhaseLocked"
            regimes_matrix[i][j] = regime

    results = {
        "experiment": "exp2_bifurcation_phase_diagram",
        "coupling_kappas": kappas,
        "stim_amplitudes": stim_amps,
        "kuramoto_order_matrix": r_matrix.tolist(),
        "regimes_matrix": regimes_matrix,
        "grid_size": grid_size,
    }

    out_file = os.path.join(output_dir, "exp2_bifurcation_phase_diagram.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[Experiment 2] Completed. Phase diagram saved to {out_file}")
    return results


if __name__ == "__main__":
    run_experiment_2()
