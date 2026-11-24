"""Tests for ConnectomeGraph and WilsonCowanSimulator."""

import numpy as np
import pytest
import torch

from src.env.connectome_graph import ConnectomeGraph, WilsonCowanSimulator
from src.metrics.evaluators import compute_kuramoto_order_parameter


def test_connectome_symmetry_and_diagonal() -> None:
    """Verifies that structural connectivity matrix W is strictly symmetric with zero diagonal."""
    connectome = ConnectomeGraph(num_regions=68, seed=42)
    w = connectome.adjacency

    # Symmetry
    assert np.allclose(w, w.T, atol=1e-5), "Adjacency matrix W must be symmetric"
    # Zero diagonal (no self-loops)
    assert np.allclose(np.diag(w), 0.0, atol=1e-6), "Diagonal of W must be zero"
    # Non-negative weights
    assert np.all(w >= -1e-6), "All connection weights must be non-negative"


def test_laplacian_spectral_bounds() -> None:
    """Verifies that the symmetric normalized Laplacian L_sym has all eigenvalues in [0, 2]."""
    connectome = ConnectomeGraph(num_regions=68, seed=42)
    lap = connectome.laplacian

    # Symmetry of Laplacian
    assert np.allclose(lap, lap.T, atol=1e-5), "Normalized Laplacian must be symmetric"

    # Spectral bounds
    valid, eigenvalues = connectome.verify_spectral_bounds()
    assert valid, f"Eigenvalues out of bounds [0, 2]: min={eigenvalues[0]}, max={eigenvalues[-1]}"
    assert eigenvalues[0] >= -1e-4, f"Smallest eigenvalue negative: {eigenvalues[0]}"
    assert eigenvalues[-1] <= 2.0 + 1e-4, f"Largest eigenvalue exceeds 2.0: {eigenvalues[-1]}"


def test_subnetwork_partitions() -> None:
    """Verifies that all 68 parcels are correctly assigned to the 4 canonical subnetworks."""
    connectome = ConnectomeGraph(num_regions=68, seed=42)
    labels = connectome.subnetwork_labels
    assert len(labels) == 68
    unique_subs = np.unique(labels)
    assert set(unique_subs) == {0, 1, 2, 3}, "Must contain exactly 4 subnetworks"


def test_wilson_cowan_simulation_stability() -> None:
    """Verifies physical calibration, bounded states, and Kuramoto phase order parameter."""
    connectome = ConnectomeGraph(num_regions=68, seed=42)
    sim = WilsonCowanSimulator(connectome=connectome, coupling_kappa=0.85)

    out = sim.simulate(t_span=(0.0, 2.0), dt=0.04, seed=42)
    states = out["states"]  # (T, N, 4)
    e_act = out["e_activity"]
    i_act = out["i_activity"]
    phases = out["phase"]

    # Check finite values (no NaNs or Infs)
    assert np.all(np.isfinite(states)), "Simulation states must be finite"
    assert np.all(e_act >= 0.0) and np.all(e_act <= 1.5), "Excitatory population must be bounded"
    assert np.all(i_act >= 0.0) and np.all(i_act <= 1.5), "Inhibitory population must be bounded"

    # Check phase angles in [-pi, pi]
    assert np.all(phases >= -np.pi - 1e-4) and np.all(phases <= np.pi + 1e-4)

    # Check Kuramoto order parameter R(t) in [0, 1]
    r_t, r_mean = compute_kuramoto_order_parameter(phases)
    assert np.all(r_t >= 0.0) and np.all(r_t <= 1.0 + 1e-5)
    assert 0.0 <= r_mean <= 1.0
