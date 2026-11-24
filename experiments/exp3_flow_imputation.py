"""Experiment 3: Optimal Transport Conditional Flow Matching (OT-CFM) Missing Sensor Imputation & Uncertainty Calibration."""

from typing import Dict
import json
import os
import numpy as np
import torch
import torch.optim as optim

from src.env.connectome_graph import ConnectomeGraph, WilsonCowanSimulator
from src.models.flow_matching import OptimalTransportCFM
from src.metrics.evaluators import compute_trajectory_mse, compute_crps


def run_experiment_3(output_dir: str = "results", quick_run: bool = False) -> Dict:
    """Evaluates OT-CFM missing sensor imputation across random and subnetwork dropouts."""
    os.makedirs(output_dir, exist_ok=True)
    device = "cpu"
    torch.manual_seed(42)
    np.random.seed(42)

    connectome = ConnectomeGraph(num_regions=68, seed=42)
    laplacian = torch.from_numpy(connectome.laplacian).float().to(device)

    # Generate trajectories
    data = WilsonCowanSimulator.load_or_generate_dataset(
        num_samples=16 if not quick_run else 4,
        num_regions=68,
        t_span=(0.0, 3.0),
        dt=0.05,
        seed=42,
    )
    trajectories = data["trajectories"].to(device)  # (B, T, N, d)

    # Flatten time into samples for flow matching frame-by-frame distribution learning
    B, T, N, d = trajectories.shape
    all_frames = trajectories.reshape(B * T, N, d)
    split_idx = int(0.8 * len(all_frames))
    train_frames = all_frames[:split_idx]
    test_frames = all_frames[split_idx:]

    cfm_model = OptimalTransportCFM(
        node_dim=d,
        hidden_dim=32,
        time_embed_dim=16,
        num_layers=3,
        spectral_orders=3,
    ).to(device)

    optimizer = optim.Adam(cfm_model.parameters(), lr=0.005)
    epochs = 15 if not quick_run else 4
    batch_size = 16

    print(f"[Experiment 3] Training OT-CFM model ({epochs} epochs)...")
    for epoch in range(epochs):
        perm = torch.randperm(len(train_frames))
        total_loss = 0.0
        batches = 0

        for i in range(0, min(len(train_frames), 64), batch_size):
            batch_x1 = train_frames[perm[i : i + batch_size]]
            # Random masking condition during training
            rand_mask = (torch.rand(len(batch_x1), N, 1, device=device) > 0.3).float()
            x_obs = batch_x1 * rand_mask

            optimizer.zero_grad()
            loss = cfm_model.compute_loss(batch_x1, x_obs, laplacian, mask=rand_mask)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

    # Evaluation scenarios
    scenarios = {
        "random_mask_25pct": 0.25,
        "random_mask_50pct": 0.50,
        "frontoparietal_network_dropout": "subnetwork",
    }
    eval_results = {}
    test_subset = test_frames[:10 if not quick_run else 2]

    for sc_name, sc_param in scenarios.items():
        if sc_param == "subnetwork":
            # Mask Frontoparietal network (indices 51 to 67)
            mask = torch.ones((len(test_subset), N, 1), device=device)
            mask[:, 51:68, :] = 0.0
        else:
            prob = float(sc_param)
            mask = (torch.rand(len(test_subset), N, 1, device=device) > prob).float()

        x_partial = test_subset * mask

        # Multi-sample imputation with uncertainty
        mu_pred, std_pred = cfm_model.impute_missing_channels(
            x_partial=x_partial,
            mask=mask,
            laplacian=laplacian,
            steps=10 if not quick_run else 5,
        )

        impute_mse = compute_trajectory_mse(test_subset, mu_pred)

        # Evaluate missing nodes only
        missing_mask = (mask == 0.0).expand_as(test_subset)
        if missing_mask.sum() > 0:
            missing_mse = float(torch.mean((test_subset[missing_mask] - mu_pred[missing_mask]) ** 2).item())
        else:
            missing_mse = impute_mse

        # Calibration: fraction of true values within 95% CI [mu - 1.96*std, mu + 1.96*std]
        lower_bound = mu_pred - 1.96 * torch.clamp(std_pred, min=1e-4)
        upper_bound = mu_pred + 1.96 * torch.clamp(std_pred, min=1e-4)
        in_ci = (test_subset >= lower_bound) & (test_subset <= upper_bound)
        ci_coverage = float(torch.mean(in_ci.float()).item())

        # CRPS calculation
        pseudo_ensemble = torch.stack([
            mu_pred + std_pred * torch.randn_like(mu_pred) for _ in range(5)
        ], dim=0)
        crps = compute_crps(test_subset, pseudo_ensemble)

        eval_results[sc_name] = {
            "overall_mse": impute_mse,
            "missing_channels_mse": missing_mse,
            "ci_95_coverage": ci_coverage,
            "crps": crps,
        }

    results = {
        "experiment": "exp3_flow_imputation",
        "eval_scenarios": eval_results,
    }

    out_file = os.path.join(output_dir, "exp3_flow_imputation.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[Experiment 3] Completed. Imputation metrics saved to {out_file}")
    return results


if __name__ == "__main__":
    run_experiment_3()
