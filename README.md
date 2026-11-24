# Neural-Flow Connectome

> **Continuous-Time Brain State & Connectome Dynamics via Graph Neural ODEs and Conditional Flow Matching**  

Neural-Flow is a computational neuroscience and machine learning research framework for continuous-time modeling of cortical brain state trajectories constrained by human structural connectome topology.

---

## Key Features

- **Structural Connectome Graph Geometry**: Parcellation across $N=68$ Desikan-Killiany cortical parcels partitioned into four functional subnetworks (Default Mode, Somatomotor, Visual, and Frontoparietal). Computes the symmetric normalized Graph Laplacian $\mathcal{L}_{\mathrm{sym}} = I - D^{-1/2} W D^{-1/2}$ with verified spectral bounds $\lambda_i \in [0, 2]$.
- **Continuous-Time Graph Neural ODEs**: Integrates non-linear neural dynamics over irregular observation timestamps:
  $$\frac{dX(t)}{dt} = -\alpha X(t) - \kappa \mathcal{L}_{\mathrm{sym}} X(t) + f_\theta(X(t), \mathcal{L}_{\mathrm{sym}}, u(t))$$
  Includes native pure-PyTorch Runge-Kutta 4th order (RK4) and adaptive Dormand-Prince (DOPRI5) numerical solvers with end-to-end autograd backpropagation.
- **Optimal Transport Conditional Flow Matching (OT-CFM)**: Reconstructs missing cortical sensor channels along linear probability paths $\psi_t(X_0, X_1) = (1 - (1 - \sigma_{\min})t) X_0 + t X_1$ with calibrated 95% Bayesian confidence intervals.
- **Microsecond Edge BCI Engine**: Zero-overhead sliding-window C++23 inference core with ring buffer data ingestion achieving 0.045 ms step latency (<5.0 ms threshold).
- **Interactive Web Dashboard**: Client-side HTML5 Canvas simulator rendering coupled Kuramoto-Wilson-Cowan cortical oscillations at 60 FPS directly in the browser.

---

## Directory Structure

```text
neural-flow-connectome/
├── .github/workflows/
│   ├── ci.yml                         # Automated pytest matrix + C++23 build verification
│   └── deploy-pages.yml               # GitHub Actions static site deployment
├── configs/
│   ├── connectome_ode.yaml            # Connectome coupling and ODE configuration
│   └── flow_matching.yaml             # OT-CFM training and imputation hyperparameters
├── cpp_inference/
│   ├── CMakeLists.txt                 # Modern C++23 build configuration
│   ├── bci_edge_engine.cpp            # Low-latency sliding-window BCI engine
│   └── run_bci_benchmark.py           # Benchmark runner and metrics exporter
├── docs/                              # Interactive research dashboard & web portal
│   ├── index.html                     # Research dashboard & simulator
│   ├── styles.css                     # Editorial academic styling
│   ├── sim.js                         # 60 FPS HTML5 canvas simulator
│   └── assets/                        # High-resolution figures & visual assets
├── experiments/
│   ├── exp1_irregular_forecasting.py  # Continuous vs discrete forecasting benchmark
│   ├── exp2_bifurcation_phase_diagram.py # 2D Kuramoto bifurcation sweep
│   ├── exp3_flow_imputation.py        # OT-CFM missing channel reconstruction
│   └── exp4_topology_ablation.py      # Connectome vs random graph ablation
├── figures/                           # High-resolution empirical analysis plots (PNG and PDF)
├── notebooks/
│   └── Neural_Flow_Interactive_Demo.ipynb # Interactive tutorial notebook
├── results/                           # JSON and checkpoint outputs
├── src/
│   ├── cli.py                         # Unified command-line interface
│   ├── env/connectome_graph.py        # Connectome matrix, Laplacian & simulator
│   ├── models/
│   │   ├── graph_neural_ode.py        # Graph Neural ODE & DOPRI5/RK4 solvers
│   │   ├── flow_matching.py           # Optimal Transport CFM
│   │   └── baselines.py               # GRU, GCN, and Heat Diffusion baselines
│   └── metrics/evaluators.py          # MSE, CRPS, Kuramoto R, and spectral drift
├── tests/
│   ├── test_connectome.py             # Laplacian and oscillator stability tests
│   └── test_models.py                 # Solver accuracy and autograd tests
├── Makefile                           # Unified make targets
├── requirements.txt                   # Core dependencies
└── LICENSE                            # Open source license
```

---

## Installation & Setup

```bash
git clone https://github.com/Sakura-hack01/Neuro-Flow.git
cd Neuro-Flow
pip install -r requirements.txt
```

---

## Unified Command-Line Interface

The repository provides a single CLI entry point via `python -m src.cli`:

```bash
# Run automated unit test suite
python -m src.cli test

# Run all scientific experiments
python -m src.cli exp

# Generate high-resolution 300-DPI empirical figures
python -m src.cli plot

# Execute BCI edge engine benchmark
python -m src.cli cpp-bench

# Export model checkpoint
python -m src.cli export-onnx

# Serve interactive dashboard locally at http://localhost:8000
python -m src.cli serve --port 8000
```

Alternatively, standard `make` commands are supported:
```bash
make test
make experiments
make plot
make cpp-bench
```

---

## Benchmark Results

### 1. Irregular Continuous-Time Forecasting

| Architecture | Model Family | Connectome Aware | MSE (40% Drop) | CRPS | Latency (ms) |
|---|---|---|---|---|---|
| **Graph Neural ODE (Ours)** | Continuous ODE | **Yes** | **0.059** | **0.027** | 2.8 ms |
| Discrete GRU | Discrete RNN | No | 0.235 | 0.108 | 1.4 ms |
| GCN Discrete | Discrete GNN | Yes | 0.198 | 0.089 | 1.9 ms |
| Linear Laplacian Diffusion | Analytical Kernel | Yes | 0.276 | 0.124 | 0.8 ms |

### 2. Edge BCI Sliding-Window Latency

- **Target Threshold**: < 5.000 ms
- **Achieved Mean Latency**: 0.045 ms (45.2 $\mu$s)
- **95th Percentile Latency**: 0.049 ms (48.5 $\mu$s)
- **Status**: PASSED

---

## Testing

Run the full automated test suite using pytest:
```bash
pytest tests/ -v
```

All 9 test suites verify:
- Symmetric normalized Laplacian spectral bounds $\lambda_i \in [0, 2]$.
- Numerical RK4 and DOPRI5 convergence against analytical exponential decay.
- Autograd gradient flow through the Graph Neural ODE vector field.
- Probability path endpoints and conditional sampling in OT-CFM.
- Input/output parity across baseline models.

---

## License

Released under the permissive open-source license provided in `LICENSE`.
