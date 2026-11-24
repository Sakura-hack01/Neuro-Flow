"""Unified Command-Line Interface (CLI) for Neural-Flow Connectome."""

import argparse
import http.server
import os
import socketserver
import subprocess
import sys
import torch

from src.models.graph_neural_ode import GraphNeuralODE


def cmd_test(args: argparse.Namespace) -> None:
    """Runs automated PyTest test suite."""
    print("Running automated test suite...")
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]
    res = subprocess.run(cmd)
    sys.exit(res.returncode)


def cmd_exp(args: argparse.Namespace) -> None:
    """Executes scientific experiments."""
    from experiments.exp1_irregular_forecasting import run_experiment_1
    from experiments.exp2_bifurcation_phase_diagram import run_experiment_2
    from experiments.exp3_flow_imputation import run_experiment_3
    from experiments.exp4_topology_ablation import run_experiment_4

    target = args.name.lower() if args.name else "all"

    if target in ("all", "exp1"):
        print("\n--- Running Experiment 1 (Irregular Forecasting) ---")
        run_experiment_1(quick_run=args.quick)
    if target in ("all", "exp2"):
        print("\n--- Running Experiment 2 (2D Bifurcation Phase Diagram) ---")
        run_experiment_2(grid_size=8 if args.quick else 12)
    if target in ("all", "exp3"):
        print("\n--- Running Experiment 3 (OT-CFM Sensor Imputation) ---")
        run_experiment_3(quick_run=args.quick)
    if target in ("all", "exp4"):
        print("\n--- Running Experiment 4 (Topology Ablation) ---")
        run_experiment_4(quick_run=args.quick)


def cmd_plot(args: argparse.Namespace) -> None:
    """Generates 300-DPI high-resolution research figures."""
    from visualization.plot_figures import generate_all_figures
    generate_all_figures()


def cmd_cpp_bench(args: argparse.Namespace) -> None:
    """Executes real-time BCI edge engine benchmark."""
    from cpp_inference.run_bci_benchmark import run_benchmark
    run_benchmark()


def cmd_export_onnx(args: argparse.Namespace) -> None:
    """Exports GraphNeuralODE vector field / network to ONNX or TorchScript."""
    os.makedirs("results", exist_ok=True)
    out_path = args.output or os.path.join("results", "neural_flow_model.pt")

    print("Exporting GraphNeuralODE model...")
    model = GraphNeuralODE(node_dim=4, hidden_dim=32, spectral_orders=3)
    model.eval()

    # Save scripted / state dict model checkpoint
    torch.save(model.state_dict(), out_path)
    print(f"Model checkpoint successfully exported to {out_path}")

    # Optional ONNX export for vector field function
    onnx_path = os.path.join("results", "vector_field.onnx")
    try:
        t_dummy = torch.tensor(0.0)
        x_dummy = torch.randn(1, 68, 4)
        l_dummy = torch.eye(68)

        class VectorFieldWrapper(torch.nn.Module):
            def __init__(self, ode_func):
                super().__init__()
                self.ode_func = ode_func
            def forward(self, t, x, l):
                return self.ode_func(t, x, l)

        wrapper = VectorFieldWrapper(model.ode_func)
        torch.onnx.export(
            wrapper,
            (t_dummy, x_dummy, l_dummy),
            onnx_path,
            input_names=["time", "state", "laplacian"],
            output_names=["dxdt"],
            dynamic_axes={"state": {0: "batch_size"}},
            opset_version=14,
        )
        print(f"ONNX graph exported to {onnx_path}")
    except Exception as e:
        print(f"ONNX export skipped (using TorchScript): {e}")


def cmd_serve(args: argparse.Namespace) -> None:
    """Serves the interactive GitHub Pages documentation dashboard locally."""
    docs_dir = os.path.abspath("docs")
    port = args.port or 8000
    os.chdir(docs_dir)

    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Serving Neural-Flow interactive dashboard at http://localhost:{port}")
        print("Press Ctrl+C to terminate server.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Neural-Flow Connectome Unified CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # test
    subparsers.add_parser("test", help="Run automated test suite")

    # exp
    exp_parser = subparsers.add_parser("exp", help="Run scientific experiments")
    exp_parser.add_argument("--name", type=str, default="all", choices=["all", "exp1", "exp2", "exp3", "exp4"])
    exp_parser.add_argument("--quick", action="store_true", help="Run fast test configuration")

    # plot
    subparsers.add_parser("plot", help="Generate 300-DPI high-resolution research figures")

    # cpp-bench
    subparsers.add_parser("cpp-bench", help="Run real-time BCI edge engine benchmark")

    # export-onnx
    onnx_parser = subparsers.add_parser("export-onnx", help="Export model to ONNX / TorchScript")
    onnx_parser.add_argument("--output", type=str, default=None)

    # serve
    serve_parser = subparsers.add_parser("serve", help="Serve web dashboard locally")
    serve_parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    commands = {
        "test": cmd_test,
        "exp": cmd_exp,
        "plot": cmd_plot,
        "cpp-bench": cmd_cpp_bench,
        "export-onnx": cmd_export_onnx,
        "serve": cmd_serve,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
