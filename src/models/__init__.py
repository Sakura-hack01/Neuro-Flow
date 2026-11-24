"""Neural network models for continuous-time connectome dynamics and flow matching."""

from .graph_neural_ode import (
    GraphNeuralODE,
    ConnectomeODEFunc,
    SpectralChebyshevConv,
    rk4_step,
    dopri5_step,
    integrate_adaptive_dopri5,
    odeint_custom,
)
from .flow_matching import (
    OptimalTransportCFM,
    ConnectomeVelocityField,
    SinusoidalTimeEmbedding,
)
from .baselines import (
    DiscreteGRU,
    GCNDiscrete,
    LinearLaplacianDiffusion,
)

__all__ = [
    "GraphNeuralODE",
    "ConnectomeODEFunc",
    "SpectralChebyshevConv",
    "rk4_step",
    "dopri5_step",
    "integrate_adaptive_dopri5",
    "odeint_custom",
    "OptimalTransportCFM",
    "ConnectomeVelocityField",
    "SinusoidalTimeEmbedding",
    "DiscreteGRU",
    "GCNDiscrete",
    "LinearLaplacianDiffusion",
]
