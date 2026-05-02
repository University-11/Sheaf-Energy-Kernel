"""Pure-Python reference implementation for the Sheaf Energy Kernel."""

from .reference import (
    EnergyBreakdown,
    SheafGraph,
    edge_restriction_grad,
    energy,
    equiprop_gradient_R,
    equiprop_update_R,
    run_to_equilibrium,
    sheaf_sweep,
    soft_threshold,
)

__all__ = [
    "EnergyBreakdown",
    "SheafGraph",
    "edge_restriction_grad",
    "energy",
    "equiprop_gradient_R",
    "equiprop_update_R",
    "run_to_equilibrium",
    "sheaf_sweep",
    "soft_threshold",
]

