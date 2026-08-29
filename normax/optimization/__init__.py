# SPDX-License-Identifier: Apache-2.0
"""
The searches a design is found by, and the one the package ships.

`auglag` is the method: constrained minimization in box bounds, knowing
nothing of what a design is, and re-exported here because that is what the
shipped search descends with. `nested` is the route it displaced, kept for
the comparison and reached by its own name.
"""

from normax.optimization.auglag import ConstrainedMaps
from normax.optimization.auglag import OptimizationBudget
from normax.optimization.auglag import OptimizationSolution
from normax.optimization.auglag import compute_penalty
from normax.optimization.auglag import measure_violation
from normax.optimization.auglag import optimize_augmented_lagrangian
from normax.optimization.auglag import recoil_point_to_last_good
from normax.optimization.auglag import update_multipliers

__all__ = [
    "ConstrainedMaps",
    "OptimizationSolution",
    "OptimizationBudget",
    "compute_penalty",
    "measure_violation",
    "optimize_augmented_lagrangian",
    "recoil_point_to_last_good",
    "update_multipliers",
]
