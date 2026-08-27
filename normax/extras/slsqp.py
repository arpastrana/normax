# Copyright 2026 Rafael Pastrana
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SLSQP over the design problem, as an oracle for the augmented Lagrangian.

The constrained solver holds every row itself, so it needs the slack's
Jacobian — one forward tangent per variable through the whole pipeline, where
the augmented route pays one reverse pass. It is kept for its convergence
certificate, and answers in the same container so the two compare like for
like.
"""

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Float
from scipy.optimize import minimize

from normax.design import DesignProblem
from normax.design import bound_variables
from normax.design import design_maps
from normax.design import evaluate_constraints
from normax.design import expand_variables
from normax.optimization import OptimizationAnswer
from normax.optimization import measure_violation

# Violation a trial point is charged when its frame cannot be factorized.
RECOIL_SLACK = 1e3


class SlsqpBudget(NamedTuple):
    """
    What an SLSQP descent may spend, and when it stops.

    Attributes
    ----------
    iterations :
        Most iterations in each round.
    rounds :
        Most restarts from the previous round's answer.
    tolerance :
        The solver's own stopping tolerance, read against the objective
        divided by its value at the start.
    """

    iterations: int
    rounds: int
    tolerance: float


def slack_rows(
    problem: DesignProblem,
) -> Callable[[Float[Array, "variables"]], Float[Array, "constraints"]]:
    """
    The inequality rows of a problem as an uncompiled function of the variables.

    Parameters
    ----------
    problem :
        The problem supplying the pipeline, the loads and the constraints.

    Returns
    -------
    slack :
        How far above zero every row sits, ready to be differentiated forward.
    """

    def slack(x: Float[Array, "variables"]) -> Float[Array, "constraints"]:
        params = expand_variables(problem, x)
        design = problem.pipeline(params, problem.loads)

        return evaluate_constraints(problem, params, design)

    return slack


def descend_slsqp(
    problem: DesignProblem,
    start: Float[np.ndarray, "variables"],
    budget: SlsqpBudget,
) -> OptimizationAnswer:
    """
    SLSQP under the rows and the bounds, restarted from its answer until quiet.

    Parameters
    ----------
    problem :
        The problem to descend.
    start :
        The variable vector to leave from.
    budget :
        Iterations per round, rounds, and the solver tolerance.

    Returns
    -------
    answer :
        The variables, the mass and violation at the start and at every
        round's answer, the evaluations spent, and whether the last round
        reported clean convergence.

    Notes
    -----
    The objective is divided by its value at the start, so the tolerance is
    relative. A trial point whose frame raises is charged a uniform enormous
    violation and the merit function walks the search back; accepted iterates
    never sit there, so the Jacobian stays unguarded.
    """
    maps = design_maps(problem)
    slack = slack_rows(problem)
    jacobian = jax.jit(jax.jacfwd(slack))
    boxes = bound_variables(problem)

    x = np.asarray(start, dtype=np.float64)
    reference = abs(float(maps.objective(jnp.asarray(x))[0])) or 1.0

    def scaled_objective(z):
        value, slope = maps.objective(jnp.asarray(z))
        scaled = float(value) / reference
        slope_scaled = np.asarray(slope, dtype=np.float64) / reference

        return scaled, slope_scaled

    def guarded_slack(z):
        try:
            return np.asarray(maps.slack(jnp.asarray(z)), dtype=np.float64)
        except (ValueError, FloatingPointError, RuntimeError):
            return np.full(rows, -RECOIL_SLACK)

    def slack_jacobian(z):
        return np.asarray(jacobian(jnp.asarray(z)), dtype=np.float64)

    violation, read = measure_violation(maps.slack, x)
    rows = read.size
    objectives = [reference]
    violations = [violation]

    held = {"type": "ineq", "fun": guarded_slack, "jac": slack_jacobian}
    options = {"maxiter": budget.iterations_warmup, "ftol": budget.violation_tol}

    spent = 0
    converged = False
    for _ in range(budget.rounds_max):
        found = minimize(
            scaled_objective,
            x,
            jac=True,
            method="SLSQP",
            bounds=boxes,
            constraints=[held],
            options=options,
        )
        x = np.asarray(found.x, dtype=np.float64)
        spent += int(found.nfev)
        converged = found.status == 0

        violation, _ = measure_violation(maps.slack, x)
        objectives.append(abs(float(maps.objective(jnp.asarray(x))[0])))
        violations.append(violation)
        if found.nit <= 1:
            break

    answer = OptimizationAnswer(
        x, np.asarray(objectives), np.asarray(violations), spent, converged
    )

    return answer
