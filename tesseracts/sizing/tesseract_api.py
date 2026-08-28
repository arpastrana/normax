# SPDX-License-Identifier: Apache-2.0
"""
T3 — The cross-section check, behind one swappable schema.

Member actions in, the sizes the check demands and the utilizations it reads
off them, one load case per call. No JAX anywhere in this module: a backend
hosts a scalar library with no derivatives of any kind and answers the
pullback with a hand-written adjoint.

**This schema is the swappable one.** It carries both questions the pipeline
asks of a check: what size these actions demand (`diameter`, `utilization`),
and how hard they work the sizes the caller already owns (`diameter_held` in,
`utilization_held` out). The `backend` input names the check, `blueprint`
being the one that ships, and is read statically as `solve` is. No buckling
length crosses — the shipped check would ignore it, and a field a check
ignores invites the belief that it participates; the wire widens when a
backend that reads one exists.
"""

from typing import Any

import numpy as np
from pydantic import BaseModel
from tesseract_core.runtime import Array
from tesseract_core.runtime import Differentiable
from tesseract_core.runtime import Float64

# The check a call gets when it names none.
BACKEND_DEFAULT = "blueprint"

# The inputs the hand adjoint covers, and the outputs it can be seeded on.
DIFFERENTIABLE_INPUTS = (
    "axial_force",
    "end_moments_major",
    "end_moments_minor",
    "diameter_held",
)
DIFFERENTIABLE_OUTPUTS = ("diameter", "utilization", "utilization_held")


class InputSchema(BaseModel):
    """
    Member actions and the catalog, one load case per call.
    """

    axial_force: Differentiable[Array[(None,), Float64]]
    """Design axial force of every member, in newtons. Tension positive."""

    end_moments_major: Differentiable[Array[(None, 2), Float64]]
    """Major-axis moment at each end of every member, in newton-millimeters."""

    end_moments_minor: Differentiable[Array[(None, 2), Float64]]
    """Minor-axis moment at each end of every member, in newton-millimeters."""

    diameter_held: Differentiable[Array[(None,), Float64]]
    """Outer diameter every member is checked at, in millimeters.

    Read only by the held check: the solve never consults it.
    """

    f_y: Float64
    """Yield strength, in newtons per square millimeter. Not differentiable."""

    gamma_m0: Float64
    """Partial factor for cross-section resistance."""

    ratio: Float64
    """Diameter-to-thickness ratio, fixing the wall."""

    diameter_min: Float64
    """Smallest diameter the section catalog offers, in millimeters."""

    backend: str = BACKEND_DEFAULT
    """Which check answers the stage, `blueprint`.

    Read statically, as `solve` is. A schema ordinarily says what a stage
    computes rather than who computes it; carrying the choice per call is what
    lets one process hold two checks and compare them.
    """

    solve: bool = True
    """Whether to run the sizing solve, or only the held check.

    Read statically. False skips the bisection and echoes the held size and
    its utilization through `diameter` and `utilization`, for the caller who
    asked nothing about sizes; the adjoint of the echo is the held check's.
    """


class OutputSchema(BaseModel):
    """
    The sizes the check requires, and how hard they work.
    """

    diameter: Differentiable[Array[(None,), Float64]]
    """Outer diameter of every member, in millimeters, floored at the minimum."""

    utilization: Differentiable[Array[(None,), Float64]]
    """Demand over resistance of every member, at the size just chosen."""

    utilization_held: Differentiable[Array[(None,), Float64]]
    """Demand over resistance of every member, at the held size that crossed."""

    clamped: Array[(None,), Float64]
    """Whether the catalog minimum decided each member's size, as zero or one.

    Non-differentiable: a cotangent on this raises rather than passing quietly.
    """


def _selected_backend(selected: str) -> Any:
    """
    The module implementing the selected backend.

    Parameters
    ----------
    selected :
        Which check answers the stage, as the call named it.

    Returns
    -------
    backend :
        The module owning that check's forward pass and its derivative.

    Raises
    ------
    ValueError
        If the selected backend does not exist.

    Notes
    -----
    Imported on use, so an image shipping one backend's dependencies still
    reads the schema. The name arrives per call, so two callers in one process
    each get the check they asked for.
    """
    if selected == "blueprint":
        import _backend_blueprint as backend  # noqa: PLC0415
    else:
        raise ValueError(f"unknown sizing backend {selected!r}")

    return backend


def apply(inputs: InputSchema) -> OutputSchema:
    """
    Run the check.

    Parameters
    ----------
    inputs :
        The member actions and the catalog.

    Returns
    -------
    outputs :
        The required sizes, both utilizations and the clamp mask.
    """
    raw = inputs.model_dump()

    return _selected_backend(inputs.backend).solve_sizes(raw)


def abstract_eval(abstract_inputs):
    """
    Output shapes and dtypes, without sizing anything.

    Parameters
    ----------
    abstract_inputs :
        The input fields, arrays replaced by their shape and dtype.

    Returns
    -------
    outputs :
        A shape and a dtype for every output field.
    """
    members = abstract_inputs.axial_force.shape[0]
    promised = {"shape": (members,), "dtype": "float64"}

    return {name: promised for name in (*DIFFERENTIABLE_OUTPUTS, "clamped")}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: list[str],
    vjp_outputs: list[str],
    cotangent_vector: dict[str, Any],
):
    """
    Pull a cotangent on the outputs back to the inputs, by hand.

    Parameters
    ----------
    inputs :
        The member actions and the catalog.
    vjp_inputs :
        Names of the input fields a derivative is taken with respect to.
    vjp_outputs :
        Names of the output fields a derivative is taken of.
    cotangent_vector :
        Cotangent on each of those outputs.

    Returns
    -------
    cotangents :
        Cotangent on each of the requested inputs.

    Raises
    ------
    ValueError
        If the clamp mask is seeded, or an input has no derivative rule.

    Notes
    -----
    What `jax.grad` calls, and the only endpoint it calls. An output not
    seeded pulls a zero, and the backend owns every rule.
    """
    unknown = set(vjp_inputs) - set(DIFFERENTIABLE_INPUTS)
    if "clamped" in vjp_outputs:
        raise ValueError("`clamped` is non-differentiable; drop it before seeding")
    if unknown:
        raise ValueError(f"no hand-written derivative covers {sorted(unknown)}")

    raw = inputs.model_dump()
    quiet = np.zeros_like(np.asarray(raw["axial_force"], dtype=np.float64))
    seeds = {
        name: np.asarray(cotangent_vector.get(name, quiet), dtype=np.float64)
        for name in DIFFERENTIABLE_OUTPUTS
    }
    pulled = _selected_backend(inputs.backend).sizes_vjp(raw, seeds)

    return {name: pulled[name] for name in vjp_inputs}
