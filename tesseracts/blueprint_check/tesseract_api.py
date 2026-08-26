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
Blueprints' EN 1993-1-1 cross-section check, behind a Tesseract schema.

No JAX anywhere in this module. The check, the bisection and the hand adjoint
are the host functions of `normax.sizing.blueprint`, plain NumPy over a scalar
library with no derivatives of any kind; this module maps the schema's fields
onto them and nothing else. The in-process sizer wraps the same functions, so
the crossed answer is the local one bit for bit.

The schema carries both questions the pipeline asks of a check: what size
these actions demand (`diameter`, `utilization`), and how hard they work the
sizes the caller already owns (`diameter_held` in, `utilization_held` out).
No buckling length crosses — the check would ignore it, and a field the check
ignores invites the belief that it participates.
"""

from typing import Any

import numpy as np
from pydantic import BaseModel
from tesseract_core.runtime import Array
from tesseract_core.runtime import Differentiable
from tesseract_core.runtime import Float64

from normax.sizing.blueprint import HostActions
from normax.sizing.blueprint import HostFamily
from normax.sizing.blueprint import SizeCotangents
from normax.sizing.blueprint import check_cotangents
from normax.sizing.blueprint import check_members
from normax.sizing.blueprint import host_actions
from normax.sizing.blueprint import host_family
from normax.sizing.blueprint import size_cotangents
from normax.sizing.blueprint import size_members

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
    Member actions and the family, one load case per call.
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
    """Smallest diameter the section family offers, in millimeters."""


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
    """Whether the catalogue minimum decided each member's size, as zero or one.

    Non-differentiable: a cotangent on this raises rather than passing quietly.
    """


def _read_family(inputs: dict[str, Any]) -> HostFamily:
    """
    The section family the flat schema scalars describe.
    """
    return host_family(
        float(inputs["ratio"]),
        float(inputs["f_y"]),
        float(inputs["gamma_m0"]),
        float(inputs["diameter_min"]),
    )


def _read_actions(inputs: dict[str, Any]) -> HostActions:
    """
    The member actions the schema arrays describe.
    """
    return host_actions(
        inputs["axial_force"], inputs["end_moments_major"], inputs["end_moments_minor"]
    )


def apply(inputs: InputSchema) -> OutputSchema:
    """
    Run the check.

    Parameters
    ----------
    inputs :
        The member actions and the family.

    Returns
    -------
    outputs :
        The required sizes, both utilizations and the clamp mask.
    """
    raw = inputs.model_dump()
    family = _read_family(raw)
    actions = _read_actions(raw)
    sized = size_members(actions, family)
    utilization_held = check_members(raw["diameter_held"], actions, family)
    outputs = {
        "diameter": sized.diameter,
        "utilization": sized.utilization,
        "utilization_held": utilization_held,
        "clamped": sized.clamped.astype(np.float64),
    }

    return outputs


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
        The member actions and the family.
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
    What `jax.grad` calls, and the only endpoint it calls. The two solve
    outputs and the held check pull back through separate host rules; an
    output not seeded pulls a zero.
    """
    unknown = set(vjp_inputs) - set(DIFFERENTIABLE_INPUTS)
    if "clamped" in vjp_outputs:
        raise ValueError("`clamped` is non-differentiable; drop it before seeding")
    if unknown:
        raise ValueError(f"no hand-written derivative covers {sorted(unknown)}")

    raw = inputs.model_dump()
    family = _read_family(raw)
    actions = _read_actions(raw)
    held = np.asarray(raw["diameter_held"], dtype=np.float64)
    quiet = np.zeros_like(actions.axial)
    seeds = {
        name: np.asarray(cotangent_vector.get(name, quiet), dtype=np.float64)
        for name in DIFFERENTIABLE_OUTPUTS
    }

    sized_seed = SizeCotangents(seeds["diameter"], seeds["utilization"])
    from_sizes = size_cotangents(actions, family, sized_seed)
    from_held = check_cotangents(held, actions, family, seeds["utilization_held"])
    gathered = {
        "axial_force": from_sizes.axial + from_held.actions.axial,
        "end_moments_major": from_sizes.end_major + from_held.actions.end_major,
        "end_moments_minor": from_sizes.end_minor + from_held.actions.end_minor,
        "diameter_held": from_held.diameter_held,
    }

    return {name: gathered[name] for name in vjp_inputs}
