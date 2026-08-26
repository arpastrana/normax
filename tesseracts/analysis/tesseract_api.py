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
T2 — Frame analysis of a form-found geometry.

A geometry and a set of sections in, the internal forces the members carry out.
Form finding is pin-jointed and reports an axial force alone; the check
downstream consumes moments, and this is where they come from.

**This schema is the swappable one.** One interface serves two solvers that
disagree about how they differentiate — a C++ solver with sensitivities compiled
into it, and a Python solver with none, given an adjoint by hand — and the
differentiable inputs are exactly the two both can supply: the coordinates and
the diameters. `NORMAX_ANALYSIS_BACKEND` names the solver, `opensees` in two
dimensions or `pynite` in three.
"""

import os
from typing import Any

import jax
import numpy as np
from _backend_common import force_outputs
from _backend_common import read_cotangent
from _backend_common import read_frame
from pydantic import BaseModel
from tesseract_core.runtime import Array
from tesseract_core.runtime import Differentiable
from tesseract_core.runtime import Float64
from tesseract_core.runtime import Int64

jax.config.update("jax_enable_x64", True)

# Environment variable naming the backend, and the one used when it is unset.
BACKEND_VARIABLE = "NORMAX_ANALYSIS_BACKEND"
BACKEND_DEFAULT = "pynite"


class InputSchema(BaseModel):
    """
    A frame: where its nodes are, what its members are, and what pushes on it.
    """

    xyz: Differentiable[Array[(None, 3), Float64]]
    """Position of every node, in millimeters. From form finding."""

    diameter: Differentiable[Array[(None,), Float64]]
    """Outer diameter of every member, in millimeters."""

    edges: Array[(None, 2), Int64]
    """The two node indices spanned by every member."""

    supports: Array[(None,), Int64]
    """Indices of the nodes whose translation is restrained."""

    loads: Array[(None, 3), Float64]
    """Force applied at every node, in newtons. One load case per call."""

    f_y: Float64
    """Yield strength, in newtons per square millimeter."""

    e_mod: Float64
    """Modulus of elasticity, in newtons per square millimeter."""

    density: Float64
    """Density, in tonnes per cubic millimeter."""

    ratio: Float64
    """Diameter-to-thickness ratio, fixing the wall of every member."""

    normal: int | None = None
    """Index of the global axis a planar frame has no thickness along.

    None for a frame that occupies all three dimensions. Read by the planar
    backend alone, whose two kept axes become the solver's own.
    """


class OutputSchema(BaseModel):
    """
    What every member carries.
    """

    axial_force: Differentiable[Array[(None,), Float64]]
    """Axial force of every member, in newtons. Tension positive."""

    end_moments_major: Differentiable[Array[(None, 2), Float64]]
    """Major-axis moment at each end of every member, in newton-millimeters."""

    end_moments_minor: Differentiable[Array[(None, 2), Float64]]
    """Minor-axis moment at each end of every member, in newton-millimeters.

    Both ends rather than a peak, because nodal loads leave the moment linear
    in between, which makes the first row of EN 1993-1-1 Table B.3 exact.
    """


def _selected_backend() -> Any:
    """
    The module implementing the selected backend.

    Raises
    ------
    ValueError
        If the selected backend does not exist.

    Notes
    -----
    Imported on use, so an image shipping one backend's dependencies still reads
    the schema, and read per call, so one process can compare the two. A backend
    owns its derivative as well as its forward pass.
    """
    selected = os.environ.get(BACKEND_VARIABLE, BACKEND_DEFAULT)

    if selected == "opensees":
        import _backend_opensees as backend  # noqa: PLC0415
    elif selected == "pynite":
        import _backend_pynite as backend  # noqa: PLC0415
    else:
        raise ValueError(f"unknown analysis backend {selected!r}")

    return backend


def apply(inputs: InputSchema) -> OutputSchema:
    """
    Analyze the frame.
    """
    frame = read_frame(inputs.model_dump())
    forces = _selected_backend().solve_forces(frame)

    return force_outputs(forces)


def abstract_eval(abstract_inputs):
    """
    Output shapes and dtypes, without analyzing anything.
    """
    members = abstract_inputs.edges.shape[0]

    return {
        "axial_force": {"shape": (members,), "dtype": "float64"},
        "end_moments_major": {"shape": (members, 2), "dtype": "float64"},
        "end_moments_minor": {"shape": (members, 2), "dtype": "float64"},
    }


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: list[str],
    vjp_outputs: list[str],
    cotangent_vector: dict[str, Any],
):
    """
    Pull a cotangent on the outputs back to the inputs.

    Parameters
    ----------
    inputs :
        The geometry, the sections, the topology and the load case.
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

    Notes
    -----
    What `jax.grad` calls. Both backends satisfy it, and how differently they
    pay for it is a result rather than an implementation detail.
    """
    frame = read_frame(inputs.model_dump())
    cotangent = read_cotangent(cotangent_vector, frame.structure.num_edges)
    pulled = _selected_backend().forces_vjp(frame, cotangent)

    return {name: np.asarray(pulled[name]) for name in vjp_inputs}
