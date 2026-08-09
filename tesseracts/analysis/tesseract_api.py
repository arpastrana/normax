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
Form finding is pin-jointed and can only report an axial force; the check
downstream consumes moments, and this is where they come from.

**This schema is the swappable one, and it is frozen.** The whole submission
turns on one interface serving two solvers that disagree about how they
differentiate — a JAX frame solver that is traced end to end, and a C++ solver
whose adjoints were hand-derived element by element over two decades and which
nothing about JAX can see into. Adding a field here is a cost paid by every
backend, so the inputs are the smallest set that describes a frame and the
differentiable ones are exactly the two a direct differentiation backend can
supply: the coordinates and the diameters.

The critical load factor of the whole frame is deliberately **not** here. It is
soft validation, it sizes nothing, and putting it in the schema would oblige
every backend to produce one. `normax.pipeline.stability` reads it beside a
finished design instead.

Set `NORMAX_ANALYSIS_BACKEND` to choose a backend. Only `smax` exists today; the
OpenSees backend arrives behind this same schema and changes nothing above it.
"""

import os
from typing import Any

import jax
import jax.numpy as jnp
from pydantic import BaseModel
from tesseract_core.runtime import Array
from tesseract_core.runtime import Differentiable
from tesseract_core.runtime import Float64
from tesseract_core.runtime import Int64

jax.config.update("jax_enable_x64", True)

BACKEND = os.environ.get("NORMAX_ANALYSIS_BACKEND", "smax")


class InputSchema(BaseModel):
    """
    A frame: where its nodes are, what its members are, and what pushes on it.
    """

    xyz: Differentiable[Array[(None, 3), Float64]]
    """Position of every node, in millimetres. From form finding."""

    diameter: Differentiable[Array[(None,), Float64]]
    """Outer diameter of every member, in millimetres.

    The coupling with the check downstream is staggered: sizing needs forces and
    forces need sizes, so this is the previous outer iterate. One pass is taken,
    not a fixed point, and what that costs is measured rather than assumed.
    """

    edges: Array[(None, 2), Int64]
    """The two node indices spanned by every member."""

    supports: Array[(None,), Int64]
    """Indices of the nodes whose translation is restrained."""

    loads: Array[(None, 3), Float64]
    """Force applied at every node, in newtons. One load case per call."""

    f_y: Float64
    """Yield strength, in newtons per square millimetre."""

    e_mod: Float64
    """Modulus of elasticity, in newtons per square millimetre."""

    density: Float64
    """Density, in tonnes per cubic millimetre."""

    ratio: Float64
    """Diameter-to-thickness ratio, fixing the wall of every member."""

    normal: int | None = None
    """Index of the global axis a planar frame has no thickness along.

    None for a frame that occupies all three dimensions. Static. A planar frame
    on pinned supports alone is a mechanism in a three-dimensional solver, since
    rotating it about the line joining its supports strains nothing.
    """


class OutputSchema(BaseModel):
    """
    What every member carries.
    """

    n_ed: Differentiable[Array[(None,), Float64]]
    """Axial force of every member, in newtons. Tension positive.

    One number per member: loads are applied at nodes alone, so nothing varies
    along a span, and the analysis is linear.
    """

    m_y_ed: Differentiable[Array[(None, 2), Float64]]
    """Major-axis moment at each end of every member, in newton-millimetres."""

    m_z_ed: Differentiable[Array[(None, 2), Float64]]
    """Minor-axis moment at each end of every member, in newton-millimetres.

    Both ends rather than a peak, because nodal loads leave the moment varying
    linearly in between. That is what makes the first row of EN 1993-1-1 Table
    B.3 exact downstream instead of approximate, and it is why a peak would be a
    lossy contract rather than a convenient one.
    """


def _forward(inputs: dict[str, Any]) -> dict[str, jnp.ndarray]:
    """
    Analyse the frame with whichever backend is selected.

    Parameters
    ----------
    inputs :
        The validated input fields.

    Returns
    -------
    outputs :
        Axial force and both end moments of every member.

    Raises
    ------
    ValueError
        If the selected backend does not exist.
    """
    if BACKEND == "smax":
        from _backend_smax import solve  # noqa: PLC0415
    else:
        raise ValueError(f"unknown analysis backend {BACKEND!r}")

    return solve(inputs)


def apply(inputs: InputSchema) -> OutputSchema:
    """
    Analyse the frame.

    Parameters
    ----------
    inputs :
        The geometry, the sections, the topology and the load case.

    Returns
    -------
    outputs :
        The internal forces of every member.
    """
    return _forward(inputs.model_dump())


def abstract_eval(abstract_inputs):
    """
    Output shapes and dtypes, without analysing anything.

    Parameters
    ----------
    abstract_inputs :
        The input fields, arrays replaced by their shape and dtype.

    Returns
    -------
    outputs :
        A shape and a dtype for every output field.

    Notes
    -----
    Required by Tesseract-JAX: JAX resolves shapes before it executes anything,
    so every endpoint below is unreachable without this one.
    """
    members = abstract_inputs.edges.shape[0]

    return {
        "n_ed": {"shape": (members,), "dtype": "float64"},
        "m_y_ed": {"shape": (members, 2), "dtype": "float64"},
        "m_z_ed": {"shape": (members, 2), "dtype": "float64"},
    }


def _differentiate(
    inputs: InputSchema,
    wrt: list[str],
    outputs: list[str],
) -> tuple[Any, list[Any]]:
    """
    The map restricted to the requested inputs and outputs, and its primals.

    Parameters
    ----------
    inputs :
        The geometry, the sections, the topology and the load case.
    wrt :
        Names of the input fields a derivative is taken with respect to.
    outputs :
        Names of the output fields a derivative is taken of.

    Returns
    -------
    restricted :
        The restricted map and the primal values of the requested inputs.
    """
    raw = inputs.model_dump()
    static = {name: value for name, value in raw.items() if name not in wrt}

    def restricted(*values):
        merged = {**static, **dict(zip(wrt, values))}
        computed = _forward(merged)

        return {name: computed[name] for name in outputs}

    return restricted, [jnp.asarray(raw[name]) for name in wrt]


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
    What `jax.grad` calls. A traced backend answers in one reverse pass whatever
    the number of coordinates; a direct differentiation backend is forward-mode
    by nature and has to assemble the same answer column by column, at a cost
    that grows with the parameter count. Both satisfy this endpoint, and how
    differently they pay for it is a result rather than an implementation
    detail.
    """
    restricted, primals = _differentiate(inputs, vjp_inputs, vjp_outputs)

    _, pullback = jax.vjp(restricted, *primals)
    cotangents = pullback(
        {name: jnp.asarray(value) for name, value in cotangent_vector.items()}
    )

    return dict(zip(vjp_inputs, cotangents))


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: list[str],
    jvp_outputs: list[str],
    tangent_vector: dict[str, Any],
):
    """
    Push a tangent on the inputs forward to the outputs.

    Parameters
    ----------
    inputs :
        The geometry, the sections, the topology and the load case.
    jvp_inputs :
        Names of the input fields a derivative is taken with respect to.
    jvp_outputs :
        Names of the output fields a derivative is taken of.
    tangent_vector :
        Tangent on each of those inputs.

    Returns
    -------
    tangents :
        Tangent on each of the requested outputs.

    Notes
    -----
    Never reached by `jax.grad`, and the natural mode for a direct
    differentiation backend, which computes a tangent per parameter directly.
    The two backends therefore meet here first, before the reverse rule.
    """
    restricted, primals = _differentiate(inputs, jvp_inputs, jvp_outputs)

    tangents = tuple(jnp.asarray(tangent_vector[name]) for name in jvp_inputs)
    _, pushed = jax.jvp(restricted, tuple(primals), tangents)

    return pushed
