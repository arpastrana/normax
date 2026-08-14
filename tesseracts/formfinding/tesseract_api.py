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
T1 — Force density form finding.

Force densities in, the geometry that carries the loads out. Under the design
load that geometry is funicular, so the network is in pure tension or pure
compression and the shape is the only thing the stage has to say.

Differentiation strategy: tracing autodiff. The equilibrium is linear in the
coordinates once the force densities are fixed, so `jax-fdm` differentiates the
solve by tracing it and no implicit rule is needed. This is the only stage of
the pipeline whose derivatives come from a tracing system, and it is the
baseline the other two are unlike.

**The handoff downstream is a geometry and nothing else** — no prestress, no
initial member forces, and no quantity derived from the coordinates. A frame
solver is handed this and finds its own internal forces, and the agreement
between those and the product of a force density and a length is a prediction
that gets tested rather than an input that gets imposed.

An edge length and an edge force are both recoverable from what does cross: a
length is a distance between two nodes, and a force is that length times a force
density. Sending either would carry arithmetic across the boundary and invite
two stages to do it differently.
"""

from typing import Any

import jax
import jax.numpy as jnp
from pydantic import BaseModel
from tesseract_core.runtime import Array
from tesseract_core.runtime import Differentiable
from tesseract_core.runtime import Float64
from tesseract_core.runtime import Int64

from normax.form_finding.fdm import equilibrium_graph
from normax.form_finding.fdm import equilibrium_state
from normax.structures import Structure

jax.config.update("jax_enable_x64", True)


class InputSchema(BaseModel):
    """
    The force densities, and the topology and loads they act on.
    """

    q: Differentiable[Array[(None,), Float64]]
    """Force density of every edge, in newtons per millimeter.

    The design variable of the whole pipeline. Negative in compression.
    """

    nodes: Array[(None, 3), Float64]
    """Starting position of every node, in millimeters.

    Read only at the supports, whose positions are held. Everywhere else the
    equilibrium discards it.
    """

    edges: Array[(None, 2), Int64]
    """The two node indices spanned by every edge."""

    supports: Array[(None,), Int64]
    """Indices of the nodes whose position is fixed."""

    loads: Array[(None, 3), Float64]
    """Force applied at every node, in newtons. Zero at the supports."""


class OutputSchema(BaseModel):
    """
    The shape that carries the loads.
    """

    xyz: Differentiable[Array[(None, 3), Float64]]
    """Position of every node at equilibrium, in millimeters."""


def _forward_pass(inputs: dict[str, Any]) -> dict[str, jnp.ndarray]:
    """
    Solve for the equilibrium geometry.

    Parameters
    ----------
    inputs :
        The validated input fields.

    Returns
    -------
    outputs :
        The equilibrium geometry.

    Notes
    -----
    The connectivity is rebuilt from the flat arrays on every call, which is
    host-side integer work and never traced. A schema carries arrays and not
    objects, and that is the price of the boundary being real.
    """
    nodes = jnp.asarray(inputs["nodes"])
    structure = Structure(
        nodes=nodes,
        edges=jnp.asarray(inputs["edges"]),
        supports=jnp.asarray(inputs["supports"]),
    )
    graph = equilibrium_graph(structure)

    state = equilibrium_state(
        jnp.asarray(inputs["q"]),
        nodes[graph.indices_fixed],
        graph,
        jnp.asarray(inputs["loads"]),
    )

    return {"xyz": state.xyz}


def apply(inputs: InputSchema) -> OutputSchema:
    """
    Form-find the network.

    Parameters
    ----------
    inputs :
        The force densities, the topology and the loads.

    Returns
    -------
    outputs :
        The equilibrium geometry.
    """
    return _forward_pass(inputs.model_dump())


def abstract_eval(abstract_inputs):
    """
    Output shapes and dtypes, without solving anything.

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
    nodes = abstract_inputs.nodes.shape[0]

    return {"xyz": {"shape": (nodes, 3), "dtype": "float64"}}


def _restrict_for_derivative(
    inputs: InputSchema,
    wrt: list[str],
    outputs: list[str],
) -> tuple[Any, list[Any]]:
    """
    The map restricted to the requested inputs and outputs, and its primals.

    Parameters
    ----------
    inputs :
        The force densities, the topology and the loads.
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

    def restricted_map(*values):
        merged = {**static, **dict(zip(wrt, values))}
        computed = _forward_pass(merged)

        return {name: computed[name] for name in outputs}

    return restricted_map, [jnp.asarray(raw[name]) for name in wrt]


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
        The force densities, the topology and the loads.
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
    What `jax.grad` calls, and the only endpoint it calls. One reverse pass
    costs a few forward solves whatever the number of force densities, which is
    the reason the design variable can be per-edge rather than global.
    """
    restricted_map, primals = _restrict_for_derivative(inputs, vjp_inputs, vjp_outputs)

    _, pullback = jax.vjp(restricted_map, *primals)
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
        The force densities, the topology and the loads.
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
    Never reached by `jax.grad`, and provided because it costs one call and
    cross-checks the reverse rule.
    """
    restricted_map, primals = _restrict_for_derivative(inputs, jvp_inputs, jvp_outputs)

    tangents = tuple(jnp.asarray(tangent_vector[name]) for name in jvp_inputs)
    _, pushed = jax.jvp(restricted_map, tuple(primals), tangents)

    return pushed
