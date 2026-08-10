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
The `smax` backend of the analysis stage, differentiated by tracing autodiff.

A JAX frame solver, so the whole assembly and solve is traceable and the
derivatives come out of the same machinery that produced them upstream. It is
the reference the second backend is measured against rather than the interesting
one: the argument the analysis stage makes is that a solver which cannot be
traced at all can sit behind this same schema.

Three dimensions throughout, which is what the gridshell needs and what a direct
differentiation backend cannot supply.

**Both derivative rules are this module's own.** The stage's endpoints ask a
backend for them rather than differentiating it, since a backend that cannot be
traced has nothing for `jax.vjp` to record. Here they are one line each; the
other backend assembles the same answers from a Jacobian it sweeps out by hand,
and the schema cannot tell the difference.
"""

from typing import Any

import jax
import jax.numpy as jnp

from normax.analysis.smax import forces
from normax.analysis.smax import prepare
from normax.ec3.sizing import Steel
from normax.ec3.sizing import Tube
from normax.structures import Structure


def solve(inputs: dict[str, Any]) -> dict[str, jnp.ndarray]:
    """
    Internal forces of the frame the inputs describe.

    Parameters
    ----------
    inputs :
        The validated input fields of the analysis schema.

    Returns
    -------
    outputs :
        Axial force and both end moments of every member.

    Notes
    -----
    The coordinates and the diameters are injected into the assembly here, so both
    are differentiable leaves rather than properties baked in when the model was
    prepared. The reference state is unstressed: the nodes displace before any
    internal force appears, and that elastic response is the whole of the gap
    between these axial forces and the ones form finding predicted.

    **The assembly is prepared per crossing.** A boundary crossing is stateless,
    so nothing a previous call prepared survives into this one and `prepare` runs
    again.

    Yield strength and density reach the material but not the answer, a linear
    elastic analysis under nodal loads having no use for either. They are carried
    so that the schema still describes the frame when self-weight or a nonlinear
    backend arrives.
    """
    xyz = jnp.asarray(inputs["xyz"])

    structure = Structure(
        nodes=xyz,
        edges=jnp.asarray(inputs["edges"]),
        supports=jnp.asarray(inputs["supports"]),
        loads=jnp.asarray(inputs["loads"]),
    )

    steel = Steel(
        f_y=inputs["f_y"],
        e_mod=inputs["e_mod"],
        density=inputs["density"],
    )
    tube = Tube(ratio=inputs["ratio"])

    model = prepare(structure, steel, tube, normal=inputs["normal"])

    member = forces(
        model,
        xyz,
        jnp.asarray(inputs["diameter"]),
        steel,
        tube,
    )

    return {
        "n_ed": member.n_ed,
        "m_y_ed": member.m_y_ed,
        "m_z_ed": member.m_z_ed,
    }


def _restricted(
    inputs: dict[str, Any],
    wrt: list[str],
    outputs: list[str],
) -> tuple[Any, list[jnp.ndarray]]:
    """
    The solve restricted to the requested inputs and outputs, and its primals.

    Parameters
    ----------
    inputs :
        The validated input fields of the analysis schema.
    wrt :
        Names of the input fields a derivative is taken with respect to.
    outputs :
        Names of the output fields a derivative is taken of.

    Returns
    -------
    restricted :
        The restricted solve and the primal values of the requested inputs.

    Notes
    -----
    Everything not differentiated is closed over rather than passed, so JAX sees
    a function of the requested arguments alone and no static field has to be
    marked as such.
    """
    static = {name: value for name, value in inputs.items() if name not in wrt}

    def restricted(*values):
        merged = {**static, **dict(zip(wrt, values))}
        computed = solve(merged)

        return {name: computed[name] for name in outputs}

    return restricted, [jnp.asarray(inputs[name]) for name in wrt]


def jvp(
    inputs: dict[str, Any],
    jvp_inputs: list[str],
    jvp_outputs: list[str],
    tangent_vector: dict[str, Any],
) -> dict[str, jnp.ndarray]:
    """
    Push a tangent on the inputs forward to the outputs.

    Parameters
    ----------
    inputs :
        The validated input fields of the analysis schema.
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
    One forward pass whatever the number of parameters, the assembly and the
    solve being traced along with everything else.
    """
    restricted, primals = _restricted(inputs, jvp_inputs, jvp_outputs)

    tangents = tuple(jnp.asarray(tangent_vector[name]) for name in jvp_inputs)
    _, pushed = jax.jvp(restricted, tuple(primals), tangents)

    return pushed


def vjp(
    inputs: dict[str, Any],
    vjp_inputs: list[str],
    vjp_outputs: list[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, jnp.ndarray]:
    """
    Pull a cotangent on the outputs back to the inputs.

    Parameters
    ----------
    inputs :
        The validated input fields of the analysis schema.
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
    One reverse pass whatever the number of coordinates, which is the property a
    direct differentiation backend has to buy with a sweep.
    """
    restricted, primals = _restricted(inputs, vjp_inputs, vjp_outputs)

    _, pullback = jax.vjp(restricted, *primals)
    cotangents = pullback(
        {name: jnp.asarray(value) for name, value in cotangent_vector.items()}
    )

    return dict(zip(vjp_inputs, cotangents))
