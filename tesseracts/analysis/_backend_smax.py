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

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from smax import CompiledStructure

from normax.analysis.smax import member_forces
from normax.analysis.smax import prepare_model
from normax.ec3.material import Steel
from normax.ec3.section import TubeCatalogue
from normax.stages import MemberForces
from normax.structures import Structure


@eqx.filter_jit
def _member_forces(
    model: CompiledStructure,
    xyz: Float[Array, "nodes 3"],
    diameters: Float[Array, "members"],
    steel: Steel,
    catalogue: TubeCatalogue,
    loads: Float[Array, "nodes 3"],
) -> MemberForces:
    """
    The analysis, compiled, from a model the caller prepared.

    Parameters
    ----------
    model :
        The prepared assembly, from `normax.analysis.smax.prepare`.
    xyz :
        Position of every node.
    diameters :
        Outer diameter of every member.
    steel :
        Material properties.
    catalogue :
        The section family, whose ratio fixes the wall thickness.
    loads :
        Force applied at every node.

    Returns
    -------
    forces :
        Axial force and both end moments of every member.

    Notes
    -----
    **Wrapped once here rather than per call, which is the whole point.** The
    compilation cache belongs to the wrapper, so a wrapper built inside `solve`
    would be a new cache every crossing and would compile afresh every time. At
    module scope the second crossing of a given shape reuses the first, and the
    cache keys itself on the shapes and dtypes of the array leaves, so a second
    load case reuses the program and a second frame size gets its own.

    **Preparation stays outside.** Compiling a frame decides its degree of freedom
    maps by reading support flags with a Python conditional, which is exactly what
    a trace cannot follow, so `prepare` cannot be inside this boundary.

    **This survives the derivative endpoints tracing it.** They call `solve`
    inside `jax.vjp` or `jax.jvp`, and a compiled call nested in a trace stays
    compiled rather than being unrolled into it.

    Every array a derivative might be taken through arrives as an argument, the
    loads included, rather than as a constant folded into the program.
    """
    return member_forces(model, xyz, diameters, steel, catalogue, loads)


def solve_forces(inputs: dict[str, Any]) -> dict[str, jnp.ndarray]:
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

    **The assembly is prepared per crossing and the solve is compiled once.** A
    boundary crossing is stateless, so nothing a previous call prepared survives
    into this one and `prepare` runs again; what does survive is the compiled
    program behind `_member_forces`, because its wrapper lives at module scope.
    Caching the prepared model across crossings would need a key over the topology
    arriving in the inputs, and is deliberately not done here.

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
    )

    steel = Steel(
        f_y=inputs["f_y"],
        e_mod=inputs["e_mod"],
        density=inputs["density"],
    )
    catalogue = TubeCatalogue(ratio=inputs["ratio"])

    model = prepare_model(structure, steel, catalogue, normal=inputs["normal"])

    member = _member_forces(
        model,
        xyz,
        jnp.asarray(inputs["diameter"]),
        steel,
        catalogue,
        jnp.asarray(inputs["loads"]),
    )

    return {
        "axial_force": member.axial_force,
        "end_moments_major": member.moment_major,
        "end_moments_minor": member.moment_minor,
    }


def _restricted_solve(
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

    def restricted_solve(*values):
        merged = {**static, **dict(zip(wrt, values))}
        computed = solve_forces(merged)

        return {name: computed[name] for name in outputs}

    return restricted_solve, [jnp.asarray(inputs[name]) for name in wrt]


def forces_jvp(
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
    restricted_solve, primals = _restricted_solve(inputs, jvp_inputs, jvp_outputs)

    tangents = tuple(jnp.asarray(tangent_vector[name]) for name in jvp_inputs)
    _, pushed = jax.jvp(restricted_solve, tuple(primals), tangents)

    return pushed


def forces_vjp(
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
    restricted_solve, primals = _restricted_solve(inputs, vjp_inputs, vjp_outputs)

    _, pullback = jax.vjp(restricted_solve, *primals)
    cotangents = pullback(
        {name: jnp.asarray(value) for name, value in cotangent_vector.items()}
    )

    return dict(zip(vjp_inputs, cotangents))
