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
T3 — EN 1993-1-1 member design, as a differentiable map.

Member actions in, the diameter the standard requires out, and a mass. This is
the component the project exists to argue about: a design standard is a
normative text, not a solver. It states resistances and leaves a human to search
for a section that carries the actions, and the reference implementations of it
are scalar, branchy code returning verdicts. Here the search is a bisection on a
monotone residual and it carries an adjoint, so the standard composes with an
autodiff form-finder instead of terminating the chain.

Differentiation strategy: an implicit tangent taken at the root of the residual,
not autodiff through the bisection. `normax.ec3.sizing` wraps the solve in a
`custom_jvp`, so tracing this module reaches the hand-derived rule rather than
fifty-five halvings of a `while_loop`.

The cross-section class is a static field rather than an array, because it
selects a clause rather than scaling a number. Fixing the diameter-to-thickness
ratio at a class limit makes the classification exact by construction, so no
branch on a traced value is ever needed.

Scope, and what is deliberately absent, is in `CLAUDE.md` §3. Clean-room from
EN 1993-1-1 by way of `docs/clauses.md`.
"""

from typing import Any

import jax
import jax.numpy as jnp
from pydantic import BaseModel
from tesseract_core.runtime import Array
from tesseract_core.runtime import Differentiable
from tesseract_core.runtime import Float64

from normax.ec3.actions import MemberActions
from normax.ec3.material import SteelGrade
from normax.ec3.section import TubeCatalogue
from normax.ec3.sizing import diameter_required
from normax.ec3.sizing import end_moments
from normax.ec3.sizing import governing_limit_state as governing_limit
from normax.ec3.sizing import mass as mass_of_tubes
from normax.ec3.sizing import utilization_design as utilization_of_tubes

jax.config.update("jax_enable_x64", True)


class InputSchema(BaseModel):
    """
    Member actions, a buckling length, and the material the tubes are cut from.
    """

    n_ed: Differentiable[Array[(None,), Float64]]
    """Design axial force of every member, in newtons. Tension positive."""

    m_y_ed: Differentiable[Array[(None, 2), Float64]]
    """Major-axis moment at each end of every member, in newton-millimetres."""

    m_z_ed: Differentiable[Array[(None, 2), Float64]]
    """Minor-axis moment at each end of every member, in newton-millimetres."""

    lengths: Differentiable[Array[(None,), Float64]]
    """Length of every member, in millimetres. Sets the mass, not the check."""

    l_cr: Differentiable[Array[(None,), Float64]]
    """Buckling length of every member, in millimetres.

    An input and never derived from the mesh. Passing the member's own length
    assumes every node is held in position by structure outside the model.
    """

    f_y: Differentiable[Float64]
    """Yield strength, in newtons per square millimetre."""

    e_mod: Differentiable[Float64]
    """Modulus of elasticity, in newtons per square millimetre."""

    density: Differentiable[Float64]
    """Density, in tonnes per cubic millimetre, so the mass comes out in tonnes."""

    gamma_m0: Differentiable[Float64]
    """Partial factor for cross-section resistance."""

    gamma_m1: Differentiable[Float64]
    """Partial factor for member instability."""

    ratio: Differentiable[Float64]
    """Diameter-to-thickness ratio, fixing the wall and so the section class."""

    alpha: Differentiable[Float64]
    """Imperfection factor of the buckling curve, EN 1993-1-1 Table 6.1."""

    diameter_min: Float64
    """Smallest diameter the section family offers, in millimetres."""

    plastic: bool
    """Whether the section is Class 1 or 2. Static: it selects a clause."""

    resultant: bool = True
    """Whether the cross-section check combines the two moments as a resultant."""


class OutputSchema(BaseModel):
    """
    The sizes EN 1993-1-1 requires, what they weigh, and how hard they work.
    """

    diameter: Differentiable[Array[(None,), Float64]]
    """Outer diameter of every member, in millimetres."""

    mass: Differentiable[Float64]
    """Total mass of the members, in tonnes. The objective of the pipeline."""

    utilization: Differentiable[Array[(None,), Float64]]
    """Demand over resistance of every member.

    One to machine precision wherever a clause decided the size, and below one
    wherever the catalogue minimum did. An invariant to assert on, not a goal.
    """

    m_y_ed: Differentiable[Array[(None,), Float64]]
    """Larger major-axis end moment of every member, in newton-millimetres."""

    m_z_ed: Differentiable[Array[(None,), Float64]]
    """Larger minor-axis end moment of every member, in newton-millimetres."""

    c_my: Differentiable[Array[(None,), Float64]]
    """Major-axis moment factor of every member, EN 1993-1-1 Table B.3."""

    c_mz: Differentiable[Array[(None,), Float64]]
    """Minor-axis moment factor of every member, EN 1993-1-1 Table B.3.

    Reported alongside the major axis rather than folded away, so a caller can
    check a finished design at a size the standard did not choose without
    analysing anything again.
    """

    governing: Array[(None,), Float64]
    """Limit state that decided every member's size, as a code.

    **Non-differentiable.** A concrete cotangent on this raises `ValueError`, so
    drop it before differentiating; only a symbolic zero is accepted. Watch it
    between optimizer steps, where repeated flips mean the design is chattering
    across a boundary of the standard.
    """


def _material(inputs: dict[str, Any]) -> tuple[SteelGrade, TubeCatalogue]:
    """
    The material and the section family, from the flat fields of the schema.

    Parameters
    ----------
    inputs :
        The validated input fields.

    Returns
    -------
    material :
        The steel and the tube family.
    """
    steel = SteelGrade(
        f_y=inputs["f_y"],
        e_mod=inputs["e_mod"],
        density=inputs["density"],
        gamma_m0=inputs["gamma_m0"],
        gamma_m1=inputs["gamma_m1"],
        alpha=inputs["alpha"],
    )
    catalogue = TubeCatalogue(
        ratio=inputs["ratio"],
        diameter_min=inputs["diameter_min"],
    )

    return steel, catalogue


def _forward(
    inputs: dict[str, Any],
    *,
    diagnostics: bool,
) -> dict[str, jnp.ndarray]:
    """
    Size every member, and weigh the result.

    Parameters
    ----------
    inputs :
        The validated input fields.
    diagnostics :
        Whether to report the governing limit state, which is not differentiated
        and so is left out of every gradient endpoint.

    Returns
    -------
    outputs :
        The output fields, the diagnostic included only when asked for.

    Notes
    -----
    EN 1993-1-1 Table B.3 lives here rather than upstream, because reading a
    design moment and an equivalent uniform moment factor out of two end moments
    is a clause of the standard and not a product of an analysis. That is what
    keeps the analysis schema free of anything a solver has no opinion on.
    """
    steel, catalogue = _material(inputs)
    plastic = inputs["plastic"]
    resultant = inputs["resultant"]

    m_y_ed = jnp.asarray(inputs["m_y_ed"])
    m_z_ed = jnp.asarray(inputs["m_z_ed"])

    m_ed, c_m = end_moments(m_y_ed[:, 0], m_y_ed[:, 1])
    m_minor, c_minor = end_moments(m_z_ed[:, 0], m_z_ed[:, 1])

    n_ed = jnp.asarray(inputs["n_ed"])
    l_cr = jnp.asarray(inputs["l_cr"])
    lengths = jnp.asarray(inputs["lengths"])

    actions = MemberActions(n_ed, m_ed, m_minor, c_m, c_minor)

    required = diameter_required(
        actions, l_cr, steel, catalogue, plastic=plastic, resultant=resultant
    )
    sized = catalogue.tube(required)
    used = utilization_of_tubes(
        sized, actions, l_cr, steel, plastic=plastic, resultant=resultant
    )

    outputs = {
        "diameter": required,
        "mass": mass_of_tubes(sized, lengths, steel),
        "utilization": used,
        "m_y_ed": m_ed,
        "m_z_ed": m_minor,
        "c_my": c_m,
        "c_mz": c_minor,
    }

    if diagnostics:
        outputs["governing"] = governing_limit(
            sized,
            actions,
            l_cr,
            steel,
            catalogue,
            plastic=plastic,
            resultant=resultant,
        )

    return outputs


def apply(inputs: InputSchema) -> OutputSchema:
    """
    Run the check.

    Parameters
    ----------
    inputs :
        The member actions and the material.

    Returns
    -------
    outputs :
        The required sizes, the mass, the utilization and the diagnostics.
    """
    return _forward(inputs.model_dump(), diagnostics=True)


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

    Notes
    -----
    Required by Tesseract-JAX: JAX resolves shapes before it executes anything,
    so every endpoint below is unreachable without this one.
    """
    members = abstract_inputs.n_ed.shape[0]

    return {
        "diameter": {"shape": (members,), "dtype": "float64"},
        "mass": {"shape": (), "dtype": "float64"},
        "utilization": {"shape": (members,), "dtype": "float64"},
        "m_y_ed": {"shape": (members,), "dtype": "float64"},
        "m_z_ed": {"shape": (members,), "dtype": "float64"},
        "c_my": {"shape": (members,), "dtype": "float64"},
        "c_mz": {"shape": (members,), "dtype": "float64"},
        "governing": {"shape": (members,), "dtype": "float64"},
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
        The member actions and the material.
    wrt :
        Names of the input fields a derivative is taken with respect to.
    outputs :
        Names of the output fields a derivative is taken of.

    Returns
    -------
    restricted :
        The restricted map and the primal values of the requested inputs.

    Raises
    ------
    ValueError
        If the governing limit state is among the outputs.

    Notes
    -----
    The diagnostic is refused here rather than silently returning a zero,
    because a cotangent on it means the caller left a non-differentiable output
    in the loss and would otherwise get a wrong answer quietly.
    """
    if "governing" in outputs:
        raise ValueError(
            "`governing` is non-differentiable; drop it before differentiating"
        )

    raw = inputs.model_dump()
    static = {name: value for name, value in raw.items() if name not in wrt}

    def restricted(*values):
        merged = {**static, **dict(zip(wrt, values))}
        computed = _forward(merged, diagnostics=False)

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
        The member actions and the material.
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
    What `jax.grad` calls, and the only endpoint it calls. Tracing reaches the
    `custom_jvp` on the sizing map, so the bisection is never differentiated
    through: the tangent comes from the implicit function theorem applied at the
    root, which needs the residual differentiable only there.
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
        The member actions and the material.
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
    cross-checks the reverse rule against the same implicit tangent.
    """
    restricted, primals = _differentiate(inputs, jvp_inputs, jvp_outputs)

    tangents = tuple(jnp.asarray(tangent_vector[name]) for name in jvp_inputs)
    _, pushed = jax.jvp(restricted, tuple(primals), tangents)

    return pushed
