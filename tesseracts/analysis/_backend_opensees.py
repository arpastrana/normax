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
The OpenSees backend of the analysis stage, differentiated by DDM.

A C++ solver behind a command interface, so nothing here is traced and both
derivative rules are assembled by hand from a Jacobian the solver sweeps out one
parameter at a time. The schema above is untouched, which is the whole claim:
the two backends disagree about how a derivative is obtained and agree about
what one is.

Two dimensions, because that is where OpenSees' Direct Differentiation Method
reaches a nodal coordinate at all. `normax.analysis.opensees` carries the model,
the parameter registration and the reasons.
"""

from typing import Any

import jax.numpy as jnp

from normax.analysis.opensees import Jacobian
from normax.analysis.opensees import forces
from normax.analysis.opensees import jacobian
from normax.analysis.opensees import prepare
from normax.ec3.material import SteelGrade
from normax.ec3.sizing import Tube
from normax.structures import Structure

# Which block of the Jacobian carries each (output, input) pair. The minor-axis
# moment has none: a plane frame carries no such moment and the one derivative
# of it that is nonzero is the block this backend cannot reach.
BLOCKS = {
    ("n_ed", "xyz"): "n_ed_xyz",
    ("n_ed", "diameter"): "n_ed_diameter",
    ("m_y_ed", "xyz"): "m_y_ed_xyz",
    ("m_y_ed", "diameter"): "m_y_ed_diameter",
}

# How many axes of a block belong to the output, so a contraction knows where
# the output ends and the input begins.
OUTPUT_RANK = {"n_ed": 1, "m_y_ed": 2, "m_z_ed": 2}


def _model(inputs: dict[str, Any]) -> tuple[Structure, SteelGrade, Tube]:
    """
    The frame the inputs describe, in the containers the backend takes.

    Parameters
    ----------
    inputs :
        The validated input fields of the analysis schema.

    Returns
    -------
    model :
        The prepared analysis model, the material and the section family.

    Notes
    -----
    Rebuilt per call rather than cached. OpenSees keeps one global model with no
    handle to it, so there is nothing a cache could hold that the next call
    would not overwrite, and preparing one settles only which plane the frame
    lies in.
    """
    structure = Structure(
        nodes=jnp.asarray(inputs["xyz"]),
        edges=jnp.asarray(inputs["edges"]),
        supports=jnp.asarray(inputs["supports"]),
        loads=jnp.asarray(inputs["loads"]),
    )

    steel = SteelGrade(
        f_y=inputs["f_y"],
        e_mod=inputs["e_mod"],
        density=inputs["density"],
    )
    tube = Tube(ratio=inputs["ratio"])

    return prepare(structure, steel, tube, normal=inputs["normal"]), steel, tube


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
    Yield strength and density reach the material and not the answer, a linear
    elastic analysis under nodal loads having no use for either, exactly as in
    the other backend. Carrying them keeps one schema describing both.
    """
    model, steel, tube = _model(inputs)

    member = forces(
        model,
        jnp.asarray(inputs["xyz"]),
        jnp.asarray(inputs["diameter"]),
        steel,
        tube,
    )

    return {
        "n_ed": member.n_ed,
        "m_y_ed": member.m_y_ed,
        "m_z_ed": member.m_z_ed,
    }


def _blocks(inputs: dict[str, Any]) -> Jacobian:
    """
    Sweep the solver once for every derivative the stage can be asked for.

    Parameters
    ----------
    inputs :
        The validated input fields of the analysis schema.

    Returns
    -------
    jacobian :
        Dense derivative blocks of the axial force and the end moments.

    Notes
    -----
    The whole Jacobian is taken whatever subset was requested. DDM forms one
    factorization and reuses it, so the parameters already registered cost a
    back-substitution each and dropping some would save a fraction of a sweep
    while making the two derivative rules disagree about what was solved.
    """
    model, steel, tube = _model(inputs)

    return jacobian(
        model,
        jnp.asarray(inputs["xyz"]),
        jnp.asarray(inputs["diameter"]),
        steel,
        tube,
    )


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
    The natural direction for this solver, and the one the two backends meet in
    first. A tangent is the sum over inputs of a block contracted with that
    input's tangent, and an output with no block for an input contributes
    nothing to it.
    """
    blocks = _blocks(inputs)
    tangents = {}

    for output in jvp_outputs:
        total = jnp.zeros(_shape(blocks, output))
        for field in jvp_inputs:
            if (output, field) not in BLOCKS:
                continue
            block = getattr(blocks, BLOCKS[(output, field)])
            tangent = jnp.asarray(tangent_vector[field])
            total = total + jnp.tensordot(block, tangent, axes=tangent.ndim)
        tangents[output] = total

    return tangents


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
    Assembled by contracting the same Jacobian from the other side, so what a
    traced backend gets in one reverse pass is bought here with a sweep whose
    cost grows in the parameter count. That difference is the measurement
    `experiments/04_backend_agreement.py` reports rather than a detail.
    """
    blocks = _blocks(inputs)
    cotangents = {}

    for field in vjp_inputs:
        total = jnp.zeros(jnp.asarray(inputs[field]).shape)
        for output in vjp_outputs:
            if (output, field) not in BLOCKS:
                continue
            block = getattr(blocks, BLOCKS[(output, field)])
            cotangent = jnp.asarray(cotangent_vector[output])
            total = total + jnp.tensordot(cotangent, block, axes=OUTPUT_RANK[output])
        cotangents[field] = total

    return cotangents


def _shape(blocks: Jacobian, output: str) -> tuple[int, ...]:
    """
    Shape of one output field, read off the blocks that produce it.

    Parameters
    ----------
    blocks :
        Dense derivative blocks of the axial force and the end moments.
    output :
        Name of the output field.

    Returns
    -------
    shape :
        Shape of that field.

    Notes
    -----
    Taken from a block rather than from the schema so that a tangent on an
    output with no blocks at all still has the right shape to be added to.
    """
    if output == "n_ed":
        return blocks.n_ed_diameter.shape[:1]

    return blocks.m_y_ed_diameter.shape[:2]
