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
The PyNite backend of the analysis stage, differentiated by an adjoint we wrote.

A space frame solver in plain Python that carries **no derivative of any kind**
and offers no way to acquire one — no tape, no tangent, no sensitivity command.
It is the strongest case the boundary has: not a solver whose gradients are
awkward to reach, one that has none. What crosses the schema here is a
derivative the solver cannot compute about its own answer.

The division of labor is worth stating exactly, because the claim is only as
good as its honesty. **PyNite** assembles the stiffness, partitions it, factors
it and solves; it reports the displacements and the matrices. **This repository**
states the element in JAX, holds it against PyNite's own matrices by test, and
differentiates equilibrium as an implicit function so one factorization answers
for every parameter. `normax.analysis.pynite` carries the rule and the reasons.

Three dimensions, which is the other half of why this exists: the planar backend
beside it refuses a geometry that leaves its plane, so a shell had nowhere to go.
The schema above is untouched — three backends now disagree about how a
derivative is obtained and agree about what one is.
"""

import hashlib
from typing import Any

import numpy as np

from normax.analysis import pynite
from normax.materials import SteelGrade
from normax.sections import TubeFamily
from normax.structures import Structure

# Which block of the Jacobian carries each (output, input) pair. Unlike the
# planar backend, every pair is present: a space frame bends both ways, so the
# minor-axis rows are a derivative and not an absence dressed as a zero.
BLOCKS = {
    ("axial_force", "xyz"): "axial_force_xyz",
    ("axial_force", "diameter"): "axial_force_diameter",
    ("end_moments_major", "xyz"): "moment_major_xyz",
    ("end_moments_major", "diameter"): "moment_major_diameter",
    ("end_moments_minor", "xyz"): "moment_minor_xyz",
    ("end_moments_minor", "diameter"): "moment_minor_diameter",
}

# How many axes of a block belong to the output, so a contraction knows where
# the output ends and the input begins.
OUTPUT_RANK = {"axial_force": 1, "end_moments_major": 2, "end_moments_minor": 2}


# One assembled and factorized frame, remembered between endpoint calls.
#
# **The expensive half of a solve does not depend on the loading.** Assembling
# the stiffness and decomposing it costs almost the whole of a forward pass,
# and every load case in one evaluation — and the adjoint that follows each —
# asks for it at the same geometry and the same diameters. So one entry
# suffices: the key deliberately excludes the loads, which is why nothing can
# evict it midway through an evaluation and why it does not need to be sized
# against the number of load cases.
#
# ⚠ **This makes the backend depend on serialized dispatch.** It is safe
# because `normax.tesseract.pin_dispatch_thread` runs every endpoint on one
# owner thread; a served runtime, or `NORMAX_PIN_DISPATCH=0`, would need a lock
# of its own. A miss only costs the preparation this would have saved, so a
# wrong guess is slow rather than wrong.
_PREPARED: dict[bytes, Any] = {}


def _fingerprint(inputs: dict[str, Any]) -> bytes:
    """
    What identifies the frame an assembly and a factorization belong to.

    Parameters
    ----------
    inputs :
        The validated input fields of the analysis schema.

    Returns
    -------
    fingerprint :
        A digest of everything the preparation reads.

    Notes
    -----
    Content, not identity: the schema rebuilds its payload per call, so nothing
    survives between them to compare by reference. Shape and dtype enter the
    digest along with the bytes, since two arrays can share bytes and mean
    different things. **The loads are deliberately absent** — the preparation
    does not read them.
    """
    digest = hashlib.blake2b(digest_size=32)
    for field in ("xyz", "diameter", "edges", "supports"):
        value = np.ascontiguousarray(inputs[field])
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    for field in ("f_y", "e_mod", "density", "ratio"):
        digest.update(repr(float(inputs[field])).encode())

    return digest.digest()


def _prepared_frame(
    problem: pynite.FrameProblem,
    inputs: dict[str, Any],
) -> Any:
    """
    The assembled and factorized frame these inputs describe.

    Parameters
    ----------
    problem :
        The frame and its section family.
    inputs :
        The validated input fields of the analysis schema.

    Returns
    -------
    prepared :
        A frame prepared at this geometry and these diameters, freshly if the
        last one asked for was a different frame.
    """
    fingerprint = _fingerprint(inputs)
    held = _PREPARED.get(fingerprint)
    if held is not None:
        return held

    prepared = pynite.prepared_frame(
        problem, np.asarray(inputs["xyz"]), np.asarray(inputs["diameter"])
    )
    _PREPARED.clear()
    _PREPARED[fingerprint] = prepared

    return prepared


def _build_model(inputs: dict[str, Any]) -> pynite.FrameProblem:
    """
    The frame the inputs describe, in the containers the backend takes.

    Parameters
    ----------
    inputs :
        The validated input fields of the analysis schema.

    Returns
    -------
    problem :
        The structure, the section family its members are drawn from, and the
        loading — everything but the two fields a derivative is taken in.

    Notes
    -----
    The solver holds no global state of its own — it builds a fresh model per
    solve — so unlike the planar backend there is no domain to wipe and no
    ordering the solver itself imposes.

    The schema's normal axis is not read. It states which plane a planar solver
    should work in, and this one has no such restriction.
    """
    structure = Structure(
        nodes=np.asarray(inputs["xyz"]),
        edges=np.asarray(inputs["edges"]),
        supports=np.asarray(inputs["supports"]),
    )

    # The schema carries no ultimate strength; a zero is loud if anything reads it.
    grade = SteelGrade(
        f_y=inputs["f_y"],
        f_u=0.0,
        e_mod=inputs["e_mod"],
        density=inputs["density"],
    )
    # An analysis reads geometry alone, so no class is derived and none is read.
    catalogue = TubeFamily(inputs["ratio"], grade)

    return pynite.FrameProblem(
        structure=structure,
        catalogue=catalogue,
        loads=np.asarray(inputs["loads"]),
    )


def solve_forces(inputs: dict[str, Any]) -> dict[str, np.ndarray]:
    """
    Internal forces of the frame the inputs describe.

    Parameters
    ----------
    inputs :
        The validated input fields of the analysis schema.

    Returns
    -------
    outputs :
        Axial force and both end moments of every member, and the shear and
        torsion the design check excludes.

    Notes
    -----
    Yield strength and density reach the material and not the answer, a linear
    elastic analysis under nodal loads having no use for either, exactly as in
    the other two backends. Carrying them keeps one schema describing all three.

    Both shears and the torsion are real numbers here rather than zeros: a space
    frame carries them, and the audit of the clause that lets the check skip
    shear is what reads them.
    """
    problem = _build_model(inputs)
    diameters = np.asarray(inputs["diameter"])

    member = pynite.member_forces(
        problem,
        np.asarray(inputs["xyz"]),
        diameters,
        problem.loads,
        _prepared_frame(problem, inputs),
    )

    return _plain_arrays(
        {
            "axial_force": member.axial_force,
            "end_moments_major": member.moment_major,
            "end_moments_minor": member.moment_minor,
            "shear_major": member.shear_major,
            "shear_minor": member.shear_minor,
            "torsion_moment": member.torsion_moment,
        }
    )


def _jacobian_blocks(inputs: dict[str, Any]) -> pynite.Jacobian:
    """
    Solve once, factor once, and answer for every derivative the stage carries.

    Parameters
    ----------
    inputs :
        The validated input fields of the analysis schema.

    Returns
    -------
    jacobian :
        Dense derivative blocks of the axial force and both end moments.

    Notes
    -----
    The whole Jacobian is taken whatever subset was requested, for the reason
    the planar backend takes its whole sweep: the expensive step is shared. Here
    it is the factorization of the free stiffness, after which each parameter
    costs a back-substitution, so dropping some would save a fraction and make
    the two product rules disagree about what was solved.
    """
    return pynite.force_jacobian(
        _build_model(inputs),
        np.asarray(inputs["xyz"]),
        np.asarray(inputs["diameter"]),
    )


def forces_jvp(
    inputs: dict[str, Any],
    jvp_inputs: list[str],
    jvp_outputs: list[str],
    tangent_vector: dict[str, Any],
) -> dict[str, np.ndarray]:
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
    """
    blocks = _jacobian_blocks(inputs)
    tangents = {}

    for output in jvp_outputs:
        total = np.zeros(_output_shape(blocks, output))
        for field in jvp_inputs:
            block = getattr(blocks, BLOCKS[(output, field)])
            tangent = np.asarray(tangent_vector[field])
            total = total + np.tensordot(block, tangent, axes=tangent.ndim)
        tangents[output] = total

    return _plain_arrays(tangents)


def forces_vjp(
    inputs: dict[str, Any],
    vjp_inputs: list[str],
    vjp_outputs: list[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, np.ndarray]:
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
    The direction a descent asks for, and the one this stage is cheapest in.
    **One solve, whatever the parameter count** — the cotangent is pulled back
    by the adjoint rule directly rather than sliced out of a materialized
    Jacobian, so a gradient costs about what the forward pass costs. That is
    the whole reason a shell can be optimized across this boundary.

    A cotangent may arrive on any subset of the differentiable outputs. The one
    handed to the rule is zero-filled over the rest, which is what a cotangent
    that was not asked for means, and the pull-back is linear so the zeros cost
    only their own arithmetic.
    """
    blocks_needed = set(vjp_inputs) - {"xyz", "diameter"}
    if blocks_needed:
        named = ", ".join(sorted(blocks_needed))
        raise ValueError(f"the stage carries no derivative in {named}")

    members = np.asarray(inputs["edges"]).shape[0]
    seeded = {
        "axial_force": np.zeros(members),
        "end_moments_major": np.zeros((members, 2)),
        "end_moments_minor": np.zeros((members, 2)),
    }
    for output in vjp_outputs:
        seeded[output] = np.asarray(cotangent_vector[output])

    problem = _build_model(inputs)
    pulled = pynite.force_cotangents(
        problem,
        np.asarray(inputs["xyz"]),
        np.asarray(inputs["diameter"]),
        pynite.ReadingCotangent(
            axial_force=seeded["axial_force"],
            moment_major=seeded["end_moments_major"],
            moment_minor=seeded["end_moments_minor"],
        ),
        _prepared_frame(problem, inputs),
    )

    cotangents = {field: getattr(pulled, field) for field in vjp_inputs}

    return _plain_arrays(cotangents)


def forces_jacobian(
    inputs: dict[str, Any],
    jac_inputs: list[str],
    jac_outputs: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    """
    Hand over every requested derivative block the one solve already assembled.

    Parameters
    ----------
    inputs :
        The validated input fields of the analysis schema.
    jac_inputs :
        Names of the input fields a derivative is taken with respect to.
    jac_outputs :
        Names of the output fields a derivative is taken of.

    Returns
    -------
    blocks :
        One array per (output, input) pair, keyed output first, each shaped as
        the output's shape followed by the input's.

    Notes
    -----
    Every requested pair exists, so nothing here fills a zero. What the planar
    backend has to explain — a block it cannot reach — has no counterpart in
    three dimensions.
    """
    blocks = _jacobian_blocks(inputs)

    jacobian = {}
    for output in jac_outputs:
        per_input = {
            field: getattr(blocks, BLOCKS[(output, field)]) for field in jac_inputs
        }
        jacobian[output] = per_input

    return _plain_arrays(jacobian)


def _output_shape(blocks: pynite.Jacobian, output: str) -> tuple[int, ...]:
    """
    Shape of one output field, read off the blocks that produce it.

    Parameters
    ----------
    blocks :
        Dense derivative blocks of the axial force and both end moments.
    output :
        Name of the output field.

    Returns
    -------
    shape :
        Shape of that field.
    """
    if output == "axial_force":
        return blocks.axial_force_diameter.shape[:1]

    return blocks.moment_major_diameter.shape[:2]


def _plain_arrays(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Every array leaf as NumPy, one or two levels down.

    Parameters
    ----------
    payload :
        Arrays, or a mapping of mappings of arrays.

    Returns
    -------
    plain :
        The same shape, with no traced array anywhere in it.

    Notes
    -----
    An endpoint is called from inside a foreign-function callback, where the
    wrapper that calls it forbids allocating a JAX array. The element
    derivatives behind this backend are taken in JAX, so the conversion is not
    a formality here.
    """
    plain = {}
    for name, value in payload.items():
        if isinstance(value, dict):
            plain[name] = {inner: np.asarray(leaf) for inner, leaf in value.items()}
        else:
            plain[name] = np.asarray(value)

    return plain
