# Copyright 2026 normax contributors
# SPDX-License-Identifier: Apache-2.0
"""T2 — Structural analysis under the asymmetric load case.

Differentiation strategy: **analytic sensitivities from a C++ solver** (OpenSees
Direct Differentiation Method), with a JAX autodiff backend as fallback and as the
cross-check.

Why this Tesseract exists: under the symmetric design load the form-found shell is
funicular and T1 already gives the complete internal force state. Under an
asymmetric load it is not, and bending appears. That is the load case that decides
the design, and it needs a real frame solver.

Why it is a real boundary: OpenSees is a large C++ codebase with no autodiff. Its
derivatives come from DDM — adjoints hand-derived element by element over two
decades of journal articles. Nothing about it is traceable by JAX. The same
optimization loop drives both backends through this one schema.

SET `NORMA_ANALYSIS_BACKEND` to "sax" (default) or "opensees".
"""

import os
from typing import Any

import jax
import jax.numpy as jnp
from pydantic import BaseModel
from tesseract_core.runtime import Array
from tesseract_core.runtime import Differentiable
from tesseract_core.runtime import Float64

jax.config.update("jax_enable_x64", True)

BACKEND = os.environ.get("NORMA_ANALYSIS_BACKEND", "sax")


# --------------------------------------------------------------------------- #
# Schemas — identical for both backends. This is the point.
# --------------------------------------------------------------------------- #
class InputSchema(BaseModel):
    xyz: Differentiable[Array[(None, 3), Float64]]
    """Node coordinates [mm], from T1."""

    diameter: Differentiable[Array[(None,), Float64]]
    """Member outer diameter [mm], from T3's previous outer iterate.

    NOTE the staggered coupling: T3 needs forces to size, T2 needs sizes to
    compute forces. The MVP passes the previous iterate and accepts a one-way
    gradient. Document this in the writeup — do not quietly ignore it.
    """

    edges: Array[(None, 2), Float64]
    supports: Array[(None,), Float64]
    """Indices of restrained nodes. Non-differentiable."""

    loads: Array[(None, 3), Float64]
    """Asymmetric load case [N]."""

    e_mod: Array[(), Float64]
    dt_ratio: Array[(), Float64]

    fd_epsilon: Array[(), Float64]
    """Finite-difference step, used only by the OpenSees fallback path.

    Non-differentiable by design, so it can be swept without touching the schema.
    Sweep it over several decades against a fixed geometry and look for the
    plateau before trusting any FD gradient — the noise floor is set by the
    solver's convergence tolerance, not by float64.
    """


class OutputSchema(BaseModel):
    n_ed: Differentiable[Array[(None,), Float64]]
    """Design axial force per member [N]. Tension positive. Feeds T3."""

    m_ed: Differentiable[Array[(None,), Float64]]
    """Peak bending moment per member [N mm].

    Not consumed by the MVP's T3 (which is axial-only). Wired now so the N+M
    interaction of §6.2.9 can be added without a schema change.
    """


# --------------------------------------------------------------------------- #
# Backend dispatch
# --------------------------------------------------------------------------- #
def _forward(inputs: dict[str, Any]) -> dict[str, jnp.ndarray]:
    if BACKEND == "sax":
        from _backend_sax import solve  # noqa: PLC0415
    elif BACKEND == "opensees":
        from _backend_opensees import solve  # noqa: PLC0415
    else:
        raise ValueError(f"Unknown backend: {BACKEND!r}")
    return solve(inputs)


def apply(inputs: InputSchema) -> OutputSchema:
    return _forward(inputs.model_dump())


def abstract_eval(abstract_inputs):
    n_edges = abstract_inputs.edges.shape[0]
    return {
        "n_ed": {"shape": (n_edges,), "dtype": "float64"},
        "m_ed": {"shape": (n_edges,), "dtype": "float64"},
    }


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: list[str],
    vjp_outputs: list[str],
    cotangent_vector: dict[str, Any],
):
    """Reverse mode.

    sax backend      -> jax.vjp, exact, ~2-3x forward.
    opensees backend -> DDM sensitivities contracted with the cotangent, or FD.

    For a scalar loss JAX issues ONE vjp call here, so a single HTTP round trip.
    Whatever the backend does internally is the entire cost.
    """
    raw = inputs.model_dump()

    if BACKEND == "sax":
        static = {k: v for k, v in raw.items() if k not in vjp_inputs}

        def f(*diff_args):
            merged = {**static, **dict(zip(vjp_inputs, diff_args))}
            out = _forward(merged)
            return {k: out[k] for k in vjp_outputs}

        primals = [jnp.asarray(raw[k]) for k in vjp_inputs]
        _, pullback = jax.vjp(f, *primals)
        cot = pullback({k: jnp.asarray(v) for k, v in cotangent_vector.items()})
        return dict(zip(vjp_inputs, cot))

    from _backend_opensees import vjp  # noqa: PLC0415

    return vjp(raw, vjp_inputs, vjp_outputs, cotangent_vector)


# --------------------------------------------------------------------------- #
# OpenSees DDM notes — read before Aug 12 (CLAUDE.md §9)
# --------------------------------------------------------------------------- #
#
# The DDM call sequence in OpenSeesPy:
#
#     ops.parameter(tag, 'element', ele_tag, 'E')   # register parameters
#     ops.sensitivityAlgorithm('-computeAtEachStep')
#     ops.analyze(1)
#     for tag in ops.getParamTags():
#         ops.sensNodeDisp(node, dof, tag)
#         ops.sensSectionForce(ele, sec, dof, tag)
#
# THE OPEN RISK: DDM parametrizes material and section properties (E, A, I, Iz,
# Iy, G, J for elastic sections). It does NOT obviously parametrize nodal
# coordinates, and T1 hands us a geometry. Resolve before building this backend:
#
#   1. Verify sensNodeDisp against central differences on one elastic
#      beam-column. If they disagree, stop — nothing downstream is trustworthy.
#   2. Confirm the element implements getResistingForceSensitivity. The presence
#      of setParameter does NOT mean DDM is enabled for that element.
#   3. Establish whether a nodal-coordinate parameter can be registered at all.
#
# Fallbacks in preference order: (a) DDM for d, geometry gradients through T1
# only; (b) DDM for section properties composed by hand with an analytic
# dN/dxyz; (c) finite differences over ~50 inputs, affordable at this scale.
#
# The headline plot for the submission is experiments/04: the same optimization,
# the same T1 and T3, gradients from JAX autodiff and from C++ DDM agreeing to
# 1e-6. That figure makes the composition argument without a caption.
