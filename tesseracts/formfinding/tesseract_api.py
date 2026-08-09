# Copyright 2026 normax contributors
# SPDX-License-Identifier: Apache-2.0
"""T1 — Force-density form-finding.

Differentiation strategy: **JAX autodiff**, with a `custom_vjp` around the linear
solve via the implicit function theorem.

Maps force densities `q` to the equilibrium geometry of a pin-jointed network and
the resulting axial forces. Under the symmetric design load this geometry is
funicular, so the member forces here are purely axial.

Boundary crossed: this is the only Tesseract in the pipeline whose derivatives come
from a tracing autodiff system.
"""

from typing import Any

import jax
import jax.numpy as jnp
from pydantic import BaseModel
from tesseract_core.runtime import Array
from tesseract_core.runtime import Differentiable
from tesseract_core.runtime import Float64

jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class InputSchema(BaseModel):
    """Force densities and fixed topology.

    NOTE: every array field must be a JAX or NumPy array when called through
    Tesseract-JAX — Python floats and lists are rejected, including scalars.
    """

    q: Differentiable[Array[(None,), Float64]]
    """Force density per edge [N/mm]. Shape (n_edges,). The design variable."""

    xyz_fixed: Array[(None, 3), Float64]
    """Coordinates of anchored nodes [mm]. Shape (n_fixed, 3)."""

    loads: Array[(None, 3), Float64]
    """Applied nodal load vector [N]. Shape (n_free, 3)."""

    edges: Array[(None, 2), Float64]
    """Edge connectivity as node index pairs. Shape (n_edges, 2).

    Float64 rather than int because Tesseract schemas are array-typed; cast to
    int32 on entry. Non-differentiable.
    """

    n_free: Array[(), Float64]
    """Number of free nodes. Non-differentiable."""


class OutputSchema(BaseModel):
    xyz: Differentiable[Array[(None, 3), Float64]]
    """Equilibrium coordinates of all nodes [mm]."""

    lengths: Differentiable[Array[(None,), Float64]]
    """Edge lengths [mm]. Feeds buckling length L_cr in T3."""

    axial: Differentiable[Array[(None,), Float64]]
    """Axial force per edge [N], sign convention: tension positive.

    Equals q * length. Under the symmetric design load this is the complete
    internal force state; T2 supplies the asymmetric load case.
    """


# --------------------------------------------------------------------------- #
# Core solve
# --------------------------------------------------------------------------- #
def _solve_fdm(
    q: jnp.ndarray,
    xyz_fixed: jnp.ndarray,
    loads: jnp.ndarray,
    edges: jnp.ndarray,
    n_free: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve the force-density equilibrium system.

    The FDM equilibrium is linear in the coordinates for fixed `q`:

        D_free @ xyz_free = loads - D_fixed @ xyz_fixed

    with D = C_free^T diag(q) C_free. Because it is linear, `jnp.linalg.solve`
    is already differentiable and no explicit `custom_vjp` is required for the
    MVP. Keep the IFT wrapper in reserve for the nonlinear (large-displacement)
    variant — see `sax` for the pattern.

    TODO: delegate to `jax_fdm` rather than reimplementing. This function exists
    so the Tesseract can be tested without the dependency resolved.
    """
    raise NotImplementedError(
        "Build the branch matrix C from `edges`, split into free/fixed columns, "
        "assemble D = C_free.T @ diag(q) @ C_free, solve for xyz_free. "
        "Verify against a known catenary before wiring anything downstream."
    )


def _forward(inputs: dict[str, Any]) -> dict[str, jnp.ndarray]:
    edges = jnp.asarray(inputs["edges"], dtype=jnp.int32)
    n_free = int(inputs["n_free"])

    xyz = _solve_fdm(
        q=inputs["q"],
        xyz_fixed=inputs["xyz_fixed"],
        loads=inputs["loads"],
        edges=edges,
        n_free=n_free,
    )
    vectors = xyz[edges[:, 1]] - xyz[edges[:, 0]]
    lengths = jnp.linalg.norm(vectors, axis=1)
    axial = inputs["q"] * lengths

    return {"xyz": xyz, "lengths": lengths, "axial": axial}


# --------------------------------------------------------------------------- #
# Tesseract endpoints
# --------------------------------------------------------------------------- #
def apply(inputs: InputSchema) -> OutputSchema:
    return _forward(inputs.model_dump())


def abstract_eval(abstract_inputs):
    """Required by Tesseract-JAX: shapes and dtypes without executing."""
    n_edges = abstract_inputs.q.shape[0]
    n_nodes = int(abstract_inputs.n_free) + abstract_inputs.xyz_fixed.shape[0]
    return {
        "xyz": {"shape": (n_nodes, 3), "dtype": "float64"},
        "lengths": {"shape": (n_edges,), "dtype": "float64"},
        "axial": {"shape": (n_edges,), "dtype": "float64"},
    }


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: list[str],
    vjp_outputs: list[str],
    cotangent_vector: dict[str, Any],
):
    """Reverse-mode AD via `jax.vjp`. Cost is ~2-3x a forward solve, independent
    of the number of force densities."""
    raw = inputs.model_dump()
    static = {k: v for k, v in raw.items() if k not in vjp_inputs}

    def f(*diff_args):
        merged = {**static, **dict(zip(vjp_inputs, diff_args))}
        out = _forward(merged)
        return {k: out[k] for k in vjp_outputs}

    primals = [jnp.asarray(raw[k]) for k in vjp_inputs]
    _, pullback = jax.vjp(f, *primals)
    cotangents = pullback({k: jnp.asarray(v) for k, v in cotangent_vector.items()})
    return dict(zip(vjp_inputs, cotangents))


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: list[str],
    jvp_outputs: list[str],
    tangent_vector: dict[str, Any],
):
    """Forward mode. Not used by `jax.grad`, but cheap to provide and useful for
    cross-checking the VJP."""
    raw = inputs.model_dump()
    static = {k: v for k, v in raw.items() if k not in jvp_inputs}

    def f(*diff_args):
        merged = {**static, **dict(zip(jvp_inputs, diff_args))}
        out = _forward(merged)
        return {k: out[k] for k in jvp_outputs}

    primals = tuple(jnp.asarray(raw[k]) for k in jvp_inputs)
    tangents = tuple(jnp.asarray(tangent_vector[k]) for k in jvp_inputs)
    _, out_tangents = jax.jvp(f, primals, tangents)
    return out_tangents
