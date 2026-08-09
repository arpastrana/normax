# Copyright 2026 normax contributors
# SPDX-License-Identifier: Apache-2.0
"""T3 — EN 1993-1-1 member design, as a differentiable map.

Differentiation strategy: **hand-derived piecewise analytic adjoint**, dispatched
on the governing limit state. Deliberately *not* autodiff over the whole map.

This is the component that makes the project's argument. A building code is a
normative text, not a solver. It has no derivatives; the reference open-source
implementations are scalar, branchy Python that return booleans. Here it carries
an adjoint and composes with an autodiff form-finder.

Scope (see CLAUDE.md §3):
  §6.2.3   tension resistance
  §6.2.4   compression resistance
  §6.3.1.1 flexural buckling, chi
  No lateral-torsional buckling: CHS is doubly symmetric, so chi_LT = 1.

CLEAN-ROOM. Implemented from EN 1993-1-1 directly. Blueprints (LGPL-2.1) appears
only in tests/, never as a source.
"""

from typing import Any

import jax
import jax.numpy as jnp
from pydantic import BaseModel
from tesseract_core.runtime import Array
from tesseract_core.runtime import Differentiable
from tesseract_core.runtime import Float64

jax.config.update("jax_enable_x64", True)

# Governing limit state codes, reported as a non-differentiable output.
GOV_TENSION = 0.0  # §6.2.3, eq. for N_t,Rd
GOV_BUCKLING = 1.0  # §6.3.1.1, chi < 1
GOV_SQUASH = 2.0  # §6.3.1.1 with the chi <= 1 cap active


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class InputSchema(BaseModel):
    n_ed: Differentiable[Array[(None,), Float64]]
    """Design axial force per member [N]. Tension positive."""

    lengths: Differentiable[Array[(None,), Float64]]
    """Member length [mm]. Buckling length is `k_cr * lengths`."""

    f_y: Array[(), Float64]
    """Yield strength [N/mm^2]. 355 for S355."""

    e_mod: Array[(), Float64]
    """Young's modulus [N/mm^2]. 210000."""

    rho: Array[(), Float64]
    """Density [t/mm^3]. 7.85e-9."""

    dt_ratio: Array[(), Float64]
    """Fixed d/t. Set to 90 * eps^2 (Class 3/4 boundary). See CLAUDE.md §3."""

    alpha_imp: Array[(), Float64]
    """Imperfection factor. 0.21 = curve a, hot-finished CHS."""

    k_cr: Array[(), Float64]
    """Buckling length factor. 1.0 for pinned-pinned."""

    gamma_m0: Array[(), Float64]
    gamma_m1: Array[(), Float64]


class OutputSchema(BaseModel):
    diameter: Differentiable[Array[(None,), Float64]]
    """Fully-stressed outer diameter [mm]."""

    mass: Differentiable[Array[(), Float64]]
    """Total structural mass [t]. The optimization objective."""

    utilization: Differentiable[Array[(None,), Float64]]
    """Should be 1.0 +/- 1e-9 by construction. An assertion target, not a goal."""

    governing: Array[(None,), Float64]
    """Limit state code per member. NON-DIFFERENTIABLE.

    Pop this before calling jax.grad: a concrete cotangent on a non-differentiable
    output raises ValueError. Log it every iteration — repeated flips between
    steps mean the optimizer is chattering across a branch boundary.
    """


# --------------------------------------------------------------------------- #
# CHS section properties (closed form, t = d / r)
# --------------------------------------------------------------------------- #
def area(d: jnp.ndarray, r: jnp.ndarray) -> jnp.ndarray:
    """A = pi d^2 (r - 1) / r^2."""
    return jnp.pi * d**2 * (r - 1.0) / r**2


def second_moment(d: jnp.ndarray, r: jnp.ndarray) -> jnp.ndarray:
    """I = (pi/64) d^4 [1 - (1 - 2/r)^4]."""
    return (jnp.pi / 64.0) * d**4 * (1.0 - (1.0 - 2.0 / r) ** 4)


# --------------------------------------------------------------------------- #
# Resistances
# --------------------------------------------------------------------------- #
def chi_buckling(d, length, r, f_y, e_mod, alpha_imp, k_cr):
    """Flexural buckling reduction factor, EN 1993-1-1 §6.3.1.1.

    lambda_bar = sqrt(A f_y / N_cr),  N_cr = pi^2 E I / L_cr^2
    Phi        = 0.5 [1 + alpha (lambda_bar - 0.2) + lambda_bar^2]
    chi        = 1 / (Phi + sqrt(Phi^2 - lambda_bar^2)),  chi <= 1

    TODO: verify the equation numbers against the standard before writing them
    into docstrings. Do not trust Blueprints' file naming for this.
    """
    a = area(d, r)
    i_sec = second_moment(d, r)
    n_cr = jnp.pi**2 * e_mod * i_sec / (k_cr * length) ** 2
    lam = jnp.sqrt(a * f_y / n_cr)
    phi = 0.5 * (1.0 + alpha_imp * (lam - 0.2) + lam**2)
    chi = 1.0 / (phi + jnp.sqrt(phi**2 - lam**2))
    return jnp.minimum(chi, 1.0), lam


def _residual(d, n_abs, length, p):
    """R(d) = N_b,Rd(d) - |N_Ed|. Strictly increasing in d, so the root is unique
    and bisection is unconditionally safe (CLAUDE.md §4)."""
    chi, _ = chi_buckling(
        d, length, p["dt_ratio"], p["f_y"], p["e_mod"], p["alpha_imp"], p["k_cr"]
    )
    return chi * area(d, p["dt_ratio"]) * p["f_y"] / p["gamma_m1"] - n_abs


# --------------------------------------------------------------------------- #
# Fully-stressed sizing map, differentiated by the implicit function theorem
# --------------------------------------------------------------------------- #
@jax.custom_vjp
def size_member(n_ed, length, p):
    """Smallest CHS diameter satisfying EN 1993-1-1 for this force and length."""
    return _size_fwd(n_ed, length, p)


def _size_fwd(n_ed, length, p):
    r, f_y = p["dt_ratio"], p["f_y"]
    n_abs = jnp.abs(n_ed)

    # Tension: closed form, no buckling. d = sqrt(N gamma_M0 r^2 / (pi (r-1) f_y))
    # Unused only because the compression branch below is still a stub; the
    # return it feeds is written out in the NotImplementedError. Do not delete.
    d_ten = jnp.sqrt(n_abs * p["gamma_m0"] * r**2 / (jnp.pi * (r - 1.0) * f_y))  # noqa: F841

    # Compression: bisection on the monotone residual.
    raise NotImplementedError(
        "Bisect _residual over [d_lo, d_hi]. Bracket from the tension solution "
        "below and ~10x above. Use lax.while_loop with a fixed iteration count "
        "so the forward pass stays jittable. Then:\n"
        "    return jnp.where(n_ed > 0, d_ten, d_comp)"
    )


def _size_fwd_res(n_ed, length, p):
    d = _size_fwd(n_ed, length, p)
    return d, (d, n_ed, length, p)


def _size_bwd(res, g):
    """The hand-derived adjoint.

    By the implicit function theorem on R(d; N, L) = 0:

        dD/dN = -(dR/dN) / (dR/dd) =  1 / (dR/dd)     [since dR/dN = -1]
        dD/dL = -(dR/dL) / (dR/dd)

    The residual is smooth and explicit, so dR/dd and dR/dL come from jax.grad of
    the *residual* — only the implicit inversion is by hand. Same pattern as the
    Newton solves in `sax`.

    Tension members take the closed-form branch: dD/dN = d / (2N), dD/dL = 0.
    """
    d, n_ed, length, p = res
    n_abs = jnp.abs(n_ed)

    dR_dd = jax.grad(_residual, argnums=0)(d, n_abs, length, p)
    dR_dL = jax.grad(_residual, argnums=2)(d, n_abs, length, p)

    dD_dN_comp = jnp.sign(n_ed) / dR_dd
    dD_dL_comp = -dR_dL / dR_dd

    is_tension = n_ed > 0
    dD_dN = jnp.where(is_tension, d / (2.0 * n_ed), dD_dN_comp)
    dD_dL = jnp.where(is_tension, 0.0, dD_dL_comp)

    return (g * dD_dN, g * dD_dL, None)


size_member.defvjp(_size_fwd_res, _size_bwd)


# --------------------------------------------------------------------------- #
# Forward
# --------------------------------------------------------------------------- #
def _forward(inputs: dict[str, Any]) -> dict[str, jnp.ndarray]:
    p = {
        k: jnp.asarray(inputs[k])
        for k in (
            "f_y",
            "e_mod",
            "dt_ratio",
            "alpha_imp",
            "k_cr",
            "gamma_m0",
            "gamma_m1",
        )
    }
    n_ed, lengths = inputs["n_ed"], inputs["lengths"]

    d = jax.vmap(size_member, in_axes=(0, 0, None))(n_ed, lengths, p)
    a = area(d, p["dt_ratio"])
    mass = inputs["rho"] * jnp.sum(a * lengths)

    chi, lam = jax.vmap(chi_buckling, in_axes=(0, 0, None, None, None, None, None))(
        d, lengths, p["dt_ratio"], p["f_y"], p["e_mod"], p["alpha_imp"], p["k_cr"]
    )

    cap = jnp.where(
        n_ed > 0,
        a * p["f_y"] / p["gamma_m0"],
        chi * a * p["f_y"] / p["gamma_m1"],
    )
    utilization = jnp.abs(n_ed) / cap

    governing = jnp.where(
        n_ed > 0,
        GOV_TENSION,
        jnp.where(chi < 1.0 - 1e-12, GOV_BUCKLING, GOV_SQUASH),
    )

    return {
        "diameter": d,
        "mass": mass,
        "utilization": utilization,
        "governing": governing,
    }


# --------------------------------------------------------------------------- #
# Tesseract endpoints
# --------------------------------------------------------------------------- #
def apply(inputs: InputSchema) -> OutputSchema:
    return _forward(inputs.model_dump())


def abstract_eval(abstract_inputs):
    n = abstract_inputs.n_ed.shape[0]
    return {
        "diameter": {"shape": (n,), "dtype": "float64"},
        "mass": {"shape": (), "dtype": "float64"},
        "utilization": {"shape": (n,), "dtype": "float64"},
        "governing": {"shape": (n,), "dtype": "float64"},
    }


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: list[str],
    vjp_outputs: list[str],
    cotangent_vector: dict[str, Any],
):
    """Reverse mode. The custom_vjp on `size_member` means jax.vjp here dispatches
    to the hand-derived adjoint rather than tracing through the bisection."""
    if "governing" in vjp_outputs:
        raise ValueError(
            "`governing` is non-differentiable — pop it before differentiating."
        )

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
