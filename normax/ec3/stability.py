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
Global stability, and the standard's second route to a member slenderness.

**Soft validation, deliberately outside the pipeline** (decided 2026-08-09).
Nothing here sizes a member, enters a gradient, or crosses a Tesseract boundary.
It is read after a design is finished, to say how far the buckling length that
produced it can be trusted. Global stability is therefore **not covered** by what
this package designs, and the writeup states that as a limitation rather than
implying otherwise.

Keeping it out is a scope decision and a cheap one. Feeding a critical load factor
back into the schema would oblige every analysis backend to supply one, which the
OpenSees backend cannot without real work, and would trade the thesis — that a
design code can carry an adjoint and compose — for a second structural feature.

EN 1993-1-1 offers two routes to the slenderness that drives the reduction factor.
§6.3.1.3 Eq. 6.50 asks a **member**: pick a buckling length, and the slenderness
follows. §6.3.4(3) asks the **structure**: find the load factor at which it
becomes unstable, and the slenderness follows from that instead. For pure
compression the two are the same equation — `α_ult,k / α_cr` reduces to
`A f_y / N_cr`, which is `λ̄²` — so either may be fed to the same `χ`.

They differ in what they are asked about, not in what they compute. A buckling
length is an assumption about how a member is held; a critical load factor is a
property of the whole frame. Where the assumption is wrong the two answers
diverge, and the size of the divergence is the size of the assumption.

**§6.3.4 is an out-of-plane clause and this is not an out-of-plane use of it.**
Its `α_cr,op` is the amplifier reaching instability in a lateral or
lateral-torsional mode, taking no account of in-plane flexural buckling, while a
planar frame's mode is in-plane by construction. The clause is cited for where
the standard writes this algebra, not as authority for the case; the identity
itself needs no source and is tested as one.
"""

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Bool
from jaxtyping import Float

# EN 1993-1-1 §5.2.1(3): the critical load factor above which second-order
# effects need not be accounted for. UK NA clause NA.2.9 moves only the plastic.
ALPHA_CR_ELASTIC = 10.0
ALPHA_CR_PLASTIC = 15.0

# EN 1993-1-1 §5.2.2(5): below this the sway amplifier is inadmissible and a
# second-order analysis is required outright.
ALPHA_CR_AMPLIFIABLE = 3.0


def slenderness_global(
    alpha_ult_k: Float[Array, "members"],
    alpha_cr: Float[Array, "members"],
) -> Float[Array, "members"]:
    """
    Slenderness from a critical load factor rather than a buckling length.

    Parameters
    ----------
    alpha_ult_k :
        Load amplifier reaching the characteristic cross-section resistance.
    alpha_cr :
        Load amplifier reaching elastic instability.

    Returns
    -------
    slenderness :
        Non-dimensional slenderness.

    Notes
    -----
    EN 1993-1-1 §6.3.4(3), whose `α_cr,op` is an out-of-plane amplifier; the
    algebra is general and a planar frame's mode is in-plane.

    The same quantity Eq. 6.50 returns from a buckling length, so its result may
    be passed to the same reduction factor. Both amplifiers are ratios to the
    same design load, so that load cancels and only the two resistances remain.
    """
    return jnp.sqrt(alpha_ult_k / alpha_cr)


def resistance_factor(
    area: Float[Array, "members"],
    f_y: float | Float[Array, ""],
    n_ed: Float[Array, "members"],
) -> Float[Array, "members"]:
    """
    Load amplifier at which a member reaches its squash resistance.

    Parameters
    ----------
    area :
        Gross cross-sectional area.
    f_y :
        Yield strength.
    n_ed :
        Design axial force, tension positive.

    Returns
    -------
    factor :
        Multiple of the design load reaching the characteristic resistance.

    Notes
    -----
    `α_ult,k` of EN 1993-1-1 §6.3.4(2), for pure compression, where the
    characteristic cross-section resistance is the squash load.

    Only the magnitude of the axial force is read, so a member in tension returns
    the amplifier of its squash load too. That is meaningless for stability and is
    the caller's business to exclude, exactly as the member check excludes tension
    from 6.3.3.

    **A member carrying no axial force returns nan, not infinity.** The amplifier
    is a ratio to the load the member carries, and there is no such ratio when it
    carries nothing — a gridshell's boundary hoops, spanning support to support,
    are the usual case. Infinity would read as a statement about that member;
    nan says the question does not apply to it, and any reduction over the
    members says so too rather than quietly absorbing it.
    """
    loaded = jnp.abs(n_ed) > 0.0
    safe = jnp.where(loaded, jnp.abs(n_ed), 1.0)

    return jnp.where(loaded, area * f_y / safe, jnp.nan)


def critical_force(
    alpha_cr: Float[Array, "members"],
    n_ed: Float[Array, "members"],
) -> Float[Array, "members"]:
    """
    Elastic critical force implied by a critical load factor.

    Parameters
    ----------
    alpha_cr :
        Load amplifier reaching elastic instability.
    n_ed :
        Design axial force, tension positive.

    Returns
    -------
    n_cr :
        Elastic critical force.

    Notes
    -----
    `α_cr = F_cr/F_Ed` of EN 1993-1-1 §5.2.1(3) read for one member: the factor
    scales the load, so it scales the member's share of it.
    """
    return alpha_cr * jnp.abs(n_ed)


def buckling_length(
    alpha_cr: Float[Array, "members"],
    n_ed: Float[Array, "members"],
    second_moment: Float[Array, "members"],
    e_mod: float | Float[Array, ""],
) -> Float[Array, "members"]:
    """
    Buckling length a critical load factor is equivalent to.

    Parameters
    ----------
    alpha_cr :
        Load amplifier reaching elastic instability.
    n_ed :
        Design axial force, tension positive.
    second_moment :
        Second moment of area.
    e_mod :
        Modulus of elasticity.

    Returns
    -------
    l_cr :
        Buckling length reproducing the same slenderness.

    Notes
    -----
    `N_cr = π² E I / L_cr²` inverted. Reports a global mode in the units a member
    check speaks, which is what makes an assumed buckling length comparable with
    the mode the structure actually has: on a shallow arch the ratio between them
    is several times, and it is mesh-independent where an assumed member length
    is not.

    **Returns nan for a member carrying no axial force**, for the reason
    `resistance_factor` does: a factor scaling the whole load says nothing about a
    member the load never reaches.
    """
    critical = critical_force(alpha_cr, n_ed)
    loaded = critical > 0.0
    safe = jnp.where(loaded, critical, 1.0)

    return jnp.where(loaded, jnp.pi * jnp.sqrt(e_mod * second_moment / safe), jnp.nan)


def utilization(
    alpha_cr: float | Float[Array, ""],
    threshold: float = ALPHA_CR_ELASTIC,
) -> Float[Array, ""]:
    """
    Demand over resistance for the global stability of a frame.

    Parameters
    ----------
    alpha_cr :
        Smallest critical load factor of the frame.
    threshold :
        Factor the frame must reach. Defaults to the elastic value.

    Returns
    -------
    utilization :
        At most one where first-order analysis is adequate.

    Notes
    -----
    EN 1993-1-1 §5.2.1(3), written as a utilization so that it reads like every
    other check here.

    A frame with a factor below one is unstable before it is loaded to its design
    value, and this returns a utilization above the threshold to say so rather
    than reporting a separate condition.
    """
    return threshold / jnp.asarray(alpha_cr)


def is_adequate(
    alpha_cr: float | Float[Array, ""],
    threshold: float = ALPHA_CR_ELASTIC,
) -> Bool[Array, ""]:
    """
    Whether first-order analysis is adequate for a frame.

    Parameters
    ----------
    alpha_cr :
        Smallest critical load factor of the frame.
    threshold :
        Factor the frame must reach. Defaults to the elastic value.

    Returns
    -------
    adequate :
        True where the frame is stiff enough for second-order effects to be
        neglected.

    Notes
    -----
    EN 1993-1-1 §5.2.1(3). **Non-differentiable**, being a verdict rather than a
    magnitude; read `utilization` when a gradient is wanted.
    """
    return utilization(alpha_cr, threshold) <= 1.0


def amplification(alpha_cr: float | Float[Array, ""]) -> Float[Array, ""]:
    """
    Factor by which sway effects are magnified in a second-order response.

    Parameters
    ----------
    alpha_cr :
        Smallest critical load factor of the frame.

    Returns
    -------
    amplification :
        Multiplier on the first-order sway effects.

    Notes
    -----
    EN 1993-1-1 §5.2.2(5), which carries no equation number of its own. It
    amplifies the horizontal loads and the equivalent loads from imperfections.

    **Admissible only for a factor above `ALPHA_CR_AMPLIFIABLE`**; the expression
    is returned regardless, since clamping it would hide the very case that needs
    a second-order analysis instead. It passes through a pole at a factor of one
    and turns negative below it, which is the arithmetic saying the frame has
    already buckled rather than a defect to be smoothed.
    """
    factor = jnp.asarray(alpha_cr)

    return 1.0 / (1.0 - 1.0 / factor)
