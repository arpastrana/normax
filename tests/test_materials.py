# SPDX-License-Identifier: Apache-2.0
"""
The steel grade, checked against the standard's own numbers.

The right-hand side is the literal a table states rather than a second
library's material record, so the grade is held to the standard and not to
another implementation of it.
"""

import jax
import jax.numpy as jnp
import pytest

from normax.materials import Steel355
from normax.materials import SteelGrade


def test_the_grade_states_its_certificate():
    # Eurocode 3 Table 3.1, S355 up to 40 mm thick, in newtons per square millimeter.
    s355 = Steel355()

    assert (s355.f_y, s355.f_u) == (355.0, 490.0)

    # Eurocode 3 3.2.6 for the modulus; the density is 7850 kg/m3 by convention.
    assert s355.e_mod == 210000.0
    assert s355.density == 7.85e-9


def test_a_bare_grade_names_its_strengths_or_is_refused():
    # A default strength is a grade chosen silently, so there is none.
    with pytest.raises(TypeError):
        SteelGrade()


def test_a_grade_defaults_only_what_every_steel_shares():
    # The modulus and the density are the same for every structural steel.
    grade = SteelGrade(f_y=460.0, f_u=540.0)

    assert grade.e_mod == 210000.0
    assert grade.density == 7.85e-9


def test_a_named_grade_survives_a_pytree_round_trip():
    # JAX rebuilds a namedtuple positionally, so keep the base signature.
    leaves, treedef = jax.tree.flatten(Steel355())
    rebuilt = jax.tree.unflatten(treedef, leaves)

    assert type(rebuilt) is Steel355
    assert rebuilt == Steel355()


def test_a_grade_carries_no_clause_field():
    assert set(SteelGrade._fields) == {"f_y", "f_u", "e_mod", "density"}


def test_every_field_of_a_grade_is_differentiable():
    def yielded(grade):
        return grade.f_y * grade.e_mod

    grade = SteelGrade(
        jnp.asarray(355.0), jnp.asarray(490.0), jnp.asarray(210000.0), 7.85e-9
    )
    gradient = jax.grad(yielded)(grade)

    assert float(gradient.f_y) == 210000.0
    assert float(gradient.e_mod) == 355.0
