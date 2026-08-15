import jax
import jax.numpy as jnp
import pytest
from ec3x.material import Steel

from normax.materials import Steel235
from normax.materials import Steel355
from normax.materials import SteelGrade
from normax.sizing.ec3 import design_steel


def test_the_default_grade_is_the_default_steel():
    # The certificate half of the two records must be the same S355, or the
    # neutral grade and the standard's material quietly describe two steels.
    grade = Steel355()
    steel = Steel()

    assert grade.f_y == steel.f_y
    assert grade.f_u == steel.f_u
    assert grade.e_mod == steel.e_mod
    assert grade.density == steel.density


def test_each_grade_states_its_certificate():
    s355 = Steel355()
    s235 = Steel235()

    assert (s355.f_y, s355.f_u) == (355.0, 490.0)
    assert (s235.f_y, s235.f_u) == (235.0, 360.0)
    assert s235.e_mod == s355.e_mod
    assert s235.density == s355.density


def test_a_bare_grade_names_its_strengths_or_is_refused():
    # A default strength is a grade chosen silently, so there is none.
    with pytest.raises(TypeError):
        SteelGrade()


def test_a_named_grade_survives_a_pytree_round_trip():
    # JAX rebuilds a namedtuple positionally as type(x)(*leaves), so the
    # subclasses must keep the base constructor signature.
    leaves, treedef = jax.tree.flatten(Steel235())
    rebuilt = jax.tree.unflatten(treedef, leaves)

    assert type(rebuilt) is Steel235
    assert rebuilt == Steel235()


def test_the_standard_reads_a_grade_without_changing_it():
    grade = SteelGrade(f_y=460.0, f_u=540.0)
    steel = design_steel(grade)

    assert steel.f_y == grade.f_y
    assert steel.f_u == grade.f_u
    assert steel.e_mod == grade.e_mod
    assert steel.density == grade.density


def test_the_standard_adds_only_its_own_factors():
    # What the reading adds is the clause half at its defaults, so a grade
    # crossed into EC3 is exactly the default Steel wherever it says nothing.
    steel = design_steel(Steel355())

    assert steel == Steel()


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
