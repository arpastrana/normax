import jax
import jax.numpy as jnp
from ec3x.material import Steel

from normax.materials import SteelGrade
from normax.sizing.ec3 import design_steel


def test_the_default_grade_is_the_default_steel():
    # The certificate half of the two records must be the same S355, or the
    # neutral grade and the standard's material quietly describe two steels.
    grade = SteelGrade()
    steel = Steel()

    assert grade.f_y == steel.f_y
    assert grade.f_u == steel.f_u
    assert grade.e_mod == steel.e_mod
    assert grade.density == steel.density


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
    steel = design_steel(SteelGrade())

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
