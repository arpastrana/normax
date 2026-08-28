# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array
from jaxtyping import Float

from normax.analysis import MemberForces
from normax.materials import Steel355
from normax.sections import TubeCatalog
from normax.sections import build_section_catalog
from normax.sizing import AbstractMemberSizer
from normax.sizing import MemberSizes
from normax.sizing.ec3 import Ec3Sizer
from normax.structures import Structure
from normax.structures import build_arch_2d

# The proof this file makes: the sizing contract is fillable without any
# standard's library, by a different design philosophy rather than a
# reimplementation — allowable-stress design, one global safety factor and no
# stability check. Nothing here selects a clause, so nothing here needs one.

NUM_EDGES = 6

# A wall proportion this test picks for itself: no class limit chose it.
RATIO = 50.0

# The classic ASD factor of safety on yield.
SAFETY_FACTOR = 1.67

# Invariant 6.5 of CLAUDE.md, philosophy-independent.
TOLERANCE_UTILIZATION = 1e-9

LENGTHS = jnp.full(NUM_EDGES, 2000.0)


class AllowableStressSizer(AbstractMemberSizer):
    """
    Allowable-stress design, as a block of the design pipeline.

    Attributes
    ----------
    structure :
        The structure whose members are sized. Read for nothing.
    catalog :
        The section catalog every member is drawn from, and its grade.
    safety_factor :
        Factor of safety on yield defining the allowable stress.

    Notes
    -----
    Deliberately naive: one allowable stress against the axial force alone —
    no buckling, no moment, no partial-factor format. With the wall
    proportional to the diameter the area is quadratic in it, so the
    fully-stressed size is a square root and no root find is needed.
    """

    structure: Structure
    catalog: TubeCatalog
    safety_factor: float = eqx.field(static=True, default=SAFETY_FACTOR)

    def allowable_stress(self) -> Float[Array, ""]:
        """
        The one number this philosophy checks against.
        """
        return jnp.asarray(self.catalog.material.f_y) / self.safety_factor

    def __call__(
        self,
        forces: MemberForces,
        buckling_length: Float[Array, "members"],
    ) -> MemberSizes:
        """
        Size every member for every load case; the length is ignored.
        """
        ratio = jnp.asarray(self.catalog.ratio)
        demanded_area = jnp.abs(forces.axial_force) / self.allowable_stress()
        diameter = jnp.sqrt(demanded_area * ratio**2 / (jnp.pi * (ratio - 1.0)))
        sections = self.catalog(diameter)
        used = jnp.abs(forces.axial_force) / (sections.area * self.allowable_stress())

        return MemberSizes(sections, used)

    def compute_utilization(
        self,
        diameters: Float[Array, "members"],
        forces: MemberForces,
        buckling_length: Float[Array, "members"],
    ) -> Float[Array, "load_cases members"]:
        """
        Check sizes the caller owns against the allowable stress.
        """
        sections = self.catalog(diameters)

        return jnp.abs(forces.axial_force) / (sections.area * self.allowable_stress())


@pytest.fixture(scope="module")
def structure():
    return build_arch_2d(num_edges=NUM_EDGES)


@pytest.fixture(scope="module")
def sizer(structure):
    return AllowableStressSizer(structure, TubeCatalog(RATIO, Steel355()))


@pytest.fixture(scope="module")
def forces():
    generator = np.random.default_rng(20260826)
    axial = jnp.asarray(generator.uniform(-6.0e5, -1.0e5, (2, NUM_EDGES)))
    major = jnp.asarray(generator.uniform(-1.0e6, 1.0e6, (2, NUM_EDGES, 2)))
    minor = jnp.asarray(generator.uniform(-1.0e5, 1.0e5, (2, NUM_EDGES, 2)))

    return MemberForces(axial, major, minor)


def test_this_file_names_no_standard_library():
    # The one EC3 name here is normax's own adapter, imported to be disagreed with.
    source = Path(__file__).read_text()
    imported = [line for line in source.splitlines() if line.startswith("from ")]

    assert not any("ec3x" in line for line in imported)


def test_the_contract_imports_no_standard():
    # `import normax.sizing` must pull neither clause library along.
    script = (
        "import sys, normax.sizing; "
        "assert 'ec3x' not in sys.modules and 'blueprints' not in sys.modules"
    )
    finished = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )

    assert finished.returncode == 0, finished.stderr


def test_a_sizer_without_a_catalog_cannot_be_built(structure):
    class CataloglessSizer(AbstractMemberSizer):
        structure: Structure

        def compute_utilization(self, diameters, forces, buckling_length):
            return jnp.zeros_like(forces.axial_force)

        def __call__(self, forces, buckling_length):
            return MemberSizes(None, jnp.zeros_like(forces.axial_force))

    with pytest.raises(TypeError):
        CataloglessSizer(structure)


def test_a_second_philosophy_fills_the_contract(sizer, forces):
    sizes = sizer(forces, LENGTHS)

    assert isinstance(sizes, MemberSizes)
    assert isinstance(sizer.catalog, TubeCatalog)
    assert np.all(np.asarray(sizes.sections.diameter) > 0.0)


def test_the_second_sizer_is_fully_stressed_too(sizer, forces):
    sizes = sizer(forces, LENGTHS)

    assert np.allclose(
        np.asarray(sizes.utilization), 1.0, rtol=0.0, atol=TOLERANCE_UTILIZATION
    )


def test_the_reread_agrees_with_the_sizes(sizer, forces):
    sizes = sizer(forces, LENGTHS)
    reread = sizer.compute_utilization(sizes.sections.diameter[0], forces, LENGTHS)

    assert np.allclose(np.asarray(reread[0]), 1.0, rtol=0.0, atol=TOLERANCE_UTILIZATION)


def test_the_check_differentiates_in_the_diameters(sizer, forces):
    def total(diameters):
        return jnp.sum(sizer.compute_utilization(diameters, forces, LENGTHS))

    gradient = jax.grad(total)(jnp.full(NUM_EDGES, 100.0))

    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.all(np.asarray(gradient) < 0.0)


def test_the_two_philosophies_disagree_about_the_sizes(structure, sizer, forces):
    # A different standard, not a reimplementation: EC3 sees buckling and
    # bending and this sizer sees neither, so compressed members differ.
    limit_state = Ec3Sizer(structure, build_section_catalog(Steel355(), 3))

    naive = sizer(forces, LENGTHS).sections.diameter
    checked = limit_state(forces, LENGTHS).sections.diameter

    assert not np.allclose(np.asarray(naive), np.asarray(checked), rtol=1e-2)
