from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array
from jaxtyping import Float

from normax.analysis import MemberForces
from normax.analysis.smax import SmaxAnalyzer
from normax.design import DesignParameters
from normax.design import StructuralDesignPipeline
from normax.design import compute_mass
from normax.design import design_envelope
from normax.form_finding.fdm import FdmFormFinder
from normax.loads import assemble_load_cases as load_cases_of
from normax.loads import loads_uniform
from normax.materials import Steel355
from normax.sections import TubeFamily
from normax.sizing import AbstractMemberSizer
from normax.sizing import MemberSizes
from normax.sizing.ec3 import Ec3Sizer
from normax.sizing.ec3 import thinnest_family
from normax.structures import Structure
from normax.structures import build_arch_2d

# The proof this file exists to make: the sizing contract is fillable without
# the EC3 library, by a *different design philosophy* rather than a
# reimplementation — allowable-stress design, the pre-limit-state format, one
# global safety factor and no stability check. Nothing here selects a clause,
# so nothing here needs one.

SPAN = 10_000.0
RISE = 3_000.0
TOTAL_LOAD = 180_000.0
NUM_EDGES = 10

# The diameter the frame is analyzed with before the check has spoken.
SEED = 100.0

# A wall proportion this test picks for itself: no class limit chose it,
# because no standard is present to have an opinion.
RATIO = 50.0

# The classic ASD factor of safety on yield.
SAFETY_FACTOR = 1.67

# Invariant 6.5 of CLAUDE.md, philosophy-independent: a fully-stressed sizer
# returns sizes worked to exactly one.
TOLERANCE_UTILIZATION = 1e-9


class AllowableStressSizer(AbstractMemberSizer):
    """
    Allowable-stress design, as a block of the design pipeline.

    Attributes
    ----------
    structure :
        The structure whose members are sized. Read for nothing.
    family :
        The section family every member is drawn from, and its grade.
    safety_factor :
        Factor of safety on yield defining the allowable stress.

    Notes
    -----
    Deliberately naive: one allowable stress, `f_y` over a factor of safety,
    against the axial force alone — no buckling, no moment interaction, no
    partial-factor format. The point is not that this is a good standard; it is
    that a philosophy this different fills the same contract, because the
    contract carries forces in, sections and a utilization out, and no field of
    either names any standard's vocabulary.

    The fully-stressed size is closed form. With the wall proportional to the
    diameter, the area is quadratic in it, so the diameter that works the
    allowable stress exactly is a square root — no residual, no root find.
    """

    structure: Structure
    family: TubeFamily
    safety_factor: float = eqx.field(static=True, default=SAFETY_FACTOR)

    def allowable_stress(self) -> Float[Array, ""]:
        """
        The one number this philosophy checks against.
        """
        return jnp.asarray(self.family.material.f_y) / self.safety_factor

    def __call__(
        self,
        forces: MemberForces,
        buckling_length: Float[Array, "members"],
    ) -> MemberSizes:
        """
        Size every member for every load case, each on its own.

        The buckling length is accepted and ignored, which is this philosophy's
        statement rather than an oversight: allowable-stress design as written
        here checks stress and nothing else.
        """
        ratio = jnp.asarray(self.family.ratio)
        demanded_area = jnp.abs(forces.axial_force) / self.allowable_stress()
        diameter = jnp.sqrt(demanded_area * ratio**2 / (jnp.pi * (ratio - 1.0)))
        sections = self.family(diameter)
        used = jnp.abs(forces.axial_force) / (sections.area * self.allowable_stress())

        return MemberSizes(sections, used)

    def utilization(
        self,
        diameters: Float[Array, "members"],
        forces: MemberForces,
        buckling_length: Float[Array, "members"],
    ) -> Float[Array, "load_cases members"]:
        """
        Re-read a finished design against the stress that sized it.
        """
        sections = self.family(diameters)

        return jnp.abs(forces.axial_force) / (sections.area * self.allowable_stress())


@pytest.fixture(scope="module")
def structure():
    return build_arch_2d(num_edges=NUM_EDGES, span=SPAN, rise=RISE)


@pytest.fixture(scope="module")
def pipeline(structure):
    grade = Steel355()
    family = TubeFamily(RATIO, grade)

    return StructuralDesignPipeline(
        FdmFormFinder(structure),
        SmaxAnalyzer(structure, family(SEED)),
        AllowableStressSizer(structure, family),
    )


@pytest.fixture(scope="module")
def params(structure):
    trial = jnp.full(NUM_EDGES, -1.0)
    shape = FdmFormFinder(structure)(trial, funicular(structure))
    reached = jnp.max(shape.xyz[:, 2])

    return DesignParameters(trial * reached / RISE, jnp.full(NUM_EDGES, SEED))


def funicular(structure):
    """
    The uniform load case the arch is form-found under.
    """
    return loads_uniform(structure, TOTAL_LOAD / (NUM_EDGES - 1))


@pytest.fixture(scope="module")
def one_case(structure):
    return load_cases_of([funicular(structure)])


def test_this_file_names_no_standard_library():
    # The drift alarm for the whole claim: the second sizer only proves the
    # seam if it is written without the EC3 library. The one EC3 name here is
    # normax's own adapter, imported to be disagreed with in the last test.
    source = Path(__file__).read_text()
    imported = [line for line in source.splitlines() if line.startswith("from ")]

    assert not any("ec3x" in line for line in imported)


def test_a_second_philosophy_fills_the_contract(pipeline, params, one_case):
    design = pipeline(params, one_case)

    assert isinstance(design.sizes, MemberSizes)
    assert np.all(np.asarray(design.sizes.sections.diameter) > 0.0)


def test_the_second_sizer_is_fully_stressed_too(pipeline, params, one_case):
    # The invariant is the sizing map's, not EN 1993-1-1's: whatever the
    # philosophy, the size returned is worked to exactly one.
    design = pipeline(params, one_case)

    assert np.allclose(
        np.asarray(design.sizes.utilization),
        1.0,
        rtol=0.0,
        atol=TOLERANCE_UTILIZATION,
    )


def test_the_reread_agrees_with_the_sizes(pipeline, params, one_case):
    design = pipeline(params, one_case)
    reread = pipeline.sizer.utilization(
        design.sizes.sections.diameter[0], design.forces, design.shape.lengths
    )

    assert np.allclose(np.asarray(reread), 1.0, rtol=0.0, atol=TOLERANCE_UTILIZATION)


def test_the_mass_still_differentiates_end_to_end(pipeline, params, one_case):
    # The composition's whole point survives the swap: force densities to a
    # mass, one exact gradient across all three blocks, no block asked how.
    def objective(q):
        design = pipeline(DesignParameters(q, params.diameters), one_case)

        return compute_mass(design_envelope(design))

    gradient = jax.grad(objective)(params.force_densities)

    assert np.all(np.isfinite(np.asarray(gradient)))
    assert float(jnp.min(jnp.abs(gradient))) > 0.0


def test_the_two_philosophies_disagree_about_the_sizes(
    structure, pipeline, params, one_case
):
    # A different standard, not a reimplementation: EC3 sees buckling and this
    # sizer does not, so a compressed arch is sized differently by the two.
    limit_state = StructuralDesignPipeline(
        pipeline.formfinder,
        pipeline.analyzer,
        Ec3Sizer(structure, thinnest_family(Steel355(), 3)),
    )

    naive = pipeline(params, one_case).sizes.sections.diameter
    checked = limit_state(params, one_case).sizes.sections.diameter

    assert not np.allclose(np.asarray(naive), np.asarray(checked), rtol=1e-2)
