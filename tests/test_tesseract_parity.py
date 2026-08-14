from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tesseract_jax import apply_tesseract

from normax.analysis.smax import SmaxAnalyzer
from normax.design import DesignParameters
from normax.design import DesignPipeline
from normax.design import calculate_mass
from normax.design import load_cases as load_cases_of
from normax.ec3.material import Steel
from normax.ec3.section import TubeCatalogue
from normax.form_finding.fdm import FdmFormFinder
from normax.sizing import Ec3Sizer
from normax.structures import Structure
from normax.structures import arch_2d
from normax.structures import loads_half_span
from normax.structures import loads_point
from normax.structures import loads_uniform
from normax.tesseract import STAGES
from normax.tesseract import Chain
from normax.tesseract import TesseractAnalyzer
from normax.tesseract import TesseractFormFinder
from normax.tesseract import TesseractSizer
from normax.tesseract import local_chain

# The same 10 m arch rising 3 m under 180 kN that the in-process pipeline is
# tested on, so the two are compared on identical ground.
SPAN = 10_000.0
RISE = 3_000.0
TOTAL_LOAD = 180_000.0
NUM_EDGES = 10

# The arch lies in the XZ plane, so it has no thickness along Y.
NORMAL = 1

# The diameter the frame is analyzed with before the check has spoken.
SEED = 100.0

# The boundary serializes float64 losslessly and both sides run the same code, so
# parity is exact rather than approximate. Both sides are compiled for this to
# hold: comparing a compiled composition against an eager oracle measures the
# arithmetic instead, which costs two orders.
TOLERANCE_PARITY = 1e-14

# An enveloped design is looser, and the cause is where the programs are cut. In
# process the three load cases compile into one program; across the boundary one
# solve is compiled and called three times, so the same sums are accumulated in
# different units.
TOLERANCE_PARITY_ENVELOPE = 1e-12

# The end moments are the exception, and the reason is the arch rather than the
# boundary. A funicular shape carries its design case axially, so the moment is a
# near-cancellation worth 4e-4 of the axial action times the length, and its
# relative precision is set by that larger scale. A single last-bit difference in
# the analysis inputs therefore reaches far further here than in the axial force
# it came from.
#
# The moment factors read a ratio of the two end moments, so they inherit it.
TOLERANCE_MOMENT = 1e-11
MOMENT_FIELDS = (
    "moment_major",
    "moment_minor",
    "moment_factor_major",
    "moment_factor_minor",
)

# Derivatives are looser than values, and not because of the boundary. Each
# stage linearizes on its own here and all three linearize together in process,
# so the same sum is accumulated in a different order and the implicit tangent
# divides by a slope that differs in its last bits.
TOLERANCE_DERIVATIVE = 5e-12

# Invariant 6.5 of CLAUDE.md. Measured at 1.8e-15 through the boundary.
TOLERANCE_UTILIZATION = 1e-9

# Relative step at which the central difference plateaus, and the agreement
# measured there, scaled by the largest component of the gradient.
STEP = 1e-5
TOLERANCE_GRADIENT = 5e-8


class ArchProblem(NamedTuple):
    """
    Everything both routes are built from, so a helper takes one argument.
    """

    structure: Structure
    chain: Chain
    steel: Steel
    params: DesignParameters


@pytest.fixture(scope="module")
def steel():
    return Steel()


@pytest.fixture(scope="module")
def chain():
    return local_chain()


@pytest.fixture(scope="module")
def structure():
    return arch_2d(num_edges=NUM_EDGES, span=SPAN, rise=RISE)


def funicular(structure):
    """
    The uniform load case the arch is form-found under.
    """
    return loads_uniform(structure, TOTAL_LOAD / (NUM_EDGES - 1))


@pytest.fixture(scope="module")
def arch(structure, chain, steel):
    """
    The arch, the three Tesseracts, and the `q` that reaches the target rise.
    """
    trial = jnp.full(NUM_EDGES, -1.0)
    shape = FdmFormFinder().compile(structure)(trial, funicular(structure))
    reached = jnp.max(shape.xyz[:, 2])

    params = DesignParameters(trial * reached / RISE, jnp.full(NUM_EDGES, SEED))

    return ArchProblem(structure, chain, steel, params)


@pytest.fixture(scope="module")
def one_case(structure):
    applied = funicular(structure)

    return load_cases_of(applied, [applied])


@pytest.fixture(scope="module")
def three_cases(structure):
    """
    Three cases of equal total: funicular, half span, and a crown point load.
    """
    spread = TOTAL_LOAD / (NUM_EDGES - 1)

    half = loads_half_span(structure, spread, factor=0.5)
    half = half * (TOTAL_LOAD / abs(float(jnp.sum(half[:, 2]))))

    point = loads_uniform(structure, spread * 0.75) + loads_point(
        structure, TOTAL_LOAD * 0.25, node=structure.crown_node()
    )

    cases = [loads_uniform(structure, spread), half, point]

    return load_cases_of(funicular(structure), cases)


def both_pipelines(arch, section_class, resultant=True):
    """
    One pipeline, twice: three blocks in process, and the same three as
    Tesseracts.

    Notes
    -----
    The claim the whole module exists to make is visible in the two calls below:
    `DesignPipeline` is the same class either way, and nothing it does depends
    on which blocks it was handed. The in-process side is compiled, so what is
    measured is the boundary rather than two fusion schedules: the Tesseract
    stages compile internally, and an eager oracle beside them would charge the
    difference to the boundary.

    **The composed side is deliberately left eager.** Compiling it works, and
    `experiments/10_arch_pipeline_tesseract.py` does exactly that, but in a
    pytest session that has already run the OpenSees backend it closes a file
    descriptor and every later test errors out of capture rather than failing an
    assertion. Nothing here needs it: the stages compile behind the boundary.
    """
    steel = arch.steel
    catalogue = TubeCatalogue.at_class_limit(steel.f_y, section_class)

    in_process = DesignPipeline(
        FdmFormFinder(),
        SmaxAnalyzer(steel, catalogue, NORMAL),
        Ec3Sizer(steel, catalogue, resultant),
    ).compile(arch.structure)

    composed = DesignPipeline(
        TesseractFormFinder(arch.chain.formfinding),
        TesseractAnalyzer(arch.chain.analysis, steel, catalogue, NORMAL),
        TesseractSizer(arch.chain.ec3, steel, catalogue, resultant),
    ).compile(arch.structure)

    return eqx.filter_jit(in_process), composed


def both(arch, loads, section_class, sharpness=None, **kwargs):
    """
    The same design taken in process and across the three Tesseracts.
    """
    in_process, composed = both_pipelines(arch, section_class)
    oracle = in_process(arch.params, loads, sharpness, **kwargs)
    crossed = composed(arch.params, loads, sharpness, **kwargs)

    return oracle, crossed


def objectives(arch, loads, section_class, sharpness=None):
    """
    The mass as a function of the force densities, by both routes.
    """
    in_process, composed = both_pipelines(arch, section_class)
    seed = arch.params.diameters

    def oracle(q):
        design = in_process(DesignParameters(q, seed), loads, sharpness)
        return calculate_mass(design)

    def crossed(q):
        design = composed(DesignParameters(q, seed), loads, sharpness)
        return calculate_mass(design)

    return oracle, crossed


def relative(oracle, composed):
    """
    Largest disagreement between two arrays, scaled by the size of the first.
    """
    left = np.asarray(oracle, dtype=np.float64)
    right = np.asarray(composed, dtype=np.float64)
    scale = max(float(np.max(np.abs(left))), np.finfo(np.float64).tiny)

    return float(np.max(np.abs(left - right))) / scale


def named_fields(container):
    """
    Every field of a result, with a nested container expanded one level.
    """
    for field in container._fields:
        value = getattr(container, field)
        if hasattr(value, "_fields"):
            for inner in value._fields:
                yield f"{field}.{inner}", getattr(value, inner)
        else:
            yield field, value


def field_by_field(oracle, composed, limit_envelope):
    """
    Every field of two designs, with the limit each one is held to.
    """
    for (label, left), (_, right) in zip(named_fields(oracle), named_fields(composed)):
        leaf = label.rpartition(".")[2]
        limit = TOLERANCE_MOMENT if leaf in MOMENT_FIELDS else limit_envelope

        yield label, left, right, limit


# --------------------------------------------------------------------------- #
# The claim the whole step exists to make
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("section_class", [2, 3])
def test_the_composed_mass_is_the_in_process_mass(arch, one_case, section_class):
    oracle, composed = both(arch, one_case, section_class)

    assert relative(calculate_mass(oracle), calculate_mass(composed)) < TOLERANCE_PARITY


@pytest.mark.parametrize("section_class", [2, 3])
def test_every_field_of_the_design_survives_the_boundary(arch, one_case, section_class):
    # Mass alone would pass on a cancellation of two errors. Comparing the
    # geometry, the member actions, the sizes and the utilization pins where any
    # disagreement entered.
    oracle, composed = both(arch, one_case, section_class)

    for label, left, right, limit in field_by_field(oracle, composed, TOLERANCE_PARITY):
        assert relative(left, right) < limit, label


@pytest.mark.parametrize("section_class", [2, 3])
def test_the_mass_gradient_survives_the_boundary(arch, one_case, section_class):
    oracle, composed = objectives(arch, one_case, section_class)

    difference = relative(
        jax.grad(oracle)(arch.params.q), jax.grad(composed)(arch.params.q)
    )

    assert difference < TOLERANCE_DERIVATIVE


def test_a_buckling_length_given_explicitly_crosses_unchanged(arch, one_case):
    # The buckling length is an input rather than a mesh length, so it has to
    # reach the check as itself and not as the member length beside it.
    buckling_length = jnp.full(NUM_EDGES, 1_000.0)
    oracle, composed = both(arch, one_case, 3, buckling_length=buckling_length)

    assert relative(calculate_mass(oracle), calculate_mass(composed)) < TOLERANCE_PARITY
    assert np.allclose(np.asarray(composed.buckling_length), 1_000.0)


def test_the_linear_sum_reading_of_the_moments_crosses_unchanged(arch, one_case):
    # `resultant` selects a clause, so it crosses as a static field and a wrong
    # default would be invisible in the mass alone.
    in_process, composed = both_pipelines(arch, 3, resultant=False)
    oracle = in_process(arch.params, one_case)
    crossed = composed(arch.params, one_case)

    assert relative(calculate_mass(oracle), calculate_mass(crossed)) < TOLERANCE_PARITY


# --------------------------------------------------------------------------- #
# The gradient, end to end
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("section_class", [2, 3])
def test_the_composed_gradient_matches_central_differences(
    arch, one_case, section_class
):
    # Parity says the boundary changed nothing. This says the thing it left
    # unchanged is right, without the in-process pipeline vouching for it.
    _, composed = objectives(arch, one_case, section_class)

    q = arch.params.q
    gradient = jax.grad(composed)(q)
    scale = float(jnp.max(jnp.abs(gradient)))

    for edge in (0, NUM_EDGES // 2, NUM_EDGES - 1):
        step = abs(float(q[edge])) * STEP
        plus = composed(q.at[edge].add(step))
        minus = composed(q.at[edge].add(-step))
        difference = float((plus - minus) / (2.0 * step))

        assert abs(float(gradient[edge]) - difference) / scale < TOLERANCE_GRADIENT


def test_the_composed_gradient_is_finite_and_nowhere_zero(arch, one_case):
    _, composed = objectives(arch, one_case, 3)

    gradient = jax.grad(composed)(arch.params.q)

    assert np.all(np.isfinite(np.asarray(gradient)))
    assert float(jnp.min(jnp.abs(gradient))) > 0.0


def test_the_chain_differentiates_in_both_directions(arch, one_case):
    # Every stage implements a tangent as well as an adjoint, and a directional
    # derivative taken forward has to equal the same direction contracted with
    # the reverse gradient.
    _, composed = objectives(arch, one_case, 3)

    q = arch.params.q
    direction = jnp.ones_like(q)
    _, forward = jax.jvp(composed, (q,), (direction,))
    reverse = jnp.sum(jax.grad(composed)(q) * direction)

    assert relative(reverse, forward) < TOLERANCE_DERIVATIVE


# --------------------------------------------------------------------------- #
# What the boundary must not quietly change
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("section_class", [2, 3])
def test_every_member_is_utilized_exactly_once_over(arch, one_case, section_class):
    _, composed = both(arch, one_case, section_class)

    assert np.allclose(
        np.asarray(composed.utilization), 1.0, rtol=0.0, atol=TOLERANCE_UTILIZATION
    )


def test_the_boundary_does_not_downcast_to_single_precision(arch, one_case):
    # Every schema declares float64. The upstream examples are float32, and a
    # float32 stage would downcast silently and cost eight digits.
    _, composed_design = both(arch, one_case, 3)
    _, composed = objectives(arch, one_case, 3)

    for label, value in named_fields(composed_design):
        assert jnp.asarray(value).dtype == jnp.float64, label

    assert jax.grad(composed)(arch.params.q).dtype == jnp.float64


def differentiable(tesseract, direction):
    """
    Which fields of a stage's schema carry a derivative, and in what shape.
    """
    schemas = tesseract.openapi_schema["components"]["schemas"]

    return schemas[f"Apply{direction}Schema"]["differentiable_arrays"]


def test_the_analysis_reports_a_moment_at_each_end_of_every_member(chain):
    # The frozen contract is both end moments and not a peak. Nodal loads leave
    # the moment linear along a member, which is what makes the first row of
    # Table B.3 exact rather than approximate, and a peak would throw away the
    # half of it the equivalent uniform moment factor is read from.
    reported = differentiable(chain.analysis, "Output")

    assert set(reported) == {
        "axial_force",
        "end_moments_major",
        "end_moments_minor",
    }
    assert reported["end_moments_major"]["shape"] == [None, 2]
    assert reported["end_moments_minor"]["shape"] == [None, 2]


def test_the_analysis_asks_for_a_derivative_in_nothing_but_shape_and_size(chain):
    # The schema has to be satisfiable by a solver whose adjoints were written
    # by hand. Direct differentiation reaches a nodal coordinate and a section
    # property, so those two are the whole of what this stage may promise.
    assert set(differentiable(chain.analysis, "Input")) == {"xyz", "diameter"}


def test_the_analysis_never_reports_a_critical_load_factor(chain):
    # Global stability is soft validation and stays outside the chain. In the
    # schema it would oblige every backend to supply one.
    schemas = chain.analysis.openapi_schema["components"]["schemas"]

    assert set(schemas["Apply_OutputSchema"]["properties"]) == {
        "axial_force",
        "end_moments_major",
        "end_moments_minor",
    }


def test_the_check_offers_a_derivative_in_every_material_property(chain):
    # Unlike the analysis, nothing here has to be reimplemented in another
    # language, so the check differentiates in everything it is given except
    # the catalogue floor and the two flags that select a clause.
    offered = set(differentiable(chain.ec3, "Input"))

    assert {"f_y", "e_mod", "gamma_m0", "gamma_m1", "ratio", "alpha"} <= offered
    assert "diameter_min" not in offered


def test_the_check_never_offers_a_derivative_in_the_governing_limit_state(chain):
    assert "governing" not in differentiable(chain.ec3, "Output")


def test_the_check_never_reports_a_mass(chain):
    # A mass is geometry rather than a resistance and EN 1993-1-1 has no opinion
    # on it, so what the standard decides is the size and the length it would be
    # multiplied by never crosses either.
    schemas = chain.ec3.openapi_schema["components"]["schemas"]

    assert "mass" not in schemas["Apply_OutputSchema"]["properties"]
    assert "lengths" not in schemas["Apply_InputSchema"]["properties"]


# --------------------------------------------------------------------------- #
# The diagnostic that must not be differentiated
# --------------------------------------------------------------------------- #
def sized_through_the_check(arch, result, catalogue):
    """
    The check alone, called across its own boundary with a finished geometry.
    """
    structure = arch.structure
    steel = arch.steel
    chain = arch.chain

    member = apply_tesseract(
        chain.analysis,
        {
            "xyz": result.xyz,
            "diameter": arch.params.diameters,
            "edges": np.asarray(structure.edges, dtype=np.int64),
            "supports": np.asarray(structure.supports, dtype=np.int64),
            "loads": np.asarray(funicular(structure), dtype=np.float64),
            "f_y": steel.f_y,
            "e_mod": steel.e_mod,
            "density": steel.density,
            "ratio": catalogue.ratio,
            "normal": NORMAL,
        },
    )

    return lambda axial_force: apply_tesseract(
        chain.ec3,
        {
            "axial_force": axial_force,
            "end_moments_major": member["end_moments_major"],
            "end_moments_minor": member["end_moments_minor"],
            "buckling_length": result.lengths,
            "f_y": steel.f_y,
            "e_mod": steel.e_mod,
            "density": steel.density,
            "gamma_m0": steel.gamma_m0,
            "gamma_m1": steel.gamma_m1,
            "ratio": catalogue.ratio,
            "alpha": steel.alpha,
            "diameter_min": catalogue.diameter_min,
            "section_class": 3,
            "resultant": True,
        },
    ), member["axial_force"]


def test_the_governing_limit_state_survives_the_boundary(arch, one_case):
    catalogue = TubeCatalogue.at_class_limit(arch.steel.f_y, 3)
    oracle, _ = both(arch, one_case, 3)
    check, axial_force = sized_through_the_check(arch, oracle, catalogue)

    sizer = Ec3Sizer(arch.steel, catalogue)
    reported = np.asarray(check(axial_force)["governing"])
    expected = np.asarray(
        sizer.governing(oracle.diameters, oracle.actions, oracle.buckling_length)
    )

    assert np.array_equal(reported, expected[0])


def test_differentiating_the_governing_limit_state_is_refused(arch, one_case):
    # A concrete cotangent on a non-differentiable output raises rather than
    # returning a zero, which is the whole reason the composition drops it.
    catalogue = TubeCatalogue.at_class_limit(arch.steel.f_y, 3)
    oracle, _ = both(arch, one_case, 3)
    check, axial_force = sized_through_the_check(arch, oracle, catalogue)

    with pytest.raises(ValueError, match="governing"):
        jax.grad(lambda forces: jnp.sum(check(forces)["governing"]))(axial_force)


# --------------------------------------------------------------------------- #
# The endpoints and the module contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("stage", STAGES)
def test_every_stage_exposes_the_endpoints_jax_needs(chain, stage):
    # `abstract_eval` is mandatory because JAX resolves shapes before it runs
    # anything, and `jax.grad` reaches for the adjoint and never the Jacobian.
    available = set(dict(zip(STAGES, chain))[stage].available_endpoints)

    assert {"apply", "abstract_eval", "vector_jacobian_product"} <= available


def test_a_python_list_is_refused_at_the_boundary(structure, chain):
    # Tesseract-JAX is stricter than Tesseract Core: every array input has to be
    # a JAX or NumPy array, scalars included.
    with pytest.raises(TypeError, match="expects an array"):
        apply_tesseract(
            chain.formfinding,
            {
                "q": [-1.0] * NUM_EDGES,
                "nodes": np.asarray(structure.nodes, dtype=np.float64),
                "edges": np.asarray(structure.edges, dtype=np.int64),
                "supports": np.asarray(structure.supports, dtype=np.int64),
                "loads": np.asarray(funicular(structure), dtype=np.float64),
            },
        )


def test_a_chain_asked_for_a_stage_that_is_not_there_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="formfinding"):
        local_chain(tmp_path)


def test_an_uncompiled_tesseract_block_refuses_to_run(arch, one_case):
    # A block that never saw a structure has no topology to serialize, and says
    # so rather than sending a schema an empty field.
    with pytest.raises(ValueError, match="not compiled"):
        TesseractFormFinder(arch.chain.formfinding)(arch.params.q, one_case.formfinding)


# --------------------------------------------------------------------------- #
# Several load cases, across the boundary
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("beta", [10.0, 500.0])
def test_every_field_of_the_enveloped_design_survives_the_boundary(
    arch, three_cases, beta
):
    # The objective the optimizer actually minimizes, which is not the one the
    # single-case parity test covers: three analyses and three checks per call,
    # aggregated above the chain.
    oracle, composed = both(arch, three_cases, 3, beta)

    for label, left, right, limit in field_by_field(
        oracle, composed, TOLERANCE_PARITY_ENVELOPE
    ):
        assert relative(left, right) < limit, label


def test_the_enveloped_mass_gradient_survives_the_boundary(arch, three_cases):
    oracle, composed = objectives(arch, three_cases, 3, 100.0)

    q = arch.params.q
    difference = relative(jax.grad(oracle)(q), jax.grad(composed)(q))

    assert difference < TOLERANCE_DERIVATIVE


def test_the_composed_envelope_form_finds_once_for_all_the_load_cases(
    arch, three_cases
):
    # The shape answers to one load case by construction, so form finding is
    # shared and only the analysis and the check are walked per case. A geometry
    # that differed between cases would mean a different structure per case.
    oracle, composed = both(arch, three_cases, 3, 500.0)

    assert relative(oracle.xyz, composed.xyz) < TOLERANCE_PARITY
    assert relative(oracle.lengths, composed.lengths) < TOLERANCE_PARITY


def test_the_composed_envelope_covers_every_load_case(arch, three_cases):
    _, composed = both(arch, three_cases, 3, 500.0)

    assert float(jnp.max(composed.utilization)) <= 1.0 + 1e-12
    assert np.all(
        np.asarray(composed.diameters)
        >= np.asarray(jnp.max(composed.required, axis=0)) - 1e-9
    )
