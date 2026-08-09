import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tesseract_jax import apply_tesseract

from normax.composition import STAGES
from normax.composition import design as design_composed
from normax.composition import local
from normax.composition import mass as mass_composed
from normax.ec3.sizing import Steel
from normax.ec3.sizing import Tube
from normax.ec3.sizing import is_plastic
from normax.formfinding import equilibrium
from normax.formfinding import graph
from normax.pipeline import design as design_in_process
from normax.pipeline import governing
from normax.pipeline import mass as mass_in_process
from normax.structures import arch

# The same 10 m arch rising 3 m under 180 kN that the in-process pipeline is
# tested on, so the two are compared on identical ground.
SPAN = 10_000.0
RISE = 3_000.0
TOTAL_LOAD = 180_000.0
NUM_EDGES = 10

# The arch lies in the XZ plane, so it has no thickness along Y.
NORMAL = 1

# The diameter the frame is analysed with before the check has spoken.
SEED = 100.0

# The boundary serializes float64 losslessly and both sides run the same code,
# so parity is exact rather than approximate. Measured at 6.7e-16 across every
# field of the design, which is the width of the numbers themselves.
TOLERANCE_PARITY = 1e-14

# Derivatives are looser than values, and not because of the boundary. Each
# stage linearizes on its own here and all three linearize together in process,
# so the same sum is accumulated in a different order and the implicit tangent
# divides by a slope that differs in its last bits. Measured at 3.6e-14 between
# the two routes and 2.7e-14 between forward and reverse mode.
TOLERANCE_DERIVATIVE = 1e-12

# Invariant 6.5 of CLAUDE.md. Measured at 1.8e-15 through the boundary.
TOLERANCE_UTILIZATION = 1e-9

# Relative step at which the central difference plateaus, and the agreement
# measured there, scaled by the largest component of the gradient.
STEP = 1e-5
TOLERANCE_GRADIENT = 5e-8


@pytest.fixture(scope="module")
def steel():
    return Steel()


@pytest.fixture(scope="module")
def chain():
    return local()


@pytest.fixture(scope="module")
def setup():
    """
    The arch, its connectivity, and the `q` that reaches the target rise.
    """
    load = TOTAL_LOAD / (NUM_EDGES - 1)
    structure = arch(num_edges=NUM_EDGES, span=SPAN, rise=RISE, load=load)
    fdm = graph(structure)

    trial = jnp.full(NUM_EDGES, -1.0)
    reached = jnp.max(equilibrium(trial, structure, fdm).xyz[:, 2])

    return structure, fdm, trial * reached / RISE


@pytest.fixture(scope="module")
def seed():
    return jnp.full(NUM_EDGES, SEED)


def both(setup, chain, steel, seed, cross_section_class, **kwargs):
    """
    The same design taken in process and across the three Tesseracts.
    """
    structure, fdm, q = setup
    tube = Tube.at_class_limit(steel.f_y, cross_section_class)
    plastic = is_plastic(cross_section_class)

    oracle = design_in_process(
        q, seed, structure, fdm, steel, tube, normal=NORMAL, plastic=plastic, **kwargs
    )
    composed = design_composed(
        q, seed, structure, chain, steel, tube, normal=NORMAL, plastic=plastic, **kwargs
    )

    return oracle, composed


def objectives(setup, chain, steel, seed, cross_section_class, **kwargs):
    """
    The mass as a function of the force densities, by both routes.
    """
    structure, fdm, _ = setup
    tube = Tube.at_class_limit(steel.f_y, cross_section_class)
    plastic = is_plastic(cross_section_class)

    def in_process(q):
        return mass_in_process(
            q,
            seed,
            structure,
            fdm,
            steel,
            tube,
            normal=NORMAL,
            plastic=plastic,
            **kwargs,
        )

    def composed(q):
        return mass_composed(
            q,
            seed,
            structure,
            chain,
            steel,
            tube,
            normal=NORMAL,
            plastic=plastic,
            **kwargs,
        )

    return in_process, composed


def relative(oracle, composed):
    """
    Largest disagreement between two arrays, scaled by the size of the first.
    """
    left = np.asarray(oracle, dtype=np.float64)
    right = np.asarray(composed, dtype=np.float64)
    scale = max(float(np.max(np.abs(left))), np.finfo(np.float64).tiny)

    return float(np.max(np.abs(left - right))) / scale


# --------------------------------------------------------------------------- #
# The claim the whole step exists to make
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cross_section_class", [2, 3])
def test_the_composed_mass_is_the_in_process_mass(
    setup, chain, steel, seed, cross_section_class
):
    oracle, composed = both(setup, chain, steel, seed, cross_section_class)

    assert relative(oracle.mass, composed.mass) < TOLERANCE_PARITY


@pytest.mark.parametrize("cross_section_class", [2, 3])
def test_every_field_of_the_design_survives_the_boundary(
    setup, chain, steel, seed, cross_section_class
):
    # Mass alone would pass on a cancellation of two errors. Comparing the
    # geometry, the member actions, the sizes and the utilization pins where any
    # disagreement entered.
    oracle, composed = both(setup, chain, steel, seed, cross_section_class)

    for field in oracle._fields:
        assert (
            relative(getattr(oracle, field), getattr(composed, field))
            < TOLERANCE_PARITY
        ), field


@pytest.mark.parametrize("cross_section_class", [2, 3])
def test_the_mass_gradient_survives_the_boundary(
    setup, chain, steel, seed, cross_section_class
):
    _, _, q = setup
    in_process, composed = objectives(setup, chain, steel, seed, cross_section_class)

    assert (
        relative(jax.grad(in_process)(q), jax.grad(composed)(q)) < TOLERANCE_DERIVATIVE
    )


def test_a_buckling_length_given_explicitly_crosses_unchanged(
    setup, chain, steel, seed
):
    # The buckling length is an input rather than a mesh length, so it has to
    # reach the check as itself and not as the member length beside it.
    l_cr = jnp.full(NUM_EDGES, 1_000.0)
    oracle, composed = both(setup, chain, steel, seed, 3, l_cr=l_cr)

    assert relative(oracle.mass, composed.mass) < TOLERANCE_PARITY
    assert np.allclose(np.asarray(composed.l_cr), 1_000.0)


def test_the_linear_sum_reading_of_the_moments_crosses_unchanged(
    setup, chain, steel, seed
):
    # `resultant` selects a clause, so it crosses as a static field and a wrong
    # default would be invisible in the mass alone.
    oracle, composed = both(setup, chain, steel, seed, 3, resultant=False)

    assert relative(oracle.mass, composed.mass) < TOLERANCE_PARITY


# --------------------------------------------------------------------------- #
# The gradient, end to end
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cross_section_class", [2, 3])
def test_the_composed_gradient_matches_central_differences(
    setup, chain, steel, seed, cross_section_class
):
    # Parity says the boundary changed nothing. This says the thing it left
    # unchanged is right, without the in-process pipeline vouching for it.
    _, _, q = setup
    _, composed = objectives(setup, chain, steel, seed, cross_section_class)

    gradient = jax.grad(composed)(q)
    scale = float(jnp.max(jnp.abs(gradient)))

    for edge in (0, NUM_EDGES // 2, NUM_EDGES - 1):
        step = abs(float(q[edge])) * STEP
        plus = composed(q.at[edge].add(step))
        minus = composed(q.at[edge].add(-step))
        difference = float((plus - minus) / (2.0 * step))

        assert abs(float(gradient[edge]) - difference) / scale < TOLERANCE_GRADIENT


def test_the_composed_gradient_is_finite_and_nowhere_zero(setup, chain, steel, seed):
    _, _, q = setup
    _, composed = objectives(setup, chain, steel, seed, 3)

    gradient = jax.grad(composed)(q)

    assert np.all(np.isfinite(np.asarray(gradient)))
    assert float(jnp.min(jnp.abs(gradient))) > 0.0


def test_the_chain_differentiates_in_both_directions(setup, chain, steel, seed):
    # Every stage implements a tangent as well as an adjoint, and a directional
    # derivative taken forward has to equal the same direction contracted with
    # the reverse gradient.
    _, _, q = setup
    _, composed = objectives(setup, chain, steel, seed, 3)

    direction = jnp.ones_like(q)
    _, forward = jax.jvp(composed, (q,), (direction,))
    reverse = jnp.sum(jax.grad(composed)(q) * direction)

    assert relative(reverse, forward) < TOLERANCE_DERIVATIVE


# --------------------------------------------------------------------------- #
# What the boundary must not quietly change
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cross_section_class", [2, 3])
def test_every_member_is_utilized_exactly_once_over(
    setup, chain, steel, seed, cross_section_class
):
    _, composed = both(setup, chain, steel, seed, cross_section_class)

    assert np.allclose(
        np.asarray(composed.utilization), 1.0, rtol=0.0, atol=TOLERANCE_UTILIZATION
    )


def test_the_boundary_does_not_downcast_to_single_precision(setup, chain, steel, seed):
    # Every schema declares float64. The upstream examples are float32, and a
    # float32 stage would downcast silently and cost eight digits.
    _, _, q = setup
    _, composed_design = both(setup, chain, steel, seed, 3)
    _, composed = objectives(setup, chain, steel, seed, 3)

    for field in composed_design._fields:
        assert jnp.asarray(getattr(composed_design, field)).dtype == jnp.float64, field

    assert jax.grad(composed)(q).dtype == jnp.float64


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

    assert set(reported) == {"n_ed", "m_y_ed", "m_z_ed"}
    assert reported["m_y_ed"]["shape"] == [None, 2]
    assert reported["m_z_ed"]["shape"] == [None, 2]


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
        "n_ed",
        "m_y_ed",
        "m_z_ed",
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


# --------------------------------------------------------------------------- #
# The diagnostic that must not be differentiated
# --------------------------------------------------------------------------- #
def sized_through_the_check(setup, chain, steel, seed, result, tube):
    """
    The check alone, called across its own boundary with a finished geometry.
    """
    structure, _, _ = setup
    edges = np.asarray(structure.edges, dtype=np.int64)
    supports = np.asarray(structure.supports, dtype=np.int64)
    loads = np.asarray(structure.loads, dtype=np.float64)

    member = apply_tesseract(
        chain.analysis,
        {
            "xyz": result.xyz,
            "diameter": seed,
            "edges": edges,
            "supports": supports,
            "loads": loads,
            "f_y": steel.f_y,
            "e_mod": steel.e_mod,
            "density": steel.density,
            "ratio": tube.ratio,
            "normal": NORMAL,
        },
    )

    return lambda n_ed: apply_tesseract(
        chain.ec3,
        {
            "n_ed": n_ed,
            "m_y_ed": member["m_y_ed"],
            "m_z_ed": member["m_z_ed"],
            "lengths": result.lengths,
            "l_cr": result.lengths,
            "f_y": steel.f_y,
            "e_mod": steel.e_mod,
            "density": steel.density,
            "gamma_m0": steel.gamma_m0,
            "gamma_m1": steel.gamma_m1,
            "ratio": tube.ratio,
            "alpha": tube.alpha,
            "diameter_min": tube.diameter_min,
            "plastic": False,
            "resultant": True,
        },
    ), member["n_ed"]


def test_the_governing_limit_state_survives_the_boundary(setup, chain, steel, seed):
    tube = Tube.at_class_limit(steel.f_y, 3)
    oracle, _ = both(setup, chain, steel, seed, 3)
    check, n_ed = sized_through_the_check(setup, chain, steel, seed, oracle, tube)

    reported = np.asarray(check(n_ed)["governing"])
    expected = np.asarray(governing(oracle, steel, tube, plastic=False))

    assert np.array_equal(reported, expected)


def test_differentiating_the_governing_limit_state_is_refused(
    setup, chain, steel, seed
):
    # A concrete cotangent on a non-differentiable output raises rather than
    # returning a zero, which is the whole reason the composition drops it.
    tube = Tube.at_class_limit(steel.f_y, 3)
    oracle, _ = both(setup, chain, steel, seed, 3)
    check, n_ed = sized_through_the_check(setup, chain, steel, seed, oracle, tube)

    with pytest.raises(ValueError, match="governing"):
        jax.grad(lambda forces: jnp.sum(check(forces)["governing"]))(n_ed)


# --------------------------------------------------------------------------- #
# The endpoints and the module contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("stage", STAGES)
def test_every_stage_exposes_the_endpoints_jax_needs(chain, stage):
    # `abstract_eval` is mandatory because JAX resolves shapes before it runs
    # anything, and `jax.grad` reaches for the adjoint and never the Jacobian.
    available = set(dict(zip(STAGES, chain))[stage].available_endpoints)

    assert {"apply", "abstract_eval", "vector_jacobian_product"} <= available


def test_a_python_list_is_refused_at_the_boundary(setup, chain):
    # Tesseract-JAX is stricter than Tesseract Core: every array input has to be
    # a JAX or NumPy array, scalars included.
    structure, _, _ = setup

    with pytest.raises(TypeError, match="expects an array"):
        apply_tesseract(
            chain.formfinding,
            {
                "q": [-1.0] * NUM_EDGES,
                "nodes": np.asarray(structure.nodes, dtype=np.float64),
                "edges": np.asarray(structure.edges, dtype=np.int64),
                "supports": np.asarray(structure.supports, dtype=np.int64),
                "loads": np.asarray(structure.loads, dtype=np.float64),
            },
        )


def test_a_chain_asked_for_a_stage_that_is_not_there_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="formfinding"):
        local(tmp_path)
