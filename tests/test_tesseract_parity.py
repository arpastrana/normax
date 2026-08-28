# SPDX-License-Identifier: Apache-2.0
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import load_tesseract_api
from tesseract_jax import apply_tesseract

from normax.analysis.smax import SmaxAnalyzer
from normax.design import DesignParameters
from normax.design import StructuralDesignPipeline
from normax.design import compute_mass
from normax.form_finding import FdmFormFinder
from normax.loads import assemble_load_cases
from normax.loads import create_load_tributary
from normax.loads import create_load_uniform
from normax.materials import Steel355
from normax.sections import build_section_catalog
from normax.sizing.blueprint import DIAMETER_MINIMUM
from normax.sizing.blueprint import GAMMA_M0
from normax.structures import build_arch_2d
from normax.structures import build_gridshell_3d
from normax.tesseract import TesseractAnalyzer
from normax.tesseract import TesseractSizer
from normax.tesseract import open_tesseract_analysis
from normax.tesseract import open_tesseract_sizing

# The same 10 m arch rising 3 m under 180 kN that the in-process pipeline is
# tested on, so the two routes are compared on identical ground.
SPAN = 10_000.0
RISE = 3_000.0
TOTAL_LOAD = 180_000.0
NUM_EDGES = 10

# The diameter the frame is analyzed with before the check has spoken.
SEED = 100.0

# A shallow cap for the space-frame route, held at its drawn rise.
SHELL_RINGS = 3
SHELL_SPOKES = 6
SHELL_RADIUS = 5_000.0
SHELL_RISE = 2_000.0
SHELL_PRESSURE = 3.0e-3

# The two analysis solvers are different programs, so parity is close rather
# than exact. Axial forces measured at 1.2e-15 (opensees) and 5.4e-15 (pynite).
TOLERANCE_AXIAL = 1e-13

# A funicular member's moment is a near-cancellation read against the axial
# scale, so it inherits that larger scale. Measured 2.1e-12 / 2.7e-13.
TOLERANCE_MOMENT = 1e-10

# The check is one shared crossed block, so any utilization disagreement is
# inherited from the analysis crossing alone. Measured 4.9e-14.
TOLERANCE_UTILIZATION = 1e-12

# Each route linearizes its own program, so the same sum is accumulated in a
# different order. Measured 3.9e-12 (wrt q) and 5.3e-14 (wrt diameters).
TOLERANCE_DERIVATIVE = 1e-10

# Relative step at which the central difference plateaus, and the agreement
# measured there, scaled by the largest component of the gradient.
STEP = 1e-4
TOLERANCE_GRADIENT = 2e-7


def relative(oracle, composed):
    """
    Largest disagreement between two arrays, scaled by the size of the first.
    """
    left = np.asarray(oracle, dtype=np.float64)
    right = np.asarray(composed, dtype=np.float64)
    scale = max(float(np.max(np.abs(left))), np.finfo(np.float64).tiny)

    return float(np.max(np.abs(left - right))) / scale


# --------------------------------------------------------------------------- #
# The arch, both routes
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def catalog():
    return build_section_catalog(Steel355(), 3)


@pytest.fixture(scope="module")
def structure():
    return build_arch_2d(num_edges=NUM_EDGES, span=SPAN, rise=RISE)


@pytest.fixture(scope="module")
def one_case(structure):
    return assemble_load_cases([create_load_uniform(structure, TOTAL_LOAD)])


@pytest.fixture(scope="module")
def force_densities(structure, one_case):
    """Force densities reaching the target rise, so the arch is the same one."""
    trial = jnp.full(NUM_EDGES, -1.0)
    shape = FdmFormFinder(structure)(trial, one_case.formfinding)

    return trial * jnp.max(shape.xyz[:, 2]) / RISE


@pytest.fixture(scope="module")
def params(force_densities):
    return DesignParameters(force_densities, jnp.full(NUM_EDGES, SEED))


@pytest.fixture(scope="module")
def opensees_client():
    return open_tesseract_analysis()


@pytest.fixture(scope="module")
def blueprint_client():
    return open_tesseract_sizing()


@pytest.fixture(scope="module")
def shared_sizer(structure, catalog):
    """One crossed check in both triples, so parity isolates the analysis."""
    return TesseractSizer(structure, catalog, backend="blueprint")


@pytest.fixture(scope="module")
def oracle_pipeline(structure, catalog, shared_sizer):
    return StructuralDesignPipeline(
        FdmFormFinder(structure),
        SmaxAnalyzer(structure, catalog(SEED)),
        shared_sizer,
    )


@pytest.fixture(scope="module")
def crossed_pipeline(structure, catalog, shared_sizer):
    analyzer = TesseractAnalyzer(structure, catalog, backend="opensees")

    return StructuralDesignPipeline(FdmFormFinder(structure), analyzer, shared_sizer)


@pytest.fixture(scope="module")
def both_designs(oracle_pipeline, crossed_pipeline, params, one_case):
    """The same design taken in process and across the two boundaries."""
    return oracle_pipeline(params, one_case), crossed_pipeline(params, one_case)


# --------------------------------------------------------------------------- #
# The claim the whole module exists to make
# --------------------------------------------------------------------------- #
def test_the_geometry_never_changes_at_the_boundary(both_designs):
    # The form finder runs in process on both routes, so the shape is bit-equal
    # and any disagreement downstream entered downstream.
    oracle, crossed = both_designs

    assert jnp.array_equal(oracle.shape.xyz, crossed.shape.xyz)
    assert jnp.array_equal(oracle.shape.lengths, crossed.shape.lengths)


def test_the_two_routes_agree_on_what_the_members_carry(both_designs):
    oracle, crossed = both_designs

    assert relative(oracle.forces.axial_force, crossed.forces.axial_force) < (
        TOLERANCE_AXIAL
    )
    assert relative(oracle.forces.moment_major, crossed.forces.moment_major) < (
        TOLERANCE_MOMENT
    )
    assert np.allclose(np.asarray(crossed.forces.moment_minor), 0.0)


def test_the_crossed_check_agrees_on_the_utilization(both_designs):
    oracle, crossed = both_designs

    assert relative(oracle.sizes.utilization, crossed.sizes.utilization) < (
        TOLERANCE_UTILIZATION
    )


def test_the_sections_and_the_mass_cross_unchanged(both_designs):
    # Both routes hand the catalog the same held diameters, so the sections and
    # the mass they weigh are identical rather than merely close.
    oracle, crossed = both_designs

    assert jnp.array_equal(
        oracle.sizes.sections.diameter, crossed.sizes.sections.diameter
    )
    assert jnp.array_equal(
        oracle.sizes.sections.thickness, crossed.sizes.sections.thickness
    )
    assert relative(compute_mass(oracle), compute_mass(crossed)) < 1e-14


# --------------------------------------------------------------------------- #
# The gradient, end to end
# --------------------------------------------------------------------------- #
def worked_utilization(pipeline, loads):
    """
    The summed utilization as a function of the parameters, one route.
    """

    def worked(design_params):
        design = pipeline(design_params, loads)

        return jnp.sum(design.sizes.utilization)

    return worked


def test_the_utilization_gradient_survives_the_boundary(
    oracle_pipeline, crossed_pipeline, params, one_case
):
    oracle = worked_utilization(oracle_pipeline, one_case)
    crossed = worked_utilization(crossed_pipeline, one_case)

    def by_densities(route):
        return lambda q: route(DesignParameters(q, params.diameters))

    q = params.force_densities
    difference = relative(
        jax.grad(by_densities(oracle))(q), jax.grad(by_densities(crossed))(q)
    )

    assert difference < TOLERANCE_DERIVATIVE


def test_the_diameter_gradient_survives_the_boundary(
    oracle_pipeline, crossed_pipeline, params, one_case
):
    # The diameters reach both crossings: the frame's stiffness on one side of
    # the analysis boundary, the held check on the other.
    oracle = worked_utilization(oracle_pipeline, one_case)
    crossed = worked_utilization(crossed_pipeline, one_case)

    def by_diameters(route):
        return lambda d: route(DesignParameters(params.force_densities, d))

    held = params.diameters
    difference = relative(
        jax.grad(by_diameters(oracle))(held), jax.grad(by_diameters(crossed))(held)
    )

    assert difference < TOLERANCE_DERIVATIVE


def test_the_crossed_gradient_matches_central_differences(
    crossed_pipeline, params, one_case
):
    # Parity says the boundary changed nothing. This says the thing it left
    # unchanged is right, without the in-process pipeline vouching for it.
    crossed = worked_utilization(crossed_pipeline, one_case)

    def objective(q):
        return crossed(DesignParameters(q, params.diameters))

    q = params.force_densities
    gradient = jax.grad(objective)(q)
    scale = float(jnp.max(jnp.abs(gradient)))

    for edge in (0, NUM_EDGES // 2):
        step = abs(float(q[edge])) * STEP
        plus = objective(q.at[edge].add(step))
        minus = objective(q.at[edge].add(-step))
        central = float((plus - minus) / (2.0 * step))

        assert abs(float(gradient[edge]) - central) / scale < TOLERANCE_GRADIENT


def test_the_boundary_does_not_downcast_to_single_precision(
    crossed_pipeline, params, one_case, both_designs
):
    # Every schema declares float64; a float32 stage would downcast silently
    # and cost eight digits.
    _, crossed = both_designs
    payloads = (
        crossed.shape.xyz,
        crossed.shape.lengths,
        crossed.forces.axial_force,
        crossed.forces.moment_major,
        crossed.forces.moment_minor,
        crossed.sizes.sections.diameter,
        crossed.sizes.utilization,
    )

    for value in payloads:
        assert jnp.asarray(value).dtype == jnp.float64

    worked = worked_utilization(crossed_pipeline, one_case)
    gradient = jax.grad(lambda q: worked(DesignParameters(q, params.diameters)))(
        params.force_densities
    )

    assert gradient.dtype == jnp.float64


# --------------------------------------------------------------------------- #
# The space frame, across the pynite boundary
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def shell():
    return build_gridshell_3d(
        num_rings=SHELL_RINGS,
        num_spokes=SHELL_SPOKES,
        radius=SHELL_RADIUS,
        rise=SHELL_RISE,
    )


@pytest.fixture(scope="module")
def shell_case(shell):
    return assemble_load_cases([create_load_tributary(shell, SHELL_PRESSURE)])


@pytest.fixture(scope="module")
def shell_params(shell, shell_case):
    """Densities holding the drawn rise, so no member leans near vertical."""
    finder = FdmFormFinder(shell)
    trial = jnp.full(shell.num_edges, -1.0)
    reached = jnp.max(finder(trial, shell_case.formfinding).xyz[:, 2])
    q = trial * reached / SHELL_RISE

    return DesignParameters(q, jnp.full(shell.num_edges, SEED))


@pytest.fixture(scope="module")
def shell_sizer(shell, catalog):
    """One crossed check in both shell triples as well."""
    return TesseractSizer(shell, catalog, backend="blueprint")


@pytest.fixture(scope="module")
def shell_oracle(shell, catalog, shell_sizer):
    return StructuralDesignPipeline(
        FdmFormFinder(shell),
        SmaxAnalyzer(shell, catalog(SEED)),
        shell_sizer,
    )


@pytest.fixture(scope="module")
def shell_crossed(shell, catalog, shell_sizer):
    analyzer = TesseractAnalyzer(shell, catalog, backend="pynite")

    return StructuralDesignPipeline(FdmFormFinder(shell), analyzer, shell_sizer)


@pytest.fixture(scope="module")
def shell_designs(shell_oracle, shell_crossed, shell_params, shell_case):
    return shell_oracle(shell_params, shell_case), shell_crossed(
        shell_params, shell_case
    )


def test_the_space_frame_routes_agree_on_what_the_members_carry(shell_designs):
    # The minor moment is left out: how one bending splits over the two local
    # axes of a tube is each solver's frame convention, not a force.
    oracle, crossed = shell_designs

    assert relative(oracle.forces.axial_force, crossed.forces.axial_force) < (
        TOLERANCE_AXIAL
    )
    assert relative(oracle.forces.moment_major, crossed.forces.moment_major) < (
        TOLERANCE_MOMENT
    )


def test_the_space_frame_check_agrees_on_the_utilization(shell_designs):
    oracle, crossed = shell_designs

    assert relative(oracle.sizes.utilization, crossed.sizes.utilization) < (
        TOLERANCE_UTILIZATION
    )


def test_the_space_frame_gradient_survives_the_boundary(
    shell_oracle, shell_crossed, shell_params, shell_case
):
    # Held to a frame-free scalar: the blueprint demand sums the two moment
    # axes linearly, so its gradient reads each solver's roll convention, and
    # only a convention-free objective compares the two adjoints themselves.
    def carried_squared(pipeline):
        def worked(q):
            design = pipeline(DesignParameters(q, shell_params.diameters), shell_case)

            return jnp.sum(design.forces.axial_force**2)

        return worked

    q = shell_params.force_densities
    slope_oracle = jax.grad(carried_squared(shell_oracle))(q)
    slope_crossed = jax.grad(carried_squared(shell_crossed))(q)

    assert relative(slope_oracle, slope_crossed) < TOLERANCE_DERIVATIVE


# --------------------------------------------------------------------------- #
# The endpoints and the schemas
# --------------------------------------------------------------------------- #
def test_every_stage_exposes_the_endpoints_jax_needs(opensees_client, blueprint_client):
    # `abstract_eval` is mandatory because JAX resolves shapes before it runs
    # anything, and `jax.grad` reaches for the adjoint.
    needed = {"apply", "abstract_eval", "vector_jacobian_product"}

    assert needed <= set(opensees_client.available_endpoints)
    assert needed <= set(blueprint_client.available_endpoints)


def test_the_analysis_asks_for_a_derivative_in_nothing_but_shape_and_size(
    opensees_client,
):
    # The schema has to be satisfiable by a solver whose adjoints were written
    # by hand, so the two fields both solvers can supply are the whole promise.
    schemas = opensees_client.openapi_schema["components"]["schemas"]

    assert set(schemas["ApplyInputSchema"]["differentiable_arrays"]) == {
        "xyz",
        "diameter",
    }


def test_the_analysis_reports_a_moment_at_each_end_of_every_member(opensees_client):
    # Both end moments and not a peak: nodal loads leave the moment linear
    # along a member, which is what makes Table B.3's first row exact.
    schemas = opensees_client.openapi_schema["components"]["schemas"]
    reported = schemas["ApplyOutputSchema"]["differentiable_arrays"]

    assert set(reported) == {
        "axial_force",
        "end_moments_major",
        "end_moments_minor",
    }
    assert reported["end_moments_major"]["shape"] == [None, 2]
    assert reported["end_moments_minor"]["shape"] == [None, 2]


def test_the_check_serves_both_questions_but_never_the_clamp_gradient(
    blueprint_client,
):
    # The schema carries the solve and the held check; the clamp mask crosses
    # as a diagnostic with no derivative to offer.
    schemas = blueprint_client.openapi_schema["components"]["schemas"]
    offered = set(schemas["ApplyOutputSchema"]["differentiable_arrays"])

    assert offered == {"diameter", "utilization", "utilization_held"}
    assert "clamped" in schemas["Apply_OutputSchema"]["properties"]


def test_the_check_module_reports_its_shapes_without_running():
    # The API module imports directly, no container and no network, and its
    # abstract evaluation answers from the member count alone.
    module = load_tesseract_api("sizing")
    abstract = types.SimpleNamespace(
        axial_force=jax.ShapeDtypeStruct((NUM_EDGES,), jnp.float64)
    )
    promised = module.abstract_eval(abstract)

    for name in ("diameter", "utilization", "utilization_held", "clamped"):
        assert promised[name] == {"shape": (NUM_EDGES,), "dtype": "float64"}


def test_a_python_list_is_refused_at_the_boundary(blueprint_client):
    # Tesseract-JAX is stricter than Tesseract Core: every array input has to
    # be a JAX or NumPy array, scalars included.
    moments = np.zeros((NUM_EDGES, 2))
    inputs = {
        "axial_force": [-1.0e5] * NUM_EDGES,
        "end_moments_major": moments,
        "end_moments_minor": moments,
        "diameter_held": np.full(NUM_EDGES, SEED),
        "f_y": jnp.asarray(355.0),
        "gamma_m0": jnp.asarray(GAMMA_M0),
        "ratio": jnp.asarray(50.0),
        "diameter_min": jnp.asarray(DIAMETER_MINIMUM),
    }

    with pytest.raises(TypeError, match="expects an array"):
        apply_tesseract(blueprint_client, inputs)
