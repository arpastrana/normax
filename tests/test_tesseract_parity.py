# SPDX-License-Identifier: Apache-2.0
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import load_tesseract_api
from tesseract_jax import apply_tesseract

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

# The same 10 m arch rising 3 m under 180 kN the rest of the suite uses, so the
# two crossed routes are compared on identical ground.
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

# Recorded from smax at tag local-dev; see docs/oracle_removal.md.
SHELL_AXIAL_NORM = 1.97649588267220708e05
SHELL_AXIAL_SAMPLED = (
    -1.15775588229506848e04,
    -6.54659209604083298e04,
    -3.49976430528596684e04,
)
SHELL_SAMPLED_MEMBERS = (0, 12, 29)
SHELL_MOMENT_NORM = 8.29685660578915966e05
SHELL_UTILIZATION_NORM = 1.21362960862376923e00

# The two analysis solvers are different programs, so parity is close rather
# than exact. Axial forces measured at 1.3e-15 (arch) and 5.4e-15 (shell).
TOLERANCE_AXIAL = 1e-13

# A funicular member's moment is a near-cancellation read against the axial
# scale, so it inherits that larger scale. Measured 1.6e-12 / 2.7e-13.
TOLERANCE_MOMENT = 1e-10

# The check is one shared crossed block, so any utilization disagreement is
# inherited from the analysis crossing alone. Measured 3.6e-14.
TOLERANCE_UTILIZATION = 1e-12

# Each route linearizes its own program, so the same sum is accumulated in a
# different order. Measured 3.9e-12 (wrt q) and 2.9e-14 (wrt diameters).
TOLERANCE_DERIVATIVE = 1e-10

# Relative step at which the central difference plateaus, and the agreement
# measured there, scaled by the largest component of the gradient.
STEP = 1e-4
TOLERANCE_GRADIENT = 2e-7


def relative(reference, compared):
    """
    Largest disagreement between two arrays, scaled by the size of the first.
    """
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(compared, dtype=np.float64)
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
def pynite_pipeline(structure, catalog, shared_sizer):
    analyzer = TesseractAnalyzer(structure, catalog, backend="pynite")

    return StructuralDesignPipeline(FdmFormFinder(structure), analyzer, shared_sizer)


@pytest.fixture(scope="module")
def opensees_pipeline(structure, catalog, shared_sizer):
    analyzer = TesseractAnalyzer(structure, catalog, backend="opensees")

    return StructuralDesignPipeline(FdmFormFinder(structure), analyzer, shared_sizer)


@pytest.fixture(scope="module")
def both_designs(pynite_pipeline, opensees_pipeline, params, one_case):
    """The same design taken across the boundary to either foreign solver."""
    return pynite_pipeline(params, one_case), opensees_pipeline(params, one_case)


# --------------------------------------------------------------------------- #
# The claim the whole module exists to make
# --------------------------------------------------------------------------- #
def test_the_geometry_never_changes_at_the_boundary(both_designs):
    # The form finder runs in process on both routes, so the shape is bit-equal
    # and any disagreement downstream entered downstream.
    through_pynite, through_opensees = both_designs

    assert jnp.array_equal(through_pynite.shape.xyz, through_opensees.shape.xyz)
    assert jnp.array_equal(through_pynite.shape.lengths, through_opensees.shape.lengths)


def test_the_two_routes_agree_on_what_the_members_carry(both_designs):
    # Two foreign solvers, one hand-adjointed and one differentiated by rules
    # compiled into it, reached across the same schema.
    through_pynite, through_opensees = both_designs

    assert relative(
        through_pynite.forces.axial_force, through_opensees.forces.axial_force
    ) < (TOLERANCE_AXIAL)
    assert relative(
        through_pynite.forces.moment_major, through_opensees.forces.moment_major
    ) < (TOLERANCE_MOMENT)
    assert np.allclose(np.asarray(through_opensees.forces.moment_minor), 0.0)
    assert np.allclose(np.asarray(through_pynite.forces.moment_minor), 0.0)


def test_the_crossed_check_agrees_on_the_utilization(both_designs):
    through_pynite, through_opensees = both_designs

    assert relative(
        through_pynite.sizes.utilization, through_opensees.sizes.utilization
    ) < (TOLERANCE_UTILIZATION)


def test_the_sections_and_the_mass_cross_unchanged(both_designs):
    # Both routes hand the catalog the same held diameters, so the sections and
    # the mass they weigh are identical rather than merely close.
    through_pynite, through_opensees = both_designs

    assert jnp.array_equal(
        through_pynite.sizes.sections.diameter, through_opensees.sizes.sections.diameter
    )
    assert jnp.array_equal(
        through_pynite.sizes.sections.thickness,
        through_opensees.sizes.sections.thickness,
    )
    assert (
        relative(compute_mass(through_pynite), compute_mass(through_opensees)) < 1e-14
    )


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
    pynite_pipeline, opensees_pipeline, params, one_case
):
    # Two adjoints written in two languages by two strategies, contracted into
    # the same reverse-mode sweep.
    through_pynite = worked_utilization(pynite_pipeline, one_case)
    through_opensees = worked_utilization(opensees_pipeline, one_case)

    def by_densities(route):
        return lambda q: route(DesignParameters(q, params.diameters))

    q = params.shape_parameters
    difference = relative(
        jax.grad(by_densities(through_pynite))(q),
        jax.grad(by_densities(through_opensees))(q),
    )

    assert difference < TOLERANCE_DERIVATIVE


def test_the_diameter_gradient_survives_the_boundary(
    pynite_pipeline, opensees_pipeline, params, one_case
):
    # The diameters reach both crossings: the frame's stiffness on one side of
    # the analysis boundary, the held check on the other.
    through_pynite = worked_utilization(pynite_pipeline, one_case)
    through_opensees = worked_utilization(opensees_pipeline, one_case)

    def by_diameters(route):
        return lambda d: route(DesignParameters(params.shape_parameters, d))

    held = params.diameters
    difference = relative(
        jax.grad(by_diameters(through_pynite))(held),
        jax.grad(by_diameters(through_opensees))(held),
    )

    assert difference < TOLERANCE_DERIVATIVE


def test_the_crossed_gradient_matches_central_differences(
    opensees_pipeline, params, one_case
):
    # Agreement between the two routes says the boundary changed nothing. This
    # says the thing it left unchanged is right, vouched for by nothing.
    crossed = worked_utilization(opensees_pipeline, one_case)

    def by_densities(q):
        return crossed(DesignParameters(q, params.diameters))

    def by_diameters(diameters):
        return crossed(DesignParameters(params.shape_parameters, diameters))

    for objective, start in (
        (by_densities, params.shape_parameters),
        (by_diameters, params.diameters),
    ):
        gradient = jax.grad(objective)(start)
        scale = float(jnp.max(jnp.abs(gradient)))

        for edge in (0, NUM_EDGES // 2):
            step = abs(float(start[edge])) * STEP
            plus = objective(start.at[edge].add(step))
            minus = objective(start.at[edge].add(-step))
            central = float((plus - minus) / (2.0 * step))

            assert abs(float(gradient[edge]) - central) / scale < TOLERANCE_GRADIENT


def test_the_boundary_does_not_downcast_to_single_precision(
    opensees_pipeline, params, one_case, both_designs
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

    worked = worked_utilization(opensees_pipeline, one_case)
    gradient = jax.grad(lambda q: worked(DesignParameters(q, params.diameters)))(
        params.shape_parameters
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
    """One crossed check for the space frame as well."""
    return TesseractSizer(shell, catalog, backend="blueprint")


@pytest.fixture(scope="module")
def shell_crossed(shell, catalog, shell_sizer):
    analyzer = TesseractAnalyzer(shell, catalog, backend="pynite")

    return StructuralDesignPipeline(FdmFormFinder(shell), analyzer, shell_sizer)


@pytest.fixture(scope="module")
def shell_design(shell_crossed, shell_params, shell_case):
    return shell_crossed(shell_params, shell_case)


def test_the_space_frame_route_carries_the_recorded_forces(shell_design):
    # No second solver reaches three dimensions here, so the claim is held to
    # the recorded answer: a norm, and three members a permutation would move.
    axial = np.asarray(shell_design.forces.axial_force, dtype=np.float64)[0]
    sampled = [float(axial[member]) for member in SHELL_SAMPLED_MEMBERS]
    moment = np.asarray(shell_design.forces.moment_major, dtype=np.float64)[0]

    assert relative(SHELL_AXIAL_NORM, float(np.linalg.norm(axial))) < TOLERANCE_AXIAL
    assert relative(SHELL_AXIAL_SAMPLED, sampled) < TOLERANCE_AXIAL
    assert relative(SHELL_MOMENT_NORM, float(np.linalg.norm(moment))) < TOLERANCE_MOMENT


def test_the_space_frame_check_carries_the_recorded_utilization(shell_design):
    utilization = np.asarray(shell_design.sizes.utilization, dtype=np.float64)[0]
    reached = float(np.linalg.norm(utilization))

    assert relative(SHELL_UTILIZATION_NORM, reached) < TOLERANCE_UTILIZATION


def test_the_space_frame_gradient_matches_central_differences(
    shell_crossed, shell_params, shell_case
):
    # Held to a frame-free scalar: the blueprint demand sums the two moment
    # axes linearly, so its gradient reads the solver's roll convention, and
    # only a convention-free objective differences the adjoint itself.
    def carried_squared(q):
        design = shell_crossed(DesignParameters(q, shell_params.diameters), shell_case)

        return jnp.sum(design.forces.axial_force**2)

    q = shell_params.shape_parameters
    gradient = jax.grad(carried_squared)(q)
    scale = float(jnp.max(jnp.abs(gradient)))

    for edge in (0, shell_params.shape_parameters.shape[0] // 2):
        step = abs(float(q[edge])) * STEP
        plus = carried_squared(q.at[edge].add(step))
        minus = carried_squared(q.at[edge].add(-step))
        central = float((plus - minus) / (2.0 * step))

        assert abs(float(gradient[edge]) - central) / scale < TOLERANCE_GRADIENT


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
