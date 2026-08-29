# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from normax.config import FormFindingConfig
from normax.config import read_run_arguments
from normax.config import read_run_config
from normax.design import DesignConstraints
from normax.design import DesignParameters
from normax.design import DesignProblem
from normax.design import StructuralDesignPipeline
from normax.design import assign_signs
from normax.design import bound_variables
from normax.design import build_design_constraints
from normax.design import create_design
from normax.design import design_maps
from normax.design import initialize_optimization_parameters
from normax.form_finding import CoefficientBounds
from normax.form_finding import DrawnShapeInitializer
from normax.form_finding import FdmFormFinder
from normax.form_finding import FixedFormFinder
from normax.form_finding import HeightsFormFinder
from normax.form_finding import UniformDensityInitializer
from normax.form_finding import build_form_finder
from normax.form_finding import build_parabolic_heights
from normax.form_finding import build_plan_basis
from normax.form_finding import select_free_nodes
from normax.loads import assemble_load_cases
from normax.loads import create_load_uniform
from normax.materials import Steel355
from normax.reporting import list_unused_settings
from normax.sections import build_section_catalog
from normax.structures import ArchDescription
from normax.structures import build_arch_2d
from normax.symmetry import SignGuard
from normax.tesseract import TesseractAnalyzer
from normax.tesseract import TesseractSizer

# A small arch under 180 kN, in millimeters and newtons.
SPAN = 4_000.0
RISE = 1_200.0
TOTAL_LOAD = 180_000.0
NUM_EDGES = 4

# The diameter every member starts at, and the floor under it.
SEED = 120.0
DIAMETER_FLOOR = 20.0

# A sample density box, not the shipped arch's — the tests that read the file
# take its numbers from the file. Meaningless on a written geometry, and
# ruinous if it ever reached one.
DENSITY_BOX = (-100.0, -1.0e-3)


@pytest.fixture(scope="module")
def structure():
    return build_arch_2d(num_edges=NUM_EDGES, span=SPAN, rise=RISE)


@pytest.fixture(scope="module")
def loads(structure):
    return assemble_load_cases([create_load_uniform(structure, TOTAL_LOAD)])


@pytest.fixture(scope="module")
def catalog():
    return build_section_catalog(Steel355(), 3)


def build_problem(structure, formfinder, catalog, loads):
    """
    A design problem over a comparison finder, whose parameters are its own.
    """
    analyzer = TesseractAnalyzer(structure, catalog, "pynite")
    sizer = TesseractSizer(structure, catalog, "blueprint")
    pipeline = StructuralDesignPipeline(formfinder, analyzer, sizer)
    constraints = DesignConstraints(DIAMETER_FLOOR, 0.0, None, None, None, None)

    return DesignProblem(structure, pipeline, loads, constraints)


def test_the_heights_finder_composes_and_the_mass_has_a_gradient(
    structure, catalog, loads
):
    finder = HeightsFormFinder(structure)
    problem = build_problem(structure, finder, catalog, loads)
    heights = jnp.asarray(structure.nodes)[select_free_nodes(structure), 2] * 1.1
    diameters = jnp.full(NUM_EDGES, SEED)

    design = problem.pipeline(DesignParameters(heights, diameters), loads)
    assert jnp.allclose(design.shape.xyz[select_free_nodes(structure), 2], heights)
    assert design.sizes.utilization.shape == (1, NUM_EDGES)

    x = jnp.concatenate([heights, diameters])
    maps = design_maps(problem)
    mass, slope = maps.objective(x)
    slack = maps.slack(x)

    assert finder.width == NUM_EDGES - 1
    assert len(bound_variables(problem)) == x.size
    assert np.isfinite(float(mass)) and float(mass) > 0.0
    assert np.all(np.isfinite(np.asarray(slope)))
    assert np.any(np.asarray(slope)[: finder.width] != 0.0)
    assert np.all(np.isfinite(np.asarray(slack)))


def test_the_fixed_finder_composes_and_moves_the_diameters_alone(
    structure, catalog, loads
):
    finder = FixedFormFinder(structure)
    problem = build_problem(structure, finder, catalog, loads)
    diameters = jnp.full(NUM_EDGES, SEED)

    design = problem.pipeline(DesignParameters(jnp.zeros(0), diameters), loads)
    assert jnp.array_equal(design.shape.xyz, jnp.asarray(structure.nodes))

    maps = design_maps(problem)
    mass, slope = maps.objective(diameters)
    slack = maps.slack(diameters)

    assert finder.count_shape_coefficients() == 0
    assert len(bound_variables(problem)) == NUM_EDGES
    assert np.isfinite(float(mass)) and float(mass) > 0.0
    assert np.all(np.asarray(slope) > 0.0)
    assert np.all(np.isfinite(np.asarray(slack)))


def described(word, basis="pivoted", start=None):
    """
    A form-finding section naming one parametrization.
    """
    return FormFindingConfig(word, basis, None, False, start or {})


def test_the_builder_names_every_parametrization_and_refuses_any_other(structure):
    built = {
        word: build_form_finder(structure, None, described(word))
        for word in ("fdm", "heights", "fixed")
    }

    assert isinstance(built["fdm"], FdmFormFinder)
    assert isinstance(built["heights"], HeightsFormFinder)
    assert isinstance(built["fixed"], FixedFormFinder)

    with pytest.raises(ValueError, match="shape parametrization"):
        build_form_finder(structure, None, described("drawn"))


def test_each_finder_boxes_the_limits_its_coefficients_are_in(structure):
    free = int(select_free_nodes(structure).size)
    limits = CoefficientBounds(DENSITY_BOX, RISE, 0.0)

    fdm = FdmFormFinder(structure)
    heights = HeightsFormFinder(structure)
    fixed = FixedFormFinder(structure)

    # A density parametrization takes the density box and nothing else: a
    # density does not say what height it reaches without a solve.
    assert fdm.bound_coefficients(limits) == [DENSITY_BOX] * NUM_EDGES

    # A heights parametrization takes the sag floor and the rise ceiling, and
    # never the density box -- which is negative at both ends and would drive
    # every node under the ground plane.
    assert heights.bound_coefficients(limits) == [(0.0, RISE)] * free

    assert fixed.bound_coefficients(limits) == []


def test_a_height_box_opens_at_whichever_end_the_file_leaves_out(structure):
    free = int(select_free_nodes(structure).size)
    heights = HeightsFormFinder(structure)

    named = CoefficientBounds(DENSITY_BOX, None, None)
    assert heights.bound_coefficients(named) == [(None, None)] * free

    capped = CoefficientBounds(DENSITY_BOX, RISE, None)
    assert heights.bound_coefficients(capped) == [(None, RISE)] * free


def test_each_finder_reads_its_own_start_off_the_fitted_densities(structure):
    q = np.full(NUM_EDGES, -80.0)
    drawn = np.asarray(structure.nodes)[select_free_nodes(structure), 2]

    assert np.array_equal(FdmFormFinder(structure).read_shape_coefficients(q), q)
    assert np.array_equal(
        HeightsFormFinder(structure).read_shape_coefficients(q), drawn
    )
    assert FixedFormFinder(structure).read_shape_coefficients(q).size == 0


def test_a_written_geometry_keeps_no_sign_guard(structure):
    guard = SignGuard(np.ones(1), np.zeros(1, dtype=int), 1.0, 1.0)

    assert FdmFormFinder(structure).read_sign_guard(guard) is guard
    assert HeightsFormFinder(structure).read_sign_guard(guard) is None
    assert FixedFormFinder(structure).read_sign_guard(guard) is None


def test_rise_and_sag_rows_are_the_same_on_every_parametrization(
    structure, catalog, loads
):
    diameters = jnp.full(NUM_EDGES, SEED)
    free = int(select_free_nodes(structure).size)
    held = DesignConstraints(DIAMETER_FLOOR, 0.0, RISE, 0.0, None, None)

    counted = {}
    for word, parameters in (
        ("heights", jnp.asarray(structure.nodes)[select_free_nodes(structure), 2]),
        ("fixed", jnp.zeros(0)),
    ):
        finder = build_form_finder(structure, None, described(word, basis=None))
        problem = build_problem(structure, finder, catalog, loads)
        problem = problem._replace(constraints=held)
        x = jnp.concatenate([parameters, diameters])
        rows = np.asarray(design_maps(problem).slack(x))
        counted[word] = rows.size

    # The utilization rows, then one rise row and one sag row per free node.
    assert counted["heights"] == NUM_EDGES + 2 * free
    assert counted["fixed"] == NUM_EDGES + 2 * free


def test_the_command_line_overrides_the_word_the_file_names(tmp_path):
    written = Path("examples/arch.yaml").read_text()
    path = tmp_path / "arch.yaml"
    path.write_text(written)

    kept = read_run_config(read_run_arguments([str(path)], path), ArchDescription)
    assert kept.form_finding.shape_parametrization == "fdm"

    argv = [str(path), "--shape-parametrization", "heights"]
    overridden = read_run_config(read_run_arguments(argv, path), ArchDescription)
    assert overridden.form_finding.shape_parametrization == "heights"

    # Everything else the file said survives the override untouched.
    assert overridden.form_finding.basis == kept.form_finding.basis
    assert overridden.constraints == kept.constraints


def test_a_file_is_read_when_the_command_line_names_none(tmp_path):
    written = Path("examples/arch.yaml").read_text()
    path = tmp_path / "arch.yaml"
    path.write_text(written)

    arguments = read_run_arguments([], path)

    assert arguments.config_path == path
    assert arguments.shape_parametrization is None


def test_a_written_route_reports_the_settings_it_did_not_read(tmp_path):
    written = Path("examples/arch.yaml").read_text()
    path = tmp_path / "arch.yaml"
    path.write_text(written)
    argv = [str(path), "--shape-parametrization", "fixed"]
    config = read_run_config(read_run_arguments(argv, path), ArchDescription)

    unused = list_unused_settings(config)

    assert set(unused) == {"basis", "density_start", "bounds", "height_start"}

    # The form-found route reads every density setting and no written one, so
    # the height start is the one thing it has to be told it ignored.
    assert list_unused_settings(
        config._replace(
            form_finding=config.form_finding._replace(shape_parametrization="fdm")
        )
    ) == ("height_start",)


# What the shipped arch carries per route: one basis coefficient and ten
# diameters end to end, nine heights and ten diameters written, ten diameters
# alone at the drawn geometry.
ARCH_VARIABLES = {"fdm": 11, "heights": 19, "fixed": 10}


@pytest.mark.parametrize("word", sorted(ARCH_VARIABLES))
def test_the_shipped_arch_carries_every_route_through_one_search(word):
    path = Path("examples/arch.yaml")
    argv = [str(path), "--shape-parametrization", word]
    config = read_run_config(read_run_arguments(argv, path), ArchDescription)
    described = config.structure
    structure = build_arch_2d(described.num_edges, described.span, described.rise)
    loads = assemble_load_cases([create_load_uniform(structure, TOTAL_LOAD)])
    catalog = build_section_catalog(Steel355(), config.sizing.section_class)

    basis = build_plan_basis(structure, None, config.form_finding.basis)
    finder = build_form_finder(structure, basis, config.form_finding)
    analyzer = TesseractAnalyzer(structure, catalog, "pynite")
    sizer = TesseractSizer(structure, catalog, "blueprint")
    pipeline = StructuralDesignPipeline(finder, analyzer, sizer)
    initializer = UniformDensityInitializer(config.form_finding.density_start)
    density_start = initializer(structure, loads.formfinding, basis, None)
    guarded = assign_signs(config.constraints, (), structure.num_edges)
    held = build_design_constraints(config.constraints, guarded, density_start)
    problem = DesignProblem(structure, pipeline, loads, held)

    diameters = np.full(described.num_edges, SEED)
    start = initialize_optimization_parameters(problem, density_start, diameters)
    boxes = bound_variables(problem)
    slack = np.asarray(design_maps(problem).slack(jnp.asarray(start)))

    assert start.size == ARCH_VARIABLES[word]
    assert len(boxes) == start.size
    assert np.all(np.isfinite(slack))

    # Every limit is read off the file rather than pinned here, so the shipped
    # config can move without this drifting into asserting a stale number.
    held_config = config.constraints
    density_box = (held_config.bounds.min, held_config.bounds.max)
    height_box = (held_config.sag_min, held_config.rise_max)

    # The density box goes on densities and the height box on heights; a shape
    # that never moves wears neither, having no coefficients.
    coefficients = boxes[: finder.count_shape_coefficients()]
    expected = {"fdm": density_box, "heights": height_box, "fixed": None}[word]
    if expected is None:
        assert coefficients == []
    else:
        assert all(box == expected for box in coefficients)

    # Each route opens where its OWN start names, which is no longer one shared
    # geometry: the arch is drawn flat, so `fixed` opens flat, `heights` opens at
    # the crown its start names, and `fdm` form-finds a rise the drawing never
    # had. Read off the file rather than pinned, so the config can move.
    design = create_design(problem, start)
    crown = float(jnp.max(design.shape.xyz[:, 2]))
    if word == "fixed":
        assert crown == pytest.approx(described.rise, abs=1e-6)
    elif word == "heights":
        named = config.form_finding.height_start["rise"]
        assert crown == pytest.approx(named, abs=1e-6)
    else:
        assert crown > described.rise


def test_a_generated_start_reaches_the_rise_it_names(structure):
    free = select_free_nodes(structure)
    lifted = build_parabolic_heights(
        np.asarray(structure.nodes), structure.supports, free, 50.0
    )

    assert lifted.shape == free.shape
    assert float(np.max(lifted)) == pytest.approx(50.0, abs=1e-9)
    assert float(np.min(lifted)) > 0.0

    # Quadratic in the plan distance to a support, so on a chain it is the
    # parabola through the two supports and the named crown, exactly.
    plan = np.asarray(structure.nodes)[free, 0]
    middle = 0.5 * SPAN
    exact = 50.0 * (1.0 - ((plan - middle) / middle) ** 2)

    assert np.allclose(lifted, exact, rtol=1e-12, atol=0.0)


def test_a_generated_start_leaves_a_flat_drawing_climbable(catalog, loads):
    # The reason the start exists. A flat drawing is a stationary point of the
    # mass AND of the utilization -- every shape derivative is exactly zero, so
    # no descent can begin from it. A named rise gives the search a slope.
    flat = build_arch_2d(num_edges=NUM_EDGES, span=SPAN, rise=0.0)
    diameters = jnp.full(NUM_EDGES, SEED)

    def read_shape_slope(start):
        described_route = described("heights", basis=None)._replace(height_start=start)
        finder = build_form_finder(flat, None, described_route)
        problem = build_problem(flat, finder, catalog, loads)
        coefficients = jnp.asarray(finder.read_shape_coefficients(jnp.zeros(0)))
        x = jnp.concatenate([coefficients, diameters])
        _, gradient = design_maps(problem).objective(x)
        width = finder.count_shape_coefficients()

        return float(np.max(np.abs(np.asarray(gradient)[:width])))

    assert read_shape_slope(None) == 0.0
    assert read_shape_slope({"rise": 50.0}) > 0.0


def test_an_unnamed_start_still_reads_the_drawn_heights(structure):
    finder = build_form_finder(structure, None, described("heights", basis=None))
    drawn = np.asarray(structure.nodes)[select_free_nodes(structure), 2]

    assert np.array_equal(finder.read_shape_coefficients(jnp.zeros(0)), drawn)


def test_an_omitted_start_is_none_and_no_config_shares_it(structure):
    # A NamedTuple evaluates its defaults once, so a mutable default would be
    # one object every config in the process holds. None cannot be mutated.
    first = FormFindingConfig("fdm", "svd", "y", False)
    second = FormFindingConfig("fdm", "pivoted", None, False)

    assert first.density_start is None
    assert second.density_start is None


def test_a_fit_reading_no_fields_accepts_an_omitted_start(structure, loads):
    # gridshell.yaml names no start, so the drawn fit is handed None and must
    # read it as an empty start rather than choking on it.
    fitted = DrawnShapeInitializer(None)
    q = fitted(structure, np.asarray(loads.formfinding), None, None)

    assert q.shape == (NUM_EDGES,)
    assert np.all(np.isfinite(q))


def test_a_start_is_still_held_to_the_fields_its_initializer_reads():
    # None reads as empty, which is right for a fit reading nothing and wrong
    # for one that reads a density — and the message names both sides either way.
    with pytest.raises(ValueError, match="must name nothing, got \\['sag'\\]"):
        DrawnShapeInitializer({"sag": 600.0})

    with pytest.raises(ValueError, match="must name force_density, got \\[\\]"):
        UniformDensityInitializer(None)


def test_folding_the_heights_needs_a_mirror_to_fold_them_by(tmp_path):
    # Asking for a fold the file names no symmetry for is a mistake worth a
    # refusal: silently folding nothing would leave the shape free to lean
    # away from a mirrored load case the file believes it no longer needs.
    written = Path("examples/arch.yaml").read_text()
    spoiled = written.replace("  mirror: x\n", "  mirror: null\n")
    path = tmp_path / "arch.yaml"
    path.write_text(spoiled)

    with pytest.raises(ValueError, match="names none"):
        read_run_config(read_run_arguments([str(path)], path), ArchDescription)


def test_the_shipped_arch_folds_its_heights_by_its_mirror(tmp_path):
    written = Path("examples/arch.yaml").read_text()
    path = tmp_path / "arch.yaml"
    path.write_text(written)

    config = read_run_config(read_run_arguments([str(path)], path), ArchDescription)

    assert config.form_finding.mirror == "x"
    assert config.form_finding.fold_heights is True
    assert config.sizing.fold_mirror is True
