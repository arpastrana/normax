import jax.numpy as jnp
import numpy as np
import pytest

from normax.analysis.smax import SmaxAnalyzer
from normax.design import DesignConstraints
from normax.design import DesignParameters
from normax.design import DesignProblem
from normax.design import StructuralDesignPipeline
from normax.design import bound_variables
from normax.design import design_maps
from normax.extras.comparison import DrawnFormFinder
from normax.extras.comparison import HeightsFormFinder
from normax.form_finding import select_free_nodes
from normax.loads import assemble_load_cases
from normax.loads import load_uniform
from normax.materials import Steel355
from normax.sections import build_section_family
from normax.sizing.ec3 import Ec3Sizer
from normax.structures import build_arch_2d

# A small arch under 180 kN, in millimeters and newtons.
SPAN = 4_000.0
RISE = 1_200.0
TOTAL_LOAD = 180_000.0
NUM_EDGES = 4

# The diameter every member starts at, and the floor under it.
SEED = 120.0
DIAMETER_FLOOR = 20.0


@pytest.fixture(scope="module")
def structure():
    return build_arch_2d(num_edges=NUM_EDGES, span=SPAN, rise=RISE)


@pytest.fixture(scope="module")
def loads(structure):
    return assemble_load_cases([load_uniform(structure, TOTAL_LOAD)])


@pytest.fixture(scope="module")
def family():
    return build_section_family(Steel355(), 3)


def build_problem(structure, formfinder, family, loads):
    """
    A design problem over a comparison finder, whose coordinates are its own.
    """
    pipeline = StructuralDesignPipeline(
        formfinder, SmaxAnalyzer(structure, family(SEED)), Ec3Sizer(structure, family)
    )
    constraints = DesignConstraints(DIAMETER_FLOOR, 0.0, None, None, None, None)

    return DesignProblem(structure, pipeline, loads, constraints)


def test_the_heights_finder_composes_and_the_mass_has_a_gradient(
    structure, family, loads
):
    finder = HeightsFormFinder(structure)
    problem = build_problem(structure, finder, family, loads)
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


def test_the_drawn_finder_composes_and_moves_the_diameters_alone(
    structure, family, loads
):
    finder = DrawnFormFinder(structure)
    problem = build_problem(structure, finder, family, loads)
    diameters = jnp.full(NUM_EDGES, SEED)

    design = problem.pipeline(DesignParameters(jnp.zeros(0), diameters), loads)
    assert jnp.array_equal(design.shape.xyz, jnp.asarray(structure.nodes))

    maps = design_maps(problem)
    mass, slope = maps.objective(diameters)
    slack = maps.slack(diameters)

    assert finder.width == 0
    assert len(bound_variables(problem)) == NUM_EDGES
    assert np.isfinite(float(mass)) and float(mass) > 0.0
    assert np.all(np.asarray(slope) > 0.0)
    assert np.all(np.isfinite(np.asarray(slack)))
