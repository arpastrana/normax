# SPDX-License-Identifier: Apache-2.0
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from normax.analysis.smax import SmaxAnalyzer
from normax.design import DesignConstraints
from normax.design import DesignParameters
from normax.design import DesignProblem
from normax.design import StructuralDesignPipeline
from normax.design import bound_variables
from normax.design import design_maps
from normax.form_finding import AbstractFormFinder
from normax.form_finding import FormFoundShape
from normax.form_finding import PlanBasis
from normax.form_finding import select_free_nodes
from normax.loads import assemble_load_cases
from normax.loads import load_uniform
from normax.materials import Steel355
from normax.sections import build_section_family
from normax.sizing.ec3 import Ec3Sizer
from normax.structures import Structure
from normax.structures import build_arch_2d
from normax.structures import compute_member_lengths

# A small arch under 180 kN, in millimeters and newtons.
SPAN = 4_000.0
RISE = 1_200.0
TOTAL_LOAD = 180_000.0
NUM_EDGES = 4

# The diameter every member starts at, and the floor under it.
SEED = 120.0
DIAMETER_FLOOR = 20.0


class HeightsFormFinder(AbstractFormFinder):
    """
    Free heights: the coordinates are the free nodes' z, in the drawn plan.

    Attributes
    ----------
    xyz :
        The drawn geometry, whose plan and supports every shape keeps.
    edges :
        The two node indices spanned by every member.
    nodes_free :
        Indices of the nodes whose height a call writes.
    width :
        How many heights a call takes.

    Notes
    -----
    Not funicular: the loads are accepted and ignored, so the frame analysis
    downstream sees whatever bending the heights raise.
    """

    xyz: Float[Array, "nodes 3"]
    edges: Int[np.ndarray, "members 2"]
    nodes_free: Int[np.ndarray, "nodes_free"]
    width: int = eqx.field(static=True)
    basis: PlanBasis | None

    def __init__(self, structure: Structure) -> None:
        """
        Build a heights finder on a drawn structure.

        Parameters
        ----------
        structure :
            The structure supplying the plan, the members and the supports.
        """
        nodes_free = select_free_nodes(structure)

        self.xyz = jnp.asarray(structure.nodes)
        self.edges = np.asarray(structure.edges)
        self.nodes_free = nodes_free
        self.width = int(nodes_free.size)
        self.basis = None

    def count_coordinates(self) -> int:
        """
        How many coordinates a call takes.
        """
        return self.width

    def __call__(
        self,
        heights: Float[Array, "nodes_free"],
        loads: Float[Array, "nodes 3"],
    ) -> FormFoundShape:
        """
        The drawn geometry with the free nodes lifted to the given heights.

        Parameters
        ----------
        heights :
            Height of every free node.
        loads :
            Accepted and ignored.

        Returns
        -------
        shape :
            The geometry, and its member lengths.
        """
        xyz = self.xyz.at[self.nodes_free, 2].set(heights)
        lengths = compute_member_lengths(xyz, self.edges)

        return FormFoundShape(xyz, lengths)


class DrawnFormFinder(AbstractFormFinder):
    """
    Sizing only: the shape is the drawn geometry, whatever it is called with.

    Attributes
    ----------
    xyz :
        The drawn geometry.
    edges :
        The two node indices spanned by every member.
    width :
        Zero: a call takes no coordinates.

    Notes
    -----
    A problem over this finder moves the diameters alone, and must set no
    sign guard, there being no densities to guard.
    """

    xyz: Float[Array, "nodes 3"]
    edges: Int[np.ndarray, "members 2"]
    width: int = eqx.field(static=True)
    basis: PlanBasis | None

    def __init__(self, structure: Structure) -> None:
        """
        Build a drawn finder on a structure.

        Parameters
        ----------
        structure :
            The structure supplying the geometry and the members.
        """
        self.xyz = jnp.asarray(structure.nodes)
        self.edges = np.asarray(structure.edges)
        self.width = 0
        self.basis = None

    def count_coordinates(self) -> int:
        """
        How many coordinates a call takes.
        """
        return self.width

    def __call__(
        self,
        coordinates: Float[Array, "0"],
        loads: Float[Array, "nodes 3"],
    ) -> FormFoundShape:
        """
        The drawn geometry as it stands.

        Parameters
        ----------
        coordinates :
            Accepted and ignored, an empty vector.
        loads :
            Accepted and ignored.

        Returns
        -------
        shape :
            The drawn geometry, and its member lengths.
        """
        lengths = compute_member_lengths(self.xyz, self.edges)

        return FormFoundShape(self.xyz, lengths)


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
