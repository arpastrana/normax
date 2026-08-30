# SPDX-License-Identifier: Apache-2.0
"""
What a design is, how three blocks compose into one, and the search over it.

A design is found by three blocks in a row: a form finder chooses the shape, a
frame analysis says what the members carry, and a code check says how hard the
sections are worked. Each block is built from a structure on the host and then
called as a pure function of design parameters, which is what an optimizer
differentiates and what compiles. This module says what the three agree on,
composes them, and states the constrained search over the composition —
nothing here asks how any block computes.
"""

from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Float

from normax.analysis import AbstractFrameAnalyzer
from normax.analysis import MemberForces
from normax.config import ConstraintsConfig
from normax.form_finding import AbstractFormFinder
from normax.form_finding import CoefficientBounds
from normax.form_finding import SignGuardSpec
from normax.form_finding import select_free_nodes
from normax.loads import LoadCases
from normax.optimization import ConstrainedMaps
from normax.optimization import OptimizationBudget
from normax.optimization import OptimizationSolution
from normax.optimization import compute_penalty
from normax.optimization import optimize_augmented_lagrangian
from normax.sections import MemberSections
from normax.sections import TubeCatalog
from normax.sizing import AbstractMemberSizer
from normax.sizing import MemberSizes
from normax.structures import DesignShape
from normax.structures import Structure
from normax.symmetry import SignGuard
from normax.symmetry import fold_values
from normax.symmetry import guard_signs
from normax.symmetry import unfold_values


class DesignParameters(NamedTuple):
    """
    The quantities that vary between evaluations of the pipeline.

    Attributes
    ----------
    shape_parameters :
        What the form finder is called with, expanded out of whatever subspace
        the search moves in — one force density per member where the shape is
        found by equilibrium, one height per free node where it is written
        down, and nothing at all where it never moves.
    diameters :
        Outer diameter every member is analyzed and checked at.

    Notes
    -----
    The first field is named for the slot rather than for one block's quantity,
    since three parametrizations fill it with three different things. What they
    share is that the form finder is called with exactly this and the search
    reaches it through exactly one linear map.
    """

    shape_parameters: Float[Array, "shape_parameters"]
    diameters: Float[Array, "members"]


class Design(NamedTuple):
    """
    One structure carried through all three blocks.

    Attributes
    ----------
    shape :
        The geometry form finding settled on, and its member lengths.
    forces :
        What every member carries under every load case, or None where the
        pipeline carried no analysis.
    sizes :
        The sections the members were given and how hard each is worked, or
        None where the pipeline carried no check.

    Notes
    -----
    One field per block, in the order they ran, and nothing no block produced:
    a mass is arithmetic over a shape and a set of sections, which
    `compute_mass` does. A field is None exactly when its block was absent, so
    a reader can tell a missing answer from a zero one.
    """

    shape: DesignShape
    forces: MemberForces | None
    sizes: MemberSizes | None


class StructuralDesignPipeline(eqx.Module):
    """
    Form finding, analysis and a code check, composed into one function.

    Attributes
    ----------
    formfinder :
        The block that chooses the shape.
    analyzer :
        The block that says what the members carry, or None to stop at a shape.
    sizer :
        The block that says how hard the sections are worked, or None to stop
        at the internal forces.

    Notes
    -----
    Each block differentiates in its own way — a traced linear solve, a traced
    or hand-adjointed assembly, an implicit tangent at the root of a residual —
    and the composition hides that: a design comes back with a gradient in
    every parameter. Replacing a block with one that crosses a Tesseract
    boundary is a different argument here and nothing else.

    **The tail may be cut, and only from the end.** A shape alone answers what
    a geometry weighs at given sections; a shape and an analysis answer what it
    carries, which is what a compliance objective reads. A check without an
    analysis is refused rather than fed zeros: a sizer needs member forces, and
    a stand-in returning none would report every member unworked, which is
    indistinguishable from a real answer and wrong.

    **Three blocks and nothing else.** A section catalog is not one of them,
    so the pipeline never holds one: the check it composes carries the catalog
    its clauses are written against, and that is where a section comes from.
    """

    formfinder: AbstractFormFinder
    analyzer: AbstractFrameAnalyzer | None
    sizer: AbstractMemberSizer | None

    def __check_init__(self) -> None:
        """
        Refuse a check with no analysis behind it.

        Raises
        ------
        ValueError
            If there is a sizer and no analyzer to feed it member forces.
        """
        if self.sizer is not None and self.analyzer is None:
            raise ValueError(
                "a sizer reads member forces, so it needs an analyzer; "
                "drop the sizer too, or give the pipeline one"
            )

    def __call__(
        self,
        params: DesignParameters,
        loads: LoadCases,
    ) -> Design:
        """
        Form-find once, then analyze and check as far as the blocks reach.

        Parameters
        ----------
        params :
            What the form finder is called with, and the diameters every member
            is analyzed and checked at.
        loads :
            The load case the shape answers to, and the ones it is checked
            against.

        Returns
        -------
        design :
            The shape, what the members carry where there is an analysis, and
            how hard the given sections are worked where there is a check.

        Notes
        -----
        Every member is assumed to buckle over its own length — a strong
        assumption that presumes every node held in position, stated once here
        and nowhere else.
        """
        shape = self.formfinder(params.shape_parameters, loads.formfinding)
        if self.analyzer is None:
            return Design(shape, None, None)

        forces = self.analyzer(shape.xyz, params.diameters, loads.analysis)
        if self.sizer is None:
            return Design(shape, forces, None)

        utilization = self.sizer.compute_utilization(
            params.diameters, forces, shape.lengths
        )
        sections = self.sizer.catalog(params.diameters)
        sizes = MemberSizes(sections, utilization)

        return Design(shape, forces, sizes)


def compute_member_mass(
    sections: MemberSections,
    lengths: Float[Array, "members"],
) -> Float[Array, ""]:
    """
    Total mass of a set of members, `rho * sum(A L)`.

    Parameters
    ----------
    sections :
        The section of every member, and the steel it is cut from.
    lengths :
        Length of every member.

    Returns
    -------
    mass :
        Total mass.
    """
    per_length = sections.material.density * sections.area

    return jnp.sum(per_length * lengths)


def compute_mass(design: Design) -> Float[Array, ""]:
    """
    Total mass of a design — the objective the whole pipeline serves.

    Parameters
    ----------
    design :
        A design with one section per member.

    Returns
    -------
    mass :
        Total mass of the members.

    Raises
    ------
    ValueError
        If the design carries no sections, its pipeline having had no check.
    """
    if design.sizes is None:
        raise ValueError("a design with no sections has no mass to read")

    return compute_member_mass(design.sizes.sections, design.shape.lengths)


def compute_member_compliance(
    sections: MemberSections,
    forces: MemberForces,
    lengths: Float[Array, "members"],
) -> Float[Array, ""]:
    """
    Strain energy stored in a set of members, summed over every load case.

    Parameters
    ----------
    sections :
        The section of every member, and the steel it is cut from.
    forces :
        Axial force and both end moments, per load case and member.
    lengths :
        Length of every member.

    Returns
    -------
    compliance :
        Total strain energy.

    Notes
    -----
    Loads are applied at nodes alone, so the axial force is constant along a
    member and the moment varies linearly between its ends. Both integrals are
    then exact in closed form and need nothing the analysis does not already
    report — no displacement crosses the boundary, and none has to:

        `N^2 L / 2EA` axially, `L (Mi^2 + Mi Mj + Mj^2) / 6EI` in bending.

    A circular hollow section has the same second moment about either axis, so
    the two bending terms share one `EI`.
    """
    e_mod = sections.material.e_mod
    stiffness_axial = e_mod * sections.area
    stiffness_bending = e_mod * sections.second_moment

    axial = forces.axial_force**2 * lengths / (2.0 * stiffness_axial)

    bending = 0.0
    for moments in (forces.moment_major, forces.moment_minor):
        near = moments[..., 0]
        far = moments[..., 1]
        squared = near**2 + near * far + far**2
        bending = bending + squared * lengths / (6.0 * stiffness_bending)

    return jnp.sum(axial + bending)


def compute_compliance(design: Design) -> Float[Array, ""]:
    """
    Strain energy of a design, summed over every load case it was analyzed for.

    Parameters
    ----------
    design :
        A design carrying member forces.

    Returns
    -------
    compliance :
        Total strain energy.

    Raises
    ------
    ValueError
        If the design carries no forces, its pipeline having had no analysis.
    """
    if design.forces is None:
        raise ValueError("a design with no member forces has no compliance to read")

    sections = design.sizes.sections if design.sizes is not None else None
    if sections is None:
        raise ValueError("compliance needs the sections the forces were found at")

    return compute_member_compliance(sections, design.forces, design.shape.lengths)


class DesignConstraints(NamedTuple):
    """
    What a design is held to beside the check.

    Attributes
    ----------
    diameter_min :
        Smallest diameter any member may take, held as a bound.
    length_min :
        Smallest length any member may keep, held as rows. Zero turns it off.
    rise_max :
        Height no free node may rise above, or None.
    sag_min :
        Height no free node may hang below, or None.
    sign_guard :
        The sign guarded densities must keep, or None.
    bounds :
        Box on the force densities where they are the coefficients, or None.

    Notes
    -----
    The length floor exists because nothing in a member check objects to a
    vanishing member: its mass is an area times a length and its buckling length
    is its own length, so as it shortens it becomes both free and unbucklable.
    """

    diameter_min: float
    length_min: float
    rise_max: float | None
    sag_min: float | None
    sign_guard: SignGuard | None
    bounds: tuple[float, float] | None


def compute_mass_problem(
    problem: "DesignProblem",
    params: DesignParameters,
) -> Float[Array, ""]:
    """
    Total mass at given parameters — the objective the package ships.

    Parameters
    ----------
    problem :
        The problem supplying the form finder, the catalog and the loads.
    params :
        What the form finder is called with, and the diameters.

    Returns
    -------
    mass :
        Total mass of the members.

    Notes
    -----
    Reads the form finder and the catalog alone rather than running the whole
    pipeline: a mass is arithmetic over sections and lengths, so an analysis
    and a check would be paid for on every objective gradient and thrown away.
    """
    pipeline = problem.pipeline
    if pipeline.sizer is None:
        raise ValueError("a mass needs sections, and this pipeline has no check")

    shape = pipeline.formfinder(params.shape_parameters, problem.loads.formfinding)
    sections = pipeline.sizer.catalog(params.diameters)

    return compute_member_mass(sections, shape.lengths)


def build_compliance_objective(
    catalog: TubeCatalog,
) -> Callable[["DesignProblem", DesignParameters], Float[Array, ""]]:
    """
    An objective that minimizes strain energy rather than mass.

    Parameters
    ----------
    catalog :
        The section catalog the members are drawn from, which the objective
        needs for `EA` and `EI` and cannot read off the blocks: a pipeline cut
        after its analysis carries no check, and a check is what holds a
        catalog.

    Returns
    -------
    objective :
        What `DesignProblem.objective` is set to for a compliance search.

    Notes
    -----
    The stiffest structure at a given set of sections rather than the lightest
    that satisfies a standard. Nothing holds the diameters here, so a
    compliance search either freezes them or bounds the mass itself.
    """

    def compute_compliance_problem(
        problem: "DesignProblem",
        params: DesignParameters,
    ) -> Float[Array, ""]:
        """
        Strain energy at given parameters, over every analyzed load case.

        Parameters
        ----------
        problem :
            The problem supplying the form finder, the analysis and the loads.
        params :
            What the form finder is called with, and the diameters.

        Returns
        -------
        compliance :
            Total strain energy.

        Raises
        ------
        ValueError
            If the pipeline carries no analysis to find the forces with.
        """
        pipeline = problem.pipeline
        if pipeline.analyzer is None:
            raise ValueError("compliance needs an analysis, and this has none")

        loads = problem.loads
        shape = pipeline.formfinder(params.shape_parameters, loads.formfinding)
        forces = pipeline.analyzer(shape.xyz, params.diameters, loads.analysis)
        sections = catalog(params.diameters)

        return compute_member_compliance(sections, forces, shape.lengths)

    return compute_compliance_problem


class DesignProblem(NamedTuple):
    """
    A structure, its blocks, its loads, and the variables a search moves.

    Attributes
    ----------
    structure :
        The structure the blocks were built from.
    pipeline :
        The three blocks, composed.
    loads :
        The case the shape answers to, and the cases it is checked against.
    constraints :
        What the design is held to beside the check.
    section_groups :
        One column per orbit of the members the diameters are folded by, or
        None to size every member on its own.
    objective :
        What the descent minimizes, called with this problem and a set of
        parameters. `compute_mass_problem` is the mass the package ships and the
        default; `build_compliance_objective` returns the compliance a pipeline
        cut after its analysis answers instead.

    Notes
    -----
    The variable vector is the form finder's coefficients followed by the folded
    diameters. Both halves expand by one linear map — the form finder's basis
    and the section groups — so a symmetric design cannot break its symmetry
    however unsymmetric the loading, and every geometry a held-plan search
    reaches keeps the drawn plan by construction. The section groups sit here
    rather than on a block because no block reads them: they say how many
    variables the search carries, not how a frame is solved or a section
    checked.
    """

    structure: Structure
    pipeline: StructuralDesignPipeline
    loads: LoadCases
    constraints: DesignConstraints
    section_groups: Float[np.ndarray, "members groups"] | None = None
    objective: Callable[["DesignProblem", DesignParameters], Float[Array, ""]] = (
        compute_mass_problem
    )


class ProblemRecord(NamedTuple):
    """
    What a run arrived at, for the report and the record to read.

    Attributes
    ----------
    problem :
        The problem the descent ran on.
    solution :
        What the descent arrived at, and the road there.
    initial :
        The design at the start.
    optimized :
        The design at the solution.
    families :
        Name and member slice of every member family, or None to read the
        design whole.
    """

    problem: DesignProblem
    solution: OptimizationSolution
    initial: Design
    optimized: Design
    families: tuple[tuple[str, slice], ...] | None


def count_shape_coefficients(problem: DesignProblem) -> int:
    """
    How many coefficients the form finder expands its parameters from.

    Parameters
    ----------
    problem :
        The problem, read for its form finder.

    Returns
    -------
    width :
        Basis width, or the member count where every density moves freely.
    """
    return problem.pipeline.formfinder.count_shape_coefficients()


def expand_variables(
    problem: DesignProblem,
    x: Float[Array, "variables"],
) -> DesignParameters:
    """
    The design parameters a variable vector stands for.

    Parameters
    ----------
    problem :
        The problem supplying the two linear maps through its pipeline.
    x :
        The density coefficients followed by the folded diameters.

    Returns
    -------
    params :
        The force density of every member, as the form finder takes them, and
        one diameter per member.
    """
    width = count_shape_coefficients(problem)
    shape_parameters = expand_shape_coefficients(problem, x[:width])
    folded = x[width:]
    section_groups = problem.section_groups
    diameters = folded if section_groups is None else section_groups @ folded

    return DesignParameters(shape_parameters, diameters)


def fold_variables(
    problem: DesignProblem,
    parameters: Float[np.ndarray, "shape_parameters"],
    diameters: Float[np.ndarray, "members"],
) -> Float[np.ndarray, "variables"]:
    """
    The variable vector a set of shape parameters and diameters folds into.

    Parameters
    ----------
    problem :
        The problem supplying the two linear maps through its pipeline.
    parameters :
        What the form finder is called with, in its own space.
    diameters :
        Outer diameter of every member.

    Returns
    -------
    x :
        The shape coefficients followed by the folded diameters, each orbit
        taking the largest diameter among its members.
    """
    coefficients = read_shape_coefficients(problem, parameters)
    folded = fold_values(diameters, problem.section_groups)

    return np.concatenate([coefficients, folded])


def read_shape_coefficients(
    problem: DesignProblem,
    parameters: Float[np.ndarray, "shape_parameters"],
) -> Float[np.ndarray, "coefficients"]:
    """
    The coefficients a set of shape parameters reads back as, on the host.

    Parameters
    ----------
    problem :
        The problem supplying the form finder.
    parameters :
        What the form finder is called with, in its own space.

    Returns
    -------
    coefficients :
        The parameters themselves, or their coefficients in the basis, or
        nothing where the shape never moves.
    """
    return problem.pipeline.formfinder.read_shape_coefficients(parameters)


def expand_shape_coefficients(
    problem: DesignProblem,
    coefficients: Float[Array, "coefficients"],
) -> Float[Array, "shape_parameters"]:
    """
    What the form finder is called with, at given coefficients.

    Parameters
    ----------
    problem :
        The problem supplying the form finder.
    coefficients :
        The basis coefficients, or the parameters where there is no basis.

    Returns
    -------
    parameters :
        What the form finder is called with, in its own space.
    """
    return problem.pipeline.formfinder.expand_shape_coefficients(coefficients)


def bound_variables(
    problem: DesignProblem,
) -> list[tuple[float | None, float | None]]:
    """
    One bound pair per variable.

    Parameters
    ----------
    problem :
        The problem, read for its counts and its constraints.

    Returns
    -------
    boxes :
        Whatever box the form finder puts on its own coefficients, then the
        diameter floor on every folded diameter.

    Notes
    -----
    The coefficient half is the finder's to state, since only it knows what its
    coefficients mean: the density box belongs on densities, the height limits
    box a parametrization whose coefficients are heights, and one that moves no
    geometry contributes no pairs at all. Every limit is handed over and the
    finder takes the ones it is in.
    """
    held = problem.constraints
    limits = CoefficientBounds(held.bounds, held.rise_max, held.sag_min)
    coefficients = problem.pipeline.formfinder.bound_coefficients(limits)
    groups = problem.structure.num_edges
    if problem.section_groups is not None:
        groups = int(problem.section_groups.shape[1])

    boxes = coefficients + [(held.diameter_min, None)] * groups

    return boxes


def evaluate_constraints(
    problem: DesignProblem,
    params: DesignParameters,
    design: Design,
) -> Float[Array, "constraints"]:
    """
    How far above zero every inequality row sits at a design.

    Parameters
    ----------
    problem :
        The problem supplying the constraints and the free nodes.
    params :
        The parameters the design was evaluated at.
    design :
        The design.

    Returns
    -------
    rows :
        The utilization rows `1 - U`, case-major, where the pipeline carried a
        check, then the rise, sag, length and sign rows the constraints ask
        for, each normalized to the utilization rows' scale.

    Raises
    ------
    ValueError
        If nothing at all would be held: an empty row set is a search with no
        constraints rather than one whose constraints are all satisfied, and
        `jnp.concatenate` on it would fail further from the cause.
    """
    held = problem.constraints
    rows = []
    if design.sizes is not None:
        rows.append(1.0 - design.sizes.utilization.ravel())

    heights = design.shape.xyz[select_free_nodes(problem.structure), 2]
    if held.rise_max is not None:
        rows.append((held.rise_max - heights) / held.rise_max)
    if held.sag_min is not None:
        scale = abs(held.sag_min) or abs(held.rise_max or 1.0)
        rows.append((heights - held.sag_min) / scale)
    if held.length_min > 0.0:
        rows.append((design.shape.lengths - held.length_min) / held.length_min)
    guard = problem.pipeline.formfinder.read_sign_guard(held.sign_guard)
    if guard is not None:
        signed = guard.signs * params.shape_parameters[guard.members]
        rows.append((signed - guard.margin) / guard.scale)

    if not rows:
        raise ValueError(
            "this problem states no constraints: it carries no check, and its "
            "config names no rise, sag, length or sign limit either"
        )

    return jnp.concatenate(rows)


def design_maps(problem: DesignProblem) -> ConstrainedMaps:
    """
    The compiled programs a constrained descent over the design calls.

    Parameters
    ----------
    problem :
        The problem to compile.

    Returns
    -------
    maps :
        The augmented objective, the mass, and the slack, each jitted.

    Notes
    -----
    The objective reads only the blocks it needs — the shipped mass takes the
    form finder and the catalog alone — while the slack runs the whole
    pipeline, and the shortest length reads the form finder alone, being what
    the descent consults before it hands a geometry to a solver that would die
    on a collapsed one rather than refuse it. The rows are aggregated inside
    the traced augmented program, so every constraint costs one reverse pass
    together — and one crossing of
    whatever boundary a block sits behind.
    """
    pipeline = problem.pipeline
    loads = problem.loads

    def design_objective(x: Float[Array, "variables"]) -> Float[Array, ""]:
        params = expand_variables(problem, x)

        return problem.objective(problem, params)

    def slack_constraints(x: Float[Array, "variables"]) -> Float[Array, "constraints"]:
        params = expand_variables(problem, x)
        design = pipeline(params, loads)

        return evaluate_constraints(problem, params, design)

    def augmented_lagrangian(x, multipliers, penalty, reference):
        penalized = compute_penalty(slack_constraints(x), multipliers, penalty)

        return design_objective(x) / reference + penalized

    def read_point(x: Float[Array, "variables"]):
        return design_objective(x), slack_constraints(x)

    def read_shortest(x: Float[Array, "variables"]) -> Float[Array, ""]:
        params = expand_variables(problem, x)
        shape = pipeline.formfinder(params.shape_parameters, loads.formfinding)

        return jnp.min(shape.lengths)

    maps = ConstrainedMaps(
        jax.jit(jax.value_and_grad(augmented_lagrangian)),
        jax.jit(jax.value_and_grad(design_objective)),
        jax.jit(slack_constraints),
        jax.jit(read_point),
        jax.jit(read_shortest),
    )

    return maps


def solve_problem(
    problem: DesignProblem,
    start: Float[np.ndarray, "variables"],
    budget: OptimizationBudget,
    progress: bool = False,
) -> OptimizationSolution:
    """
    Descend the problem's objective under the check and the constraints.

    Parameters
    ----------
    problem :
        The problem to descend, read for its objective and its constraints.
    start :
        The variable vector to leave from.
    budget :
        What the descent may spend, and when it stops.
    progress :
        Whether the descent draws a progress bar while it runs.

    Returns
    -------
    solution :
        The parameters it stopped on, the objective and violation of every
        round, and how it ended.

    Notes
    -----
    The parameters come back rather than a design: `create_design` is what
    turns them into one, at the start and at the answer alike.
    """
    maps = design_maps(problem)
    boxes = bound_variables(problem)

    return optimize_augmented_lagrangian(maps, start, boxes, budget, progress)


def envelope_diameters(
    problem: DesignProblem,
    coefficients: Float[np.ndarray, "coefficients"],
    seeded: Float[np.ndarray, "members"],
) -> Float[np.ndarray, "members"]:
    """
    The diameters a frozen-seed analysis asks of every member, enveloped.

    Parameters
    ----------
    problem :
        The problem supplying the blocks and the loads.
    coefficients :
        Where the shape is found.
    seeded :
        Outer diameter every member is first analyzed at.

    Returns
    -------
    diameters :
        The largest diameter any load case demands of each member, floored —
        the classical design-office move, analyze at a guess and size to the
        forces, which is where a search starts.

    Raises
    ------
    ValueError
        If the pipeline carries no analysis or no check, an envelope being a
        reading off all three blocks.
    """
    pipeline = problem.pipeline
    if pipeline.analyzer is None or pipeline.sizer is None:
        raise ValueError("an envelope needs all three blocks, and this has not")

    q = expand_shape_coefficients(problem, jnp.asarray(coefficients))
    shape = pipeline.formfinder(q, problem.loads.formfinding)
    forces = pipeline.analyzer(shape.xyz, seeded, problem.loads.analysis)
    sizes = pipeline.sizer(forces, shape.lengths)
    demanded = np.asarray(jnp.max(sizes.sections.diameter, axis=0))

    return np.maximum(demanded, problem.constraints.diameter_min)


def initialize_optimization_parameters(
    problem: DesignProblem,
    q: Float[np.ndarray, "members"],
    diameters: Float[np.ndarray, "members"],
) -> Float[np.ndarray, "variables"]:
    """
    The design parameters a search leaves from, at given force densities.

    Parameters
    ----------
    problem :
        The problem supplying the maps and the blocks.
    q :
        Force density of every member at the start.
    diameters :
        Outer diameter every member starts at, from the config's diameter
        initializer, floored where the constraints ask for more.

    Returns
    -------
    parameters :
        The densities' coefficients, and the diameters folded.

    Notes
    -----
    The diameters are taken as given rather than sized to it. Enveloping a frozen
    analysis would open the search on a fully-stressed design, which is a
    second sizing rule beside the one the constraints already state.
    """
    diameters = np.maximum(diameters, problem.constraints.diameter_min)

    return fold_variables(problem, q, diameters)


def create_design(
    problem: DesignProblem,
    x: Float[np.ndarray, "variables"],
) -> Design:
    """
    The design a variable vector stands for, made once through the three blocks.

    Parameters
    ----------
    problem :
        The problem the vector belongs to.
    x :
        The density coefficients followed by the folded diameters.

    Returns
    -------
    design :
        The shape, the forces and the checked sections at that point.
    """
    params = expand_variables(problem, jnp.asarray(x))

    return problem.pipeline(params, problem.loads)


def unfold_diameters(
    problem: DesignProblem,
    x: Float[np.ndarray, "variables"],
) -> Float[np.ndarray, "members"]:
    """
    One diameter per member, off a variable vector, on the host.

    Parameters
    ----------
    problem :
        The problem supplying the folding.
    x :
        The density coefficients followed by the folded diameters.

    Returns
    -------
    diameters :
        Outer diameter of every member.
    """
    folded = np.asarray(x)[count_shape_coefficients(problem) :]

    return unfold_values(folded, problem.section_groups)


def build_design_constraints(
    config: ConstraintsConfig,
    guarded: SignGuardSpec | None,
    density_start: Float[np.ndarray, "members"],
) -> DesignConstraints:
    """
    What the design is held to, read off a run config and the start.

    Parameters
    ----------
    config :
        The floors, the height limits and the density box the file names.
    guarded :
        Which members must keep a sign and by how much, or None for no guard.
    density_start :
        Force density of every member at the start, which the guard is scaled
        against.

    Returns
    -------
    constraints :
        Everything the descent is held to beside the check.

    Notes
    -----
    The guard is scaled here, once, off the densities the descent leaves from,
    and then held fixed. Reading it off each iterate instead would shrink the
    margin as the densities shrink, so the row would chase the design rather
    than constrain it. A margin of zero or less asks for no rows at all — the
    start is still signed, by the initializer, but nothing holds it there.
    """
    guard = None
    if guarded is not None and guarded.margin_fraction > 0.0:
        guard = guard_signs(
            density_start,
            guarded.signs,
            guarded.members,
            guarded.margin_fraction,
        )

    bounds = None if config.bounds is None else (config.bounds.min, config.bounds.max)
    constraints = DesignConstraints(
        config.diameter_min,
        config.length_min,
        config.rise_max,
        config.sag_min,
        guard,
        bounds,
    )

    return constraints


# The sign a guarded family keeps, by the word a run config uses.
SIGN_WORDS = {"tension": 1.0, "compression": -1.0}


def assign_signs(
    config: ConstraintsConfig,
    families: tuple[tuple[str, slice], ...],
    num_members: int,
) -> SignGuardSpec | None:
    """
    Which members the start must sign, read off a run config by family name.

    Parameters
    ----------
    config :
        The constraints section, read for the sign guard and its margin.
    families :
        Name and member slice of every family the structure has.
    num_members :
        How many members the structure has, closing any open-ended slice.

    Returns
    -------
    guarded :
        Signs and indices of the guarded members with the margin, or None for
        no guard.

    Raises
    ------
    ValueError
        If the guard names a family the structure lacks, or a sign that is not
        `tension` or `compression`.
    """
    if config.sign_guard is None:
        return None

    named = dict(families)
    signs = []
    members = []
    for family, word in config.sign_guard.items():
        if family not in named:
            raise ValueError(f"no family {family!r} to guard, known: {sorted(named)}")
        if word not in SIGN_WORDS:
            raise ValueError(f"sign must be one of {sorted(SIGN_WORDS)}, got {word!r}")
        indices = np.arange(*named[family].indices(num_members))
        signs.append(np.full(indices.size, SIGN_WORDS[word]))
        members.append(indices)

    guarded = SignGuardSpec(
        np.concatenate(signs), np.concatenate(members), config.sign_margin_fraction
    )

    return guarded
