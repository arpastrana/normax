# Copyright 2026 Rafael Pastrana
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
from normax.form_finding import FormFoundShape
from normax.form_finding import SignGuardSpec
from normax.form_finding import select_free_nodes
from normax.loads import LoadCases
from normax.optimization import ConstrainedMaps
from normax.optimization import OptimizationAnswer
from normax.optimization import OptimizationBudget
from normax.optimization import compute_penalty
from normax.optimization import descend_augmented_lagrangian
from normax.sections import MemberSections
from normax.sizing import AbstractMemberSizer
from normax.sizing import MemberSizes
from normax.structures import Structure
from normax.symmetry import SignGuard
from normax.symmetry import fold_values
from normax.symmetry import unfold_values


class DesignParameters(NamedTuple):
    """
    The quantities that vary between evaluations of the pipeline.

    Attributes
    ----------
    coordinates :
        What the form finder is called with — the force density of every
        member, with any held-plan subspace already expanded.
    diameters :
        Outer diameter every member is analyzed and checked at.
    """

    coordinates: Float[Array, "coordinates"]
    diameters: Float[Array, "members"]


class Design(NamedTuple):
    """
    One structure carried through all three blocks.

    Attributes
    ----------
    shape :
        The geometry form finding settled on, and its member lengths.
    forces :
        What every member carries under every load case.
    sizes :
        The sections the members were given, and how hard each is worked.

    Notes
    -----
    One field per block, in the order they ran, and nothing no block produced:
    a mass is arithmetic over two of these fields, which `compute_mass` does.
    """

    shape: FormFoundShape
    forces: MemberForces
    sizes: MemberSizes


class StructuralDesignPipeline(eqx.Module):
    """
    Form finding, analysis and a code check, composed into one function.

    Attributes
    ----------
    formfinder :
        The block that chooses the shape.
    analyzer :
        The block that says what the members carry.
    sizer :
        The block that says how hard the sections are worked.
    spread :
        One column per orbit of the members the diameters are folded by, or
        None to size every member on its own.

    Notes
    -----
    Each block differentiates in its own way — a traced linear solve, a traced
    or hand-adjointed assembly, an implicit tangent at the root of a residual —
    and the composition hides that: a design comes back with a gradient in
    every parameter. Replacing a block with one that crosses a Tesseract
    boundary is a different argument here and nothing else.
    """

    formfinder: AbstractFormFinder
    analyzer: AbstractFrameAnalyzer
    sizer: AbstractMemberSizer
    spread: Float[np.ndarray, "members patterns"] | None = None

    def __call__(
        self,
        params: DesignParameters,
        loads: LoadCases,
    ) -> Design:
        """
        Form-find once, analyze every load case, and check every member.

        Parameters
        ----------
        params :
            The form finder's coordinates, and the diameters every member is
            analyzed and checked at.
        loads :
            The load case the shape answers to, and the ones it is checked
            against.

        Returns
        -------
        design :
            The shape, what the members carry, and how hard the given sections
            are worked under every load case.

        Notes
        -----
        Every member is assumed to buckle over its own length — a strong
        assumption that presumes every node held in position, stated once here
        and nowhere else.
        """
        shape = self.formfinder(params.coordinates, loads.formfinding)
        forces = self.analyzer(shape.xyz, params.diameters, loads.analysis)
        utilization = self.sizer.compute_utilization(
            params.diameters, forces, shape.lengths
        )
        sections = self.sizer.family(params.diameters)
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
    """
    return compute_member_mass(design.sizes.sections, design.shape.lengths)


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
        Box on the force densities where they are the coordinates, or None.

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

    Notes
    -----
    The variable vector is the form finder's coordinates followed by the folded
    diameters. Both halves expand by one linear map — the form finder's basis
    and the pipeline's spread — so a symmetric design cannot break its symmetry
    however unsymmetric the loading, and every geometry a held-plan search
    reaches keeps the drawn plan by construction.
    """

    structure: Structure
    pipeline: StructuralDesignPipeline
    loads: LoadCases
    constraints: DesignConstraints


class DesignRecord(NamedTuple):
    """
    What a run arrived at, for the report, the record and the viewer to read.

    Attributes
    ----------
    problem :
        The problem the descent ran on.
    answer :
        What the descent arrived at, and the road there.
    initial :
        The design at the start.
    optimized :
        The design at the answer.
    families :
        Name and member slice of every member family, or none to read the
        design whole.
    """

    problem: DesignProblem
    answer: OptimizationAnswer
    initial: Design
    optimized: Design
    families: tuple[tuple[str, slice], ...]


def count_coordinates(problem: DesignProblem) -> int:
    """
    How many coordinates the form finder is called with.

    Parameters
    ----------
    problem :
        The problem, read for its form finder.

    Returns
    -------
    width :
        Basis width, or the member count where every density moves freely.
    """
    return problem.pipeline.formfinder.count_coordinates()


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
        The coordinates followed by the folded diameters.

    Returns
    -------
    params :
        The coordinates as the form finder takes them, and one diameter per
        member.
    """
    width = count_coordinates(problem)
    coordinates = read_member_densities(problem, x[:width])
    folded = x[width:]
    diameters = (
        folded if problem.pipeline.spread is None else problem.pipeline.spread @ folded
    )

    return DesignParameters(coordinates, diameters)


def fold_variables(
    problem: DesignProblem,
    q: Float[np.ndarray, "members"],
    diameters: Float[np.ndarray, "members"],
) -> Float[np.ndarray, "variables"]:
    """
    The variable vector a set of densities and diameters folds into.

    Parameters
    ----------
    problem :
        The problem supplying the two linear maps through its pipeline.
    q :
        Force density of every member.
    diameters :
        Outer diameter of every member.

    Returns
    -------
    x :
        The coordinates followed by the folded diameters, each orbit taking the
        largest diameter among its members.
    """
    coordinates = read_coordinates(problem, q)
    folded = fold_values(diameters, problem.pipeline.spread)

    return np.concatenate([coordinates, folded])


def read_coordinates(
    problem: DesignProblem,
    q: Float[np.ndarray, "members"],
) -> Float[np.ndarray, "coordinates"]:
    """
    The coordinates a set of force densities reads back as, on the host.

    Parameters
    ----------
    problem :
        The problem supplying the form finder.
    q :
        Force density of every member.

    Returns
    -------
    coordinates :
        The densities themselves, or their coordinates in the basis.
    """
    return problem.pipeline.formfinder.read_coordinates(q)


def read_member_densities(
    problem: DesignProblem,
    coordinates: Float[Array, "coordinates"],
) -> Float[Array, "members"]:
    """
    The force density of every member at given coordinates.

    Parameters
    ----------
    problem :
        The problem supplying the form finder.
    coordinates :
        The basis coordinates, or the densities where there is no basis.

    Returns
    -------
    q :
        Force density of every member, as the form finder is called with.
    """
    return problem.pipeline.formfinder.expand_coordinates(coordinates)


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
        The density box on the coordinates where the densities are the
        coordinates, nothing on subspace coordinates, and the diameter floor
        on every folded diameter.
    """
    held = problem.constraints
    boxed = (None, None) if held.bounds is None else held.bounds
    width = count_coordinates(problem)
    patterns = problem.structure.num_edges
    if problem.pipeline.spread is not None:
        patterns = int(problem.pipeline.spread.shape[1])

    boxes = [boxed] * width + [(held.diameter_min, None)] * patterns

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
        The utilization rows `1 - U`, case-major, then the rise, sag, length
        and sign rows the constraints ask for, each normalized to the
        utilization rows' scale.
    """
    held = problem.constraints
    rows = [1.0 - design.sizes.utilization.ravel()]

    heights = design.shape.xyz[select_free_nodes(problem.structure), 2]
    if held.rise_max is not None:
        rows.append((held.rise_max - heights) / held.rise_max)
    if held.sag_min is not None:
        scale = abs(held.sag_min) or abs(held.rise_max or 1.0)
        rows.append((heights - held.sag_min) / scale)
    if held.length_min > 0.0:
        rows.append((design.shape.lengths - held.length_min) / held.length_min)
    if held.sign_guard is not None:
        guard = held.sign_guard
        signed = guard.signs * params.coordinates[guard.members]
        rows.append((signed - guard.margin) / guard.scale)

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
    The mass reads the form finder and the family alone; the slack runs the
    whole pipeline. The rows are aggregated inside the traced augmented program,
    so every constraint costs one reverse pass together — and one crossing of
    whatever boundary a block sits behind.
    """
    pipeline = problem.pipeline
    loads = problem.loads

    def weigh(x: Float[Array, "variables"]) -> Float[Array, ""]:
        params = expand_variables(problem, x)
        shape = pipeline.formfinder(params.coordinates, loads.formfinding)
        sections = pipeline.sizer.family(params.diameters)

        return compute_member_mass(sections, shape.lengths)

    def slack(x: Float[Array, "variables"]) -> Float[Array, "constraints"]:
        params = expand_variables(problem, x)
        design = pipeline(params, loads)

        return evaluate_constraints(problem, params, design)

    def augmented_lagrangian(x, multipliers, penalty, reference):
        penalized = compute_penalty(slack(x), multipliers, penalty)

        return weigh(x) / reference + penalized

    maps = ConstrainedMaps(
        jax.jit(jax.value_and_grad(augmented_lagrangian)),
        jax.jit(jax.value_and_grad(weigh)),
        jax.jit(slack),
    )

    return maps


def optimize_design(
    problem: DesignProblem,
    start: Float[np.ndarray, "variables"],
    budget: OptimizationBudget,
) -> OptimizationAnswer:
    """
    Descend the mass under the check and the constraints, from a start.

    Parameters
    ----------
    problem :
        The problem to descend.
    start :
        The variable vector to leave from.
    budget :
        What the descent may spend, and when it stops.

    Returns
    -------
    answer :
        The variables, the mass and violation of every round, and how it ended.
    """
    maps = design_maps(problem)
    boxes = bound_variables(problem)

    return descend_augmented_lagrangian(maps, start, boxes, budget)


def envelope_diameters(
    problem: DesignProblem,
    coordinates: Float[np.ndarray, "coordinates"],
    seed: float,
) -> Float[np.ndarray, "members"]:
    """
    The diameters a frozen-seed analysis asks of every member, enveloped.

    Parameters
    ----------
    problem :
        The problem supplying the blocks and the loads.
    coordinates :
        Where the shape is found.
    seed :
        Outer diameter the frame is analyzed at.

    Returns
    -------
    diameters :
        The largest diameter any load case demands of each member, floored —
        the classical design-office move, analyze at a guess and size to the
        forces, which is where a search starts.
    """
    pipeline = problem.pipeline
    seeded = jnp.full(problem.structure.num_edges, seed)
    q = read_member_densities(problem, jnp.asarray(coordinates))
    shape = pipeline.formfinder(q, problem.loads.formfinding)
    forces = pipeline.analyzer(shape.xyz, seeded, problem.loads.analysis)
    sizes = pipeline.sizer(forces, shape.lengths)
    demanded = np.asarray(jnp.max(sizes.sections.diameter, axis=0))

    return np.maximum(demanded, problem.constraints.diameter_min)


def initialize_optimization_variables(
    problem: DesignProblem,
    q: Float[np.ndarray, "members"],
    seed: float,
) -> Float[np.ndarray, "variables"]:
    """
    The variable vector a search leaves from, at given force densities.

    Parameters
    ----------
    problem :
        The problem supplying the maps and the blocks.
    q :
        Force density of every member at the start.
    seed :
        Outer diameter the frame is first analyzed at.

    Returns
    -------
    x :
        The densities' coordinates, and the enveloped diameters folded.
    """
    coordinates = read_coordinates(problem, q)
    diameters = envelope_diameters(problem, coordinates, seed)

    return fold_variables(problem, q, diameters)


def evaluate_design(
    problem: DesignProblem,
    x: Float[np.ndarray, "variables"],
) -> Design:
    """
    The design a variable vector stands for, evaluated once.

    Parameters
    ----------
    problem :
        The problem the vector belongs to.
    x :
        The coordinates followed by the folded diameters.

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
        The coordinates followed by the folded diameters.

    Returns
    -------
    diameters :
        Outer diameter of every member.
    """
    return unfold_values(
        np.asarray(x)[count_coordinates(problem) :], problem.pipeline.spread
    )


def build_design_constraints(
    config: ConstraintsConfig,
    guard: SignGuard | None,
) -> DesignConstraints:
    """
    What the design is held to, read off a run config.

    Parameters
    ----------
    config :
        The floors, the height limits and the density box the file names.
    guard :
        The sign guard the start scaled, or None for none.

    Returns
    -------
    constraints :
        Everything the descent is held to beside the check.
    """
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
