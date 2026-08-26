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
The load cases a truss or a shell is designed against.
"""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Float

from normax.loads import LoadCases
from normax.loads import assemble_load_cases
from normax.loads import create_loads_point
from normax.loads import create_loads_tributary
from normax.searches.config import LoadConfig
from normax.searches.config import ShellConfig
from normax.searches.config import ShellLoads
from normax.searches.config import TaskConfig
from normax.searches.settings import CASE_NAMES
from normax.searches.settings import SHELL_NAMES
from normax.structures import Structure


def build_load_cases(
    structure: Structure,
    weight: LoadConfig,
    num_bays: int,
) -> LoadCases:
    """
    Four cases of equal total, every one on the bottom chord alone.

    Parameters
    ----------
    structure :
        The truss to load.
    weight :
        The total and the asymmetry factor.
    num_bays :
        Number of bottom-chord segments, locating the interior deck nodes.

    Returns
    -------
    loads :
        The uniform deck the shape answers to, the two half-span cases, and
        a fraction of the total concentrated at the midspan deck node.

    Notes
    -----
    The arch experiments' load family, moved onto the deck: the top chord
    carries nothing directly, matching a bridge whose traffic runs on the
    bottom chord. The three distributed cases are rescaled to the shared
    total so none wins by simply carrying less; the point case carries its
    own fraction of it.
    """
    if num_bays % 2:
        raise ValueError(f"num_bays must be even for a midspan node, got {num_bays}")

    interior = np.arange(1, num_bays)
    along = np.asarray(structure.nodes)[interior, 0]
    middle = 0.5 * float(np.asarray(structure.nodes)[num_bays, 0])

    def deck_case(weights: Float[np.ndarray, "interior"]) -> Float[Array, "nodes 3"]:
        scaled = weights * (weight.total / float(weights.sum()))
        cases = [
            create_loads_point(structure, float(load), node=int(node))
            for node, load in zip(interior, scaled)
        ]

        return jnp.sum(jnp.stack(cases), axis=0)

    uniform = deck_case(np.ones(interior.size))
    near = deck_case(np.where(along <= middle, 1.0, weight.half_factor))
    concentrated = weight.total * weight.point_factor
    point = create_loads_point(structure, concentrated, node=num_bays // 2)
    cases = [uniform, near, point]
    if weight.mirrored_case:
        far = deck_case(np.where(along >= middle, 1.0, weight.half_factor))
        cases.insert(2, far)

    return assemble_load_cases(cases)


def load_names(weight: LoadConfig) -> tuple[str, ...]:
    """
    Name of every load case built, in build order.

    Parameters
    ----------
    weight :
        The load description, read for whether the mirrored case is built.

    Returns
    -------
    names :
        The case names, keeping their identities when one is deleted.
    """
    if weight.mirrored_case:
        return CASE_NAMES

    return (CASE_NAMES[0], CASE_NAMES[1], CASE_NAMES[3])


class LoadPlan(NamedTuple):
    """
    Every case a run is checked against, named, and what they each weigh.

    Attributes
    ----------
    cases :
        The case the shape is found under, and the stack every search is
        checked against.
    names :
        Name of every case, in build order.
    total :
        Downward force each distributed case carries, the scale the start's
        equilibrium gap is reported against.

    Notes
    -----
    A profile's one job on the loading side is to return this: how a family
    spreads a load over its own nodes is the one part of a load case no
    shared flow can know, while the three things read afterwards are the same
    for every family.
    """

    cases: LoadCases
    names: tuple[str, ...]
    total: float


def truss_loads(structure: Structure, config: TaskConfig) -> LoadPlan:
    """
    The deck cases of experiments 18 and 19, gathered into a plan.

    Parameters
    ----------
    structure :
        The truss to load.
    config :
        The run description, read for the loads and the bay count.

    Returns
    -------
    plan :
        The four deck cases, or three where the mirrored one is deleted.
    """
    weight = config.loads
    cases = build_load_cases(structure, weight, config.structure.num_bays)

    return LoadPlan(cases, load_names(weight), weight.total)


def tributary_areas(sketch: ShellConfig) -> Float[np.ndarray, "nodes"]:
    """
    Plan area every node of a polar cap carries.

    Parameters
    ----------
    sketch :
        The cap the generator was asked to draw.

    Returns
    -------
    areas :
        Plan area of every node, the apex first where there is one and then
        ring by ring.

    Notes
    -----
    Each ring owns the annulus reaching halfway to its neighbours, split
    evenly between its spokes; the apex owns the disc inside the first such
    boundary. The areas therefore sum to the whole plan exactly, which is what
    makes the supports' share readable as the difference between the stated
    pressure's total and the total actually applied.

    **An oculus is open, so it carries nothing.** The first ring then owns
    only the annulus outside itself, and the areas sum to the plan less the
    hole — the run's stated pressure buys less total load than the same
    pressure on a closed cap, which is part of what the opening costs.
    """
    rings = sketch.num_rings
    spokes = sketch.num_spokes

    rhos = sketch.radius * np.arange(1, rings + 1) / rings
    inner = np.concatenate([[0.0], 0.5 * (rhos[:-1] + rhos[1:])])
    outer = np.concatenate([0.5 * (rhos[:-1] + rhos[1:]), [sketch.radius]])

    inner[0] = rhos[0] if sketch.oculus else 0.5 * rhos[0]

    annuli = np.pi * (outer**2 - inner**2) / spokes
    ring_areas = np.repeat(annuli, spokes)
    if sketch.oculus:
        return ring_areas

    apex = np.pi * inner[0] ** 2

    return np.concatenate([[apex], ring_areas])


def sector_areas(
    sketch: ShellConfig,
    weight: ShellLoads,
    areas: Float[np.ndarray, "nodes"],
    center: int,
) -> Float[np.ndarray, "nodes"]:
    """
    The tributary areas a drift over one sector loads each node through.

    Parameters
    ----------
    sketch :
        The cap the generator was asked to draw.
    weight :
        The loading, read for the sector width and what the rest keeps.
    areas :
        Plan area of every node, as the tributary rule shares it out.
    center :
        Spoke the sector is centred on.

    Returns
    -------
    drifting :
        Each node's area, kept whole inside the sector and scaled by
        `drift_factor` outside it. A crown node, sitting on every sector's
        axis, is always inside.

    Notes
    -----
    **The drift grades rather than spotlights.** The sector keeps the full
    pressure and the rest of the plan keeps its fraction, which is the
    trusses' half-span construction read onto a disc. Emptying the plan
    outside the sector instead would concentrate the whole roof's load on a
    slice of it once rescaled — a stress test rather than a snow load, and one
    whose feasible set is measurably harder to descend.
    """
    spokes = sketch.num_spokes
    reach = weight.sector_spokes // 2

    offset = (np.arange(spokes) - center + reach) % spokes
    within = offset <= 2 * reach
    tiled = np.tile(within, sketch.num_rings)
    inside = tiled if sketch.oculus else np.concatenate([[True], tiled])

    return np.where(inside, areas, weight.drift_factor * areas)


def shell_loads(structure: Structure, config: TaskConfig) -> LoadPlan:
    """
    A uniform pressure, a drift over one sector, and that drift reflected.

    Parameters
    ----------
    structure :
        The shell to load.
    config :
        The run description, read for the pressure and the sector.

    Returns
    -------
    plan :
        The three cases, every one of them carrying the same total.

    Notes
    -----
    **The stated pressure and the applied total are two different numbers.**
    The pressure acts on the whole plan, but the boundary ring's tributary
    share sits on supported nodes and goes straight to ground, so the total
    the structure carries is what is left. That remainder is the plan's total,
    and both drift cases are rescaled onto it so no case wins by carrying
    less.

    **The second drift is the first one's mirror image**, built by reflecting
    the sector's centre rather than by permuting the case, so the two are the
    same construction at two centres and their asymmetries cancel over the
    pair. A design folded about that plane therefore loses nothing: what one
    case asks of a member, the other asks of its mirror twin.
    """
    sketch = config.structure
    weight = config.loads
    areas = tributary_areas(sketch)

    uniform = create_loads_tributary(structure, weight.pressure, jnp.asarray(areas))
    total = float(jnp.sum(jnp.abs(uniform)))

    if not weight.asymmetric_cases:
        return LoadPlan(assemble_load_cases([uniform]), SHELL_NAMES[:1], total)

    center = weight.sector_center
    reflected = (-center) % sketch.num_spokes

    drifts = []
    for spoke in (center, reflected):
        drifting = sector_areas(sketch, weight, areas, spoke)
        drift = create_loads_tributary(
            structure, weight.pressure, jnp.asarray(drifting)
        )
        carried = float(jnp.sum(jnp.abs(drift)))
        drifts.append(drift * (total / carried))

    cases = assemble_load_cases([uniform, *drifts])

    return LoadPlan(cases, SHELL_NAMES, total)
