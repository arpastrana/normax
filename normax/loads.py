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
The load cases a structure is shaped by and checked against, and their axis.

A load case is an array of nodal forces and nothing else, so a case built here
adds to any other and none of them belongs to a structure. Every generator takes
a structure and returns a pattern over its nodes, zeroed at the supports, which
is what lets a name in a configuration file select one.

**Everything that knows what a load case axis means lives here**, which is three
operations: stacking several cases of a container into one, taking one back out,
and counting them. They are generic over the container because none of them
reads a field — a pytree map is the whole implementation — so the analysis
stage's internal forces and the check's design actions share them rather than
each carrying its own. Two stages that stacked their own would be free to
disagree about the order of that axis, which is a bug no shape catches.

A leaf otherwise. Nothing here computes an equilibrium, a resistance or a
geometry, so any stage can speak about load cases without importing the stage
beside it.
"""

from collections.abc import Sequence
from typing import NamedTuple
from typing import TypeVar

import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float

from normax.structures import Structure

# Any container whose fields take a leading load case axis, at either rank.
LoadCaseAxis = TypeVar("LoadCaseAxis", bound=tuple)


def loads_uniform(
    structure: Structure,
    load: float,
) -> Float[Array, "nodes 3"]:
    """
    A downward point load of the same size on every free node.

    Parameters
    ----------
    structure :
        The structure to load.
    load :
        Magnitude of the downward point load.

    Returns
    -------
    loads :
        Force applied at every node.

    Notes
    -----
    The load case a funicular structure is form-found under, so the geometry carries
    it in pure tension or compression and the members see no bending. Every
    other load case is a departure from it, and the bending that appears is what a
    frame analysis exists to report.
    """
    return _nodal_loads(structure, jnp.ones(structure.num_nodes) * load)


def loads_half_span_mirrored(
    structure: Structure,
    load: float,
) -> Float[Array, "nodes 3"]:
    """
    A downward point load on the far half of the span, and nothing on the near one.

    Parameters
    ----------
    structure :
        The structure to load.
    load :
        Magnitude of the downward point load on the loaded half.

    Returns
    -------
    loads :
        Force applied at every node.

    Notes
    -----
    The exact reflection of `loads_half_span` on a structure whose nodes are
    symmetric about midspan, a node sitting exactly at midspan being loaded
    either way. One asymmetric case on its own biases a search towards the half
    it leaves light; the pair does not.
    """
    return loads_half_span(structure, load, axis=0, factor=0.0, mirrored=True)


def loads_half_span(
    structure: Structure,
    load: float,
    *,
    axis: int = 0,
    factor: float = 0.0,
    mirrored: bool = False,
) -> Float[Array, "nodes 3"]:
    """
    A downward point load on one half of the span and a fraction on the other.

    Parameters
    ----------
    structure :
        The structure to load.
    load :
        Magnitude of the downward point load on the loaded half.
    axis :
        Index of the global axis the span is measured along.
    factor :
        Fraction of that load carried by the other half.
    mirrored :
        Whether to load the far half instead of the near one.

    Returns
    -------
    loads :
        Force applied at every node.

    Raises
    ------
    ValueError
        If the axis is not 0, 1 or 2.
    """
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1 or 2, got {axis}")

    along = structure.nodes[:, axis]
    middle = 0.5 * (jnp.min(along) + jnp.max(along))
    loaded = along >= middle if mirrored else along <= middle
    applied = jnp.where(loaded, load, load * factor)

    return _nodal_loads(structure, applied)


def loads_point(
    structure: Structure,
    load: float,
    *,
    node: int,
) -> Float[Array, "nodes 3"]:
    """
    A single downward point load at one node.

    Parameters
    ----------
    structure :
        The structure to load.
    load :
        Magnitude of the downward point load.
    node :
        Index of the node carrying it.

    Returns
    -------
    loads :
        Force applied at every node.

    Notes
    -----
    Adds to any other load case, being an array like the rest, so a concentrated
    load on top of a distributed one is a sum and needs no separate generator.
    A load placed on a support is discarded, since the support carries it
    straight to ground.
    """
    magnitudes = jnp.zeros(structure.num_nodes).at[node].set(load)

    return _nodal_loads(structure, magnitudes)


def _nodal_loads(
    structure: Structure,
    magnitudes: Float[Array, "nodes"],
) -> Float[Array, "nodes 3"]:
    """
    Downward forces of given magnitudes, zeroed at the supports.

    Parameters
    ----------
    structure :
        The structure supplying the node count and the supported nodes.
    magnitudes :
        Size of the downward force at every node.

    Returns
    -------
    loads :
        Force applied at every node.
    """
    vertical = jnp.zeros((structure.num_nodes, 3)).at[:, 2].set(-magnitudes)

    return vertical.at[structure.supports, :].set(0.0)


LOAD_CASE_REGISTRY = {
    "uniform": loads_uniform,
    "half_span": loads_half_span,
    "half_span_mirrored": loads_half_span_mirrored,
}


def create_loads_by_name(
    name: str,
    structure: Structure,
    magnitude: float,
) -> Float[Array, "nodes 3"]:
    """
    Build a named load case carrying a given total.

    Parameters
    ----------
    name :
        Name of the pattern, one of the keys of `LOAD_CASE_REGISTRY`.
    structure :
        The structure to load.
    magnitude :
        Total downward force the case carries.

    Returns
    -------
    loads :
        Force applied at every node.

    Raises
    ------
    ValueError
        If no pattern goes by that name.

    Notes
    -----
    **The magnitude is a total and not a nodal force**, which is what makes two
    cases comparable: a case on half the span carries the same total over fewer
    nodes rather than the same nodal force over fewer of them. The generator is
    called at unit load for its shape alone and the result scaled, so how a
    pattern spreads itself stays the generator's business.

    Named rather than passed as a function so that a load case can be written
    down in a configuration file, which is the only reason this indirection
    exists.
    """
    if name not in LOAD_CASE_REGISTRY:
        known = ", ".join(sorted(LOAD_CASE_REGISTRY))
        raise ValueError(f"unknown load case {name!r}, expected one of {known}")

    pattern = LOAD_CASE_REGISTRY[name](structure, 1.0)
    carried = jnp.abs(jnp.sum(pattern[:, 2]))
    applied = pattern * (magnitude / carried)

    return applied


class LoadCases(NamedTuple):
    """
    What a structure is shaped by, and what it is then checked against.

    Attributes
    ----------
    formfinding :
        Force applied at every node in the load case the shape answers to.
    analysis :
        Force applied at every node in every load case the members carry.

    Notes
    -----
    **One case shapes the structure and several check it, and the asymmetry is
    the point.** A funicular shape carries exactly one load case axially, which
    is what makes it funicular; every other case is a departure from it, and the
    bending that appears is the reason a frame analysis is in the pipeline at
    all. Form-finding under each case in turn would mean a different structure
    per case rather than one structure asked to survive all of them.

    The form-finding case is usually also one of the checked ones. Nothing here
    requires it, since a structure may be shaped by a load case that is never
    checked on its own.
    """

    formfinding: Float[Array, "nodes 3"]
    analysis: Float[Array, "load_cases nodes 3"]


def assemble_load_cases(
    load_cases: Sequence[Float[Array, "nodes 3"]],
) -> LoadCases:
    """
    The load case a structure is shaped by, and the ones it is checked against.

    Parameters
    ----------
    load_cases :
        Force applied at every node, one entry per checked load case.

    Returns
    -------
    loads :
        The checked cases stacked along a leading axis, and the first of them
        again as the case the shape answers to.

    Notes
    -----
    **The first case is the one the structure is form-found under**, which is
    the convention a list of cases has to carry somehow and the only one that
    needs no second argument. A structure shaped by a case it is never checked
    against is expressible by building the container directly.

    Stacking here rather than at every call site is what keeps a load case axis
    from being assembled differently by two callers.
    """
    formfinding_case = load_cases[0]
    analysis_cases = jnp.stack(list(load_cases))

    return LoadCases(formfinding_case, analysis_cases)


def stack_load_cases(per_case: Sequence[LoadCaseAxis]) -> LoadCaseAxis:
    """
    Several load cases of one container, stacked into one container.

    Parameters
    ----------
    per_case :
        The container of every load case, in order.

    Returns
    -------
    stacked :
        The same container, every field carrying a leading load case axis.

    Notes
    -----
    **The one place a load case axis is added.** A solver or a check answers one
    case at a time and never sees the axis; this is what puts the answers side
    by side, in the order the cases were given.

    A pytree map is what makes it generic: it reads the container's structure
    rather than its field names, so a container that gains a field needs no
    change here and one holding a nested container stacks to the same depth.
    """
    return jax.tree.map(lambda *cases: jnp.stack(cases), *per_case)


def select_load_case(stacked: LoadCaseAxis, load_case: int) -> LoadCaseAxis:
    """
    One load case of a stacked container.

    Parameters
    ----------
    stacked :
        A container whose every field carries a leading load case axis.
    load_case :
        Index of the load case to read.

    Returns
    -------
    selected :
        The same container, for that load case alone and without the axis.

    Notes
    -----
    The inverse of `stack_load_cases`, and generic for the same reason. What
    comes back is the rank a clause and a solver both work at, neither of them
    having anything to say about the other cases.
    """
    return jax.tree.map(lambda field: field[load_case], stacked)


def count_load_cases(stacked: LoadCaseAxis) -> int:
    """
    How many load cases a stacked container carries.

    Parameters
    ----------
    stacked :
        A container whose every field carries a leading load case axis.

    Returns
    -------
    count :
        Number of load cases.

    Notes
    -----
    Read from the first leaf, every field of a stacked container sharing the
    axis by construction. A static Python integer, so it may drive a loop
    inside a traced function.
    """
    leaves = jax.tree.leaves(stacked)

    return int(leaves[0].shape[0])
