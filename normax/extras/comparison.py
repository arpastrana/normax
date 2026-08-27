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
Form finders that make the two comparison searches a swapped block.

The end-to-end search moves force densities through the force density method.
Its two foils move something else through the same pipeline: free heights,
which write the nodes' z directly and are not funicular, and nothing at all,
which sizes the drawn geometry as it stands. Both are form finders here, so a
`DesignProblem` over either differs from the headline in one block and an
identity basis of the right width.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from normax.form_finding import AbstractFormFinder
from normax.form_finding import FormFoundShape
from normax.form_finding import PlanBasis
from normax.form_finding import select_free_nodes
from normax.structures import Structure
from normax.structures import compute_member_lengths


def identity_basis(width: int) -> PlanBasis:
    """
    The basis a `DesignProblem` over one of these finders is built with.

    Parameters
    ----------
    width :
        How many coordinates the finder is called with.

    Returns
    -------
    basis :
        Identity columns, so the coordinates expand to themselves and the
        problem reads its width off the basis.
    """
    return PlanBasis(np.eye(width), None)


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
