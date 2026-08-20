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
What a form-found shape is, and what any form finder must answer.

A form finder maps a parametrization to the geometry that carries a load case
without bending. This module says what that geometry is and what the contract
looks like; `normax.form_finding.fdm` holds the force density method as one
implementation, re-exported at the bottom so call sites import two levels deep.

**The contract is separated from the method for the same reason the analysis
stage separates its own.** A design composes blocks and has to name their types,
so a container sharing a module with a solver would drag that solver into
everything downstream — and into every container image that imports it.
"""

import abc
from typing import NamedTuple

import equinox as eqx
from jaxtyping import Array
from jaxtyping import Float


class FormFoundShape(NamedTuple):
    """
    The geometry a form finder settles on, and what its members measure there.

    Attributes
    ----------
    xyz :
        Position of every node at equilibrium.
    lengths :
        Length of every member.

    Notes
    -----
    **The handoff downstream is a geometry** — no prestress and no initial
    member forces. A frame analysis is given this, starts from an unstressed
    reference state, and finds its own axial forces; that those agree with the
    form finder's is a prediction that gets tested rather than an input that
    gets imposed. The form finder's own forces are absent for that reason, and
    they cost nothing to recover: an edge carries the product of its force
    density and its length.

    **The lengths are here because measuring needs the connectivity**, which a
    form finder holds and nothing downstream does. They are read as the length a
    member buckles over and as the `L` of `ρ Σ A L`, and a length is geometry
    rather than any stage's opinion, so a block reporting one is reporting what
    it measured rather than what it decided.
    """

    xyz: Float[Array, "nodes 3"]
    lengths: Float[Array, "members"]


class AbstractFormFinder(eqx.Module):
    """
    A parametrization of the shapes a structure may take in equilibrium.

    Notes
    -----
    Maps force densities and a load case to a geometry that carries that case
    without bending. Concrete form finders differ in which quantities they treat
    as independent, not in the mechanics they encode, which is why they share
    one shape and one call signature.

    Built from the structure it is to shape, and from nothing else that varies.
    """

    @abc.abstractmethod
    def __call__(
        self,
        q: Float[Array, "members"],
        loads: Float[Array, "nodes 3"],
    ) -> FormFoundShape:
        """
        Find the shape that carries a load case at given force densities.

        Parameters
        ----------
        q :
            Force density of every member. Negative in compression.
        loads :
            Force applied at every node.

        Returns
        -------
        shape :
            The geometry at equilibrium, and its member lengths.
        """


# The force density method, re-exported so call sites import two levels deep.
from normax.form_finding.fdm import DensityFit
from normax.form_finding.fdm import FdmFormFinder
from normax.form_finding.fdm import PivotedBasis
from normax.form_finding.fdm import SubspaceFormFinder
from normax.form_finding.fdm import density_basis
from normax.form_finding.fdm import equilibrium_gap
from normax.form_finding.fdm import equilibrium_graph
from normax.form_finding.fdm import equilibrium_state
from normax.form_finding.fdm import fit_densities
from normax.form_finding.fdm import pivoted_basis
from normax.form_finding.fdm import plan_equilibrium
from normax.form_finding.fdm import positions_vertical
