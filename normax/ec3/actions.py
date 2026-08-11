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
What a member carries, in the terms EN 1993-1-1 states its checks in.

A leaf. An analysis produces these and every check consumes them, so holding
them together is what keeps the two from disagreeing about how many there are.
"""

from typing import NamedTuple

from jaxtyping import Array
from jaxtyping import Float


class MemberActions(NamedTuple):
    """
    The design actions on one member.

    Attributes
    ----------
    n_ed :
        Axial force, tension positive.
    m_y_ed :
        Design bending moment about the major axis.
    m_z_ed :
        Design bending moment about the minor axis.
    c_my :
        Equivalent uniform moment factor for major-axis bending.
    c_mz :
        Equivalent uniform moment factor for minor-axis bending.

    Notes
    -----
    The reduction from two end moments to a design moment and a factor is
    EN 1993-1-1 Table B.3, so an analysis stops one step short of this and the
    step belongs to the check. `normax.ec3.sizing.end_moments` performs it.

    Sign conventions differ between the checks that read this, and none of them
    is applied here: the cross-section check of 6.2.9 reads the axial force with
    its sign, while the member check of 6.3.3 reads a compression magnitude and
    is switched off in tension. What is stored is what the analysis produced.

    Both moment factors default to one, the value they take under a uniform
    moment and the largest Table B.3 permits, so a member given no factor is
    checked conservatively rather than favourably.
    """

    n_ed: Float[Array, "members"]
    m_y_ed: float | Float[Array, "members"] = 0.0
    m_z_ed: float | Float[Array, "members"] = 0.0
    c_my: float | Float[Array, "members"] = 1.0
    c_mz: float | Float[Array, "members"] = 1.0
