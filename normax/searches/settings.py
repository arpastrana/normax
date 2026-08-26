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
The constants every search is measured and bounded by.
"""

from pathlib import Path

import jax

from normax.materials import Steel355
from normax.optimization import AugmentedBudget

CASE_NAMES = (
    "LC1 uniform deck",
    "LC2 half span",
    "LC3 half span mirrored",
    "LC4 midspan point",
)


# The shell's cases: a pressure, and a drift with its own mirror image, so the
# pair is jointly symmetric about the plane the design is folded by.
SHELL_NAMES = (
    "LC1 uniform pressure",
    "LC2 sector drift",
    "LC3 mirrored drift",
)


# Relative steps the central difference sweeps, and the worst scaled error the
# directional derivative may show at its plateau.
GRADIENT_STEPS = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)


TOLERANCE_GRADIENT = 1e-6


# Worst constraint violation an answer may show — SLSQP holds its constraints
# to its own ftol, measured orders below this headroom.
TOLERANCE_FEASIBILITY = 1e-6


# How exactly the signed lens densities live in the searched basis, and how
# exactly the full form-finding solve reproduces the drawn lens from them.
TOLERANCE_PROJECTION = 1e-9


TOLERANCE_SHAPE = 1e-8


# How exactly the start's density fit balances the lens, scaled by the load.
TOLERANCE_FIT = 1e-11


# How exactly one load case must reflect onto another before its rows are
# dropped as a reindexing, as a share of the largest force in any case.
TOLERANCE_MIRRORED = 1e-12


# A member is counted fully stressed above this envelope utilization, and
# counted at the floor within this distance of the bound.
ACTIVE_UTILIZATION = 0.999


FLOOR_SLACK = 1e-6


# Violation a trial point is charged when its frame cannot be factorized —
# enormous against the order-one slack rows, so the line search recoils.
RECOIL_SLACK = 1e3


METHOD_SLSQP = "slsqp"


METHOD_AUGMENTED = "augmented"


METHOD_ORDER = (METHOD_SLSQP, METHOD_AUGMENTED)


# What the polish after an augmented descent is allowed. A landing that is
# already stationary needs a handful of iterations; the cap is what stops a
# polish with real work left from quietly becoming the descent.
POLISH_ITERATIONS = 300


POLISH_ROUNDS = 2


# Violation above which an augmented landing is refused rather than polished. A
# constrained solver started from a grossly infeasible point does not repair
# it — it wanders, and reports where it wandered to.
POLISH_ADMISSION = 1e-3


# What an augmented descent runs on when a run description names the method and
# no budget. The opening penalty is the one setting that decides the answer
# rather than the speed, and no value is safe on every structure: 0.01 wins on
# the 16x16 cap and leaves a nine-member Vierendeel overstressed. The default
# is the conservative end, so a file wanting the fast one says so.
AUGMENTED_DEFAULT = AugmentedBudget(
    rounds=10,
    iterations=400,
    settled=100,
    opening=3,
    penalty=0.1,
    growth=10.0,
    ceiling=1.0e8,
    tolerance=1.0e-6,
    quiet=1.0e-6,
)


# Fixed, so a multi-start run is a measurement rather than a lottery.
SCATTER_SEED = 20260820


# Slack a scattered landing may sit at and still be counted feasible enough to
# compete; the run's own checks hold the winner to the real tolerance.
SCATTER_SLACK = -1e-6


# Growth passes a repair is allowed before it gives up on a landing. Capacity
# is strictly increasing in the diameter, so each pass moves the right way; a
# pass is needed at all only because a fatter member attracts more force.
REPAIR_PASSES = 8


# A member is governed by every case within this distance of its worst:
# mirror-paired cases tie exactly on self-mirrored members, and splitting a
# tie by index order would misreport a symmetric design as lopsided.
TIE_MARGIN = 1e-9


FIGURES = Path(__file__).resolve().parents[2] / "figures"


# Where a descent's answer is kept, so that looking at a design again is a
# read rather than a rerun.
DESIGNS = Path(__file__).resolve().parents[2] / "designs"


# Both searches compile a gradient and a Jacobian program; the persistent cache
# keeps reruns from paying the compilations again.
COMPILATION_CACHE = Path(__file__).resolve().parents[2] / ".jax_cache"
COMPILATION_CACHE.mkdir(exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(COMPILATION_CACHE))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)


# The fixture every pinned tolerance was measured at, so code rather than file.
GRADE = Steel355()


SECTION_CLASS = 3


SEARCH_FORMFOUND = "end to end"


SEARCH_HEIGHTS = "free heights"


SEARCH_DRAWN = "sizing only"


SEARCH_ORDER = (SEARCH_FORMFOUND, SEARCH_HEIGHTS, SEARCH_DRAWN)


# The truss is planar in XZ, so the axial force and the moment about y are
# the whole of what a member carries.
FORCE_DIAGRAMS = ("nx", "my")


# EN 1993-1-1 §6.1's recommended value, as every sizer in the repo states it.
GAMMA_M0_SHEAR = 1.0


def searches_present(keyed: dict[str, object]) -> tuple[str, ...]:
    """
    Which searches a keyed collection holds, in the shared order.

    Parameters
    ----------
    keyed :
        Anything keyed by search — maps, starts, answers or reads.

    Returns
    -------
    searches :
        The searches present, ordered as `SEARCH_ORDER` orders them.

    Notes
    -----
    Every table and every check reads its search list off the collection it is
    handed rather than off `SEARCH_ORDER`, which is what lets a solo run write
    the same report with one row in it. The order is still the shared one, so
    a full run's tables are unchanged to the character.
    """
    return tuple(search for search in SEARCH_ORDER if search in keyed)


# What the analysis stage may be answered by, and what a file gets unasked.
ANALYSIS_SMAX = "smax"


ANALYSIS_SMAX_CROSSED = "smax_tesseract"


ANALYSIS_OPENSEES = "opensees"


ANALYSIS_PYNITE = "pynite"


ANALYSIS_BACKENDS = (
    ANALYSIS_SMAX,
    ANALYSIS_SMAX_CROSSED,
    ANALYSIS_OPENSEES,
    ANALYSIS_PYNITE,
)


# Which of them reach the solver across a boundary rather than in process.
ANALYSIS_PLANAR = (ANALYSIS_OPENSEES,)


# What the check may be answered by. Blueprints is reachable only crossed.
SIZING_EC3 = "ec3"


SIZING_BLUEPRINT = "blueprint_tesseract"


SIZING_BACKENDS = (SIZING_EC3, SIZING_BLUEPRINT)
