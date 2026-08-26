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
The three constrained searches to a design, shared across the examples.

What a run needs is re-exported here so a caller imports two levels deep. The
viewer is deliberately absent: it lives in `normax.viewing`, needs the optional
`viz` group, and nothing in this package imports it.
"""

from normax.searches.config import TaskConfig
from normax.searches.config import augmented_budget
from normax.searches.config import parse_shell
from normax.searches.config import parse_truss
from normax.searches.descent import descend_all
from normax.searches.descent import descend_augmented_search
from normax.searches.descent import descent_plan
from normax.searches.descent import scattered_points
from normax.searches.driver import StructureProfile
from normax.searches.driver import run_searches
from normax.searches.folding import ChordSigns
from normax.searches.folding import folding_maps
from normax.searches.folding import lens_geometry
from normax.searches.folding import signed_shift
from normax.searches.loads import shell_loads
from normax.searches.loads import truss_loads
from normax.searches.maps import HeightTruss
from normax.searches.maps import search_boxes
from normax.searches.maps import search_maps
from normax.searches.maps import search_starts
from normax.searches.maps import shell_heights
from normax.searches.maps import truss_heights
from normax.searches.problem import DesignProblem
from normax.searches.problem import StartPoint
from normax.searches.problem import ViewRequest
from normax.searches.problem import prepare_problem
from normax.searches.reporting import read_answer
from normax.searches.reporting import search_reads
from normax.searches.reporting import shell_extent
from normax.searches.reporting import truss_extent
from normax.searches.settings import AUGMENTED_DEFAULT
from normax.searches.settings import FIGURES
from normax.searches.settings import FORCE_DIAGRAMS
from normax.searches.settings import POLISH_ADMISSION
from normax.searches.settings import POLISH_ITERATIONS
from normax.searches.settings import SEARCH_DRAWN
from normax.searches.settings import SEARCH_FORMFOUND
from normax.searches.settings import SEARCH_HEIGHTS
from normax.searches.settings import SEARCH_ORDER

__all__ = [
    "AUGMENTED_DEFAULT",
    "ChordSigns",
    "DesignProblem",
    "FIGURES",
    "FORCE_DIAGRAMS",
    "HeightTruss",
    "POLISH_ADMISSION",
    "POLISH_ITERATIONS",
    "SEARCH_DRAWN",
    "SEARCH_FORMFOUND",
    "SEARCH_HEIGHTS",
    "SEARCH_ORDER",
    "StartPoint",
    "StructureProfile",
    "TaskConfig",
    "ViewRequest",
    "augmented_budget",
    "descend_all",
    "descend_augmented_search",
    "descent_plan",
    "folding_maps",
    "lens_geometry",
    "parse_shell",
    "parse_truss",
    "prepare_problem",
    "read_answer",
    "run_searches",
    "scattered_points",
    "search_boxes",
    "search_maps",
    "search_reads",
    "search_starts",
    "shell_extent",
    "shell_heights",
    "shell_loads",
    "signed_shift",
    "truss_extent",
    "truss_heights",
    "truss_loads",
]
