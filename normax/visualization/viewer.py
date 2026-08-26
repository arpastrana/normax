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
Answers drawn in the frame solver's own terms.
"""

import jax.numpy as jnp
import vix
from smax import LoadCase

from normax.analysis import AbstractFrameAnalyzer
from normax.analysis import SmaxAnalyzer
from normax.analysis import frame_model
from normax.searches.problem import DesignProblem
from normax.searches.problem import ViewRequest
from normax.searches.settings import FORCE_DIAGRAMS
from normax.sections import MemberSections


def response_analyzer(
    problem: DesignProblem,
    sections: MemberSections,
) -> AbstractFrameAnalyzer:
    """
    The analyzer a drawing reads its response from.

    Parameters
    ----------
    problem :
        The prepared structure and its pipeline.
    sections :
        The tubes the design landed on.

    Returns
    -------
    analyzer :
        The stage's own analyzer where it can report a full response, and a
        traced one built for the occasion where it cannot.

    Notes
    -----
    **A drawing wants more than the schema carries.** Deformation and force
    diagrams are read off a whole response field, and the analysis schema serves
    member end forces alone — so no crossed backend can answer this, and asking
    one is how the viewer used to fail. Keyed on whether a response can be had
    rather than on which backend it is, the same way the shear fallback is, so a
    backend that grows the ability stops needing the substitute.

    **The substitute is sound because the design is not the drawing's.** The
    geometry and the diameters were decided by whatever pipeline the run
    described; this only recomputes a response at that finished design, and the
    backends agree there to a part in a million million.
    """
    analyzer = problem.pipeline.analyzer
    if hasattr(analyzer, "solve_response"):
        return analyzer

    return SmaxAnalyzer(problem.structure, sections)


def view_answers(request: ViewRequest) -> None:
    """
    Open named answers in a viewer, in the frame solver's own terms.

    Parameters
    ----------
    request :
        What to draw: the prepared structure, each search's answer read back,
        which searches to show, and the run's viewer section.

    Notes
    -----
    **A response per case is a response per solve.** Drawing one case is the
    difference between opening a scene and assembling the frame again for
    every condition it was checked against, and a reader comparing shapes
    rarely wants more than the one the shape was found under.

    The sections are the answer's own diameters walled by the family every
    block was built on, not a re-sizing: an envelope would replace them with
    the sizer's demand and draw a structure the report never mentioned.

    Each response comes from `SmaxAnalyzer.solve_response`, the same injected
    assembly and solve the member forces were read from, so the diagrams are
    the analysis rather than a retelling.

    Each response carries its displaced shape at true scale, which on a stiff
    truss is a shape a slider has to open up rather than one that reads off the
    screen unaided.

    Every registration is named apart. A viewer's `add` replaces a same-named
    one, so a loads group sharing its response's name would tear that response
    down instead of joining it. Each response also names its parent outright,
    which is required rather than tidy: the searches share a scene, so the frame
    a response belongs to is ambiguous otherwise.

    The caller names the searches, one or several. Several share a scene and
    nearly a location, so they are told apart by switching frames off from the
    panel rather than by looking.

    Support reactions are asked for by name and refused, so a design is read
    by what its members carry rather than by what its supports push back. The
    viewer registers the glyphs whatever it is told and only draws them when
    asked, so this pins the answer rather than skipping the work.

    Blocks until the window closes.
    """
    problem = request.problem
    reads = request.reads
    searches = request.searches
    viewer_config = request.viewer_config

    viewer = vix.Viewer(show_reactions=False)
    load_case = viewer_config.load_case

    for search in searches:
        read = reads[search]
        xyz = jnp.asarray(read.xyz)
        sections = problem.pipeline.sizer.family(jnp.asarray(read.diameters))
        frame = frame_model(problem.structure, xyz, sections)
        drawn = response_analyzer(problem, sections)
        viewer.add(frame, name=search)

        for index, case_name in enumerate(problem.case_names):
            if not case_name.startswith(load_case):
                continue
            case_loads = problem.loads.analysis[index]
            response = drawn.solve_response(
                xyz,
                sections.diameter,
                case_loads,
            )
            viewer.add(
                response,
                name=f"{search} — {case_name}",
                structure=search,
                show_deformation=True,
                show_forces=FORCE_DIAGRAMS,
            )

            loads_drawn = LoadCase.from_array(case_loads, frame)
            viewer.add(
                loads_drawn,
                name=f"{search} — {case_name} — loads at {viewer_config.load_scale:g}x",
                structure=search,
                load_scale=viewer_config.load_scale,
            )

    viewer.show()
