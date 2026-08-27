# SPDX-License-Identifier: Apache-2.0
"""
Designs drawn in the frame solver's own terms.
"""

from typing import Any

import vix
from smax import LoadCase

from normax.analysis.smax import SmaxAnalyzer
from normax.analysis.smax import assemble_frame_model
from normax.config import RunConfig
from normax.design import Design
from normax.design import DesignRecord
from normax.loads import LoadCases
from normax.loads import label_load_cases
from normax.structures import Structure
from normax.tesseract import TesseractAnalyzer

# The internal-force diagrams drawn beside every response.
FORCE_DIAGRAMS = ("nx", "my")


def view_designs(
    structure: Structure,
    analyzer: SmaxAnalyzer | TesseractAnalyzer,
    loads: LoadCases,
    designs: dict[str, Design],
    case_names: tuple[str, ...],
) -> None:
    """
    Inspect designs interactively, in the frame solver's own terms.

    Parameters
    ----------
    structure :
        The structure supplying the connectivity and the supported nodes.
    analyzer :
        The analysis block, whose model builder and solve the viewer reads.
    loads :
        The checked load cases, each solved and drawn for every design.
    designs :
        The designs to draw, keyed by the name each appears under.
    case_names :
        Name of every checked case, naming its response in the viewer.

    Notes
    -----
    Each response is the solver's own, at the design's geometry and sections,
    so the diagrams are the analysis rather than a retelling. An analyzer that
    cannot report a whole response is stood in for by a traced one built at
    the same structure; the design itself is only redrawn. Every registration
    is named apart, since the viewer's `add` replaces a same-named one.
    Blocks until the window closes.
    """
    if isinstance(analyzer, TesseractAnalyzer):
        analyzer = SmaxAnalyzer(structure, analyzer.family(100.0))

    viewer = vix.Viewer(show_reactions=False)

    for name, design in designs.items():
        xyz = design.shape.xyz
        sections = design.sizes.sections
        frame = assemble_frame_model(structure, xyz, sections)
        viewer.add(frame, name=name)

        for index, case_name in enumerate(case_names):
            case_loads = loads.analysis[index]
            response = analyzer.solve_response(xyz, sections.diameter, case_loads)
            viewer.add(
                response,
                name=f"{name} — {case_name}",
                structure=name,
                show_deformation=False,
                show_forces=FORCE_DIAGRAMS,
            )

            loads_drawn = LoadCase.from_array(case_loads, frame)
            viewer.add(
                loads_drawn,
                name=f"{name} — {case_name} — loads",
                structure=name,
            )

    viewer.show()


def view_design(record: DesignRecord, config: RunConfig[Any]) -> None:
    """
    A run's start and answer in the viewer, or nothing when the run asks none.

    Parameters
    ----------
    record :
        What the run arrived at.
    config :
        The run config, naming its load cases and whether it ends in a
        viewer.

    Notes
    -----
    Blocks until the window closes, so a run reports and exports first.
    """
    if not config.output.viewer:
        return

    problem = record.problem
    designs = {"start": record.initial, "answer": record.optimized}
    labels = label_load_cases(config.load_cases)
    analyzer = problem.pipeline.analyzer
    view_designs(problem.structure, analyzer, problem.loads, designs, labels)
