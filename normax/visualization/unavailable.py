# SPDX-License-Identifier: Apache-2.0
"""
The viewer's entry points, on an install that cannot draw them.

Stand-ins matching the real signatures, so an install without the viewer's
packages imports and runs everything that does not ask to see a design.
"""

from typing import Any

from normax.analysis import AbstractFrameAnalyzer
from normax.config import RunConfig
from normax.design import Design
from normax.design import DesignRecord
from normax.loads import LoadCases
from normax.structures import Structure

# What is missing, and what installs it.
ABSENT = "the interactive viewer needs smax and vix, neither of them installed"


def view_designs(
    structure: Structure,
    analyzer: AbstractFrameAnalyzer,
    loads: LoadCases,
    designs: dict[str, Design],
    case_names: tuple[str, ...],
) -> None:
    """
    Decline to draw designs, the packages that draw them being absent.

    Parameters
    ----------
    structure :
        The structure a viewer would read the connectivity from.
    analyzer :
        The analysis block a viewer would read the responses from.
    loads :
        The checked load cases a viewer would solve.
    designs :
        The designs a viewer would draw.
    case_names :
        Name of every checked case.

    Raises
    ------
    ImportError
        Always, a request to draw being answerable no other way.
    """
    raise ImportError(ABSENT)


def view_design(record: DesignRecord, config: RunConfig[Any]) -> None:
    """
    Nothing when the run asks for no viewer, and an error when it asks.

    Parameters
    ----------
    record :
        What the run arrived at.
    config :
        The run config, whose output section says whether a viewer was
        asked for.

    Raises
    ------
    ImportError
        If the run asks for a viewer this install cannot open.

    Notes
    -----
    A run that never wanted a viewer is unaffected, so the packages are
    optional in fact and not only in name. One that wanted one is told, at the
    end of the run rather than at import, so the report and the record are
    already written.
    """
    if not config.output.viewer:
        return

    raise ImportError(ABSENT)
