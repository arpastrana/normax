# SPDX-License-Identifier: Apache-2.0
"""
What a run leaves behind on disk.

`records` writes a finished run — its record and its figures — and is what
an example calls. `replay` reads a recorded nested search back, turning the
force densities it walked through into the designs they stood for.
"""

from normax.exporting.records import ExportTarget
from normax.exporting.records import export_design

__all__ = [
    "ExportTarget",
    "export_design",
]
