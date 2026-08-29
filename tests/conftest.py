# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path

# Nothing here is gated any more. Until 2026-08-28 this file skipped a test file
# whose imports reached `smax` or `ec3x` — two private JAX oracles pinned to a
# local path, absent from CI — which meant CI ran 126 of 395 tests. The oracles
# are gone (docs/oracle_removal.md); every package the suite now needs is a main
# dependency, so a clean `uv sync` runs the whole thing, without Docker.


def load_tesseract_api(name):
    p = Path(__file__).parent.parent / "tesseracts" / name / "tesseract_api.py"
    spec = importlib.util.spec_from_file_location(f"ta_{name}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
