import importlib.util
from pathlib import Path

# TEMPORARY, until smax is published. It is not on PyPI yet, so it and jax-fdm
# sit in the "pipeline" dependency group and CI installs "dev" alone. Delete
# this block and move both into the project dependencies once smax is public.
PIPELINE_PACKAGES = ("jax_fdm", "smax")

# A test importing either package, directly or through normax.formfinding,
# normax.analysis or normax.pipeline, belongs here. Omitting one turns CI red at
# collection rather than passing quietly, so this list fails loudly when stale.
PIPELINE_TESTS = ("test_equilibrium_consistency.py",)

collect_ignore = []
if any(importlib.util.find_spec(name) is None for name in PIPELINE_PACKAGES):
    collect_ignore.extend(PIPELINE_TESTS)


def load_tesseract_api(name):
    p = Path(__file__).parent.parent / "tesseracts" / name / "tesseract_api.py"
    spec = importlib.util.spec_from_file_location(f"ta_{name}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
