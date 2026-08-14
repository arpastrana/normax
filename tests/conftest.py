import importlib.util
from pathlib import Path

# TEMPORARY, until smax is published. It is not on PyPI yet, so it and jax-fdm
# sit in the "pipeline" dependency group and CI installs "dev" alone. Delete
# this block and move both into the project dependencies once smax is public.
PIPELINE_PACKAGES = ("jax_fdm", "smax")

# A test importing either package, directly or through normax.form_finding,
# normax.analysis or normax.design, belongs here. Omitting one turns CI red at
# collection rather than passing quietly, so this list fails loudly when stale.
PIPELINE_TESTS = (
    "test_analysis_prepared.py",
    "test_equilibrium_consistency.py",
    "test_pipeline.py",
    "test_design.py",
    "test_tesseract_parity.py",
)

# openseespy is the "spike" optional extra and CI never installs it, so the
# second analysis backend is skipped wherever it is absent. It needs smax too,
# being tested against it, and so is listed under both guards.
OPENSEES_PACKAGES = ("openseespy",)

OPENSEES_TESTS = ("test_backend_opensees.py",)

collect_ignore = []
if any(importlib.util.find_spec(name) is None for name in PIPELINE_PACKAGES):
    collect_ignore.extend(PIPELINE_TESTS)
    collect_ignore.extend(OPENSEES_TESTS)
if any(importlib.util.find_spec(name) is None for name in OPENSEES_PACKAGES):
    collect_ignore.extend(OPENSEES_TESTS)


def load_tesseract_api(name):
    p = Path(__file__).parent.parent / "tesseracts" / name / "tesseract_api.py"
    spec = importlib.util.spec_from_file_location(f"ta_{name}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
