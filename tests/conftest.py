import importlib.util
from pathlib import Path

import jax

# Most of a run is XLA recompiling the same programs; see CHANGELOG for the cost.
COMPILATION_CACHE = Path(__file__).resolve().parent.parent / ".jax_cache"
COMPILATION_CACHE.mkdir(exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(COMPILATION_CACHE))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

# TEMPORARY, until smax and ec3x are published. None of the three is on PyPI
# yet, so all sit in the "pipeline" dependency group and CI installs "dev"
# alone. Delete this block and move them into the project dependencies once
# they are public.
PIPELINE_PACKAGES = ("jax_fdm", "smax", "ec3x")

# A test importing any of those packages, directly or through
# normax.form_finding, normax.analysis, normax.sizing.ec3 or the backends,
# belongs here. Omitting one turns CI red at collection rather than passing
# quietly, so this list fails loudly when stale.
PIPELINE_TESTS = (
    "test_analysis_prepared.py",
    "test_equilibrium_consistency.py",
    "test_pipeline.py",
    "test_design.py",
    "test_replay.py",
    "test_second_sizer.py",
    "test_tesseract_parity.py",
)

# These two compare normax's neutral containers against ec3x's, so they need
# ec3x alone — an environment with ec3x but no frame solver still runs them.
EC3X_PACKAGES = ("ec3x",)

EC3X_TESTS = (
    "test_materials.py",
    "test_sections.py",
)

# openseespy is the "spike" optional extra and CI never installs it, so the
# second analysis backend is skipped wherever it is absent. It needs smax too,
# being tested against it, and so is listed under both guards.
OPENSEES_PACKAGES = ("openseespy",)

OPENSEES_TESTS = ("test_backend_opensees.py",)

# blue-prints (LGPL-2.1, experiment-only) is imported by normax.tesseract while
# the blueprint sizer is prototyped, so every test importing that module — or
# the sizer itself — is skipped without it, and no other pipeline test is.
BLUEPRINT_PACKAGES = ("blueprints",)

BLUEPRINT_TESTS = (
    "test_backend_opensees.py",
    "test_blueprint_sizer.py",
    "test_tesseract_parity.py",
)

collect_ignore = []
if any(importlib.util.find_spec(name) is None for name in PIPELINE_PACKAGES):
    collect_ignore.extend(PIPELINE_TESTS)
    collect_ignore.extend(OPENSEES_TESTS)
    collect_ignore.extend(BLUEPRINT_TESTS)
if any(importlib.util.find_spec(name) is None for name in OPENSEES_PACKAGES):
    collect_ignore.extend(OPENSEES_TESTS)
if any(importlib.util.find_spec(name) is None for name in EC3X_PACKAGES):
    collect_ignore.extend(EC3X_TESTS)
if any(importlib.util.find_spec(name) is None for name in BLUEPRINT_PACKAGES):
    collect_ignore.extend(BLUEPRINT_TESTS)


def load_tesseract_api(name):
    p = Path(__file__).parent.parent / "tesseracts" / name / "tesseract_api.py"
    spec = importlib.util.spec_from_file_location(f"ta_{name}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
