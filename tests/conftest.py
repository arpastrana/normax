import importlib.util
from pathlib import Path

import jax

# Most of a run is XLA recompiling the same programs, so the cache is shared.
COMPILATION_CACHE = Path(__file__).resolve().parent.parent / ".jax_cache"
COMPILATION_CACHE.mkdir(exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(COMPILATION_CACHE))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

# TEMPORARY, until smax and ec3x are gone. Neither is on PyPI, both are pinned
# to a local path in the "local-dev" group, and CI installs "dev" alone. They
# are JAX-native, which is what makes them oracles rather than backends, and
# the plan is to delete them rather than publish them.
ORACLE_PACKAGES = ("smax", "ec3x")

# Tests importing an oracle, directly or through normax.viewer,
# normax.analysis.smax or normax.sizing.ec3. Omitting one turns CI red at
# collection rather than passing quietly.
ORACLE_TESTS = (
    "test_analysis_prepared.py",
    "test_blueprint_sizer.py",
    "test_design.py",
    "test_equilibrium_consistency.py",
    "test_extras_comparison.py",
    "test_extras_nested.py",
    "test_extras_replay.py",
    "test_materials.py",
    "test_pipeline.py",
    "test_second_sizer.py",
    "test_sections.py",
    "test_tesseract_parity.py",
)

# The two crossed analysis backends are tested against the traced oracle, so
# their tests sit under both guards.
OPENSEES_PACKAGES = ("openseespy",)

OPENSEES_TESTS = ("test_backend_opensees.py",)

# The import name is Pynite; the distribution is pynitefea.
PYNITE_PACKAGES = ("Pynite",)

PYNITE_TESTS = (
    "test_backend_pynite.py",
    "test_frame_convention.py",
)

# blue-prints (LGPL-2.1) is imported unmodified as a pip package.
BLUEPRINT_PACKAGES = ("blueprints",)

BLUEPRINT_TESTS = (
    "test_backend_opensees.py",
    "test_blueprint_sizer.py",
    "test_tesseract_parity.py",
)

collect_ignore = []
if any(importlib.util.find_spec(name) is None for name in ORACLE_PACKAGES):
    collect_ignore.extend(ORACLE_TESTS)
    collect_ignore.extend(OPENSEES_TESTS)
    collect_ignore.extend(PYNITE_TESTS)
if any(importlib.util.find_spec(name) is None for name in OPENSEES_PACKAGES):
    collect_ignore.extend(OPENSEES_TESTS)
if any(importlib.util.find_spec(name) is None for name in PYNITE_PACKAGES):
    collect_ignore.extend(PYNITE_TESTS)
if any(importlib.util.find_spec(name) is None for name in BLUEPRINT_PACKAGES):
    collect_ignore.extend(BLUEPRINT_TESTS)


def load_tesseract_api(name):
    p = Path(__file__).parent.parent / "tesseracts" / name / "tesseract_api.py"
    spec = importlib.util.spec_from_file_location(f"ta_{name}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
