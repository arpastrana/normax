# SPDX-License-Identifier: Apache-2.0
"""
Backpropagating through the building code.
"""

from pathlib import Path

import jax

# Most of a run is XLA recompiling the same programs, so the cache is shared.
COMPILATION_CACHE = Path(__file__).resolve().parent.parent / ".jax_cache"

jax.config.update("jax_enable_x64", True)
try:
    COMPILATION_CACHE.mkdir(exist_ok=True)
except OSError:
    pass
else:
    jax.config.update("jax_compilation_cache_dir", str(COMPILATION_CACHE))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
