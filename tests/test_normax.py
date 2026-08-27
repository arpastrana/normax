# SPDX-License-Identifier: Apache-2.0
import jax.numpy as jnp

import normax  # noqa: F401


def test_x64_is_enabled():
    assert jnp.zeros(1).dtype == jnp.float64
