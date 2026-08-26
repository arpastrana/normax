# Copyright 2026 Rafael Pastrana
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Smoke test of the Tesseract toolchain: Docker, tesseract-core, tesseract-jax.

Not a pytest test and not part of the suite. Run it by hand when the toolchain
might have moved, and especially before the composition work begins — a broken
Docker setup is cheap to fix now and expensive to discover later.

Requires the upstream example image to have been built first:

    docker run --rm hello-world
    git clone https://github.com/pasteurlabs/tesseract-jax /tmp/tj
    uv run tesseract build /tmp/tj/examples/simple/vectoradd_jax
    uv run python experiments/00_toolchain_smoke.py

The upstream example returns several fields; the one to check is
`vector_add.result`, the elementwise sum [5.0, 7.0, 9.0]. Anything else means the
toolchain is broken rather than this package.

**The forward pass is the less important half.** What P3 depends on is `jax.grad`
crossing the boundary, so this also differentiates through the Tesseract and
under `jit`. A toolchain that applies but will not differentiate looks healthy
right up to the moment it matters.

Note the upstream image is float32, so its gradient comes back float32. That is a
property of the example, not of this toolchain — our own Tesseracts declare
Float64 (invariant 3).
"""

import jax
import jax.numpy as jnp
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

from normax.reporting import Report

A = jnp.array([1.0, 2.0, 3.0])
B = jnp.array([4.0, 5.0, 6.0])

IMAGE = "vectoradd_jax"


def main(verbose: bool = True) -> None:
    """
    Apply the upstream vector-addition Tesseract, then differentiate through it.
    """
    report = Report(verbose)

    with Tesseract.from_image(IMAGE, user="root") as tesseract:

        def total(a):
            inputs = {"a": {"v": a}, "b": {"v": B}}
            out = apply_tesseract(tesseract, inputs)

            return jnp.sum(out["vector_add"]["result"])

        inputs = {"a": {"v": A}, "b": {"v": B}}
        applied = apply_tesseract(tesseract, inputs)
        gradient = jax.grad(total)(A)
        compiled = jax.jit(jax.grad(total))(A)

        entries = (
            ("apply", f"{applied}"),
            ("sum", f"{total(A)}, expected 21.0"),
            ("grad", f"{gradient}, expected [1. 1. 1.]"),
            ("jit(grad)", f"{compiled}, expected [1. 1. 1.]"),
        )

        report.write_heading(f"The upstream {IMAGE} image, applied and differentiated")
        report.write_entries(entries)


if __name__ == "__main__":
    main()
