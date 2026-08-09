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

Expected output is the elementwise sum, [5.0, 7.0, 9.0]. Anything else means
the toolchain is broken rather than this package.
"""

import jax.numpy as jnp
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract


def main() -> None:
    """
    Apply the upstream vector-addition Tesseract and print the result.
    """
    inputs = {
        "a": {"v": jnp.array([1.0, 2.0, 3.0])},
        "b": {"v": jnp.array([4.0, 5.0, 6.0])},
    }

    with Tesseract.from_image("vectoradd_jax", user="root") as tesseract:
        print(apply_tesseract(tesseract, inputs))


if __name__ == "__main__":
    main()
