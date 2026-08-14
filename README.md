# normax

[![Tests](https://github.com/arpastrana/normax/actions/workflows/test.yml/badge.svg)](https://github.com/arpastrana/normax/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/arpastrana/normax/blob/main/LICENSE)

Backpropagating through structural engineering codes

Force densities to a funicular shape, a frame analysis to member forces, and
EN 1993-1-1 to the sections it requires — composed into one function with exact
gradients throughout. The building code is a normative text rather than a solver:
it has no derivatives of its own, and giving it one is what lets it sit in an
optimization loop beside an autodiff form-finder.

## Installation

Not published. Clone the repository and install with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/arpastrana/normax
cd normax
uv sync --group dev --group pipeline
```

The `pipeline` group carries `jax-fdm`, `smax` and `matplotlib`, which the
composed pipeline and the experiments need. The `spike` extra carries
`openseespy` for the second analysis backend, and CI never installs it:

```bash
uv sync --extra spike
```

## Usage

```python
import jax
import jax.numpy as jnp

from normax.analysis.smax import SmaxAnalyzer
from normax.ec3.material import SteelGrade
from normax.ec3.section import TubeCatalogue
from normax.form_finding import FdmFormFinder
from normax.pipeline import DesignPipeline
from normax.pipeline import calculate_mass
from normax.sizing import Ec3Sizer
from normax.design import DesignParameters
from normax.design import load_cases
from normax.structures import arch_2d
from normax.structures import loads_uniform

steel = SteelGrade()
catalogue = TubeCatalogue.at_class_limit(steel.f_y, 3)

structure = arch_2d(num_edges=20, span=10_000.0, rise=3_000.0)
uniform = loads_uniform(structure, 9_474.0)
loads = load_cases(uniform, [uniform])

pipeline = DesignPipeline(
    FdmFormFinder(),
    SmaxAnalyzer(steel, catalogue, normal=1),
    Ec3Sizer(steel, catalogue),
).compile(structure)

seed = jnp.full(20, 100.0)


def total(q):
    return calculate_mass(pipeline(DesignParameters(q, seed), loads))


q = jnp.full(20, -60.0)
print(total(q))  # tonnes of steel EN 1993-1-1 requires
print(jax.grad(total)(q))  # its gradient in the force densities
```

**The pipeline is three swappable blocks.** Each one is configured, compiled
against a structure on the host, and then called: `compile` is where every piece
of software gets to see the structure in its own terms — a form finder wants
connectivity matrices, a frame solver wants an assembly and degree of freedom
maps, a code check wants nothing at all. What is left is a function of design
parameters and load cases, and that is what an optimizer differentiates.

Swapping a block is a constructor argument. `normax.tesseract` holds the same
three reached across a Tesseract boundary, and `DesignPipeline` cannot tell the
difference — `tests/test_tesseract_parity.py` runs one pipeline over both sets
and measures it. See `experiments/` for the arch optimization, the two analysis
backends against each other, and `experiments/101_api.py` for the whole API in
one file.

## Development

```bash
uv sync --group dev --group pipeline
uv run pytest
```

Add dependencies with `uv add` or `uv add --dev` rather than by editing
`pyproject.toml`. Install the formatting hooks before the first commit; their
pinned ruff is what CI uses:

```bash
uv run pre-commit install
```
