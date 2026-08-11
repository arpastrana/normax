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

from normax.analysis.smax import prepare_model
from normax.ec3.material import SteelGrade
from normax.ec3.section import TubeCatalogue
from normax.formfinding import equilibrium_graph
from normax.pipeline import ProblemSetup
from normax.pipeline import total_mass
from normax.structures import arch_2d

steel = SteelGrade()
catalogue = TubeCatalogue.at_class_limit(steel.f_y, 3)

structure = arch_2d(num_edges=20, span=10_000.0, rise=3_000.0, load=9_474.0)

problem = ProblemSetup(
    structure,
    equilibrium_graph(structure),
    prepare_model(structure, steel, catalogue, normal=1),
    steel,
    catalogue,
)

seed = jnp.full(20, 100.0)


def total(q):
    return total_mass(
        q,
        seed,
        problem,
        section_class=catalogue.section_class(steel.f_y),
    )


q = jnp.full(20, -60.0)
print(total(q))  # tonnes of steel EN 1993-1-1 requires
print(jax.grad(total)(q))  # its gradient in the force densities
```

`ProblemSetup` is built once on the host and reused; only the force densities and
the analysed diameters enter the traced call. See `experiments/` for the arch
optimization, the two analysis backends measured against each other, and the same
pipeline composed across three Tesseracts.

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
