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
from normax.design import DesignParameters
from normax.design import StructuralDesignPipeline
from normax.design import compute_mass
from normax.design import design_envelope
from normax.ec3.material import Steel
from normax.ec3.section import TubeCatalogue
from normax.form_finding.fdm import FdmFormFinder
from normax.loads import assemble_load_cases
from normax.loads import loads_uniform
from normax.sizing import Ec3Sizer
from normax.structures import build_arch_2d

steel = Steel()
catalogue = TubeCatalogue.at_class_limit(steel, 3)

structure = build_arch_2d(num_edges=20, span=10_000.0, rise=3_000.0)
uniform = loads_uniform(structure, 9_474.0)
loads = assemble_load_cases([uniform])

pipeline = StructuralDesignPipeline(
    FdmFormFinder(structure),
    SmaxAnalyzer(structure, catalogue, normal=1),
    Ec3Sizer(structure, catalogue),
)

seed = jnp.full(20, 100.0)


def total(q):
    design = pipeline(DesignParameters(q, seed), loads)

    return compute_mass(design_envelope(design))


q = jnp.full(20, -60.0)
print(total(q))  # tonnes of steel EN 1993-1-1 requires
print(jax.grad(total)(q))  # its gradient in the force densities
```

**The pipeline is three swappable blocks.** Each one is built from a structure on
the host and then called, and the split is where every piece of software gets to
see the structure in its own terms — a form finder wants connectivity matrices, a
frame solver wants an assembly and degree of freedom maps, a code check wants
nothing at all. What is left is a function of design parameters and load cases,
and that is what an optimizer differentiates.

**A section family is one argument, and it carries its own grade.** A catalogue's
ratio is a class limit read at a yield strength, so the grade and the class are the
family's identity rather than companions handed in beside it; calling the catalogue
at a diameter generates the tube, which carries both onward. Nothing downstream can
pair a wall with the class of a different one.

Swapping a block is a constructor argument. `normax.tesseract` holds the same three
reached across a Tesseract boundary, and `StructuralDesignPipeline` cannot tell the
difference — `tests/test_tesseract_parity.py` runs one pipeline over both sets and
measures it. See `experiments/` for the arch optimization, the two analysis
backends against each other, and `experiments/101_api.py` for the whole API in one
file.

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
