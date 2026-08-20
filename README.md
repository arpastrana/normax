# normax

[![Tests](https://github.com/arpastrana/normax/actions/workflows/test.yml/badge.svg)](https://github.com/arpastrana/normax/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/arpastrana/normax/blob/main/LICENSE)

Backpropagating through structural engineering codes

Force densities to a funicular shape, a frame analysis to member forces, and
EN 1993-1-1 to the sections it requires — composed into one function with exact
gradients throughout. The building code is a normative text rather than a solver:
it has no derivatives of its own, and giving it one is what lets it sit in an
optimization loop beside an autodiff form-finder. The three blocks differentiate
three different ways — implicit-function-theorem rules on the form-finding and
sizing solves, a frame analysis swappable between JAX tracing and OpenSees'
analytic sensitivities, and a hand-derived piecewise adjoint on the code check —
and one composed function is what the optimizer sees.

## Installation

Not published. Clone the repository and install with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/arpastrana/normax
cd normax
uv sync --group dev --group pipeline
```

The `pipeline` group carries the form finder and the frame solver, the `viz`
group the interactive viewer, and the `spike` extra `openseespy` for the second
analysis backend.

## Usage

```python
import jax
import jax.numpy as jnp

from normax.analysis import SmaxAnalyzer
from normax.design import DesignParameters
from normax.design import StructuralDesignPipeline
from normax.design import compute_mass
from normax.form_finding import FdmFormFinder
from normax.loads import create_loads_uniform
from normax.materials import Steel355
from normax.sizing import Ec3Sizer
from normax.sizing import build_section_family
from normax.structures import build_arch_2d


num_edges = 20
section_class = 3
diameter_start = 100.0
force_density_start = -50.0

material = Steel355()
section_family = build_section_family(material, section_class)
diameters = jnp.full(num_edges, diameter_start)
force_densities = jnp.full(num_edges, force_density_start)
tubes = section_family(diameters)

structure = build_arch_2d(num_edges, span=10_000.0, rise=3_000.0)
loads_uniform = create_loads_uniform(structure, 15_000.0)

pipeline = StructuralDesignPipeline(
    FdmFormFinder(structure),
    SmaxAnalyzer(structure, tubes),
    Ec3Sizer(structure, section_family),
)

def code_mass(force_densities):
    parameters = DesignParameters(force_densities, diameters)
    design = pipeline(parameters, loads_uniform)
    return compute_mass(design)

print(code_mass(force_densities))  # tonnes of steel EN 1993-1-1 requires
print(jax.grad(code_mass)(force_densities))  # its gradient in the force densities
```

The pipeline is three swappable blocks, each built from a structure on the
host and then called; what is left is a function of design parameters and load
cases, and that is what an optimizer differentiates. `normax.tesseract` holds
the same three blocks reached across a Tesseract boundary, and the pipeline
cannot tell the difference. The check also runs the other way around: `Ec3Sizer.compute_utilization`
has exactly a constraint function's signature, so a constrained optimizer can
hold `utilization <= 1` with analytic Jacobians while shape and sections move together.
See `experiments/101_api.py` for the whole API in one file,
`experiments/103_simultaneous_api.py` for the constrained formulation, and
`experiments/04_backend_agreement.py` for the two analysis backends against
each other.

## What the gradient buys

`normax/structures.py` generates the structures the claims are measured on — a
funicular arch, a Warren truss, a Vierendeel truss, and a gridshell cap — and
the experiments race parametrizations against each other on them, so what the
gradient buys is a comparison rather than a demo. A form finder acts as a
shape prior: descending one force density through it is start-proof where
descending every free node height stalls in bending. On a truss, holding the
plan leaves a null space of force densities to search, and
`SubspaceFormFinder` makes its basis coordinates the design variables.
`experiments/truss_routes.py` then races three routes over the same members,
loads and check — the whole pipeline end to end, free heights without the form
finder, and sizing alone at the drawn geometry — and moving the geometry buys
the larger share of the mass on both trusses. Numbers, tolerances and the full
protocol are in the accompanying paper and in `CHANGELOG.md`; every experiment
reads its settings from the YAML beside it and reproduces headless:

```bash
uv run --group pipeline --group viz python experiments/18_warren_optimize.py
```

## Limitations

**The nested route's gradient omits `∂d/∂q`.** In the fully-stressed
formulation the diameters the analysis runs at stay at their seed while the
force densities move, so the feedback from a chosen size back into the forces
that chose it is a path the reverse pass never enters. The cost is measured,
and two closures exist: staggered re-sectioning, and the simultaneous
formulation above, which makes the design self-consistent by construction.

**Shear and torsion are not designed for, and the exclusion is measured rather
than assumed.** The check covers axial force with bending and leaves out
EN 1993-1-1 §6.2.6–6.2.8, as clause 6.2.10 permits while the design shear
stays under half the plastic shear resistance. `experiments/20_shear_audit.py`
reads that fraction off every converged design and no structure here
approaches the threshold; `docs/shear_design.md` records what designing for
shear would take.

**No lateral-torsional buckling check, by construction.** Every member is a
circular hollow section, which is doubly symmetric, so lateral-torsional
buckling does not occur and §6.3.2 is not implemented.

**Self-weight does not feed back into the loads.** The load cases are stated
once and never re-assembled from the sections a design chose.

**No buildability constraint.** Diameters are continuous and per member; the
continuous optimum is a lower bound on any catalog design.

**Global stability is not checked.** Every member is verified over its own
buckling length; nothing computes a critical load factor of the whole frame
per §5.2. Frame-stability checks are future work.

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
