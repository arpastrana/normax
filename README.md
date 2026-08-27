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
uv sync
```

Everything the pipeline needs — the form finder, both host frame solvers, the
Blueprints check, the viewer — is a regular dependency. The `local-dev` group
pins the path-installed oracle packages the parity tests compare against;
without it those tests skip themselves and the rest of the suite runs.

## Usage

```python
import jax
import jax.numpy as jnp

from normax.config import AnalysisConfig
from normax.config import FormFindingConfig
from normax.config import SizingConfig
from normax.design import DesignParameters
from normax.design import compute_mass
from normax.form_finding import UniformDensityInitializer
from normax.loads import assemble_load_cases
from normax.loads import load_uniform
from normax.materials import Steel355
from normax.sections import build_section_family
from normax.structures import build_arch_2d
from normax.tesseract import build_pipeline

num_edges = 20
section_class = 3
diameter_start = 100.0
force_density_start = -50.0

structure = build_arch_2d(num_edges, span=10_000.0, rise=3_000.0)
case_uniform = load_uniform(structure, 15_000.0)
loads = assemble_load_cases([case_uniform])

family = build_section_family(Steel355(), section_class)
initializer = UniformDensityInitializer(force_density_start)
form_finding = FormFindingConfig(None, None, initializer)
analysis = AnalysisConfig(diameter_start, "opensees")
sizing = SizingConfig(section_class, "blueprint", False, False)
pipeline = build_pipeline(structure, family, form_finding, analysis, sizing)

diameters = jnp.full(num_edges, diameter_start)

def code_mass(force_densities):
    parameters = DesignParameters(force_densities, diameters)
    design = pipeline(parameters, loads)
    return compute_mass(design)

force_densities = jnp.full(num_edges, force_density_start)
print(code_mass(force_densities))  # tonnes of steel at the given sections
print(jax.grad(code_mass)(force_densities))  # its gradient in the force densities
```

The pipeline is three swappable blocks, each built from a structure on the
host and then called; what is left is a function of design parameters and load
cases, and that is what an optimizer differentiates. Form finding traces a
linear solve in this process. The frame analysis and the code check each cross
a Tesseract boundary to a host that does not differentiate itself — OpenSees
for a planar frame, PyNite for a space frame, Blueprints for the check — and
come back with a hand-written adjoint; swapping a block for one that runs in
process is a different word in the config and nothing else. The design that
comes back also carries `design.sizes.utilization`, how hard EN 1993-1-1 works
every member under every load case, which is exactly a constraint function —
the augmented Lagrangian in `normax.design.optimize_design` holds
`utilization <= 1` with it while shape and sections move together.
`examples/arch.py` is the whole project in one file; the other three examples
share its shape and add a held-plan subspace, symmetry folding and sign
guards:

```bash
uv run python examples/arch.py
uv run python examples/warren.py
uv run python examples/vierendeel.py
uv run python examples/gridshell.py
```

Each reads the YAML beside it, prints what the descent bought, and writes its
figures and a `data/*.npz` record; the file's `output` block turns the report,
the export and the viewer on and off.

## What the gradient buys

`normax/structures.py` generates the structures the claims are measured on — a
funicular arch, a Warren truss, a Vierendeel truss, and a gridshell cap — and
the experiments race parametrizations against each other on them, so what the
gradient buys is a comparison rather than a demo. A form finder acts as a
shape prior: descending one force density through it is start-proof where
descending every free node height stalls in bending. On a truss, holding the
plan leaves a null space of force densities to search, and
`normax.form_finding.build_plan_basis` makes its basis coordinates the design
variables. The rival routes — free heights without the form finder, and sizing
alone at the drawn geometry — are form finders too, in
`normax/extras/comparison.py`, so racing them swaps one block and nothing
else; moving the geometry buys the larger share of the mass on both trusses.
Numbers, tolerances and the full protocol are in the accompanying paper and in
`CHANGELOG.md`.

## Limitations

**The nested fully-stressed route, kept in `normax/extras/`, omits `∂d/∂q`.**
The shipped search is simultaneous — diameters are variables beside the force
densities, so the design is self-consistent by construction and the gradient
is complete. In the nested add-on the diameters the analysis runs at stay at
their seed while the densities move, and that feedback path is one the reverse
pass never enters; its cost is measured, and staggered re-sectioning closes it.

**Shear and torsion are not designed for, and the exclusion is measured rather
than assumed.** The check covers axial force with bending and leaves out
EN 1993-1-1 §6.2.6–6.2.8, as clause 6.2.10 permits while the design shear
stays under half the plastic shear resistance. `experiments/validation/20_shear_audit.py`
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
uv sync
uv run pytest
```

Add dependencies with `uv add` or `uv add --dev` rather than by editing
`pyproject.toml`. Install the formatting hooks before the first commit; their
pinned ruff is what CI uses:

```bash
uv run pre-commit install
```
