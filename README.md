# normax

[![Tests](https://github.com/arpastrana/normax/actions/workflows/test.yml/badge.svg)](https://github.com/arpastrana/normax/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/arpastrana/normax/blob/main/LICENSE)

Backpropagating through structural engineering codes

Force densities to a funicular shape, a frame analysis to member forces, and
EN 1993-1-1 to the sections it requires — composed into one function with exact
gradients throughout. The building code is a normative text rather than a solver:
it has no derivatives of its own, and giving it one is what lets it sit in an
optimization loop beside an autodiff form-finder. The three blocks differentiate
three different ways — implicit-function-theorem rules on the form-finding
solve, sensitivities compiled into a C++ solver years before this pipeline
existed, and hand-derived adjoints on the frame analysis and the code check —
and one composed function is what the optimizer sees.

## Installation

Not published. Clone the repository and install with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/arpastrana/normax
cd normax
uv sync
```

Everything the pipeline needs — the form finder, both host frame solvers and
the Blueprints check — is a regular dependency. The `local-dev` group pins the
packages that do not ship: the two oracles the parity tests compare against,
and the viewer. Without it those tests skip themselves, the viewer stands
itself in, and the rest of the suite runs.

## Usage

```python
import jax
import jax.numpy as jnp

from normax.design import DesignParameters
from normax.design import StructuralDesignPipeline
from normax.design import compute_mass
from normax.form_finding import FdmFormFinder
from normax.loads import assemble_load_cases
from normax.loads import create_load_uniform
from normax.materials import Steel355
from normax.sections import build_section_catalog
from normax.structures import build_arch_2d
from normax.tesseract import TesseractAnalyzer
from normax.tesseract import TesseractSizer

num_edges = 20
diameter_start = 100.0
force_density_start = -50.0
section_class = 3
material = Steel355()

# Problem setup
structure = build_arch_2d(num_edges, span=10_000.0, rise=3_000.0)
load_case = create_load_uniform(structure, 15_000.0)
loads = assemble_load_cases([load_case])
section_catalog = build_section_catalog(material, section_class)

# The three main computation blocks of the structural design pipeline
form_finder = FdmFormFinder(structure)
analyzer = TesseractAnalyzer(structure, section_catalog, backend="opensees")
sizer = TesseractSizer(structure, section_catalog, backend="blueprint")

# One ring to rule them all
pipeline = StructuralDesignPipeline(form_finder, analyzer, sizer)

# Design parameters: one force density and one diameter per member
force_densities = jnp.full(num_edges, force_density_start)
diameters = jnp.full(num_edges, diameter_start)


# "How much does your building weigh, Mr. Foster?" (Fuller, 1978)
def compute_lawful_mass(force_densities):
    parameters = DesignParameters(force_densities, diameters)
    design = pipeline(parameters, loads)
    return compute_mass(design), design


# One pass for the mass, its gradient, and the design the two describe
compute_mass_and_gradient = jax.value_and_grad(compute_lawful_mass, has_aux=True)
(mass, design), gradient = compute_mass_and_gradient(force_densities)

# Is every member within what EN 1993-1-1 allows?
is_design_safe = jnp.all(design.sizes.utilization <= 1.0)

print(mass)  # tonnes of steel
print(gradient)  # the mass' gradient
print(is_design_safe)  # True, the norm check passes on every member
```

The pipeline is three swappable blocks, each built from a structure on the
host and then called; what is left is a function of design parameters and load
cases, and that is what an optimizer differentiates. Form finding traces a
linear solve in this process. The frame analysis and the code check each cross
a Tesseract boundary to a host that does not differentiate itself — OpenSees
for a planar frame, PyNite for a space frame, Blueprints for the check — and
come back with a hand-written adjoint. The blocks are constructed one by one
above to show what a pipeline is made of, and the four examples do the same,
naming the held-plan basis and the symmetry folding they build a form finder
with rather than settling them behind a call. Swapping the
planar solver for the space-frame one is a different word in the config and
nothing else. The design that comes back also carries
`design.sizes.utilization`, how hard EN 1993-1-1 works every member under every
load case, which is exactly a constraint function —
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
each claim was measured by racing parametrizations against each other on them,
so what the gradient buys is a comparison rather than a demo. A form finder
acts as a shape prior: descending one force density through it is start-proof
where descending every free node height stalls in bending. On a truss, holding the
plan leaves a null space of force densities to search, and
`normax.form_finding.build_plan_basis` makes its basis coordinates the design
variables. The rival routes — free heights without the form finder, and sizing
alone at the drawn geometry — are form finders too, implemented against the
same interface in `tests/test_comparison.py`, which is where the swap is
checked: racing them swaps one block and nothing else, and moving the geometry
buys the larger share of the mass on both trusses.
Numbers, tolerances and the full protocol are in the accompanying paper and in
`CHANGELOG.md`; `experiments/validation/` keeps the checks that still run
against the shipped API, and `docs/retired_experiments.md` says where the rest
went and how to get them back.

## Limitations

**The nested fully-stressed route, kept in `normax/optimization/nested.py`,
omits `∂d/∂q`.**
The shipped search is simultaneous — diameters are variables beside the force
densities, so the design is self-consistent by construction and the gradient
is complete. In the nested add-on the diameters the analysis runs at stay at
their seed while the densities move, and that feedback path is one the reverse
pass never enters; its cost is measured, and staggered re-sectioning closes it.

**Shear and torsion are not designed for, and the exclusion is measured rather
than assumed.** The check covers axial force with bending and leaves out
EN 1993-1-1 §6.2.6–6.2.8, as clause 6.2.10 permits while the design shear
stays under half the plastic shear resistance. That fraction was read off every
converged design rather than bounded, and no structure here approaches the
threshold; `docs/shear_design.md` records the measurement and what designing
for shear would take.

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
