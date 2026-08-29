# normax

[![Tests](https://github.com/arpastrana/normax/actions/workflows/test.yml/badge.svg)](https://github.com/arpastrana/normax/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/arpastrana/normax/blob/main/LICENSE)

> Backpropagating through structural engineering norms.

This project composes three traditionally separate stages of structural design into a single differentiable program for meter-scale structures composed of beam members.
In the first stage, a form-finding solver maps force densities to a funicular geometry.
Next, a structural analysis solver then transforms that geometry into member forces, and the Eurocode 3 (EN 1993-1-1) finally converts forces into law-compliant member sections that are safe for construction.
The result of this composition is one function that can be optimized end-to-end with exact gradients.

The interesting part is that none of these components was designed to be differentiated in the same way.
The form-finding solve is differentiated natively and implicitly through JAX.
Meanwhile, the structural analysis solver for planar systems uses sensitivities compiled into its C++ implementation years before this pipeline existed, and the its 3D version in addition to the code checks both use hand-derived adjoints.
Eurocode 3 is especially instructive: a building code is a normative specification, not a numerical solver, so it has no natural notion of a derivative.
Giving the code check a differentiable computational representation is what allows it to participate in the same optimization loop as an autodiff-native form-finder.

To a gradient-based optimizer, however, these distinctions disappear as Tesseract provides convenient interfaces to glue together these seemingly disparate pieces of software across eras, programming languages, and differentiation schemes.
This glue is a two-way street.
Not only it allows to feed information across structural design stages through forward computation, but it critically empowers them to backpropagate gradients (very useful serach directions in high dimensional search spaces, if you ask me) through the entire pipeline end-to-end.
That is where the magic happens, as derivatives promise to unify currently disjoint engineering stages into a streamlined process that aims to accelerate design optimization cycles toward building safe yet material-efficient bridges, roofs, and buildings.

## Installation

Clone the repository and install with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/arpastrana/normax
cd normax
uv sync
```

The form-finding solver, the structural analysis backends, and the structural
engineering norm verifier are all installed as regular dependencies. Nothing
else is needed: `uv sync` followed by `uv run pytest` runs the whole suite, and
no step of it requires Docker — the Tesseract stages are imported into the test
process.

## An example

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
the augmented Lagrangian in `normax.design.solve_problem` holds
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
the export and the viewer on and off. Adding `--shape-parametrization heights`
or `fixed` races the same structure, loads, analysis and check against a
geometry that is written down rather than found.

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
same interface in `normax/form_finding.py` and reached from the same example by
a word on the command line, so racing them swaps one block and nothing else:

```bash
uv run python examples/arch.py                                  # 0.150150 t
uv run python examples/arch.py --shape-parametrization heights  # 0.144128 t
uv run python examples/arch.py --shape-parametrization fixed    # 0.157469 t
```

All three open on the same 2500 mm parabola at 0.291664 t, so what separates
them is the search and not the start, and the mass falls monotonically with the
shape freedom each is given: none, one force density, nine node heights.

Moving the geometry buys the larger share of the mass on both trusses, and the
arch says what that freedom is worth at the margin. Its held plan leaves **one**
independent density, so free heights searches nine degrees of freedom against
the form finder's one — and buys 4% for them, landing at much the same rise
(1467 mm against 1397 mm) by a search costing an order of magnitude more
iterations. A shape prior is not free, but on this structure it is nearly free.
Numbers, tolerances and the full protocol are in the accompanying paper and in
`CHANGELOG.md`; `validation/` keeps the checks that still run against the
shipped API, and `docs/retired_experiments.md` says where the rest went and how
to get them back.

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

**Commercial engineering software integration is future work.** Products such
as SkyCiv or Dlubal could provide valuable independent analysis and design
checks, first as validators of a finished design and eventually as pipeline
backends. The present bottlenecks are common to black-box commercial systems:
their APIs expose less calculation state and implementation detail than an
exact adjoint needs, generally return forward results without sensitivities,
and may restrict section parametrization to the software's own catalogs. API
keys, licensing, usage limits, network latency and service availability also
make an inner optimization loop harder to reproduce and test offline. A final
batched validation is therefore the practical first integration; an in-loop
backend can follow when a product exposes enough information to define and
verify its derivative contract.

**Every adjoint is verified against finite differences of its own forward pass,
rather than against a second implementation.** During development the crossed
stack was checked against two in-process JAX implementations — a frame solver
and an EN 1993-1-1 check written independently of it — and they agreed to
1.3e-14 on gradients and 6.7e-16 on every field crossing the boundary. Neither
could ship, so both were deleted before submission; `docs/oracle_removal.md`
records the reasoning and the tag `local-dev` marks the tree where that
agreement reproduces. What ships is held to central differences of the crossed
primal, to closed-form section algebra, and to references frozen at that tag.
That is weaker in one respect — a second implementation can disagree in a way a
difference cannot — and stronger in another, since a difference of a function's
own primal cannot inherit a mistake two implementations share.

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
