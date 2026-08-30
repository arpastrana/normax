# normax

[![Tests](https://github.com/arpastrana/normax/actions/workflows/test.yml/badge.svg)](https://github.com/arpastrana/normax/actions/workflows/test.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Tesseract Hackathon 2026](https://img.shields.io/badge/Tesseract_Hackathon_2026-Track_01-6f42c1.svg)](https://pasteurlabs.ai/tesseract-hackathon-2026/)

> Backpropagating through structural engineering codes.

**Normax is a Tesseract Hackathon 2026 submission for Track 01: Inverse Design
& Shape Optimization.** It turns structural form finding, structural analysis,
and Eurocode cross-section checks into one differentiable design program.

```text
[jax-fdm form finding] → [OpenSees / PyNite structural analysis] → [Blueprints code check]
          ↑                                                               │
          └─────────── mass + code gradients flow back ────────────────────┘
```

Geometry and member actions move to the right. Gradients of steel mass and code
utilization move back to the left, so all three stages participate in one
design decision.

## Motivation

Safety is not negotiable. Waste is. Structural codes turn hard-won experience
into rules for buildings, roofs, and bridges. Construction also produces about
37% of global CO₂ emissions and consumes nearly half of extracted materials,
according to the
[UN Environment Programme's 2025–2026 global status
report](https://www.unep.org/resources/report/global-status-report-buildings-and-construction-2025-2026).
The goal is not to trade safety for sustainability. It is to remove material
while keeping safety inside the optimization.

Practice splits the work. Form finding chooses geometry. Structural analysis
computes its response. A code check judges the result. Data moves forward, but
useful feedback rarely returns. Late compliance can enlarge a section. It
cannot easily revise the shape that created the demand.

Normax makes the handoff differentiable. It joins jax-fdm, OpenSees or PyNite,
and Blueprints' EN 1993-1-1 check. Code utilization can then shape geometry, not
merely reject it. This is the key step. Codes contain branches, envelopes,
classes, and discrete choices. Their full design spaces are not smooth. Normax
differentiates a stated continuous slice: S355 circular hollow sections at a
fixed cross-section class.

**Only the joined program lets geometry and sections respond together to the
same loads and code constraints.** The experiments compare that search with
fixed-geometry sizing and free nodal heights. Final reruns are in progress, so
the table below keeps earlier values out of the submission record.

<!-- FINAL: HERO_ANIMATION: add figures/hero.gif, then uncomment the line below. -->
<!-- ![A Normax optimization morphing a structure while member utilization changes](figures/hero.gif) -->

## The optimization problem

For force densities $\mathbf q$, diameters $\mathbf d$, and load cases
$\mathcal L$, Normax composes three maps:

$$
\begin{aligned}
(\mathbf x,\boldsymbol\ell) &= \mathcal F(\mathbf q)
&& \text{form finding: geometry and member lengths}, \\
\mathbf s &= \mathcal A(\mathbf x,\mathbf d,\mathcal L)
&& \text{structural analysis: member actions}, \\
\mathbf u &= \mathcal C_{\mathrm{EC3}}(\mathbf s,\mathbf d,\boldsymbol\ell)
&& \text{code check: utilization}.
\end{aligned}
$$

The simultaneous design problem is

$$
\begin{aligned}
\min_{\mathbf q,\mathbf d}\quad
& m(\mathbf q,\mathbf d)
= \rho\sum_{e=1}^{n} A_{\mathrm{CHS}}(d_e)\,\ell_e(\mathbf q) \\
\text{subject to}\quad
& u_{k,e}(\mathbf q,\mathbf d) \le 1
&& \forall\ \text{load cases } k\ \text{and members } e, \\
& \mathbf g_{\mathrm{geometry}}(\mathbf x) \le \mathbf 0,
\qquad \mathbf q\in\mathcal Q,\quad \mathbf d\in\mathcal D.
\end{aligned}
$$

Every symbol has a concrete role:

| Symbol | Meaning |
|---|---|
| $\mathbf q$, $\mathcal Q$ | member force densities and their admissible set |
| $\mathbf d$, $\mathcal D$ | member diameters and their admissible box |
| $\mathcal L$, $k$ | prescribed load cases and one load-case index |
| $e$, $n$ | one member index and the number of members |
| $\mathcal F$ | jax-fdm form-finding map |
| $\mathbf x$, $\boldsymbol\ell$, $\ell_e$ | nodal coordinates, all member lengths, and the length of member $e$ |
| $\mathcal A$, $\mathbf s$ | OpenSees or PyNite structural-analysis map and its member actions |
| $\mathcal C_{\mathrm{EC3}}$, $\mathbf u$, $u_{k,e}$ | Blueprints code-check map, its utilization matrix, and one load-case/member utilization |
| $\rho$, $A_{\mathrm{CHS}}(d_e)$ | steel density and circular-hollow-section area at diameter $d_e$ |
| $m$ | total steel mass |
| $\mathbf g_{\mathrm{geometry}}$, $\mathbf 0$ | geometric inequality vector and its feasible upper bound |

Normax differentiates this actual composition. Code utilization, geometry,
signs, and box bounds constrain the mass objective.

## Software stack

Tesseract and Tesseract-JAX provide the shared schema and JAX boundary. Each
numerical stage keeps its own implementation and derivative strategy:

| Stage | Software | Language | Differentiation used by Normax |
|---|---|---|---|
| Form finding | [jax-fdm](https://github.com/arpastrana/jax_fdm) | Python and JAX | native JAX reverse mode through the equilibrium solve |
| Structural analysis (2D) | [OpenSees](https://opensees.berkeley.edu/) | C++ core with a Python interface | native Direct Differentiation Method forward sensitivities assembled into a VJP |
| Structural analysis (3D) | [PyNite](https://github.com/JWock82/Pynite) | Python | no native derivatives, so Normax supplies an implicit structural adjoint |
| Code compliance | [Blueprints](https://github.com/Blueprints-org/blueprints) | Python | no native derivatives, so Normax supplies a hand-derived VJP |

## Results

Final reruns will populate this table. Masses are tonnes of steel. Every result
must meet the same utilization tolerance, loads, and section model as its
baseline.

| Structure | Analysis | Fixed geometry | Form + sizing | Material reduction |
|---|---|---:|---:|---:|
| Arch | OpenSees | TBD <!-- FINAL: ARCH_FIXED_MASS_T --> | TBD <!-- FINAL: ARCH_FDM_MASS_T --> | TBD <!-- FINAL: ARCH_SAVINGS_PCT --> |
| Warren truss | OpenSees | TBD <!-- FINAL: WARREN_FIXED_MASS_T --> | TBD <!-- FINAL: WARREN_FDM_MASS_T --> | TBD <!-- FINAL: WARREN_SAVINGS_PCT --> |
| Vierendeel truss | OpenSees | TBD <!-- FINAL: VIERENDEEL_FIXED_MASS_T --> | TBD <!-- FINAL: VIERENDEEL_FDM_MASS_T --> | TBD <!-- FINAL: VIERENDEEL_SAVINGS_PCT --> |
| Gridshell | PyNite | TBD <!-- FINAL: GRIDSHELL_FIXED_MASS_T --> | TBD <!-- FINAL: GRIDSHELL_FDM_MASS_T --> | TBD <!-- FINAL: GRIDSHELL_SAVINGS_PCT --> |

<!-- FINAL: ARCH_DESIGNS: add figures/arch_designs.png, then uncomment below. -->
<!-- ![Initial and optimized arch designs](figures/arch_designs.png) -->

<!-- FINAL: WARREN_DESIGNS: add figures/warren_designs.png, then uncomment below. -->
<!-- ![Initial and optimized Warren truss designs](figures/warren_designs.png) -->

<!-- FINAL: ARCH_OPTIMIZATION: add figures/arch_optimization.png, then uncomment below. -->
<!-- ![Arch objective and constraint history](figures/arch_optimization.png) -->

The final configurations, comparison protocol, tolerances, complete tables, and
figure provenance will be published with the frozen submission results.

## Installation

Normax requires Python 3.12. Clone the repository and let
[`uv`](https://docs.astral.sh/uv/) create the project environment:

```bash
git clone https://github.com/arpastrana/normax.git
cd normax
uv sync
```

## Quickstart

Suppose a pedestrian bridge must cross a ten-metre ravine in the Rocky
Mountains. The terrain fixes two rocky abutments. The deck supplies a load. The
steel arch that will serve as its backbone is still negotiable. We want one
function that finds its equilibrium geometry, runs structural analysis, checks
EN 1993-1-1, and tells us how mass changes with the force densities.

Normax keeps that function compact and legible. Tesseract makes the analysis and
code implementations look like ordinary JAX calls:

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

Run the same snippet from the repository:

```bash
uv run python examples/readme.py
```

[`examples/readme.py`](examples/readme.py) evaluates mass and utilization, then
differentiates through both Tesseract boundaries.

To use the other analysis backend, replace one line:

```python
analyzer = TesseractAnalyzer(structure, section_catalog, backend="pynite")
```

The pipeline, schema, and JAX-facing API stay fixed. The solvers come from
different eras. [OpenSees](https://opensees.berkeley.edu/) is a long-established
C++ research framework with solver sensitivities.
[PyNite](https://github.com/JWock82/Pynite) is a structural analysis solver in
plain Python with no derivative API. Normax gives each backend its own
derivative rule behind the same call.

The snippet evaluates one design and its gradient. Let the optimizer move force
densities and diameters with:

```bash
uv run python examples/arch.py
```

## More examples

Once the arch runs, three more vignettes exercise the same pipeline on different
structural systems. Each reads the YAML beside it and exports the configured
data and figures:

```bash
uv run python examples/warren.py
uv run python examples/vierendeel.py
uv run python examples/gridshell.py
```

For the planar examples, use the same model and switch only the shape
parametrization to reproduce the baselines:

```bash
uv run python examples/arch.py --shape-parametrization fixed
uv run python examples/arch.py --shape-parametrization heights
```

The options mean:

- `fdm`: optimize reduced force-density coordinates and recover equilibrium.
- `heights`: optimize free node heights directly.
- `fixed`: hold geometry fixed and optimize sections.

## Technical notes and guides

Normax grew through measured decisions, discarded approaches, and explicit
scope cuts. These notes preserve the derivations and engineering work behind the
small public API:

- [Development roadmap](docs/ROADMAP.md)
- [Initialization and parametrization](docs/initialization.md)
- [Gridshell experiments](docs/gridshell_findings.md)
- [Building the PyNite backward pass](docs/fast_backward_pass.md)
- [Parallel gradients and backend choice](docs/parallel_gradients.md)
- [Tesseract stdio concurrency defect](docs/tesseract_stdio_race.md)
- [Owning materials and sections](docs/sections_extraction.md)
- [The retired `ec3x` extraction](docs/ec3x_extraction.md)
- [Removing private validation oracles](docs/oracle_removal.md)
- [Shear-design scope](docs/shear_design.md)
- [Retired experiments and recovery](docs/retired_experiments.md)
- [Development history](CHANGELOG.md)

## Verification

Tests run the Tesseract APIs in process. They need neither Docker nor a network
service.

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Focused scripts compare each reverse rule with solver identities, element
formulas, closed forms, and central differences:

```bash
uv run python validation/opensees_ddm.py
uv run python validation/pynite_adjoint.py
uv run python validation/blueprint_adjoint.py
```

These commands are the reproducibility entry points. The frozen submission will
add expected outputs, platform notes, and environment details.

## One program, three kinds of differentiation

Each stage appears as one function despite different languages, data models,
and differentiation strategies.

```text
Forward pass

            force densities q               section diameters d
                  │                                  │
                  ▼                                  │
       JAX force-density form finding                │
       equilibrium solve + held-plan basis           │
                  │ geometry x, lengths L            │
                  └────────────────┬─────────────────┘
                                   ▼
                     Tesseract: structural analysis
                       OpenSees (2D) / PyNite (3D)
                                   │ member actions
                                   ▼
                     Tesseract: section check
                       Blueprints / EN 1993-1-1
                                   │ utilization U
                  ┌────────────────┴─────────────────┐
                  │                                  │
          mass m(x, L, d)                   constraints U ≤ 1
                  └────────────────┬─────────────────┘
                                   ▼
                     augmented-Lagrangian optimizer
```

The optimizer sees one composition. Its reverse pass follows the same route in
the other direction:

```text
Backward pass

 objective m                         constraints U ≤ 1
      │ direct cotangents                     │
      │ on geometry and d                     ▼
      │                    Blueprints check VJP: hand adjoint
      │                                       │ actions, d
      │                                       ▼
      │                    structural-analysis VJP
      │                      OpenSees: solver sensitivities
      │                      PyNite: implicit structural adjoint
      │                                       │ geometry, d
      └───────────────────┬───────────────────┘
                          ▼
        JAX / implicit form-finding pullback + accumulation
                          │
                          ▼
                  gradients ∂L/∂q and ∂L/∂d
```

This is not a surrogate. Every design evaluation calls the crossed analysis and
Blueprints check. Backpropagation calls their derivative endpoints.

## The advantages of Tesseract for structural engineering

No single autodiff system owns this calculation:

- jax-fdm differentiates through its equilibrium solve.
- OpenSees exposes compiled forward sensitivities through the analysis
  Tesseract.
- PyNite has no derivative API. Normax supplies an implicit element-level
  reverse rule.
- Blueprints evaluates EN 1993-1-1 in scalar Python. Normax supplies its
  hand-derived pullback.

Tesseract makes the crossed stages JAX-callable blocks with explicit schemas and
vector-Jacobian products. Switching OpenSees for PyNite changes one backend, not
the pipeline. Without that boundary, Normax would have to replace the software
it claims to optimize through.

## Scope and limitations

Normax is a research prototype, not a certification tool. Its claims apply only
to the stated models and loads.

- Member checks cover EN 1993-1-1 cross-section resistance under axial force
  with biaxial bending for S355 circular hollow sections. The shipped check
  does not implement member flexural buckling under §6.3.1, shear, or torsion.
- Circular hollow sections avoid lateral-torsional buckling by construction.
  Other section families, global frame stability, and critical load factors are
  not supported.
- Loads are prescribed. Changed section sizes do not feed self-weight back into
  the load cases.
- Diameters are continuous and member-wise, without catalog rounding,
  connection design, fabrication, or other buildability constraints.
- The nested fully-stressed experimental route omits the `∂d/∂q` coupling. The
  headline route instead optimizes shape variables and diameters simultaneously
  so that path is present.
- Commercial engineering software is not integrated. Batched validation is the
  practical first step because most commercial APIs do not expose the solver
  state or sensitivities needed by an in-loop differentiable backend.
- Local Tesseract dispatch is serialized because the hosted solvers and runtime
  redirection have mutable, thread-sensitive state.

These exclusions define the experiment. They are not safety claims.

## Repository map

| Path | Purpose |
|---|---|
| [`normax/design.py`](normax/design.py) | pipeline, objectives, constraints, and optimization problem |
| [`normax/form_finding.py`](normax/form_finding.py) | force-density, free-height, and fixed-geometry parametrizations |
| [`normax/tesseract.py`](normax/tesseract.py) | JAX-facing structural-analysis and sizing blocks |
| [`tesseracts/analysis`](tesseracts/analysis) | OpenSees and PyNite Tesseract API and backends |
| [`tesseracts/sizing`](tesseracts/sizing) | Blueprints check Tesseract API and adjoint |
| [`examples`](examples) | four reproducible design studies and their YAML configurations |
| [`validation`](validation) | focused numerical and derivative checks |
| [`tests`](tests) | unit, composition, parity, and regression tests |

## License

Normax is released under the [Apache License 2.0](LICENSE). Third-party solvers,
libraries, and standards retain their own terms. Citation metadata is in
[`CITATION.cff`](CITATION.cff).
