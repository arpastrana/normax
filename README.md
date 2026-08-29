# normax

[![Tests](https://github.com/arpastrana/normax/actions/workflows/test.yml/badge.svg)](https://github.com/arpastrana/normax/actions/workflows/test.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Tesseract Hackathon 2026](https://img.shields.io/badge/Tesseract_Hackathon_2026-Track_01-6f42c1.svg)](https://pasteurlabs.ai/tesseract-hackathon-2026/)

> Backpropagating through structural engineering codes.

**Normax is a Tesseract Hackathon 2026 submission for Track 01 — Inverse Design
& Shape Optimization.** It turns structural form finding, frame analysis, and
Eurocode cross-section checks into one differentiable design program.

Structural design is usually a relay: propose a geometry, analyze it, size its
members, and repeat. Information moves forward, but the reason a design is
heavy does not move backward through the whole chain. Normax connects those
stages so a constrained optimizer can change both the geometry and circular
hollow-section diameters using gradients of the final design objective and
constraints.

The headline experiments compare that coupled search with the same structures,
loads, analysis, and checks under fixed-geometry and free-height baselines.
Final reruns are in progress; the submission numbers and assets are marked
explicitly below rather than mixed with earlier experimental results.

<!-- FINAL: HERO_ANIMATION — add figures/hero.gif, then uncomment the line below. -->
<!-- ![A Normax optimization morphing a structure while member utilization changes](figures/hero.gif) -->

## One program, three kinds of differentiation

Normax presents each stage as a function, even though the implementations
disagree about languages, data models, and differentiation strategies.

```text
Forward pass

 force densities / shape parameters q       section diameters d
                  │                                  │
                  ▼                                  │
       JAX force-density form finding                │
       equilibrium solve + held-plan basis           │
                  │ geometry x, lengths L            │
                  └────────────────┬─────────────────┘
                                   ▼
                     Tesseract: frame analysis
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
      │                    Blueprints check VJP — hand adjoint
      │                                       │ actions, d
      │                                       ▼
      │                    frame-analysis VJP
      │                      OpenSees: solver sensitivities
      │                      PyNite: implicit frame adjoint
      │                                       │ geometry, d
      └───────────────────┬───────────────────┘
                          ▼
        JAX / implicit form-finding pullback + accumulation
                          │
                          ▼
                  gradients ∂L/∂q and ∂L/∂d
```

This is not a differentiable surrogate for the engineering pipeline. The
optimizer evaluates the crossed OpenSees or PyNite analysis and the crossed
Blueprints check on every design evaluation, then calls their derivative
endpoints during backpropagation.

## Why Tesseract is load-bearing

No single autodiff system owns this calculation:

- Force-density form finding is native JAX and differentiates through a linear
  equilibrium solve.
- OpenSees is a compiled structural solver with forward sensitivities; its
  planar backend exposes those through the analysis Tesseract.
- PyNite is a Python space-frame solver with no native derivative API; Normax
  supplies an implicit, element-level reverse rule behind the same schema.
- Blueprints evaluates the EN 1993-1-1 cross-section formulas as scalar Python;
  Normax gives the check a hand-derived pullback behind a sizing Tesseract.

Tesseract makes the two crossed stages ordinary JAX-callable blocks with
explicit schemas and vector-Jacobian products. Replacing OpenSees with PyNite
changes the selected backend, not the design pipeline or optimizer. Without
those boundaries and derivative contracts, Normax would need a parallel
reimplementation of each solver and check in one autodiff-native stack; the
experiment would no longer be optimizing through the software it claims to
compose.

## Results

The table will be populated from the final frozen protocol. All masses are
tonnes of steel, and every reported solution must satisfy the documented
utilization tolerance under the same loads and section model as its baseline.

| Structure | Analysis | Fixed geometry | Form + sizing | Material reduction |
|---|---|---:|---:|---:|
| Arch | OpenSees | TBD <!-- FINAL: ARCH_FIXED_MASS_T --> | TBD <!-- FINAL: ARCH_FDM_MASS_T --> | TBD <!-- FINAL: ARCH_SAVINGS_PCT --> |
| Warren truss | OpenSees | TBD <!-- FINAL: WARREN_FIXED_MASS_T --> | TBD <!-- FINAL: WARREN_FDM_MASS_T --> | TBD <!-- FINAL: WARREN_SAVINGS_PCT --> |
| Vierendeel truss | OpenSees | TBD <!-- FINAL: VIERENDEEL_FIXED_MASS_T --> | TBD <!-- FINAL: VIERENDEEL_FDM_MASS_T --> | TBD <!-- FINAL: VIERENDEEL_SAVINGS_PCT --> |
| Gridshell | PyNite | TBD <!-- FINAL: GRIDSHELL_FIXED_MASS_T --> | TBD <!-- FINAL: GRIDSHELL_FDM_MASS_T --> | TBD <!-- FINAL: GRIDSHELL_SAVINGS_PCT --> |

<!-- FINAL: ARCH_DESIGNS — add figures/arch_designs.png, then uncomment below. -->
<!-- ![Initial and optimized arch designs](figures/arch_designs.png) -->

<!-- FINAL: WARREN_DESIGNS — add figures/warren_designs.png, then uncomment below. -->
<!-- ![Initial and optimized Warren truss designs](figures/warren_designs.png) -->

<!-- FINAL: ARCH_OPTIMIZATION — add figures/arch_optimization.png, then uncomment below. -->
<!-- ![Arch objective and constraint history](figures/arch_optimization.png) -->

The final configurations, comparison protocol, tolerances, complete tables, and
figure provenance will be published with the frozen submission results.

## Quickstart

Normax requires Python 3.12. Clone the repository and let
[`uv`](https://docs.astral.sh/uv/) create the project environment:

```bash
git clone https://github.com/arpastrana/normax.git
cd normax
uv sync
uv run python examples/readme.py
```

[`examples/readme.py`](examples/readme.py) is the smallest executable version
of the composition: it builds an arch, connects the three stages, evaluates its
mass and EN 1993-1-1 utilization, and takes a JAX gradient through both
Tesseract boundaries. It is kept as runnable source instead of duplicated as a
long README snippet.

Run a complete constrained optimization with:

```bash
uv run python examples/arch.py
```

Each headline example reads the YAML file beside it, reports the optimization,
and can export its data and figures as configured in that file:

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

- `fdm`: optimize a reduced set of force densities, then obtain geometry from
  equilibrium;
- `heights`: optimize the free node heights directly;
- `fixed`: keep the drawn geometry and optimize sections only.

## Verification

The default installation runs the Tesseract APIs locally in the test process;
Docker and a network service are not required.

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Focused validation scripts exercise the derivative rules against solver
identities, element formulas, closed-form section algebra, and central
differences of the corresponding forward pass:

```bash
uv run python validation/opensees_ddm.py
uv run python validation/pynite_adjoint.py
uv run python validation/blueprint_adjoint.py
```

These commands are the reproducibility entry points; the frozen submission will
add expected outputs, platform notes, and environment-capture instructions.

## Scope and limitations

Normax is a research prototype, not a complete building-code certification
tool. Its comparisons are useful only within the stated models and loads.

- Member checks cover EN 1993-1-1 cross-section resistance under axial force
  with biaxial bending for S355 circular hollow sections. The shipped check
  does not implement member flexural buckling under §6.3.1, shear, or torsion.
- Circular hollow sections avoid lateral-torsional buckling by construction;
  other section families are not supported. Global frame stability and a
  whole-structure critical load factor are not checked.
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

These exclusions are part of the reported protocol, not claims about safety.

## Repository map

| Path | Purpose |
|---|---|
| [`normax/design.py`](normax/design.py) | pipeline, objectives, constraints, and optimization problem |
| [`normax/form_finding.py`](normax/form_finding.py) | force-density, free-height, and fixed-geometry parametrizations |
| [`normax/tesseract.py`](normax/tesseract.py) | JAX-facing analysis and sizing blocks |
| [`tesseracts/analysis`](tesseracts/analysis) | OpenSees and PyNite Tesseract API and backends |
| [`tesseracts/sizing`](tesseracts/sizing) | Blueprints check Tesseract API and adjoint |
| [`examples`](examples) | four reproducible design studies and their YAML configurations |
| [`validation`](validation) | focused numerical and derivative checks |
| [`tests`](tests) | unit, composition, parity, and regression tests |

Selected technical note:

- [Building a fast backward pass](docs/fast_backward_pass.md)

## License

Normax is released under the [Apache License 2.0](LICENSE). Third-party solvers,
libraries, and standards remain subject to their own licenses and terms.
