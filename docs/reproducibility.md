# Reproducing normax

This guide separates a quick check from the optimization runs behind the final
results. Run every command from a clean clone.

## Environment

Normax requires Python 3.12 and uses
[uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/arpastrana/normax.git
cd normax
uv sync --frozen
uv run python -V
```

The committed `uv.lock` freezes the submission environment. Record the commit
with every published result. Capture the installed package list when reporting
outside that lock:

```bash
git rev-parse HEAD
uv pip freeze
```

Examples import local Tesseract clients into the Python process. Docker is not
required. The first JAX call may compile and is not a steady-state timing.

## Fast verification

Run the suite:

```bash
uv run pytest
```

Pytest uses all available cores. Use `uv run pytest -n0` for a serial run.

Then exercise one forward and backward pass:

```bash
uv run python examples/readme.py
```

It prints steel mass, its force-density gradient, and whether every reported
utilization is at most one. It does not optimize or write artifacts.

## Headline examples

Each driver reads the YAML beside it, optimizes shape and size, prints a report,
then exports the configured artifacts:

```bash
uv run python examples/arch.py
uv run python examples/warren.py
uv run python examples/vierendeel.py
uv run python examples/gridshell.py
```

These are studies, not smoke tests. Cost depends on convergence, compilation,
hardware, and resolved packages. Judge a run by its termination and violation
fields, not a portable time estimate. The gridshell is the largest case.
Animation happens after optimization and may continue after the answer prints.

### Compare parametrizations

One driver exposes all three routes:

```bash
uv run python examples/arch.py --shape-parametrization fdm
uv run python examples/arch.py --shape-parametrization heights
uv run python examples/arch.py --shape-parametrization fixed
```

Substitute `warren.py`, `vierendeel.py`, or `gridshell.py` for the other cases.
Read [results.md](results.md) before comparing them. Their design spaces differ,
and an infeasible low point is not a result.

## Artifacts

An exported run writes a NumPy record under `data/` and figures under
`figures/`. Alternative routes gain `_heights` or `_fixed`:

```text
data/<name>.npz
figures/<name>_designs.png
figures/<name>_load_cases.png
figures/<name>_optimization.png
figures/<name>_optimization.mp4

data/<name>_heights.npz
figures/<name>_heights_*.png
figures/<name>_heights_optimization.mp4

data/<name>_fixed.npz
figures/<name>_fixed_*.png
figures/<name>_fixed_optimization.mp4
```

A redraw can write two more per route: `<stem>_optimization.gif`, every frame at
the film's own width, and `<stem>_optimization_web.gif`, reduced for a page.

`<name>` is `arch`, `warren`, `vierendeel`, or `gridshell`. The gridshell config
records inner iterations and enables MP4 output; animation begins only after
the accepted answer has printed. Repeating a route replaces its files.

Generated data, images, and video are ignored by default. Curated submission
assets need narrow `.gitignore` exceptions or an explicit forced add. Tests do
not depend on generated files.

Each `.npz` stores `parameters`, `iterates`, `objectives`, `violations`, and
`round_index`. With `trace_iterations` enabled, the path contains inner
iterations. Otherwise it contains augmented-Lagrangian rounds.

Any example's accepted archives can be redrawn with one shared framing, curve
range, and animation pace across its baselines, without rerunning any
optimization:

```bash
uv run python redraw_designs.py                     # every example, drawings only
uv run python redraw_designs.py gridshell --film    # one of them, animation too
uv run python redraw_designs.py gridshell --film --gif   # and a faithful GIF of it
uv run python redraw_designs.py gridshell --web     # and one light enough to embed
```

Name `arch`, `warren`, `vierendeel`, or `gridshell` to redraw a subset. Add
`--film` for the animations, which cost minutes rather than seconds. `--gif`
converts each film at its own width and frame rate, which on a dense drawing
runs to a hundred megabytes and is meant for review rather than a page; `--web`
writes a second GIF beside it, narrowed and thinned until a page can carry it.
Both read the film already on disk rather than rendering again.

The three planar examples are drawn to one shared box, so an arch and a truss
can be set side by side at one scale; the shell is framed on its own. Each
example is rebuilt exactly as its own script builds it -- verified against every
archive's recorded objective to 4.1e-16 -- and the recorded parameters and
objective histories are never changed. The accepted masses and matching log
names are recorded in [results.md](results.md#gridshell-result-record).

### What the README embeds

Six of the generated files are committed by name, against the blanket ignore,
because a page cannot render what a clone has to generate first. Every other
figure a run writes stays ignored.

```text
figures/{arch,warren,vierendeel}_designs.png        the three planar comparisons
figures/gridshell_optimization_web.gif              end-to-end
figures/gridshell_heights_optimization_web.gif      heights + sections
figures/gridshell_fixed_optimization_web.gif        sections only
```

The three GIFs are `--web` output taken unaltered, so
`uv run python redraw_designs.py gridshell --film --web` reproduces them from
the archives. Replacing any of the six is a redraw followed by a commit of the
file itself; the `.gitignore` exceptions are listed one per line so that
nothing joins them by accident.

## Focused gradient validation

These studies expose the main derivative checks:

```bash
uv run python validation/opensees_ddm.py
uv run python validation/pynite_adjoint.py --quiet
uv run python validation/strut_gradients.py
uv run python validation/interaction_gradients.py
uv run python validation/blueprint_adjoint.py
uv run python validation/load_case_envelope.py
uv run python validation/class_ratio_sweep.py
uv run python validation/sizing_formulations.py
```

They compare crossed forward passes and reverse rules with central differences,
closed forms, implicit derivatives, host calculations, and frozen reference
norms. [blueprints_backward_pass.md](blueprints_backward_pass.md) derives the
code-check pullback. [fast_backward_pass.md](fast_backward_pass.md) derives the
PyNite adjoint and records its timings.

## Reporting rules

- Record the commit, resolved packages, route, YAML, final mass, worst
  violation, and termination reason.
- For multiple starts, record the generation rule and seed.
- Compare accepted feasible landings. Ignore lighter infeasible visits.
- Expect exact repeatability only inside one environment. Small floating-point
  changes can select another basin in a non-convex search.
- Treat results as local optima with continuous diameters, not global proofs or
  catalog-ready designs.
- The Blueprints backend checks Eurocode 3 cross-section resistance for axial
  force with biaxial bending on the fixed Class 3 CHS family. See
  [results.md](results.md#scope-and-limitations) for omitted checks.
- Local solver dispatch is serialized. See
  [tesseract_stdio_race.md](tesseract_stdio_race.md) for the upstream defect.
