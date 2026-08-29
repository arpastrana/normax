# Reproducing normax

This page separates a quick functional check from the longer optimization runs
that produce the submission results. Commands assume a clean clone and are run
from the repository root.

## Environment

Normax requires Python 3.12. The project uses
[uv](https://docs.astral.sh/uv/) to create the environment and install the form
finder, both structural-analysis backends, Blueprints, Tesseract Core and the
development tools.

```bash
git clone https://github.com/arpastrana/normax.git
cd normax
uv sync
uv run python -V
```

The repository currently has no lockfile. `uv sync` therefore resolves the
declared dependency ranges at install time; record the commit and resolved
environment with any published result:

```bash
git rev-parse HEAD
uv pip freeze
```

No Docker service is needed. The examples use local Tesseract clients imported
into the Python process. The first call through JAX may include compilation, so
it should not be interpreted as a steady-state timing.

## Fast verification

Run the complete test suite:

```bash
uv run pytest
```

Pytest uses all available cores by default. For easier diagnosis or a machine
on which process-level parallelism is constrained, run the same suite serially:

```bash
uv run pytest -n0
```

Then exercise one forward and backward pass through the three-stage pipeline:

```bash
uv run python examples/readme.py
```

The last command prints the steel mass, its gradient with respect to the force
densities, and whether every reported cross-section utilization is at most one.
It does not run an optimization or write artifacts.

## Headline examples

Each example reads the YAML file beside it, runs the simultaneous shape-and-size
optimization, prints a report and exports its record and figures:

```bash
uv run python examples/arch.py
uv run python examples/warren.py
uv run python examples/vierendeel.py
uv run python examples/gridshell.py
```

These are optimization studies, not smoke tests. Their cost depends on the
structure, optimizer convergence, JAX compilation, CPU and resolved dependency
versions. The gridshell is much larger than the planar examples. No portable
wall-clock estimate is asserted here; allow each process to finish and judge a
result from its printed convergence and violation fields.

The YAML `output` block controls reporting, export and animation. The checked-in
configs export all four examples; the gridshell disables animation, while the
three planar examples enable it. Animation rendering happens after the search
and can remain busy even though the numerical answer has already printed.

### Compare the three parametrizations

The default `fdm` route optimizes force densities and diameters end to end. The
same driver can instead optimize free nodal heights and diameters, or keep the
drawn geometry fixed and optimize diameters only:

```bash
uv run python examples/arch.py --shape-parametrization fdm
uv run python examples/arch.py --shape-parametrization heights
uv run python examples/arch.py --shape-parametrization fixed
```

Replace `arch.py` with `warren.py`, `vierendeel.py` or `gridshell.py` for the
other structures. See [results.md](results.md) before interpreting these runs:
the routes have different design spaces, and a local optimum or an infeasible
low point is not a headline result.

## Artifacts

An exported run writes a NumPy record under `data/` and still figures under
`figures/`. The form-found route keeps the bare structure name; the alternatives
gain `_heights` or `_fixed` so the three runs do not overwrite one another.

For a structure named `<name>`, expect:

```text
data/<name>.npz
figures/<name>_designs.png
figures/<name>_load_cases.png
figures/<name>_optimization.png
figures/<name>_optimization.mp4      # only when output.animate is true

data/<name>_heights.npz
figures/<name>_heights_*.png
figures/<name>_heights_optimization.mp4

data/<name>_fixed.npz
figures/<name>_fixed_*.png
figures/<name>_fixed_optimization.mp4
```

`<name>` is `arch`, `warren`, `vierendeel` or `gridshell`. The gridshell does
not write an MP4 with its checked-in config. Re-running one route replaces that
route's files, so copy any record that must be preserved before repeating it.
Generated result files are intentionally not required by the test suite.
The repository ignores generated image, video and `data/` files by default;
curated submission assets must be added deliberately (for example with a narrow
`.gitignore` exception) rather than assumed to be tracked after a run.

Each `.npz` contains the final parameter vector and the recorded optimization
path: `parameters`, `iterates`, `objectives`, `violations` and `round_index`.
Where the YAML enables `trace_iterations`, the path contains inner iterations;
otherwise it contains outer augmented-Lagrangian rounds.

## Focused gradient validation

The tests verify the crossed forward passes, reverse rules and composed
gradients. The following executable studies make the most relevant comparisons
visible as tables:

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

They include central differences of the crossed forward passes, closed-form or
implicit checks where available, boundary-versus-host comparisons, and frozen
norms recorded before the private development oracles were removed. The
[backward-pass note](fast_backward_pass.md) explains the PyNite adjoint and its
performance measurements.

## Reproduction notes

- The checked-in headline configs contain no random seed or random start. A
  repeated run in the same environment is expected to reproduce the same
  trajectory. Across dependency builds, platforms or CPU implementations,
  floating-point differences can steer the non-convex search into another
  basin even when each forward evaluation is deterministic.
- A reported result must include the Git commit, resolved package versions,
  route, YAML config, final mass, worst constraint violation and termination
  reason. For a multi-start study, also include the start-generation rule and
  seed.
- Compare feasible landings, not the lowest mass visited during a run. The
  augmented Lagrangian deliberately visits infeasible points before returning
  to the constraint surface.
- Results are local optima under continuous per-member diameters. They are not
  proofs of a global optimum and not catalog-rounded fabrication designs.
- Normax's shipped Blueprints backend performs an EN 1993-1-1 cross-section
  resistance check for axial force with biaxial bending. Its exact scope and
  the omitted engineering checks are stated in [results.md](results.md#scope-and-limitations).
- Local Tesseract execution serializes normax's solver dispatch. The upstream
  stdio concurrency defect found during development is documented in
  [tesseract_stdio_race.md](tesseract_stdio_race.md).
