# Changelog

This file records externally meaningful changes to Normax. It intentionally
omits the implementation diary, intermediate benchmark numbers, and superseded
experiments preserved in Git history.

Normax is still pre-release. There are no published releases or release dates
to report yet.

## Unreleased

### Headline results

The current examples optimize structural geometry and circular hollow-section
diameters together under multiple load cases and utilization constraints.
Headline values are being rerun against the final submission environment and
will be inserted only after that protocol is frozen.

| Example | Form-found design | Fixed-geometry baseline | Material saved |
|---|---:|---:|---:|
| Arch | <!-- FINAL: ARCH_FDM_MASS_T --> | <!-- FINAL: ARCH_FIXED_MASS_T --> | <!-- FINAL: ARCH_SAVINGS_PCT --> |
| Warren truss | <!-- FINAL: WARREN_FDM_MASS_T --> | <!-- FINAL: WARREN_FIXED_MASS_T --> | <!-- FINAL: WARREN_SAVINGS_PCT --> |
| Vierendeel truss | <!-- FINAL: VIERENDEEL_FDM_MASS_T --> | <!-- FINAL: VIERENDEEL_FIXED_MASS_T --> | <!-- FINAL: VIERENDEEL_SAVINGS_PCT --> |
| Gridshell | <!-- FINAL: GRIDSHELL_FDM_MASS_T --> | <!-- FINAL: GRIDSHELL_FIXED_MASS_T --> | <!-- FINAL: GRIDSHELL_SAVINGS_PCT --> |

<!-- FINAL: HEADLINE_PROTOCOL_AND_TOLERANCES -->

### Added

- A differentiable structural-design pipeline with three explicit stages:
  form finding, frame analysis, and steel cross-section checking.
- End-to-end constrained optimization of force-density or shape parameters and
  member diameters. Mass and every constraint are differentiated through the
  same composed program used for evaluation.
- An augmented-Lagrangian optimizer with bound handling, normalized constraint
  rows, convergence budgets, recovery from invalid trial points, and an
  explicit termination record.
- Three comparable shape parametrizations:
  - force-density form finding with a held-plan basis;
  - directly written free-node heights;
  - fixed geometry for a sizing-only baseline.
- Symmetry-aware parameter folding for member groups, force-density bases, and
  written heights. Mirrors and fabrication rotations remain separate concepts.
- Load-case generation for uniform, half-span, deck, deck-point, tributary, and
  sector pressure patterns.
- Constraints for utilization, rise, sag, minimum member length, parameter
  bounds, and sign conventions.
- Initializers for diameters, force densities, and parabolic height fields.
  Every search now begins at the requested seed rather than at an implicit
  optimizer default.
- Four submission examples sharing one execution shape:
  - a planar arch;
  - a Warren truss;
  - a Vierendeel truss;
  - a spatial gridshell.
- YAML configuration for the structures, load cases, backends, constraints,
  optimization budgets, and output settings used by each example.

### Tesseract composition

- A frame-analysis Tesseract with one schema and two selectable backends:
  - OpenSees for planar frames, differentiated with OpenSees DDM;
  - PyNite for space frames, differentiated with an equilibrium adjoint
    implemented and verified in this repository.
- A sizing Tesseract backed by Blueprints' EN 1993-1-1 cross-section clauses.
  Its reverse rule differentiates the scalar Python check and the diameter
  solve without requiring Blueprints to use JAX.
- Explicit reverse-mode endpoints for both crossed stages. One
  vector-Jacobian product pulls a scalar objective or aggregated constraint
  vector back through a stage.
- Backend selection as a schema input, with host-side validation of supported
  names. Swapping OpenSees and PyNite does not change the pipeline contract.
- Local loading of the Tesseract API modules for tests and examples, so the
  crossed interfaces are exercised without requiring Docker.
- A process-wide serialization guard for local Tesseract endpoints whose host
  solvers or standard streams are not safe under overlapping callbacks.

### Optimization and performance

- Simultaneous shape-and-sizing optimization replaced the earlier collection of
  route-specific search harnesses.
- Constraint rows are folded into the augmented objective before
  differentiation, reducing a full Jacobian to the vector-Jacobian product the
  optimizer actually consumes.
- Form finding, objectives, constraint maps, and replay readings are compiled
  at their stable boundaries.
- Multiple load cases share prepared analysis state and, where supported, a
  factorization.
- The PyNite reverse pass reuses its prepared frame and solved states rather
  than rebuilding a model for each coordinate perturbation.
- The Blueprints backend avoids constructing clause objects inside every
  bisection step while verifying that its fast evaluator agrees with the
  library's public class.
- Sizing forward and reverse calls share solved states through a bounded cache.
- Topology, section-family data, and other static inputs are prepared outside
  the traced optimization loop.
- Parallel finite-difference and process-pool alternatives were measured and
  declined where callback overhead, the Python interpreter lock, or solver
  thread safety made them slower or unsafe.

### Reporting and reproducibility

- A structured problem record containing the opening design, optimization
  solution, round history, optional iteration history, and final evaluated
  design.
- Reproducible NPZ exports for optimization parameters and the finest recorded
  history resolution.
- Text reports for the run configuration, backends, load cases, opening and
  final designs, constraint violation, and convergence.
- Design figures with shared utilization scales, comparable member-width
  mapping, visible supports, and an outline of the opening geometry.
- Separate figures for governing load cases and optimization history.
- Optimization plots showing true constraint violation and objective values,
  with outer-round starts marked on the same trajectory.
- MP4 optimization animations generated with the bundled imageio-ffmpeg
  executable, avoiding a system FFmpeg dependency.
- Replay utilities that reconstruct designs from a recorded optimization walk
  without storing solver-specific response objects.
- Validation scripts for analysis adjoints, Blueprints gradients, load-case
  envelopes, cross-section formulations, interaction equations, and excluded
  limit states.
- A focused account of the PyNite adjoint and callback performance in
  `docs/fast_backward_pass.md`.
- A documented comparison protocol and acceptance criteria in
  `docs/results.md`.

### Verification

- Reverse-mode derivatives are checked against central differences of the
  crossed forward programs.
- The OpenSees and PyNite backends are checked against each other where both can
  represent the same planar problem.
- Tesseract transport tests compare crossed values and gradients with direct
  calls to the backend implementations.
- Structural equilibrium, frame conventions, load-case envelopes, symmetry
  bases, section properties, optimizer constraints, record replay, and the
  shared pipeline components have dedicated tests.
- Closed-form checks cover section geometry, simple frame behavior, and
  selected code-equation branches.
- The test suite runs against the shipped public dependencies; private JAX
  oracle packages are no longer required or conditionally skipped.
- CI runs lint, formatting checks, and the test suite on pushes to the primary
  branch and on pull requests.

### Changed

- The public API was condensed around `DesignProblem`,
  `StructuralDesignPipeline`, stage contracts, and one shared optimization
  method.
- Each stage owns its input/output contract:
  - form finding returns geometry;
  - analysis returns member actions;
  - sizing returns utilization information;
  - mass is computed from geometry and sections outside the code check.
- Analysis and sizing backends now sit behind stable, swappable schemas rather
  than leaking solver-specific objects through the optimization code.
- Section grouping is a fabrication constraint owned by the problem, not by a
  particular sizing backend.
- The examples build their own structures and initializers while sharing the
  same pipeline, optimization, reporting, and export APIs.
- The arch, Warren truss, and Vierendeel truss now use the same three total-load
  cases: full deck, near half-deck, and a midspan point load. Mirror folding
  makes the omitted far half-deck case a reindexing rather than a missing load.
- Load cases are arrays with one explicit leading axis throughout the pipeline.
- Configuration names now describe the problem rather than the implementation
  technique, and unknown backend names fail before a crossed call.
- The optimization record distinguishes outer augmented-Lagrangian rounds from
  inner solver iterations.
- Output filenames include the shape parametrization, allowing several routes
  for one structure to coexist.
- Figure styling is local to Normax figures rather than changing Matplotlib's
  process-wide theme.
- Package and function names were standardized around clear action verbs and
  American English.

### Fixed

- Corrected circular-section moment demand to use the resultant at each member
  end rather than summing axis maxima that could come from different ends.
  This makes the demand independent of arbitrary local-axis roll.
- Folded directly written heights across requested mirror orbits. Previously a
  nominally symmetric height search could produce a design that did not satisfy
  the mirrored load case it omitted.
- Applied mirror folding to the arch's diameter and shape variables.
- Added a pre-solve collapsed-geometry guard so a zero-length member is never
  handed to a host solver.
- Preserved a separate, physical minimum-member-length constraint instead of
  treating the numerical collapse guard as a design rule.
- Recovered from non-finite or failed trial evaluations by recoiling toward the
  last valid point, while refusing invalid starting geometries.
- Corrected optimizer initialization so calibrated seeds, rather than a generic
  bound midpoint, determine where each search starts.
- Corrected load magnitudes to represent totals consistently across patterns.
- Corrected support fixities and frame conventions for planar structures.
- Corrected shear reporting and a factor-of-two error in a shear-bound
  validation sweep.
- Corrected end-to-end unit conversions and kept the Tesseract schemas explicit
  about millimetres, newtons, and moments.
- Removed stale numeric claims after changes to moment reduction,
  initialization, symmetry folding, and search protocols.

### Removed

- Private JAX frame-analysis and steel-check packages that had served as
  development oracles. The submission now evaluates and differentiates the
  public crossed backends it claims to use.
- The duplicate in-process Blueprints sizing wrapper and other paths that could
  bypass the sizing Tesseract during a featured run.
- The legacy multi-route search framework, stored landing database, and large
  experiment harness.
- Superseded experiment artifacts and development-only viewers from the
  installed package.
- Jacobian and Jacobian-vector-product endpoints not needed by the scalar
  constrained optimization path.
- Duplicate stage containers, builders, envelope implementations, and
  solver-specific result objects.

### Known limitations

- Blueprints supplies the cross-section checks used here but not EN 1993-1-1
  §6.3.1 member buckling. Headline results must not be described as complete
  member-stability designs.
- Shear and torsion are reported by the analysis stage but are not included in
  the current sizing constraint. Validation scripts quantify the excluded
  shear demand.
- The OpenSees backend differentiates planar frames only. Spatial examples use
  PyNite and the repository's verified adjoint.
- Both Tesseract stages expose reverse mode, which is the mode needed by the
  optimizer; forward-mode `jax.jvp` is not implemented across either stage.
- The force-density, free-height, and fixed parametrizations describe different
  feasible sets and optimization landscapes. A lighter landing does not by
  itself prove one parametrization universally superior.
- Geometry optimization is non-convex and can be sensitive to initialization.
  Final claims must state the start protocol, budget, feasibility tolerance,
  and whether they compare best runs or a fixed seed.
- Local in-process Tesseract dispatch is serialized because overlapping host
  callbacks can race on standard streams and some solver state. The
  reproduction and proposed upstream fix are documented separately.
- A minimum member length and explicit bounds regularize degenerate structural
  layouts; they are modeling assumptions and must accompany reported results.

## 0.1.0 (planned)

No `0.1.0` release has been tagged or published. The initial release is
expected to consist of the verified Unreleased work above after the final
headline results and reproducibility protocol replace their placeholders.
