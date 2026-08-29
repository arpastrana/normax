# Results and experiment protocol

This document is the source of truth for what the headline comparisons mean.
The final submission numbers are intentionally left as explicit placeholders
until the current code, configs and dependency environment have been frozen and
all routes have been rerun. A table containing `<!-- FINAL: ... -->` is not yet
citable.

## Question

Normax tests whether a useful structural shape prior can be optimized through a
real analysis solver and a real engineering-code implementation, rather than
stopping differentiation at either boundary. The objective is steel mass. The
design constraints include the reported cross-section utilization, geometric
bounds and, where configured, force-density sign guards. Shape and section
diameters are optimized simultaneously in the headline end-to-end route.

The examples cover four different design conditions:

| structure | role in the study | analysis backend |
|---|---|---|
| arch | one force-density coordinate against free node heights | OpenSees |
| Warren truss | a triangulated planar frame with a held plan | OpenSees |
| Vierendeel truss | a bending-dominated planar frame without diagonals | OpenSees |
| gridshell | a larger space frame with folded section families | PyNite |

Every headline route uses Blueprints for the crossed EN 1993-1-1 cross-section
check.

## The comparison

For each structure, one driver and one YAML description are run with three
interchangeable shape parametrizations:

| label in this document | CLI value | variables that move | interpretation |
|---|---|---|---|
| end to end | `fdm` | held-plan force-density coordinates and diameters | form finding, analysis and checking participate in one gradient |
| free heights | `heights` | permitted node heights and diameters | a less structured shape space, without the funicular prior |
| sizing only | `fixed` | diameters | the drawn geometry is held fixed |

The same structure topology, load cases, material, section class, analysis
backend, code-check backend, section grouping and optimizer implementation are
used within each three-way comparison. The YAML and command-line override are
the executable protocol; copy the exact final YAML with a reported result.

The arch, Warren truss, and Vierendeel truss also share one planar load
vocabulary and the same totals: 180 kN over the full deck, 90 kN over the near
half, and 45 kN at midspan. The arch is drawn flat, so its deck mask reaches
every free node. A separate far-half case is omitted only because the node and
section variables are folded by the midspan mirror, making that case a
reindexing of the near half. The spatial gridshell keeps its own pressure-load
protocol.

The design spaces are deliberately not equal. Free heights generally contains
the shapes reachable through the force-density parametrization and has more
variables. Sizing only has no shape freedom and measures the value of moving
the geometry relative to the particular drawing; it is a baseline, not a rival
shape-optimization method. Initializers are parametrization-specific, so a
fair report must state each route's initial geometry rather than assume that
the coordinate vectors imply the same start.

## Acceptance protocol

A landing enters the final table only when all of the following hold:

1. It was produced from the submission commit and the checked-in or archived
   final YAML, with the resolved dependency versions recorded.
2. The driver completed normally and reported its termination reason.
3. Its worst constraint violation is within the YAML's
   `optimization.violation_tol`; a low-mass point visited before feasibility is
   not an answer.
4. The mass is recomputed from the returned final parameters, rather than read
   from an intermediate line-search evaluation.
5. Its `.npz` record and figures are retained with a stem identifying the
   structure and route.
6. Where more than one start is used, every route receives a declared start
   budget and the reported statistic says whether it is the nominal, best or
   median feasible landing. Best must be compared with best, not with another
   route's nominal start.

Evaluation counts are useful evidence about search effort but are not a
hardware-independent runtime. The problems are non-convex; these are local
optima, not global-optimality certificates.

## Final headline table

Fill mass in tonnes, worst utilization, and objective evaluations from the
final report. Derived savings should be computed only after the source masses
have been frozen.

| structure | route | final mass [t] | worst utilization | evaluations |
|---|---|---:|---:|---:|
| arch | end to end | <!-- FINAL: ARCH_FDM_MASS_T --> | <!-- FINAL: ARCH_FDM_WORST_U --> | <!-- FINAL: ARCH_FDM_EVALUATIONS --> |
| arch | free heights | <!-- FINAL: ARCH_HEIGHTS_MASS_T --> | <!-- FINAL: ARCH_HEIGHTS_WORST_U --> | <!-- FINAL: ARCH_HEIGHTS_EVALUATIONS --> |
| arch | sizing only | <!-- FINAL: ARCH_FIXED_MASS_T --> | <!-- FINAL: ARCH_FIXED_WORST_U --> | <!-- FINAL: ARCH_FIXED_EVALUATIONS --> |
| Warren | end to end | <!-- FINAL: WARREN_FDM_MASS_T --> | <!-- FINAL: WARREN_FDM_WORST_U --> | <!-- FINAL: WARREN_FDM_EVALUATIONS --> |
| Warren | free heights | <!-- FINAL: WARREN_HEIGHTS_MASS_T --> | <!-- FINAL: WARREN_HEIGHTS_WORST_U --> | <!-- FINAL: WARREN_HEIGHTS_EVALUATIONS --> |
| Warren | sizing only | <!-- FINAL: WARREN_FIXED_MASS_T --> | <!-- FINAL: WARREN_FIXED_WORST_U --> | <!-- FINAL: WARREN_FIXED_EVALUATIONS --> |
| Vierendeel | end to end | <!-- FINAL: VIERENDEEL_FDM_MASS_T --> | <!-- FINAL: VIERENDEEL_FDM_WORST_U --> | <!-- FINAL: VIERENDEEL_FDM_EVALUATIONS --> |
| Vierendeel | free heights | <!-- FINAL: VIERENDEEL_HEIGHTS_MASS_T --> | <!-- FINAL: VIERENDEEL_HEIGHTS_WORST_U --> | <!-- FINAL: VIERENDEEL_HEIGHTS_EVALUATIONS --> |
| Vierendeel | sizing only | <!-- FINAL: VIERENDEEL_FIXED_MASS_T --> | <!-- FINAL: VIERENDEEL_FIXED_WORST_U --> | <!-- FINAL: VIERENDEEL_FIXED_EVALUATIONS --> |
| gridshell | end to end | <!-- FINAL: GRIDSHELL_FDM_MASS_T --> | <!-- FINAL: GRIDSHELL_FDM_WORST_U --> | <!-- FINAL: GRIDSHELL_FDM_EVALUATIONS --> |
| gridshell | free heights | <!-- FINAL: GRIDSHELL_HEIGHTS_MASS_T --> | <!-- FINAL: GRIDSHELL_HEIGHTS_WORST_U --> | <!-- FINAL: GRIDSHELL_HEIGHTS_EVALUATIONS --> |
| gridshell | sizing only | <!-- FINAL: GRIDSHELL_FIXED_MASS_T --> | <!-- FINAL: GRIDSHELL_FIXED_WORST_U --> | <!-- FINAL: GRIDSHELL_FIXED_EVALUATIONS --> |

| derived comparison | value |
|---|---:|
| arch: end to end vs sizing only | <!-- FINAL: ARCH_SAVINGS_PCT --> |
| Warren: end to end vs sizing only | <!-- FINAL: WARREN_SAVINGS_PCT --> |
| Vierendeel: end to end vs sizing only | <!-- FINAL: VIERENDEEL_SAVINGS_PCT --> |
| gridshell: end to end vs sizing only | <!-- FINAL: GRIDSHELL_SAVINGS_PCT --> |

Also record the final geometry descriptors used in prose and captions:

| quantity | value |
|---|---:|
| arch end-to-end crown/rise [mm] | <!-- FINAL: ARCH_FDM_RISE_MM --> |
| arch free-heights crown/rise [mm] | <!-- FINAL: ARCH_HEIGHTS_RISE_MM --> |
| gridshell end-to-end rise [mm] | <!-- FINAL: GRIDSHELL_FDM_RISE_MM --> |

## What the evidence can support

The strongest evidence is architectural: the optimizer's actual objective and
constraint evaluation cross the structural-analysis and Blueprints Tesseract
boundaries, and reverse mode returns through them. Tesseract is therefore part
of the production optimization path, not a separate parity demonstration.

The final numeric claims should be narrower:

- Compare end to end with sizing only to quantify what moving geometry bought
  relative to the named drawing.
- Compare end to end with free heights to discuss the effect of the funicular
  shape prior, including design-space dimension, feasibility and evaluation
  count. Do not imply that the prior must be lighter: free heights usually has
  the larger feasible set.
- Describe a result as robust to initialization only if the final rerun includes
  a declared multi-start or start-sensitivity study. Determinism of one start is
  not robustness.
- Attach load magnitude and section floor to any material-saving percentage;
  the advantage of geometry can vary with load and with members resting at the
  minimum diameter.

Historical experiments that predate the final moment-demand reduction or the
current public examples are excluded from the headline table. Git history
preserves them, but their absolute masses must not be mixed with final runs.

## Validation evidence

The shipped verification is layered:

- backend tests compare OpenSees Direct Differentiation Method sensitivities
  and the PyNite hand-written implicit adjoint with central differences of
  their own forward solves;
- sizing tests and executable studies compare the Blueprints hand adjoint with
  central differences, explicit implicit-function derivatives and closed-form
  section algebra where available;
- composition tests differentiate objectives through all three blocks and
  compare the result with central differences of the complete crossed forward
  pass;
- parity tests check the host and Tesseract representations agree on forward
  values and on the available reverse rule;
- the complete test suite runs without the two private JAX implementations used
  during development.

Before those private implementations were removed, the crossed stack agreed
with them to 1.3e-14 on gradients and 6.7e-16 on values crossing the boundary.
The `local-dev` Git tag records that development state. The current tree keeps
only public dependencies: it replaces the private oracles with finite
differences of the crossed primal, analytic invariants, host-versus-boundary
checks and a small number of frozen reference norms. This is an honest change
in evidence: a finite difference cannot detect every error that a genuinely
independent implementation might, while differencing a backend's own primal
also cannot inherit an adjoint error shared by two implementations.

Commands for the suite and executable validation studies are in
[reproducibility.md](reproducibility.md#focused-gradient-validation). The
derivation and performance record for the PyNite reverse rule are in
[fast_backward_pass.md](fast_backward_pass.md).

## Scope and limitations

### The headline search is simultaneous; the nested add-on is not

The results above use the simultaneous augmented-Lagrangian search: shape
coordinates and diameters are decision variables together, so stiffness
feedback is inside the differentiated problem. The optional nested route in
`normax/optimization/nested.py` analyzes each inner descent at frozen seed
diameters and refreshes those diameters only between rounds. Its gradient
therefore omits the nested `∂d/∂q` feedback path. Staggered re-sectioning closes
the returned design's forward coupling, but does not turn that inner gradient
into the derivative of the fully coupled problem. Do not use the nested route
for a headline gradient claim.

### The EC3 claim is a cross-section check, not a complete building-code design

The shipped Blueprints backend evaluates EN 1993-1-1 cross-section resistance
for axial force and biaxial bending using the relevant 6.2 expressions for the
configured circular hollow sections. It does **not** implement member flexural
buckling under §6.3.1. The validation study writes out a §6.3.1 comparison from
the standard to expose that gap; it does not add buckling to the reported
designs.

Shear, torsion and their interactions are also outside the sizing map. The
validation study reconstructs the shear demand that the declined clause would
have seen, but that does not make shear part of the reported design check. A
final claim that shear would not change the studied diameters must be tied to a
rerun of the frozen headline designs and its observed `V_Ed/V_pl,Rd` value.
<!-- FINAL: HEADLINE_MAX_SHEAR_RATIO_AND_SCOPE -->

Torsion is zero for the planar, nodally loaded examples, but that does not
establish the same for arbitrary space frames.

Circular hollow sections are doubly symmetric, so lateral-torsional buckling is
not the governing omitted mode for these section shapes. That does not replace
the missing flexural-member and global-stability checks.

### Loads, fabrication and stability

- Self-weight does not feed back into the load cases. Loads are assembled once
  from the YAML and are not rebuilt as sections change.
- Diameters are continuous and may vary by member subject to configured symmetry
  groups. There is no discrete product-catalog, connection, erection or other
  buildability constraint. The continuous result is a lower bound on a
  fabricated catalog design.
- Local cross-section utilization is constrained, but global frame stability
  and a whole-structure critical load factor are not checked.
- The analysis is linear elastic and uses the supports, load cases and model
  assumptions encoded by each example. Results do not establish robustness to
  unmodeled load combinations, imperfections or second-order effects.

### Numerical interpretation

The optimizer can visit infeasible points lighter than its accepted answer, and
different floating-point builds can choose different local basins. Report the
accepted feasible landing, exact environment and start protocol. A single run
does not establish a distribution, and none of the comparisons proves global
optimality.
