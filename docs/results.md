# Results and experiment protocol

This file defines every headline comparison. The arch, Warren, and Vierendeel
tables contain accepted planar runs from the frozen configurations on commit
`cb1c074`. The PyNite gridshell study remains in progress and is not yet a
result.

## Research question

Can a structural shape prior be optimized through real analysis and code-check
software? Normax minimizes steel mass while constraining cross-section
utilization, geometry, and configured force-density signs. The end-to-end route
optimizes force densities and diameters together.

| structure | role | analysis |
|---|---|---|
| arch | one force-density coordinate against free heights | OpenSees |
| Warren truss | triangulated planar frame with a held plan | OpenSees |
| Vierendeel truss | bending-dominated planar frame | OpenSees |
| gridshell | space frame with folded section families | PyNite |

Every route uses Blueprints for the crossed Eurocode 3 cross-section check.

## Comparison

Each driver and YAML pair exposes three shape parametrizations:

| label | CLI value | variables | interpretation |
|---|---|---|---|
| end to end | `fdm` | held-plan force-density coordinates and diameters | all three stages share one gradient |
| free heights | `heights` | permitted node heights and diameters | larger shape space without the funicular prior |
| sizing only | `fixed` | diameters | common starting geometry stays fixed |

Within a three-way comparison, topology, loads, material, section class,
backends, section groups, optimizer, starting geometry, and starting diameters
are fixed. The end-to-end route reproduces that geometry through force
densities, the free-height route writes it as nodal heights, and sizing only
holds it. The final YAML and CLI form the executable protocol.

The planar cases use 180 kN over the full deck, 90 kN over the near half, and
90 kN at midspan. Midspan symmetry makes the far-half case a reindexing of the
near half. The gridshell uses its own pressure loads.

The design spaces are intentionally unequal. Free heights usually contains the
force-density shapes and adds variables. Sizing only measures the value of
moving the common starting geometry. It is a baseline, not a competing shape
method. Every report states the start so that equality can be checked rather
than assumed.

## Acceptance protocol

A landing enters the final table only when:

1. It comes from the submission commit and final YAML.
2. Resolved package versions are recorded.
3. The driver exits normally and reports its reason.
4. Worst violation meets `optimization.violation_tol`.
5. Mass is recomputed from the returned parameters.
6. The `.npz` record and figures identify structure and route.
7. Multi-start comparisons give each route the same declared start budget and
   compare the same statistic.

Evaluation counts measure search effort, not portable runtime. These are local
optima. They are not global certificates.

## Final headline table

Mass is reported in tonnes. Worst utilization and objective evaluations come
from each accepted report. The remaining placeholders belong only to the
in-progress gridshell study.

| structure | route | final mass [t] | worst utilization | evaluations |
|---|---|---:|---:|---:|
| arch | end to end | 0.171684 | 1.000000 | 129 |
| arch | free heights | 0.154561 | 1.000000 | 758 |
| arch | sizing only | 0.517654 | 1.000000 | 81 |
| Warren | end to end | 0.050743 | 1.000001 | 2243 |
| Warren | free heights | 0.051188 | 1.000000 | 1856 |
| Warren | sizing only | 0.071797 | 1.000000 | 208 |
| Vierendeel | end to end | 0.120819 | 1.000001 | 3110 |
| Vierendeel | free heights | 0.136498 | 1.000000 | 2193 |
| Vierendeel | sizing only | 0.277435 | 1.000000 | 1222 |
| gridshell | end to end | <!-- FINAL: GRIDSHELL_FDM_MASS_T --> | <!-- FINAL: GRIDSHELL_FDM_WORST_U --> | <!-- FINAL: GRIDSHELL_FDM_EVALUATIONS --> |
| gridshell | free heights | <!-- FINAL: GRIDSHELL_HEIGHTS_MASS_T --> | <!-- FINAL: GRIDSHELL_HEIGHTS_WORST_U --> | <!-- FINAL: GRIDSHELL_HEIGHTS_EVALUATIONS --> |
| gridshell | sizing only | <!-- FINAL: GRIDSHELL_FIXED_MASS_T --> | <!-- FINAL: GRIDSHELL_FIXED_WORST_U --> | <!-- FINAL: GRIDSHELL_FIXED_EVALUATIONS --> |

| derived comparison | value |
|---|---:|
| arch: end to end vs sizing only | 66.83% less mass |
| Warren: end to end vs sizing only | 29.32% less mass |
| Vierendeel: end to end vs sizing only | 56.45% less mass |
| gridshell: end to end vs sizing only | <!-- FINAL: GRIDSHELL_SAVINGS_PCT --> |

| geometry descriptor | value |
|---|---:|
| arch end-to-end rise [mm] | 1397.6 |
| arch free-heights rise [mm] | 1632.2 |
| gridshell end-to-end rise [mm] | <!-- FINAL: GRIDSHELL_FDM_RISE_MM --> |

## Permitted claims

The architectural claim is direct. Objective and constraint evaluations cross
the analysis and Blueprints boundaries, then reverse mode returns through them.
Tesseract sits inside the optimization path.

Numeric claims must stay narrower:

- Compare end to end with sizing only to measure the value of moving the common
  starting geometry.
- Compare end to end with free heights to study the shape prior, dimension,
  feasibility, and evaluation count. Do not assume the prior must be lighter.
- Claim robustness only after a declared multi-start study.
- Attach loads and section floors to every saving percentage.

Do not mix historical masses with the final table. Earlier experiments used a
different moment reduction or different public examples.

## Validation evidence

The shipped evidence has four layers:

- OpenSees DDM and the PyNite adjoint against central differences of their own
  forward solves.
- Blueprints adjoints against central differences, implicit derivatives, and
  closed-form section algebra.
- Full compositions against central differences of the crossed forward pass.
- Host and Tesseract parity for values and reverse rules.

The four layers are deliberately different. Central differences expose local
derivative errors. Closed forms guard section algebra. Host and Tesseract parity
guards the boundary. Frozen norms catch drift. None is a universal proof, so
reported claims remain scoped to the checked configurations.

See [reproducibility.md](reproducibility.md#focused-gradient-validation) for
commands and [fast_backward_pass.md](fast_backward_pass.md) for the PyNite
derivation.

## Scope and limitations

### Simultaneous and nested searches differ

Headline results use simultaneous shape and diameter variables. Stiffness
feedback is therefore inside the differentiated problem.

The optional route in `normax/optimization/nested.py` freezes seed diameters
during each inner descent and refreshes them between rounds. Its gradient omits
the nested `∂d/∂q` path. Staggered re-sectioning repairs the returned forward
design, not the inner derivative. Do not use this route for a headline gradient
claim.

### Eurocode 3 scope

Blueprints evaluates Eurocode 3 cross-section resistance under axial force and
biaxial bending for circular hollow sections. Every headline configuration
fixes Class 3 before optimization, precomputes its limiting
$d/t=90(235/f_y)$ ratio, and varies only outer diameter. It does not check
member flexural buckling under §6.3.1. A validation comparison exposes that gap
without adding buckling to the design. The
[backward-rule guide](blueprints_backward_pass.md) derives this exact slice.

Shear, torsion, and their interactions are also absent. A final shear claim
needs the frozen designs and their observed `V_Ed/V_pl,Rd`.
<!-- FINAL: HEADLINE_MAX_SHEAR_RATIO_AND_SCOPE -->

Planar nodal loads produce zero torsion. This says nothing about arbitrary space
frames. Circular hollow sections avoid lateral-torsional buckling, but not the
omitted flexural-member or global-stability checks.

### Other exclusions

- Self-weight does not update the prescribed loads.
- Diameters are continuous. Product catalogs, connections, erection, and other
  buildability constraints are absent.
- Global stability and critical load factors are not checked.
- Analysis is linear elastic. Claims exclude unmodeled combinations,
  imperfections, and second-order effects.

### Numerical interpretation

Report the accepted feasible landing, environment, and start protocol. The
optimizer can visit lighter infeasible points. Floating-point builds can select
different local basins. One run does not define a distribution or prove global
optimality.
