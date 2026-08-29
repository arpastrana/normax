# Where a search starts, and which parametrizations care

*Measured 2026-08-29 on `examples/arch.yaml` — ten members, 10 m span, 180 kN
uniform plus two half-span cases, S355 at the Class 3 limit, crossed OpenSees
analysis and crossed Blueprints check. Reproduce with
`uv run python examples/arch.py --shape-parametrization fdm|heights|fixed`.*

This is the seed of the robustness-to-initialization study the paper wants. It
is kept separate from `CHANGELOG.md` because it is a result rather than a
decision, and because one of its findings inverts a claim the repo used to make.

---

## 1. A flat drawing is a stationary point, not a bad guess

The written-heights route reads its start off the drawn geometry. Drawn flat,
every shape derivative is **exactly zero**:

    d(mass)/d(height)               0.000000e+00   all nine free nodes
    d(worst utilization)/d(height)  0.000000e+00   all nine
    d(mass)/d(diameter)             2.034961e-04   the only live direction

Two reasons, and neither is a matter of tuning.

**Length is first-order insensitive to transverse motion.** Mass is
`rho * sum A(d) L` and `L = sqrt(dx^2 + dz^2)`, so `dL/dz` is proportional to
`dz/L` and vanishes at `dz = 0`. This holds whatever the load and whatever the
check.

**Flat is a symmetry point of the mechanics.** Raising the nodes gives an arch,
lowering them gives a cable, and both shed bending equally, so the response is
even in `z` and its derivative vanishes at the origin. A linear analysis makes
this exact: at rise 0 the axial force is identically zero and the bending is at
its maximum (2.5e8 N mm against 6.2e5 at rise 2500).

**What the search does with that.** Nothing. From flat, the heights route
returns **bit-identical results to the fixed route** — same 0.491027 t, same
diameters 272.5 to 383.0 mm — with its nine variables provably inert. From a
1 mm start it is *pulled back* into the stationary point and lands on the same
number.

`form_finding.height_start: {rise: 50.0}` exists for this reason:
`build_parabolic_heights` generates a lift at a named crown, quadratic in each
free node's plan distance to the nearest support, rather than reading a drawing
that may have no shape in it.

## 2. Above a small threshold, free heights is start-insensitive

| start rise | heights answer | its final crown |
|---|---|---|
| 0 mm | 0.491027 t | 0.0 — never moved |
| 1 mm | 0.491027 t | 0.0 — fell back |
| 50 mm | 0.144129 t | 1464.8 mm |
| 250 mm | 0.144129 t | 1464.8 mm |
| 1000 mm | 0.144129 t | 1464.9 mm |
| 2500 mm | 0.144128 t | 1466.8 mm |

The escape threshold is between **1 mm and 50 mm**, which is 0.5% of the span
and 3% of the crown the search ends at. Above it the route lands on the same
answer from every start tried, spanning 50x in rise.

**This inverts the earlier record.** Experiment 15 concluded that free heights
is "start-ruled — flat line never leaves the ground, random lands 17% high". On
the current augmented Lagrangian with calibrated budgets, the only failing start
is the degenerate one, and it fails for a provable reason rather than a
basin-radius one. Do not build an argument on free heights being fragile to
initialization; it is fragile at exactly one point.

**The form-found route reads no drawn height at all.** `fdm` returns 0.150150 t
whether the arch is drawn flat or at 2500 mm — identical to the digit. Only the
rivals are start-sensitive, which matters for the fairness of any comparison
drawn from this file.

## 3. Converged against converged

| route | mass | evaluations | ended | worst U |
|---|---|---|---|---|
| `fdm` | 0.150150 t | **202** | converged at round 8 | 1.000000 |
| `heights` | **0.144129 t** | **901** | converged, needs `rounds_max: 20` | 1.000000 |
| `fixed` | 0.491027 t | 145 | converged | 0.999999 |

`heights` is **4.0% lighter for 4.46x the evaluations.** That it wins on mass is
not a surprise and should not be reported as one: any shape the form finder can
reach is a shape the written route can write, so its feasible set strictly
contains the other's and it can only be lighter or equal. If `fdm` ever came out
lighter, that would be evidence of a convergence failure in `heights`.

What is worth reporting is how little the extra freedom buys. **Nine shape
variables against one** — the arch's held plan leaves exactly one independent
force density — for 4%, landing 5% apart in rise (1464.8 against 1396.9 mm). The
funicular manifold passes very close to the unconstrained optimum, which is the
claim a shape prior rests on.

At the shipped `rounds_max: 10`, `heights` stops on its round budget at the same
0.144129 t but at worst utilization 1.000004, marginally infeasible. The mass
was already found; the extra ten rounds buy feasibility, not mass.

## 4. At equal budget the mass claim does not go the convenient way

Scaling `iterations_warmup` and `iterations_after_warmup` together:

| budget | `fdm` evals / mass | `heights` evals / mass / worst U |
|---|---|---|
| tiny | 159 / 0.150150 converged | 145 / 0.144196 / 1.000011 |
| small | 164 / 0.150150 converged | 332 / 0.144187 / 1.000003 |
| medium | 202 / 0.150150 converged | 643 / 0.144130 / 1.000052 |
| large | 202 / 0.150150 converged | 780 / 0.144129 / 1.000007 |

**`fdm`'s mass is identical at every budget**, converged from 159 evaluations up.
But `heights` is within **0.05%** of its converged answer after 145 evaluations,
so its 4.46x is spent *certifying feasibility to 1e-6*, not searching — and it
is lighter at every budget. Its violation also wanders non-monotonically with
budget.

**So the defensible end-to-end claims on this arch are budget-insensitivity,
start-insensitivity, and one variable against nine. Not a lighter design, and
not a faster search.** Recorded plainly here so nobody re-derives it hopefully.

## 5. What the `fixed` route is, and what it is not

Drawn flat, `fixed` sizes a straight pin-ended beam: **0.491027 t against
0.150150, a factor of 3.27.** That is a legitimate and striking number for
*having no shape freedom at all* — sizing cannot move the geometry, which is the
project's whole premise.

It is not a competing design method, and it must not be labelled as one. Across
the draws tried, `fixed` ranges from 0.151959 t (rise 1000 mm) to 0.517654 t
(rise 50 mm) — a **3.4x spread on how well somebody drew the shape**. That
spread, against a form-found route that is invariant to the drawing, is the
honest form of the argument.

## 6. Where a mass claim honestly lives

Not on this arch. The gridshell's held-plan basis is 13-dimensional against
hundreds of free heights — a far larger asymmetry than the arch's one against
nine — and experiment 23 recorded end-to-end *winning* there, 0.53% lighter and
3.1x fewer iterations on the braced diagrid. That comparison is untested since
the 2026-08-28 moment reduction and is the next thing to measure.
