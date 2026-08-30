# Which way the truss bulges

**Decided 2026-08-30: the trusses arch upward.** Both truss runs now open on an
arched sketch and guard every family they can, and the Vierendeel guards its
verticals. The arch is untouched, a chain already arches by its density box.

The cause of a sagging answer is the **start sketch**, not the sign guard, and
that is the whole finding. The load sits on the bottom chord, so
`sketch_lens(sag=+500)` fits the loaded chord as a tension catenary and the
downward bulge is seeded at iteration 0. Nothing downstream lifts it back out: a
compression guard on the sagging sketch only freezes the top chord flat and
leaves the bottom at its floor.

## The recipe

```yaml
# examples/vierendeel.yaml
density_start: {sag: -500.0, rise: 500.0, held_plan: true}
sign_guard: {bottom chord: compression, top chord: compression, verticals: tension}
sign_margin_fraction: 0.1
```

A **negative** sag lifts the bottom chord, so the sketch arches at constant depth
and both chords are fitted in compression. The Warren takes the same start with
three differences, each forced rather than chosen: `held_plan` stays false,
because the held arched fit leaves **zero** self-stresses and no shift can then
sign anything; the diagonals are not guarded and cannot be, because each one
reverses sign across midspan, which is what a Warren does; and they need no
guard, because every diagonal has plan projection and no panel of that truss can
collapse or invert.

## What it costs and what it buys

| structure | fdm | heights | fixed | heights vs fdm | geometry bought |
|---|---|---|---|---|---|
| arch | 0.171684 | 0.154561 | 0.517654 | -9.97% | 66.8% |
| warren | 0.050743 | 0.051085 | 0.071797 | +0.67% | 29.3% |
| vierendeel | 0.120819 | 0.136736 | 0.277435 | +13.17% | 56.5% |

Against the sagging configuration this replaces, the Vierendeel pays 0.99%
(0.119635 to 0.120819) and widens its lead over written heights from 12.71% to
13.17%; the Warren comes out 0.13% lighter, and its drawn baseline improves a
great deal, 0.079787 to 0.071797, because an arched sketch is a better truss than
a lens. Masses are single-start local optima on a landscape whose basins spread
about 12%, so read the sign of a difference this small and not its size.

## Two defects the experiment exposed

**`length_min` cannot see an inverted panel.** The row is built on
`design.shape.lengths`, a Euclidean norm, so a panel at depth -3000 satisfies a
1000 mm floor. Guard the chords in compression and leave the verticals free and
the descent turns five of the nine Vierendeel panels inside out to buy lever arm,
reporting a violation of 2.8e-07 while doing it. Guarding the verticals is what
stops it here, and it is not a proof: a deeper arched sketch crossed anyway with
every vertical density positive, because a vertical's density sign says
tension or compression and not which of its nodes is higher. A signed depth row
is the repair, and is not written.

**The guard margin is one number for families that differ by thirty times.**
`guard_signs` scales the margin off the median of every guarded member, and the
Vierendeel's chords carry about 205 against its verticals' 12.8. So a vertical
guard is inadmissible above about 0.11 -- no shift along the single self-stress
signs every family at once -- while below 0.1 the verticals are left weak enough
that the panels invert again. The working value sits at the top edge of its own
admissible window. A per-family margin would remove the knife edge.

## What was measured and rejected

The inverted guard, `{bottom chord: compression, top chord: tension}`, also
arches and needs no vertical guard, but holds equilibrium by the near
cancellation of two enormous chord forces -- -3.7e+06 against +2.2e+06, where the
adopted recipe uses -8.96 and -196.76. Every variant of it lands on the same
shallow near-constant-depth arch whatever its start, because the geometry is the
residual of that cancellation, and all of them weigh 0.19 t or more.

Guarding the verticals on the **sagging** start is worse than either: it is
inadmissible at margin 0.1, and forcing it in at 0.02 costs 0.357020 t, three
times the adopted answer, and still sags. The sagging optimum needs a
sign-flipped midspan vertical -- its densities run +9.16 +5.63 +5.10 **-51.16**
+5.10 +5.63 +9.16 -- and that flipped panel is exactly what lets its top chord
dive to zero and open 2566 mm of depth at the quarter points.

Run logs are `data/runlogs/{arch,sign}_*.log`, the variant configs and the fit
probes are in `data/runlogs/configs/`, and the table is
`data/runlogs/table_arched.txt`.
