# What the crossed shell's descents did, and how to get it back

The three routes of `experiments/gridshell_16_crossed.yaml`, descended through
**jax-fdm in process, PyNite across the analysis schema and Blueprints across the
check**, by augmented Lagrangian with no SLSQP polish. 2026-08-26.

## The expensive part, kept

`*_starts.json` and `*.log` — **24 fixed-seed scattered starts per route, 4h20 of
compute.** Every landing: mass, worst violation, violated rows, evaluations, wall
clock, and the design variables. This is the measurement the reported numbers
come from and it is not cheap to reproduce, so it is kept rather than regenerated.

| route | best [t] | median | spread | feasible |
|---|---|---|---|---|
| end to end | 0.074724 | 0.075659 | 53.7% | 23/24 |
| free heights | 0.136011 | 0.144273 | 24.0% | 24/24 |
| sizing only | 0.145735 | 0.148004 | 2.3% | 24/24 |

## The cheap part, regenerable

`*.npz` — the **path** of each route's winning start, one frame per objective
evaluation, for the descent animations. Roughly 2 MB and about nine minutes of
compute, so these are **not** kept in git.

Regenerate with `experiments/28_descent_paths.py`, from the repository root:

    uv run --group pipeline --extra spike python experiments/28_descent_paths.py It works because the starts are drawn from a
fixed seed (`SCATTER_SEED` in `experiments/design_routes.py`), so a winner can be
reconstructed exactly; all three re-descents reproduced their recorded mass to
every digit, which is what says the reconstruction is the same descent rather
than a near neighbour.

Each file holds `opening` (the start), `steps` (every evaluation's variables),
`masses` (mass per frame), `rounds` (the per-round trail the method reports),
`landing`, `route` and `start`.

## Two things to know before plotting

**The paths dive below where they land, and that is the method working.** End to
end reaches 0.073284 mid-descent and settles at 0.074724; sizing only touches
0.116707 and settles at 0.145735. A small opening penalty lets the mass lead
until the multipliers pull it back onto the constraint surface, so those minima
are infeasible points and not better answers. **Plot the violation beside the
mass** or the picture says the opposite of what happened.

Sizing only starts at 7.89 t and ends at 0.146, a sixty-fold sweep — that axis
wants to be logarithmic.

**A frame is an evaluation, not an accepted step.** The line search evaluates
points it then rejects, so consecutive frames can move backwards. Fine for a
morph; for a monotone curve use `rounds`, or take a running minimum.
