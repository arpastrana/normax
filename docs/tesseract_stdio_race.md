# Tesseract Core stdio concurrency defect

Normax exposed a process-wide stdio race while local Tesseract endpoints were
allowed to overlap. The defect is tracked upstream as
[tesseract-core #709](https://github.com/pasteurlabs/tesseract-core/issues/709).
This short note records why it matters to normax; the issue contains the
standalone public-API reproduction and current upstream discussion.

## The defect

`LocalClient.run_tesseract` enters the runtime's `start_run` path for every
endpoint call. That path temporarily redirects file descriptors 1 and 2 with
`os.dup2`, saves their previous targets, and restores them by hand. File
descriptors are process-global. Two overlapping redirections are safe only if
their exits happen in strict last-in-first-out order; independent threads do
not guarantee that ordering.

The minimum reproduction in #709 creates an in-memory Tesseract through the
public API and calls one shared local client from four ordinary Python threads.
Against the affected runtime, worker calls can raise `Bad file descriptor` and
stdout or stderr can be restored to the wrong target. The reproduction does not
depend on normax, JAX, OpenSees or Blueprints.

The proposed minimal runtime fix is a module-level `threading.RLock` held for
the entire stdio-redirection lifetime, including restoration and cleanup. It
must be reentrant because an existing nested-run path can enter `start_run`
again on one thread. That fix serializes local endpoint bodies; preserving
concurrency would require a more involved ownership design for process-global
descriptors.

## How normax found it

The retired optimization path requested a dense constraint Jacobian, producing
one reverse crossing per constraint row. During that work normax measured
overlapping local dispatch while its planar analysis Tesseract hosted OpenSees,
whose native domain is also process-global and mutable. Native runs could end
with `exit 139` and no Python traceback.

These are two distinct hazards exposed by the same overlap:

- Tesseract Core's save-and-restore protocol could corrupt process stdio.
- Concurrent calls could mutate OpenSees' single native model while another
  call was solving or reading sensitivities.

The dense-Jacobian formulation increased the number of crossings; it did not
create either unsafe global resource. A later smoke test confirmed the larger
crossing count but did not reproduce concurrent scheduling, so normax treats
that formulation as an exposure amplifier rather than the cause.

## Current mitigation and scope

Normax serializes crossed solver dispatch on an owner thread. The current
augmented-Lagrangian formulation also aggregates constraint rows into one
scalar before reverse mode, requiring one cotangent crossing per stage instead
of a dense row-by-row pullback. Together these choices avoid the overlapping
local calls that exposed the failure in normax's tested execution path.

This is a mitigation, not a claim that the affected Tesseract runtime is safe
for arbitrary concurrent local clients. It also does not make a native library
thread-safe: even with stdio protected upstream, a Tesseract hosting a solver
with process-global state may still require serialization or thread affinity of
its own.

For reproducible normax commands and environment notes, see
[reproducibility.md](reproducibility.md).
