# Finding and mitigating a Tesseract concurrency race

Normax exposed a process-wide stdio race in overlapping local Tesseract calls.
The upstream report is
[tesseract-core #709](https://github.com/pasteurlabs/tesseract-core/issues/709).
That issue contains the standalone reproduction and current discussion.

## Defect

Each `LocalClient.run_tesseract` call enters `start_run`. The runtime redirects
file descriptors 1 and 2 with `os.dup2`, saves their targets, then restores them
by hand. File descriptors belong to the process. Overlap is safe only when calls
exit in strict last-in-first-out order. Threads provide no such guarantee.

The reproduction creates an in-memory Tesseract through the public API and
shares one local client across four threads. Affected runtimes can raise
`Bad file descriptor` or restore stdout and stderr to the wrong targets. Normax,
JAX, OpenSees, and Blueprints are not required.

The proposed minimal fix holds a module-level `threading.RLock` for the full
redirection lifetime, including cleanup. It must be reentrant because a nested
run can enter `start_run` twice on one thread. This serializes local endpoint
bodies. Preserving concurrency needs a deeper ownership model for process-wide
descriptors.

## How normax found it

A retired dense-Jacobian path made one reverse crossing per constraint row. Its
calls overlapped while the planar Tesseract hosted OpenSees, which also owns
mutable process-wide state. Native runs could end with `exit 139` and no Python
traceback.

The overlap exposed two separate hazards:

- Tesseract Core could corrupt process stdio.
- Concurrent calls could mutate one OpenSees model during a solve.

The dense Jacobian increased exposure. It caused neither unsafe resource.

## Mitigation

Normax dispatches crossed solvers on one owner thread. Its augmented Lagrangian
also reduces all constraint rows to one scalar before reverse mode. Each stage
therefore needs one cotangent crossing, not one crossing per row.

This protects normax's tested path. It does not make arbitrary concurrent local
clients safe. An upstream stdio lock also cannot make a hosted native library
thread-safe.

See [reproducibility.md](reproducibility.md) for commands and environment notes.
