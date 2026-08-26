import jax.numpy as jnp
import matplotlib
import numpy as np
import pytest
from matplotlib.figure import Figure

from normax.analysis.smax import SmaxAnalyzer
from normax.design import DesignParameters
from normax.design import StructuralDesignPipeline
from normax.design import compute_mass
from normax.extras.nested import Trajectory
from normax.extras.nested import design_envelope
from normax.extras.nested import minimize_bounded
from normax.extras.nested import penalized_mass
from normax.extras.nested import size_design
from normax.extras.replay import figure_trajectory
from normax.extras.replay import load_trajectory
from normax.extras.replay import replay_trajectory
from normax.extras.replay import save_trajectory
from normax.form_finding import FdmFormFinder
from normax.loads import assemble_load_cases
from normax.loads import load_half_span
from normax.loads import load_uniform
from normax.materials import Steel355
from normax.sections import build_section_family
from normax.sizing.ec3 import Ec3Sizer
from normax.structures import build_arch_2d

matplotlib.use("Agg")

# A small arch under 180 kN, in millimeters and newtons.
SPAN = 4_000.0
RISE = 1_200.0
TOTAL_LOAD = 180_000.0
NUM_EDGES = 4

# The diameter the frame is analyzed with before the check has spoken.
SEED = 100.0

# Sharpness of the envelope in the enveloped tests.
SHARPNESS = 50.0

# The floor under the shortest member, and how hard it is held there.
FLOOR_LENGTH = 100.0
FLOOR_SHARPNESS = 20.0
FLOOR_WEIGHT = 0.1

# Bounds wide enough that a short descent is not pinned by them.
BOUNDS = (-500.0, -1.0)
ITERATIONS = 3

# The replay runs the same float64 math through different jit boundaries.
TOLERANCE_REPLAY = 1e-10

# Below this, a utilization is exactly one up to round-off.
TOLERANCE_ROUNDOFF = 1e-9


@pytest.fixture(scope="module")
def structure():
    return build_arch_2d(num_edges=NUM_EDGES, span=SPAN, rise=RISE)


@pytest.fixture(scope="module")
def loads(structure):
    uniform = load_uniform(structure, TOTAL_LOAD)
    lopsided = load_half_span(structure, TOTAL_LOAD)

    return assemble_load_cases([uniform, lopsided])


@pytest.fixture(scope="module")
def pipeline(structure):
    family = build_section_family(Steel355(), 3)

    return StructuralDesignPipeline(
        FdmFormFinder(structure),
        SmaxAnalyzer(structure, family(SEED)),
        Ec3Sizer(structure, family),
    )


@pytest.fixture(scope="module")
def diameters():
    return jnp.full(NUM_EDGES, SEED)


def run_search(pipeline, loads, diameters, sharpness):
    """
    A short bounded descent whose trajectory the replay is measured against.
    """

    def weigh_shape(force_densities):
        design = size_design(
            pipeline, DesignParameters(force_densities, diameters), loads
        )
        sized = design_envelope(design, sharpness)
        mass = compute_mass(sized)

        return penalized_mass(
            mass,
            sized.shape.lengths,
            FLOOR_LENGTH,
            beta=FLOOR_SHARPNESS,
            weight=FLOOR_WEIGHT,
        )

    stamped = 0.0 if sharpness is None else sharpness

    return minimize_bounded(
        weigh_shape,
        jnp.full(NUM_EDGES, -100.0),
        bounds=BOUNDS,
        iterations=ITERATIONS,
        sharpness=stamped,
    )


def recompute_objective(history):
    """
    The penalized objective, re-read off a replayed history step by step.
    """
    weighed = [
        penalized_mass(
            history.mass[step],
            history.lengths[step],
            FLOOR_LENGTH,
            beta=FLOOR_SHARPNESS,
            weight=FLOOR_WEIGHT,
        )
        for step in range(history.mass.shape[0])
    ]

    return jnp.stack(weighed)


def test_artifact_roundtrip(tmp_path):
    trajectory = Trajectory(
        q=jnp.asarray([[-1.0, -2.0], [-3.0, -4.0]]),
        mass=jnp.asarray([5.0, 4.0]),
        beta=jnp.asarray([50.0, 50.0]),
    )
    config_text = "structure:\n  span: 4000.0\n  # a comment that must survive\n"
    path = tmp_path / "search.npz"

    save_trajectory(path, trajectory, config_text)
    artifact = load_trajectory(path)

    assert np.array_equal(artifact.trajectory.q, trajectory.q)
    assert np.array_equal(artifact.trajectory.mass, trajectory.mass)
    assert np.array_equal(artifact.trajectory.beta, trajectory.beta)
    assert artifact.config_text == config_text


def test_replay_matches_recorded_objective(pipeline, loads, diameters):
    found = run_search(pipeline, loads, diameters, SHARPNESS)
    history = replay_trajectory(pipeline, loads, found.trajectory, diameters)
    recomputed = recompute_objective(history)

    gap = jnp.abs(recomputed / found.trajectory.mass - 1.0)
    assert float(jnp.max(gap)) < TOLERANCE_REPLAY


def test_replay_under_hard_envelope(pipeline, loads, diameters):
    found = run_search(pipeline, loads, diameters, None)
    history = replay_trajectory(pipeline, loads, found.trajectory, diameters)
    recomputed = recompute_objective(history)

    gap = jnp.abs(recomputed / found.trajectory.mass - 1.0)
    assert float(jnp.max(gap)) < TOLERANCE_REPLAY

    # The hard envelope leaves the governing case exactly satisfied.
    hardest = jnp.max(history.utilization, axis=1)
    assert float(jnp.max(jnp.abs(hardest - 1.0))) < TOLERANCE_ROUNDOFF


def test_mixed_sharpness_is_refused(pipeline, loads, diameters):
    trajectory = Trajectory(
        q=jnp.full((2, NUM_EDGES), -100.0),
        mass=jnp.zeros(2),
        beta=jnp.asarray([0.0, SHARPNESS]),
    )

    with pytest.raises(ValueError, match="mixes"):
        replay_trajectory(pipeline, loads, trajectory, diameters)


def test_history_invariants(pipeline, loads, diameters):
    found = run_search(pipeline, loads, diameters, SHARPNESS)
    history = replay_trajectory(pipeline, loads, found.trajectory, diameters)

    # The smooth envelope never understates a size, so nothing is overworked.
    assert float(jnp.max(history.utilization)) < 1.0 + TOLERANCE_ROUNDOFF

    # The governing column agrees with the sizes the cases demanded on their own.
    design = size_design(
        pipeline, DesignParameters(found.trajectory.q[0], diameters), loads
    )
    expected = jnp.argmax(design.sizes.sections.diameter, axis=0)
    assert np.array_equal(history.governing[0], expected)


def test_the_trajectory_figure_draws_one_curve_per_run():
    stamped = Trajectory(
        q=jnp.zeros((3, 2)),
        mass=jnp.asarray([3.0, 2.0, 1.0]),
        beta=jnp.asarray([1.0, 2.0, 4.0]),
    )
    unstamped = stamped._replace(beta=jnp.zeros(3))

    colored = figure_trajectory([stamped, stamped], concatenated=True)
    plain = figure_trajectory([unstamped], titles=("walk",))

    assert isinstance(colored, Figure)
    assert len(colored.axes[0].lines) >= 2
    assert plain.axes[0].get_legend().get_texts()[0].get_text() == "walk"
