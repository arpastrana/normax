import jax
import jax.numpy as jnp
import numpy as np
import pytest

from normax.optimization import annealing_schedule
from normax.optimization import minimize_bounded
from normax.optimization import optimize_annealed
from normax.optimization import penalized_mass
from normax.optimization import shortest_member

# A bowl whose minimum is known exactly, so the driver is tested against
# arithmetic rather than against the pipeline it usually drives.
CENTER = jnp.asarray([-2.0, -3.0, -1.5])
START = jnp.asarray([-8.0, -8.0, -8.0])
BOUNDS = (-10.0, -0.1)


def bowl(q, beta=1.0):
    return beta * jnp.sum((q - CENTER) ** 2)


def wall(q, beta=1.0):
    # Minimized outside the box, so the answer sits on the bound and the bound
    # is what has to hold it there.
    return beta * jnp.sum((q + 50.0) ** 2)


# --------------------------------------------------------------------------- #
# The annealing schedule
# --------------------------------------------------------------------------- #
def test_the_schedule_starts_and_ends_where_it_is_told():
    schedule = annealing_schedule(10.0, 500.0, 5)

    assert float(schedule[0]) == pytest.approx(10.0)
    assert float(schedule[-1]) == pytest.approx(500.0)
    assert schedule.shape == (5,)


def test_the_schedule_rises_by_a_constant_ratio():
    # Geometric, because what the envelope gives away falls as the reciprocal of
    # the sharpness, so equal ratios buy equal fractions of what is left.
    schedule = np.asarray(annealing_schedule(10.0, 500.0, 6))
    ratios = schedule[1:] / schedule[:-1]

    assert np.allclose(ratios, ratios[0])


def test_a_schedule_of_one_round_is_just_its_start():
    assert np.asarray(annealing_schedule(7.0, 7.0, 1)) == pytest.approx(7.0)


@pytest.mark.parametrize("start,stop", [(0.0, 100.0), (10.0, -1.0), (-5.0, 5.0)])
def test_the_schedule_refuses_a_sharpness_that_is_not_positive(start, stop):
    with pytest.raises(ValueError, match="positive"):
        annealing_schedule(start, stop, 4)


def test_the_schedule_refuses_to_have_no_rounds():
    with pytest.raises(ValueError, match="rounds"):
        annealing_schedule(10.0, 500.0, 0)


# --------------------------------------------------------------------------- #
# One descent
# --------------------------------------------------------------------------- #
def test_the_descent_finds_a_minimum_it_can_reach():
    walked = minimize_bounded(bowl, START, bounds=BOUNDS, iterations=50)

    assert np.allclose(np.asarray(walked.q[-1]), np.asarray(CENTER), atol=1e-6)


def test_the_descent_never_goes_uphill():
    walked = minimize_bounded(bowl, START, bounds=BOUNDS, iterations=50)
    masses = np.asarray(walked.mass)

    assert np.all(np.diff(masses) <= 1e-12)


def test_the_descent_records_the_starting_point_first():
    # The trajectory is what a figure plots, so it has to begin where the search
    # began rather than after its first step.
    walked = minimize_bounded(bowl, START, bounds=BOUNDS, iterations=5)

    assert np.allclose(np.asarray(walked.q[0]), np.asarray(START))
    assert float(walked.mass[0]) == pytest.approx(float(bowl(START)))


def test_the_recorded_mass_belongs_to_the_recorded_iterate():
    walked = minimize_bounded(bowl, START, bounds=BOUNDS, iterations=50)

    for step, mass in zip(walked.q, walked.mass):
        assert float(mass) == pytest.approx(float(bowl(step)), rel=1e-12)


def test_the_bounds_hold_every_iterate():
    walked = minimize_bounded(wall, START, bounds=BOUNDS, iterations=30)
    visited = np.asarray(walked.q)

    assert np.all(visited >= BOUNDS[0] - 1e-12)
    assert np.all(visited <= BOUNDS[1] + 1e-12)


def test_a_minimum_outside_the_box_lands_on_the_bound():
    walked = minimize_bounded(wall, START, bounds=BOUNDS, iterations=30)

    assert np.allclose(np.asarray(walked.q[-1]), BOUNDS[0])


def test_spending_no_iterations_leaves_the_starting_point():
    # L-BFGS-B takes a step before it honours a limit of zero, and that step is
    # a clipped trial point rather than an improvement, so the driver refuses to
    # report it as an answer.
    walked = minimize_bounded(bowl, START, bounds=BOUNDS, iterations=0)

    assert np.allclose(np.asarray(walked.q[-1]), np.asarray(START))
    assert walked.q.shape[0] == 1


def test_the_last_iterate_is_the_answer_the_search_returned():
    # The point reported last and the point kept as best differ whenever the
    # search stops part-way through a line search, and callers read the last.
    walked = minimize_bounded(bowl, START, bounds=BOUNDS, iterations=2)
    masses = np.asarray(walked.mass)

    assert float(masses[-1]) == min(masses)


# --------------------------------------------------------------------------- #
# The annealed descent
# --------------------------------------------------------------------------- #
def test_the_annealed_descent_reports_every_round():
    schedule = annealing_schedule(1.0, 4.0, 3)
    walked = optimize_annealed(bowl, START, schedule, bounds=BOUNDS, iterations=10)

    assert set(np.unique(np.asarray(walked.beta))) == set(
        float(sharpness) for sharpness in schedule
    )


def test_each_round_starts_where_the_last_one_stopped():
    # Warm starting is the whole point of a schedule: a sharper round refines a
    # design rather than rediscovering it.
    schedule = annealing_schedule(1.0, 4.0, 3)
    walked = optimize_annealed(bowl, START, schedule, bounds=BOUNDS, iterations=10)

    betas = np.asarray(walked.beta)
    visited = np.asarray(walked.q)
    boundaries = np.flatnonzero(np.diff(betas))

    for edge in boundaries:
        assert np.allclose(visited[edge], visited[edge + 1])


def test_the_annealed_descent_reaches_the_same_minimum():
    schedule = annealing_schedule(1.0, 4.0, 3)
    walked = optimize_annealed(bowl, START, schedule, bounds=BOUNDS, iterations=25)

    assert np.allclose(np.asarray(walked.q[-1]), np.asarray(CENTER), atol=1e-6)


def test_the_sharpness_reaches_the_objective():
    # A round has to be taken under its own sharpness, or annealing does nothing
    # at all and the schedule is decoration. Read off the objective rather than
    # inside it: the sharpness arrives traced, so only its effect is concrete.
    walked = optimize_annealed(
        bowl, START, annealing_schedule(2.0, 8.0, 3), bounds=BOUNDS, iterations=3
    )

    assert sorted(set(np.asarray(walked.beta).tolist())) == pytest.approx(
        [2.0, 4.0, 8.0]
    )
    for q, beta, mass in zip(walked.q, walked.beta, walked.mass):
        assert float(mass) == pytest.approx(float(bowl(q, beta)), rel=1e-12)


def test_the_schedule_traces_the_objective_once():
    # The sharpness parameterizes one compiled program rather than selecting
    # between one per round. Capturing it again would compile the pipeline as
    # many times as the schedule is long, which is most of what a descent costs.
    traces = 0

    def counted(q, beta):
        nonlocal traces
        traces += 1
        return bowl(q, beta)

    optimize_annealed(
        counted, START, annealing_schedule(2.0, 8.0, 5), bounds=BOUNDS, iterations=3
    )

    assert traces == 1


def test_a_schedule_of_plain_floats_also_traces_once():
    # A float leaf is static under `eqx.filter_jit`, so a sequence that is not
    # converted to an array compiles a program per round without saying so.
    traces = 0

    def counted(q, beta):
        nonlocal traces
        traces += 1
        return bowl(q, beta)

    optimize_annealed(counted, START, [2.0, 4.0, 8.0], bounds=BOUNDS, iterations=3)

    assert traces == 1


# --------------------------------------------------------------------------- #
# The length floor
# --------------------------------------------------------------------------- #
LENGTHS = jnp.asarray([500.0, 700.0, 300.0, 900.0])


def test_the_smooth_minimum_never_overstates_the_shortest():
    # Understating is the safe direction for a floor: a constraint built on it
    # bites slightly early rather than slightly late.
    for beta in (5.0, 20.0, 100.0):
        assert float(shortest_member(LENGTHS, beta)) <= float(jnp.min(LENGTHS))


def test_the_smooth_minimum_approaches_the_shortest_as_it_sharpens():
    blunt = float(shortest_member(LENGTHS, 5.0))
    sharp = float(shortest_member(LENGTHS, 200.0))

    assert blunt < sharp <= float(jnp.min(LENGTHS))
    assert sharp == pytest.approx(float(jnp.min(LENGTHS)), rel=1e-6)


def test_the_smooth_minimum_respects_its_bound():
    # It falls below the true shortest by at most the member count raised to the
    # reciprocal of the sharpness, the same bound the envelope carries.
    for beta in (5.0, 20.0, 100.0):
        ratio = float(jnp.min(LENGTHS)) / float(shortest_member(LENGTHS, beta))

        assert 1.0 <= ratio <= float(len(LENGTHS) ** (1.0 / beta)) + 1e-12


def test_the_smooth_minimum_scales_with_the_structure():
    # Taken in the logarithm, so the sharpness means the same thing whatever the
    # structure is measured in.
    assert float(shortest_member(LENGTHS * 1000.0, 20.0)) == pytest.approx(
        1000.0 * float(shortest_member(LENGTHS, 20.0))
    )


def test_a_design_clear_of_the_floor_is_not_penalized():
    mass = jnp.asarray(0.05)

    assert float(penalized_mass(mass, LENGTHS, 200.0, beta=50.0, weight=10.0)) == (
        pytest.approx(float(mass))
    )


def test_a_design_below_the_floor_is_penalized():
    mass = jnp.asarray(0.05)
    inflated = float(penalized_mass(mass, LENGTHS, 600.0, beta=50.0, weight=10.0))

    assert inflated > float(mass)


def test_the_penalty_grows_as_the_shortest_member_shrinks():
    mass = jnp.asarray(0.05)
    lengths = [LENGTHS.at[2].set(value) for value in (280.0, 200.0, 120.0)]
    inflated = [
        float(penalized_mass(mass, load_case, 600.0, beta=50.0, weight=10.0))
        for load_case in lengths
    ]

    assert inflated[0] < inflated[1] < inflated[2]


def test_the_penalty_is_flat_where_it_starts():
    # Squared, so the objective is not kinked at the floor and a search may
    # approach it from either side.
    mass = jnp.asarray(0.05)
    floor = float(shortest_member(LENGTHS, 50.0))

    slope = jax.grad(lambda x: penalized_mass(mass, x, floor, beta=50.0, weight=10.0))(
        LENGTHS
    )

    assert np.allclose(np.asarray(slope), 0.0, atol=1e-9)


def test_the_penalty_is_a_fraction_and_not_a_mass():
    # Multiplicative and reading a ratio, so it needs no mass scale and means
    # the same thing on a structure of any size.
    light = jnp.asarray(0.05)
    heavy = jnp.asarray(5.0)

    ratio_light = (
        float(penalized_mass(light, LENGTHS, 600.0, beta=50.0, weight=10.0)) / 0.05
    )
    ratio_heavy = (
        float(penalized_mass(heavy, LENGTHS, 600.0, beta=50.0, weight=10.0)) / 5.0
    )

    assert ratio_light == pytest.approx(ratio_heavy)


def test_the_penalty_gradient_reaches_only_the_shortest_members():
    mass = jnp.asarray(0.05)

    slope = np.asarray(
        jax.grad(lambda x: penalized_mass(mass, x, 600.0, beta=50.0, weight=10.0))(
            LENGTHS
        )
    )

    assert np.all(np.isfinite(slope))
    assert abs(slope[2]) > abs(slope[3])
