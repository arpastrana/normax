import math

import pytest
from blueprints.codes.eurocode.en_1993_1_1_2005.chapter_6_ultimate_limit_state.formula_6_6 import (
    Form6Dot6DesignPlasticResistanceGrossCrossSection,
)
from blueprints.codes.eurocode.en_1993_1_1_2005.chapter_6_ultimate_limit_state.formula_6_7 import (
    Form6Dot7DesignUltimateResistanceNetCrossSection,
)
from blueprints.codes.eurocode.en_1993_1_1_2005.chapter_6_ultimate_limit_state.formula_6_10 import (
    Form6Dot10NcRdClass1And2And3,
)
from blueprints.codes.eurocode.en_1993_1_1_2005.chapter_6_ultimate_limit_state.formula_6_13 import (
    Form6Dot13MCRdClass1And2,
)
from blueprints.codes.eurocode.en_1993_1_1_2005.chapter_6_ultimate_limit_state.formula_6_14 import (
    Form6Dot14MCRdClass3,
)
from blueprints.codes.eurocode.en_1993_1_1_2005.chapter_6_ultimate_limit_state.formula_6_18 import (
    Form6Dot18DesignPlasticShearResistance,
)
from blueprints.codes.eurocode.en_1993_1_1_2005.chapter_6_ultimate_limit_state.formula_6_41 import (
    Form6Dot41BiaxialBendingCheck,
)
from blueprints.codes.eurocode.en_1993_1_1_2005.chapter_6_ultimate_limit_state.formula_6_42 import (
    Form6Dot42LongitudinalStressClass3CrossSections,
)
from blueprints.codes.eurocode.en_1993_1_1_2005.chapter_6_ultimate_limit_state.formula_6_44 import (
    Form6Dot44CombinedCompressionBendingClass4CrossSections,
)
from blueprints.structural_sections.steel.standard_profiles import chs

from normax.ec3.resistance import area_shear
from normax.ec3.resistance import moment_resultant
from normax.ec3.resistance import resistance_bending_elastic
from normax.ec3.resistance import resistance_bending_plastic
from normax.ec3.resistance import resistance_compression
from normax.ec3.resistance import resistance_fracture
from normax.ec3.resistance import resistance_shear
from normax.ec3.resistance import resistance_tension
from normax.ec3.resistance import resistance_yielding
from normax.ec3.resistance import utilization_elastic
from normax.ec3.section import area
from normax.ec3.section import diameter_inner
from normax.ec3.section import modulus_elastic
from normax.ec3.section import modulus_plastic
from normax.ec3.section import radius_of_gyration
from normax.ec3.section import second_moment
from normax.ec3.section import thickness

# Blueprints is LGPL-2.1 and normax is Apache-2.0, so it is a dev dependency
# and appears only here, as a NUMERICAL ORACLE. Its formula classes are called
# and their results compared; its formula source is never read, copied or
# ported. EN 1993-1-1 itself, via references/9780727741721.pdf, is the source
# for every implementation in normax/ec3/.
#
# Blueprints has no section 6.3 member buckling (its chapter jumps from
# formula_6_45 to formula_6_54) and no cross-section classification, so chi,
# lambda_bar and the class limits are checked against the guide by hand
# instead. That gap is exactly what normax fills.
#
# Blueprints works in mm, N and MPa, as we do. Its formula classes subclass
# float, so they drop straight into pytest.approx.

AREAS = [1000.0, 4406.0, 5000.0, 7367.034773, 9310.0, 20000.0]
GRADES = [235.0, 275.0, 355.0, 420.0, 460.0]
ULTIMATES = [360.0, 430.0, 490.0]
PARTIAL_FACTORS = [1.0, 1.1, 1.25]

# Blueprints meshes each profile as a polygon rather than using the closed
# form, so the agreement is limited by its discretization, not by us. The
# error grows as the diameter shrinks: 2.5e-3 at CHS 21.3, 5e-6 at CHS 508.
PROFILE_NAMES = [
    "CHS 21.3x2.3",
    "CHS 244.5x5",
    "CHS 244.5x10",
    "CHS 244.5x25",
    "CHS 508x20",
]
MESH_TOLERANCE = 5e-3


@pytest.fixture(scope="module")
def profiles():
    table = {profile.name: profile for profile in chs.CHS}

    return {name: table[name] for name in PROFILE_NAMES}


@pytest.fixture(scope="module")
def meshed(profiles):
    return {name: profile.section_properties() for name, profile in profiles.items()}


# ---- 6.2.3 and 6.2.4 ---- #


@pytest.mark.parametrize("gamma_m0", PARTIAL_FACTORS)
@pytest.mark.parametrize("f_y", GRADES)
@pytest.mark.parametrize("gross", AREAS)
def test_eq_6_6_agrees(gross, f_y, gamma_m0):
    oracle = Form6Dot6DesignPlasticResistanceGrossCrossSection(
        a=gross,
        f_y=f_y,
        gamma_m0=gamma_m0,
    )

    assert resistance_yielding(gross, f_y, gamma_m0) == pytest.approx(float(oracle))


@pytest.mark.parametrize("gamma_m2", PARTIAL_FACTORS)
@pytest.mark.parametrize("f_u", ULTIMATES)
@pytest.mark.parametrize("net", AREAS)
def test_eq_6_7_agrees(net, f_u, gamma_m2):
    oracle = Form6Dot7DesignUltimateResistanceNetCrossSection(
        a_net=net,
        f_u=f_u,
        gamma_m2=gamma_m2,
    )

    assert resistance_fracture(net, f_u, gamma_m2) == pytest.approx(float(oracle))


@pytest.mark.parametrize("gamma_m0", PARTIAL_FACTORS)
@pytest.mark.parametrize("f_y", GRADES)
@pytest.mark.parametrize("gross", AREAS)
def test_eq_6_10_agrees(gross, f_y, gamma_m0):
    oracle = Form6Dot10NcRdClass1And2And3(a=gross, f_y=f_y, gamma_m0=gamma_m0)

    assert resistance_compression(gross, f_y, gamma_m0) == pytest.approx(float(oracle))


@pytest.mark.parametrize("f_u", ULTIMATES)
@pytest.mark.parametrize("f_y", GRADES)
def test_tension_resistance_min_agrees(f_y, f_u):
    gross, net, gamma_m0, gamma_m2 = 5000.0, 4406.0, 1.0, 1.1

    yielding = Form6Dot6DesignPlasticResistanceGrossCrossSection(
        a=gross,
        f_y=f_y,
        gamma_m0=gamma_m0,
    )
    fracture = Form6Dot7DesignUltimateResistanceNetCrossSection(
        a_net=net,
        f_u=f_u,
        gamma_m2=gamma_m2,
    )
    oracle = min(float(yielding), float(fracture))

    resistance = resistance_tension(gross, net, f_y, f_u, gamma_m0, gamma_m2)

    assert resistance == pytest.approx(oracle)


# ---- The guide's worked examples, double-oracled ---- #


def test_worked_example_6_1_tension_agrees():
    yielding = Form6Dot6DesignPlasticResistanceGrossCrossSection(
        a=5000.0, f_y=265.0, gamma_m0=1.0
    )
    fracture = Form6Dot7DesignUltimateResistanceNetCrossSection(
        a_net=4406.0, f_u=430.0, gamma_m2=1.1
    )

    assert float(yielding) * 1e-3 == pytest.approx(1325.0, rel=1e-3)
    assert float(fracture) * 1e-3 == pytest.approx(1550.0, rel=1e-3)
    assert resistance_yielding(5000.0, 265.0, 1.0) == pytest.approx(float(yielding))
    assert resistance_fracture(4406.0, 430.0, 1.1) == pytest.approx(float(fracture))


def test_worked_example_6_2_compression_agrees():
    oracle = Form6Dot10NcRdClass1And2And3(a=9310.0, f_y=355.0, gamma_m0=1.0)

    assert float(oracle) * 1e-3 == pytest.approx(3305.0, rel=1e-3)
    assert resistance_compression(9310.0, 355.0, 1.0) == pytest.approx(float(oracle))


def test_worked_example_chs_compression_agrees():
    oracle = Form6Dot10NcRdClass1And2And3(a=7367.034773, f_y=355.0, gamma_m0=1.0)

    assert float(oracle) * 1e-3 == pytest.approx(2616.0, rel=1e-2)
    assert resistance_compression(7367.034773, 355.0, 1.0) == pytest.approx(
        float(oracle)
    )


# ---- CHS geometry against the standard profile table ---- #


@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_thickness_agrees(profiles, name):
    profile = profiles[name]
    ratio = profile.outer_diameter / profile.wall_thickness

    assert thickness(profile.outer_diameter, ratio) == pytest.approx(
        profile.wall_thickness
    )


@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_inner_diameter_agrees(profiles, name):
    profile = profiles[name]
    ratio = profile.outer_diameter / profile.wall_thickness

    assert diameter_inner(profile.outer_diameter, ratio) == pytest.approx(
        profile.inner_diameter
    )


@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_area_agrees(profiles, name):
    profile = profiles[name]
    ratio = profile.outer_diameter / profile.wall_thickness

    assert area(profile.outer_diameter, ratio) == pytest.approx(
        profile.area, rel=MESH_TOLERANCE
    )


@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_second_moment_agrees(profiles, meshed, name):
    profile = profiles[name]
    ratio = profile.outer_diameter / profile.wall_thickness

    ours = second_moment(profile.outer_diameter, ratio)

    assert ours == pytest.approx(meshed[name].ixx_c, rel=MESH_TOLERANCE)
    assert ours == pytest.approx(meshed[name].iyy_c, rel=MESH_TOLERANCE)


@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_modulus_elastic_agrees(profiles, meshed, name):
    profile = profiles[name]
    ratio = profile.outer_diameter / profile.wall_thickness

    ours = modulus_elastic(profile.outer_diameter, ratio)

    assert ours == pytest.approx(meshed[name].zxx_plus, rel=MESH_TOLERANCE)


@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_modulus_plastic_agrees(profiles, meshed, name):
    profile = profiles[name]
    ratio = profile.outer_diameter / profile.wall_thickness

    ours = modulus_plastic(profile.outer_diameter, ratio)

    assert ours == pytest.approx(meshed[name].sxx, rel=MESH_TOLERANCE)


@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_radius_of_gyration_agrees(profiles, meshed, name):
    profile = profiles[name]
    ratio = profile.outer_diameter / profile.wall_thickness

    ours = radius_of_gyration(profile.outer_diameter, ratio)

    assert ours == pytest.approx(meshed[name].rx_c, rel=MESH_TOLERANCE)


# ---- 6.2.5 and 6.2.9, brought into scope by the N+M expansion ---- #
#
# Blueprints gained relevance when bending was admitted: it has Eqs. 6.13,
# 6.14, 6.31 to 6.44 and, importantly, 6.41 with the exponents as arguments.
# It still has nothing in section 6.3, so normax/ec3/interaction.py has no
# oracle here at all, and it has no circular-hollow reduced moment either --
# a grep of the whole chapter for "circular" and "1.7" hits only 6.41's
# documentation. That expression is unnumbered in EN, and Blueprints names its
# modules by equation number.

MODULI = [214.2e3, 335.9e3, 414981.0, 550236.0, 2194e3]


@pytest.mark.parametrize("gamma_m0", PARTIAL_FACTORS)
@pytest.mark.parametrize("f_y", GRADES)
@pytest.mark.parametrize("modulus", MODULI)
def test_eq_6_13_agrees(modulus, f_y, gamma_m0):
    oracle = Form6Dot13MCRdClass1And2(w_pl=modulus, f_y=f_y, gamma_m0=gamma_m0)

    assert resistance_bending_plastic(modulus, f_y, gamma_m0) == pytest.approx(
        float(oracle)
    )


@pytest.mark.parametrize("gamma_m0", PARTIAL_FACTORS)
@pytest.mark.parametrize("f_y", GRADES)
@pytest.mark.parametrize("modulus", MODULI)
def test_eq_6_14_agrees(modulus, f_y, gamma_m0):
    oracle = Form6Dot14MCRdClass3(w_el_min=modulus, f_y=f_y, gamma_m0=gamma_m0)

    assert resistance_bending_elastic(modulus, f_y, gamma_m0) == pytest.approx(
        float(oracle)
    )


BIAXIAL = [
    (0.0, 0.0),
    (40e6, 0.0),
    (0.0, 40e6),
    (60e6, 60e6),
    (90e6, 40e6),
    (95e6, 30e6),
    (120e6, 10e6),
    (100e6, 100e6),
]


@pytest.mark.parametrize("m_y, m_z", BIAXIAL)
def test_eq_6_41_verdict_agrees_with_the_resultant(m_y, m_z):
    # For a circular hollow section both exponents are two and the two reduced
    # moment resistances are equal, so Eq. 6.41 collapses exactly to comparing
    # a resultant. This checks that collapse against Blueprints' own
    # implementation of the general equation, which takes alpha and beta as
    # arguments. It returns a bool, so verdicts are compared, not values.
    reduced = 120e6

    oracle = Form6Dot41BiaxialBendingCheck(
        my_ed=m_y,
        m_n_y_rd=reduced,
        mz_ed=m_z,
        m_n_z_rd=reduced,
        alpha=2.0,
        beta=2.0,
    )
    ours = moment_resultant(m_y, m_z) <= reduced

    assert bool(ours) == bool(oracle)


@pytest.mark.parametrize("m_y, m_z", BIAXIAL)
def test_eq_6_41_ratio_agrees_with_the_resultant(m_y, m_z):
    # The verdict alone is a weak comparison, so also check the quantity: the
    # square root of Eq. 6.41's left-hand side is the normalized resultant.
    reduced = 120e6
    equation = (m_y / reduced) ** 2 + (m_z / reduced) ** 2

    assert moment_resultant(m_y, m_z) / reduced == pytest.approx(math.sqrt(equation))


def test_eq_6_41_rejects_a_section_beyond_its_resistance():
    reduced = 120e6

    oracle = Form6Dot41BiaxialBendingCheck(
        my_ed=110e6,
        m_n_y_rd=reduced,
        mz_ed=110e6,
        m_n_z_rd=reduced,
        alpha=2.0,
        beta=2.0,
    )

    assert not bool(oracle)
    assert moment_resultant(110e6, 110e6) > reduced


def test_the_fixture_section_is_in_the_profile_table(profiles):
    profile = profiles["CHS 244.5x10"]

    assert profile.outer_diameter == pytest.approx(244.5)
    assert profile.wall_thickness == pytest.approx(10.0)
    assert profile.area == pytest.approx(
        math.pi * 10.0 * (244.5 - 10.0), rel=MESH_TOLERANCE
    )


# ---- 6.18, the shear resistance ---- #
#
# Added 2026-08-09. The audit that opened this file recorded the chapter as
# jumping from formula_6_45 to formula_6_54, which is true of section 6.3 and
# not of 6.2: Blueprints covers 6.1 through 6.45 almost completely. Eqs. 6.18,
# 6.42, 6.43 and 6.44 were all available and unused.


@pytest.mark.parametrize("area", [1e3, 4690.0, 7367.03, 2e4])
def test_shear_resistance_agrees(area):
    ours = resistance_shear(area_shear(area), 355.0, 1.0)
    oracle = Form6Dot18DesignPlasticShearResistance(
        a_v=float(area_shear(area)), f_y=355.0, gamma_m0=1.0
    )

    assert float(ours) == pytest.approx(float(oracle))


@pytest.mark.parametrize("gamma_m0", [1.0, 1.1, 1.25])
def test_shear_resistance_agrees_across_partial_factors(gamma_m0):
    ours = resistance_shear(4690.0, 355.0, gamma_m0)
    oracle = Form6Dot18DesignPlasticShearResistance(
        a_v=4690.0, f_y=355.0, gamma_m0=gamma_m0
    )

    assert float(ours) == pytest.approx(float(oracle))


# ---- 6.44, the linear-sum reading of the elastic interaction ---- #
#
# Blueprints implements 6.42 with the stress as an INPUT, so it does not say how
# that stress is assembled and cannot settle the reading. It does write out the
# Class 4 analogue, Eq. 6.44, and that is an explicit three-term linear sum.
# With effective properties set to gross ones and no neutral-axis shift, 6.44
# reduces exactly to our linear-sum elastic check, which makes it an oracle for
# that reading. See docs/clauses.md, 6.2.9.2.

ELASTIC_ACTIONS = [
    (-5e5, 4e7, 1.5e7),
    (-5e5, 4e7, 4e7),
    (5e5, 4e7, 0.0),
    (0.0, 6e7, 2e7),
    (-9e5, 8e7, 6e7),
]


@pytest.mark.parametrize("n_ed, m_y_ed, m_z_ed", ELASTIC_ACTIONS)
def test_the_linear_sum_reading_agrees_with_equation_6_44(n_ed, m_y_ed, m_z_ed):
    area, modulus = 7367.03, 414981.0
    ours = utilization_elastic(
        n_ed, m_y_ed, m_z_ed, area, modulus, 355.0, 1.0, resultant=False
    )
    oracle = Form6Dot44CombinedCompressionBendingClass4CrossSections(
        n_ed=abs(n_ed),
        a_eff=area,
        f_y=355.0,
        gamma_m0=1.0,
        m_y_ed=m_y_ed,
        e_ny=0.0,
        w_eff_y_min=modulus,
        m_z_ed=m_z_ed,
        e_nz=0.0,
        w_eff_z_min=modulus,
    )

    assert float(ours) == pytest.approx(
        float(
            oracle._evaluate_lhs(
                n_ed=abs(n_ed),
                a_eff=area,
                f_y=355.0,
                gamma_m0=1.0,
                m_y_ed=m_y_ed,
                e_ny=0.0,
                w_eff_y_min=modulus,
                m_z_ed=m_z_ed,
                e_nz=0.0,
                w_eff_z_min=modulus,
            )
        )
    )


@pytest.mark.parametrize("n_ed, m_y_ed, m_z_ed", ELASTIC_ACTIONS)
def test_the_resultant_reading_never_exceeds_the_linear_sum(n_ed, m_y_ed, m_z_ed):
    # The two readings of 6.42 that docs/clauses.md records. The sum is the
    # conservative one, by at most the square root of two in the moment term.
    area, modulus = 7367.03, 414981.0
    summed = utilization_elastic(
        n_ed, m_y_ed, m_z_ed, area, modulus, 355.0, 1.0, resultant=False
    )
    resultant = utilization_elastic(
        n_ed, m_y_ed, m_z_ed, area, modulus, 355.0, 1.0, resultant=True
    )

    assert float(resultant) <= float(summed) + 1e-12


# ---- 6.42, the stress limit itself ---- #


def test_the_elastic_check_is_the_stress_limit_of_equation_6_42():
    # Blueprints takes the stress as given and only compares it, so this pins
    # the normalization rather than the assembly: our utilization times the
    # design strength is the stress its comparison accepts at exactly unity.
    area, modulus = 7367.03, 414981.0
    ours = utilization_elastic(-5e5, 4e7, 1.5e7, area, modulus, 355.0, 1.0)
    stress = float(ours) * 355.0 / 1.0

    assert bool(
        Form6Dot42LongitudinalStressClass3CrossSections(
            sigma_x_ed=stress, f_y=355.0, gamma_m0=1.0
        )
    ) is (float(ours) <= 1.0)
