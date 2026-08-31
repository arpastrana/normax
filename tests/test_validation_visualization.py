# SPDX-License-Identifier: Apache-2.0
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from normax.visualization import draw_code_validation
from normax.visualization import draw_pipeline_validation
from normax.visualization import draw_pynite_validation


def assert_paper_theme(figure):
    """Every validation panel is boxed and typeset in Computer Modern."""
    for ax in figure.axes:
        assert all(spine.get_visible() for spine in ax.spines.values())
        assert ax.title.get_fontfamily()[0] == "cmr10"


def test_pipeline_validation_is_a_pure_three_panel_figure():
    figure = draw_pipeline_validation(
        reverse=(0.9, -0.4, 0.2, -0.1),
        central=(0.900001, -0.399999, 0.200001, -0.100001),
        parameter_kinds=("force density", "force density", "diameter", "diameter"),
        error_labels=("density", "diameter", "complete"),
        errors=(1e-8, 2e-8, 2e-8),
        tolerances=(1e-6, 1e-6, 1e-6),
        timing_labels=("forward", "reverse", "central FD"),
        timing_seconds=(0.01, 0.05, 0.4),
    )

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 3
    assert_paper_theme(figure)
    plt.close(figure)


def test_code_validation_is_a_pure_three_panel_figure():
    figure = draw_code_validation(
        force_errors=(1e-10, 2e-10, 4e-10),
        moment_errors=(2e-9, 1e-9, 3e-9),
        case_labels=("1", "2", "3"),
        sharpness=(5.0, 25.0, 100.0),
        envelope_excess=(0.2, 0.01, 0.001),
        envelope_bound=(0.22, 0.04, 0.01),
        check_labels=("route", "difference", "branch"),
        check_errors=(1e-14, 1e-9, 1e-10),
        check_tolerances=(1e-12, 1e-7, 1e-6),
    )

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 3
    assert_paper_theme(figure)
    plt.close(figure)


def test_pynite_validation_is_a_pure_three_panel_figure():
    figure = draw_pynite_validation(
        steps=(1e-5, 1e-4, 1e-3, 1e-2),
        node_errors=(1e-5, 1e-7, 1e-9, 1e-8),
        diameter_errors=(1e-4, 1e-6, 2e-9, 1e-8),
        route_labels=("nodes", "diameters", "boundary"),
        route_errors=(1e-9, 2e-9, 1e-14),
        route_tolerances=(1e-8, 1e-8, 1e-11),
        timing_labels=("forward", "3 cases", "adjoint", "central FD"),
        timing_seconds=(0.04, 0.08, 0.05, 100.0),
        finite_difference_measured=True,
    )

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 3
    assert_paper_theme(figure)
    plt.close(figure)
