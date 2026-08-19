import numpy as np

from nba_prop_quant.ctmc import (
    ctmc_over_under_push,
    transition_distribution_matrix,
    transition_distribution_poisson,
)


def test_ctmc_matrix_matches_poisson_closed_form():
    matrix = transition_distribution_matrix(
        intensity_per_minute=0.4,
        remaining_minutes=5.0,
        max_increment=30,
    )
    closed = transition_distribution_poisson(
        intensity_per_minute=0.4,
        remaining_minutes=5.0,
        max_increment=30,
    )
    assert np.allclose(matrix[:-1], closed[:-1], atol=1e-10)
    assert abs(matrix.sum() - 1.0) < 1e-10


def test_ctmc_live_line_probabilities_sum_to_one():
    over, under, push = ctmc_over_under_push(
        current_count=18,
        line=22.0,
        expected_remaining_count=5.0,
    )
    assert abs(over + under + push - 1.0) < 1e-12
