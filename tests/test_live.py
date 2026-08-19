import numpy as np

from nba_prop_quant.live import (
    INGARCH11,
    live_mean_projection,
    regulation_minutes_remaining,
    remaining_minutes_projection,
)


def test_regulation_clock_math():
    assert regulation_minutes_remaining(1, "06:00") == 42.0
    assert regulation_minutes_remaining(4, "02:30") == 2.5


def test_live_projection_respects_current_total():
    model = INGARCH11(omega=0.05, alpha=0.1, beta=0.7)
    mean = live_mean_projection(
        current_count=12,
        remaining_minutes=10.0,
        pregame_rate_per_minute=0.5,
        ingarch=model,
        recent_counts_per_bin=np.array([0, 1, 0, 1], dtype=float),
    )
    assert mean >= 12.0


def test_blowout_reduces_remaining_minutes():
    normal = remaining_minutes_projection(36, 20, score_margin=5)
    blowout = remaining_minutes_projection(36, 20, score_margin=30)
    assert blowout < normal
