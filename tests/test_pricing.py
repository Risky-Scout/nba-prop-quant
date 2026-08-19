from nba_prop_quant.pricing import (
    american_implied_probability,
    devig_two_way,
    fair_american,
)


def test_american_odds_math():
    assert abs(american_implied_probability(-110) - 0.5238095238) < 1e-8
    over, under = devig_two_way(-110, -110)
    assert abs(over - 0.5) < 1e-12
    assert abs(under - 0.5) < 1e-12
    assert fair_american(0.5) == -100
