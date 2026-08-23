from nba_prop_quant.fair_price_feed import (
    stable_event_seed,
)


def test_stable_event_seed_exact_formula() -> None:
    prop_type = "rebounds_assists"
    prop_code = sum(
        (index + 1) * ord(char)
        for index, char in enumerate(prop_type)
    )
    expected = int(
        (
            73
            + 1_000_003 * 123456
            + 9_176 * 237
            + 37 * prop_code
        )
        % (2**32 - 1)
    )
    assert stable_event_seed(
        123456,
        237,
        prop_type,
        73,
    ) == expected
