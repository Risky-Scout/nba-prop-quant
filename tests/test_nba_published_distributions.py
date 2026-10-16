"""The published distributions: are they the ones the prices came out of?

Every assertion here is against a slate the frozen serving path actually
priced, with the frozen bundle's own marginals, copula and dependence policy.
Nothing is fixtured: the priced parquet is ``scripts/15_price_markets.py``'s
output, the document is ``ops/nba_distribution_publication.py``'s output over
that parquet, and what is compared is two numbers that production produced.

The suite is in four parts.

WHAT A DISTRIBUTION HAS TO BE
-----------------------------

Integer outcomes from zero upward, no repeats, ascending, every probability
finite and non-negative, and the listed mass plus the tail bucket summing to
one. A record failing any of those is not a thinner distribution, it is not a
distribution, and the renderer would draw it anyway.

WHETHER IT IS THE PRICE'S OWN DISTRIBUTION
------------------------------------------

Three claims, and they are not the same claim.

Mass is arithmetic internal to the record. The *mean* is a claim about the
pricing path: the record's mean must be the mean the line was priced against,
which for a single stat is the published ``model_projection``. And the *line
probabilities* are the claim that matters most -- summing the published
record's mass around a line has to reproduce the probability the pricing
engine computed for that line, for every line on the slate.

That third claim is deliberately checked against the uncalibrated
distributional probabilities, which is what a distribution determines. The
probability a reader is shown for the wager is the calibrated one, and the
calibration is a monotone map applied to a scalar: it does not define a
distribution, so forcing the record to integrate to it would mean publishing a
distribution the model never held. The panel says so in words; this says so in
numbers.

WHICH ENGINE, AND SAYING SO
---------------------------

``pricing.price_combo_lines`` prices a combination analytically when its
frozen dependence coefficient is zero and from seeded draws when it is not.
Four of the frozen policy's combinations are the former and
``rebounds_assists`` is the latter, so both branches are exercised on a real
slate. A simulated record is labelled as simulated, and the parity assertion
for it is equality against the pricing engine's own function over the very
same draws -- same seed, same count -- rather than agreement between two
independently seeded runs.

WHAT IS NOT COVERED, AND WHY
----------------------------

``stocks``. The frozen market-probability calibration policy in the installed
bundle does not carry it, so the frozen pricing path drops every quote on it
and production publishes no ``stocks`` price. There is no price for a
distribution to sit beside, and manufacturing one would be publishing a
forecast for a wager the model is not offering an opinion on. A test asserts
that this is the reason rather than an oversight.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]

for _relative in ("ops", "src", "tests"):
    if str(PROJECT / _relative) not in sys.path:
        sys.path.insert(0, str(PROJECT / _relative))

import nba_distribution_publication as stage  # noqa: E402
import nba_distribution_slate_fixture as coverage  # noqa: E402

from nba_prop_quant import pricing  # noqa: E402

#: How far a published number may sit from the pricing engine's own, and why.
#:
#: ``MASS`` is floating-point summation over a few dozen terms.
#:
#: ``LINE`` and ``MEAN`` are both the frozen marginal routine's own truncation:
#: ``_marginal_pmf_for_row`` lists outcomes up to the 1-1e-10 quantile and
#: renormalises, so its PMF differs from the closed form it came from by about
#: that much, and every number derived from it inherits exactly that. These are
#: not tolerances on an approximation introduced here -- there is no
#: approximation introduced here -- they are the width of the frozen engine's
#: own numerics.
MASS_TOLERANCE = 1e-12
LINE_TOLERANCE = 1e-9
MEAN_TOLERANCE = 1e-6

#: Every prop type the frozen policy covers, which is what the slate has to
#: carry for this suite to be about the contract rather than about a subset.
EXPECTED_SINGLE: tuple[str, ...] = (
    "assists",
    "blocks",
    "points",
    "rebounds",
    "steals",
    "threes",
)

EXPECTED_COMBO: tuple[str, ...] = (
    "points_assists",
    "points_rebounds",
    "points_rebounds_assists",
    "rebounds_assists",
)


# ======================================================================
# the slate
# ======================================================================


@pytest.fixture(scope="module")
def slate(tmp_path_factory) -> dict:
    """One real served slate and the document derived from it.

    Module-scoped because serving it runs the real prediction and pricing
    scripts over a frozen bundle, and every test below is a different question
    about the same run.
    """
    bundle = coverage.skip_without_bundle()

    return coverage.serve(bundle, tmp_path_factory.mktemp("distributions") / "run")


@pytest.fixture(scope="module")
def records(slate: dict) -> dict:
    return slate["document"]["distributions"]


def published(record) -> stage.PublishedDistribution:
    """A published record read back into the dataclass that wrote it.

    So the arithmetic below runs on the published numbers rather than on the
    objects that produced them. A record that lost a field in serialisation
    fails here.
    """
    return stage.PublishedDistribution(
        kind=record["distribution_type"],
        support=tuple(record["distribution_support"]),
        probability=tuple(record["distribution_probability"]),
        tail_outcome=record["distribution_tail_outcome"],
        tail_probability=record["distribution_tail_probability"],
        mean=record["distribution_mean"],
        most_likely_low=record["distribution_most_likely_low"],
        most_likely_high=record["distribution_most_likely_high"],
        most_likely_mass=record["distribution_most_likely_mass"],
        dependence_lambda=record["dependence_lambda"],
        simulation_draws=record["simulation_draws"],
        simulation_seed=record["simulation_seed"],
    )


def group_of(priced: pd.DataFrame, key: str) -> pd.DataFrame:
    game_id, player_id, prop_type = key.split("|")

    return priced[
        (priced["game_id"] == int(game_id))
        & (priced["player_id"] == int(player_id))
        & (priced["prop_type"] == prop_type)
    ]


# ======================================================================
# 1. the slate is worth asserting on
# ======================================================================


def test_1_the_slate_carries_every_prop_the_frozen_bundle_prices(slate, records):
    """Every covered prop type is priced, and has a record.

    Asserted before anything else: a parity suite over a slate missing half
    the prop types proves the half it has and says nothing about the rest, and
    nothing in the output distinguishes that from a full pass.
    """
    priced = set(slate["priced"]["prop_type"].unique())

    for prop_type in (*EXPECTED_SINGLE, *EXPECTED_COMBO):
        assert prop_type in priced, prop_type

    keyed = {key.split("|")[2] for key in records}

    assert keyed == priced

    # One record per player, stat and game -- not per quote. Several
    # sportsbooks pricing one wager is several prices out of one forecast.
    wagers = slate["priced"].groupby(
        list(stage.DISTRIBUTION_KEY_FIELDS)
    ).ngroups

    assert len(records) == wagers > len(EXPECTED_SINGLE)


def test_2_stocks_is_absent_because_the_frozen_policy_does_not_cover_it(slate):
    """The one combination with no record, and the reason it has none.

    ``pricing`` supports it; the frozen calibration policy does not carry it;
    the pricing path prices the intersection. So production publishes no
    ``stocks`` price, and a distribution published beside no price would be a
    forecast for a wager nobody is being offered.
    """
    policy = json.loads(
        (
            slate["model_dir"] / "market_probability_calibration_policy.json"
        ).read_text(encoding="utf-8")
    )

    for prop_type in coverage.UNPRICEABLE:
        assert prop_type in pricing.COMBO_COMPONENTS or (
            prop_type in pricing.PROP_TO_TARGET
        )

        assert prop_type not in policy["props"]

        assert prop_type not in set(slate["priced"]["prop_type"].unique())

    # And it was quoted, so its absence is the pricing path's decision rather
    # than this fixture never having offered it.
    assert "stocks" not in {prop for _, prop, _ in coverage.ADDED_PROPS}


# ======================================================================
# 3-8: what every record has to be
# ======================================================================


def test_3_every_support_is_whole_non_negative_ascending_and_unique(records):
    """Case 1, 3. Outcomes of a counting stat, in order, each once."""
    for key, record in records.items():
        support = record["distribution_support"]

        assert support, key

        assert all(isinstance(outcome, int) for outcome in support), key

        assert all(outcome >= 0 for outcome in support), key

        assert support == sorted(support), key

        assert len(set(support)) == len(support), key

        assert support[0] == 0, key

        assert support == list(range(len(support))), key


def test_4_every_probability_is_finite_and_non_negative(records):
    """Case 2. A probability that is not a number is not a thinner forecast."""
    for key, record in records.items():
        probability = record["distribution_probability"]

        assert len(probability) == len(record["distribution_support"]), key

        for value in (*probability, record["distribution_tail_probability"]):
            assert isinstance(value, float), key

            assert np.isfinite(value), key

            assert value >= 0.0, key


def test_5_the_listed_mass_and_the_tail_bucket_sum_to_one(records):
    """Case 4. Nothing is discarded; the remainder is stated, not dropped."""
    for key, record in records.items():
        total = sum(record["distribution_probability"])

        total += record["distribution_tail_probability"]

        assert abs(total - 1.0) <= MASS_TOLERANCE, (key, total)


def test_6_the_tail_bucket_sits_above_the_last_listed_outcome(records):
    """Case 4's other half. ``N+`` means above ``N-1``, and says which N."""
    for key, record in records.items():
        outcome = record["distribution_tail_outcome"]

        if outcome is None:
            # A distribution whose whole support is listed has no remainder,
            # so the bucket would be an outcome with no mass behind it.
            assert record["distribution_tail_probability"] == 0.0, key

            continue

        assert outcome == record["distribution_support"][-1] + 1, key

        assert record["distribution_tail_label"] == f"{outcome}+", key

        assert record["distribution_tail_probability"] > 0.0, key


def test_7_the_highlighted_range_is_the_narrowest_run_holding_the_target(records):
    """Case 10. A stated rule over the published numbers, not a judgement.

    The range is recomputed here from the record's own listed probabilities
    and bucket, by the rule the panel states, and required to be the published
    pair. No word like "likely" is involved and none is published: it is an
    interval and the mass it holds.
    """
    for key, record in records.items():
        weights = list(record["distribution_probability"])

        if record["distribution_tail_outcome"] is not None:
            weights.append(record["distribution_tail_probability"])

        low, high, held = stage.most_likely_range(weights)

        assert record["distribution_most_likely_low"] == low, key

        assert record["distribution_most_likely_high"] == high, key

        assert record["distribution_most_likely_mass"] == pytest.approx(held)

        assert low <= high

        assert held >= stage.MOST_LIKELY_MASS or (
            low == 0 and high == len(weights) - 1
        ), key

        assert record["distribution_most_likely_target_mass"] == (
            stage.MOST_LIKELY_MASS
        )


def test_8_a_record_says_which_engine_produced_it_and_is_labelled_for_it(records):
    """Cases 9 and 16. A finite set of draws is not an analytical PMF.

    Both branches are on this slate, so this is not a test of a constant: the
    analytical records carry the plain label, the simulated ones say they are
    simulation-based, and nothing anywhere is called an exact PMF.
    """
    kinds = {record["distribution_type"] for record in records.values()}

    assert kinds == {
        stage.KIND_MARGINAL,
        stage.KIND_CONVOLUTION,
        stage.KIND_SIMULATION,
    }

    for key, record in records.items():
        kind = record["distribution_type"]

        analytical = kind in stage.ANALYTICAL_KINDS

        assert record["distribution_analytical"] is analytical, key

        assert record["distribution_label"] == (
            stage.ANALYTICAL_LABEL if analytical else stage.SIMULATION_LABEL
        ), key

        assert "exact" not in record["distribution_label"].lower(), key

        if analytical:
            assert record["simulation_draws"] is None, key

            assert record["simulation_seed"] is None, key
        else:
            assert record["simulation_draws"] == (
                stage.DEFAULT_COMBO_SIMULATIONS
            ), key

            assert isinstance(record["simulation_seed"], int), key


# ======================================================================
# 9-12: whether it is the price's own distribution
# ======================================================================


def test_9_a_single_stats_record_is_the_frozen_marginals_own_pmf(slate, records):
    """Case 7. Not merely close to it: the same array, element for element.

    The record is rebuilt by calling the pricing engine's own routine on the
    same row and the same mean, and required to be identical. A second
    implementation of a negative binomial that agreed to six places would
    fail here, which is the point.
    """
    import joblib

    marginals = joblib.load(slate["model_dir"] / "marginals.joblib")

    checked = 0

    for key, record in records.items():
        prop_type = key.split("|")[2]

        if prop_type not in pricing.PROP_TO_TARGET:
            continue

        group = group_of(slate["priced"], key)

        row = group.iloc[0]

        mu = float(row[stage.PROJECTION_COLUMN])

        engine = pricing._marginal_pmf_for_row(
            marginals[pricing.PROP_TO_TARGET[prop_type]], mu, row
        )

        listed = list(record["distribution_probability"])

        assert listed == [float(value) for value in engine[: len(listed)]], key

        assert record["distribution_tail_probability"] == pytest.approx(
            float(engine[len(listed) :].sum()), abs=0.0, rel=1e-12
        ), key

        checked += 1

    assert checked >= len(EXPECTED_SINGLE)


def test_10_a_records_mean_is_the_mean_the_line_was_priced_against(slate, records):
    """Case 5. ``sum(k * P(X = k))`` over the complete distribution.

    Taken before the tail is bucketed and published as its own field, because
    the bucket states how much mass is above an outcome and not where in the
    tail it sits -- a mean recomputed from the published rows would be the
    mean of the display. The published mean is required to be the canonical
    projection, which is the number the panel shows as Model Projection.
    """
    checked = 0

    for key, record in records.items():
        prop_type = key.split("|")[2]

        if prop_type not in pricing.PROP_TO_TARGET:
            continue

        projection = float(
            group_of(slate["priced"], key).iloc[0][stage.PROJECTION_COLUMN]
        )

        assert record["distribution_mean"] == pytest.approx(
            projection, abs=MEAN_TOLERANCE
        ), key

        checked += 1

    assert checked >= len(EXPECTED_SINGLE)


def test_11_summing_the_record_reproduces_the_priced_line_probability(
    slate, records
):
    """Case 6, and case 17 for the combinations. Every line on the slate.

    The published record's mass is summed around each quoted line by the
    pricing engine's own ``_pmf_over_under_push``, and the three numbers are
    required to be the uncalibrated ``p_over`` / ``p_under`` / ``p_push`` the
    pricing run wrote into that row. This is the assertion that makes the
    panel a view of the price rather than a second opinion beside it.
    """
    checked = 0

    for key, record in records.items():
        distribution = published(record)

        for _, row in group_of(slate["priced"], key).iterrows():
            over, under, push = stage.line_probabilities(
                distribution, float(row["line_value"])
            )

            assert over == pytest.approx(
                float(row["p_over"]), abs=LINE_TOLERANCE
            ), key

            assert under == pytest.approx(
                float(row["p_under"]), abs=LINE_TOLERANCE
            ), key

            assert push == pytest.approx(
                float(row["p_push"]), abs=LINE_TOLERANCE
            ), key

            checked += 1

    assert checked == len(slate["priced"])


def test_12_the_record_is_not_the_calibrated_probability_and_does_not_claim_to_be(
    slate, records
):
    """Case 6's companion. The two numbers differ, and the panel says why.

    A record forced to integrate to the calibrated probability would be a
    distribution the model never held: the calibration is a scalar map fitted
    on realised outcomes, and nothing in it defines a coherent calibrated PMF
    over the support. So the distribution reproduces the raw probability and
    the panel discloses that the displayed wager probability may include
    calibration.
    """
    import wizardofodds_player_props_page as page

    assert "calibration" in page.DISTRIBUTION_NOTE.lower()

    differing = 0

    for key, record in records.items():
        distribution = published(record)

        for _, row in group_of(slate["priced"], key).iterrows():
            raw = float(row["q_over_nonpush"])

            calibrated = float(row["calibrated_q_over_nonpush"])

            over, under, _ = stage.line_probabilities(
                distribution, float(row["line_value"])
            )

            nonpush = over + under

            assert over / nonpush == pytest.approx(raw, abs=LINE_TOLERANCE), key

            if abs(raw - calibrated) > 1e-6:
                differing += 1

    # Not a hypothetical: on this slate the frozen calibration does move the
    # number, which is why the two cannot both be what the record integrates
    # to and why the disclosure is required rather than decorative.
    assert differing > 0


# ======================================================================
# 13-18: the combinations
# ======================================================================


def test_13_the_frozen_policy_decides_which_engine_prices_a_combination(
    slate, records
):
    """Case 2 of the combination requirements. The branch is not taken here.

    ``pricing.price_combo_lines`` compares the frozen dependence coefficient
    against its own threshold; this reads the same coefficient from the same
    file and follows it. So the assertion is that the record's engine is the
    one the policy implies, for every combination on the slate.
    """
    policy = json.loads(
        (slate["model_dir"] / "combo_dependence_policy.json").read_text(
            encoding="utf-8"
        )
    )

    seen: dict[str, str] = {}

    for key, record in records.items():
        prop_type = key.split("|")[2]

        if prop_type not in pricing.COMBO_COMPONENTS:
            continue

        frozen = float(policy["combos"][prop_type]["production_lambda"])

        assert record["dependence_lambda"] == frozen, prop_type

        expected = (
            stage.KIND_CONVOLUTION
            if frozen <= stage.INDEPENDENCE_LAMBDA
            else stage.KIND_SIMULATION
        )

        assert record["distribution_type"] == expected, prop_type

        seen[prop_type] = expected

    assert set(seen) == set(EXPECTED_COMBO)

    # Both branches, on one real slate, from one real policy.
    assert set(seen.values()) == {stage.KIND_CONVOLUTION, stage.KIND_SIMULATION}


def test_14_an_independent_combination_is_the_engines_own_convolution(
    slate, records
):
    """Case 1 and 4. The convolution ``pricing`` performs, not one like it.

    Component marginals convolved by ``pricing.independent_combo_pmf``, which
    is the function the price came out of. Exactly equal, because it is the
    same call on the same row.
    """
    import joblib

    marginals = joblib.load(slate["model_dir"] / "marginals.joblib")

    mu_columns = stage.mu_columns_for(sorted(marginals))

    checked = 0

    for key, record in records.items():
        prop_type = key.split("|")[2]

        if record["distribution_type"] != stage.KIND_CONVOLUTION:
            continue

        row = group_of(slate["priced"], key).iloc[0]

        engine = pricing.independent_combo_pmf(
            row, prop_type, dict(marginals), dict(mu_columns)
        )

        listed = list(record["distribution_probability"])

        assert listed == [float(value) for value in engine[: len(listed)]], key

        assert record["distribution_tail_probability"] == pytest.approx(
            float(engine[len(listed) :].sum()), abs=0.0, rel=1e-12
        ), key

        checked += 1

    assert checked > 0


def test_15_a_dependent_combination_is_the_frequencies_of_the_priced_draws(
    slate, records
):
    """Cases 3, 5, 6 and 8. The same draws, not a second simulation.

    The seed is regenerated by ``stable_event_seed`` out of
    ``scripts/15_price_markets.py`` -- the function the pricing run used, read
    from the file that used it -- and the draw count is the one the record
    states. Those draws are then put through
    ``pricing.empirical_over_under_push``, the function that priced the line,
    and the record's own mass around the same line has to equal it.

    No independence shortcut is permitted anywhere in this path, and the
    assertion would catch one: convolving the two marginals of a pair with a
    dependence coefficient of 0.85 does not reproduce these frequencies, and
    the test below asserts it does not.
    """
    import joblib

    marginals = joblib.load(slate["model_dir"] / "marginals.joblib")

    copula = joblib.load(slate["model_dir"] / "copula.joblib")

    mu_columns = stage.mu_columns_for(sorted(marginals))

    entry_point = stage.load_pricing_entry_point(PROJECT)

    checked = 0

    for key, record in records.items():
        if record["distribution_type"] != stage.KIND_SIMULATION:
            continue

        game_id, player_id, prop_type = key.split("|")

        seed = entry_point.stable_event_seed(
            int(game_id), int(player_id), prop_type, stage.DEFAULT_SEED
        )

        assert record["simulation_seed"] == seed, key

        group = group_of(slate["priced"], key)

        row = group.iloc[0]

        values = pricing.simulate_combo_values(
            row=row,
            prop_type=prop_type,
            marginals=dict(marginals),
            mu_columns=dict(mu_columns),
            copula=copula,
            dependence_lambda=record["dependence_lambda"],
            simulations=record["simulation_draws"],
            seed=seed,
        )

        empirical = stage.empirical_distribution(
            values, dependence_lambda=record["dependence_lambda"], seed=seed
        )

        assert list(record["distribution_probability"]) == list(
            empirical.probability
        ), key

        assert record["distribution_tail_probability"] == pytest.approx(
            empirical.tail_probability, abs=0.0, rel=1e-12
        ), key

        # And the pricing engine's own function over those same draws.
        for _, quote in group.iterrows():
            line = float(quote["line_value"])

            priced_triple = stage.pricing_line_probabilities(values, line)

            from_record = stage.line_probabilities(published(record), line)

            for derived, engine in zip(from_record, priced_triple):
                assert derived == pytest.approx(engine, abs=MASS_TOLERANCE), key

            checked += 1

        # The independence shortcut, for the record, is a different answer.
        independent = pricing.independent_combo_pmf(
            row, prop_type, dict(marginals), dict(mu_columns)
        )

        overlap = min(len(independent), len(empirical.probability))

        assert not np.allclose(
            independent[:overlap],
            np.asarray(empirical.probability[:overlap]),
            atol=1e-4,
        ), key

    assert checked > 0


def test_16_the_stage_is_deterministic_over_one_priced_slate(slate):
    """Case 9 of both lists. Two derivations, byte-identical.

    Run twice over the same parquet with the same arguments, and the documents
    have to be the same bytes -- including the simulated records, whose seeds
    are a function of the event rather than of the run.
    """
    first = stage.build_document(
        priced_path=slate["priced_path"],
        model_dir=slate["model_dir"],
        slate_date=coverage.SLATE_DATE,
    )

    second = stage.build_document(
        priced_path=slate["priced_path"],
        model_dir=slate["model_dir"],
        slate_date=coverage.SLATE_DATE,
    )

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)

    assert json.dumps(first, sort_keys=True) == json.dumps(
        slate["document"], sort_keys=True
    )


def test_17_the_document_is_provenanced_to_the_parquet_it_was_derived_from(slate):
    """Case 10. A document beside the wrong slate's prices is refusable.

    The digest is the whole reason the document has a header: the publisher is
    handed a serving receipt that digests the priced parquet, and it refuses a
    document whose digest is not that one. A stale file from an earlier run
    cannot be published beside a newer slate's prices.
    """
    document = slate["document"]

    assert document["priced_sha256"] == stage.sha256_file(slate["priced_path"])

    assert document["priced_relative"] == slate["priced_path"].name

    assert document["slate_date"] == coverage.SLATE_DATE

    assert document["seed"] == stage.DEFAULT_SEED

    assert document["combo_simulations"] == stage.DEFAULT_COMBO_SIMULATIONS

    assert document["contract_version"] == stage.DOCUMENT_CONTRACT_VERSION


def test_18_the_stage_defines_no_probability_mathematics_of_its_own(slate):
    """Case 8 of the single-stat list, as a property of the source.

    The parity assertions above are the real proof, but they are proof about
    one slate. This is about the module: every distribution it publishes is
    produced by a named import from ``nba_prop_quant.pricing``, and a second
    implementation added later would have to be called from somewhere this
    lists.
    """
    source = (PROJECT / "ops" / "nba_distribution_publication.py").read_text(
        encoding="utf-8"
    )

    imported = source.split("from nba_prop_quant.pricing import (", 1)

    assert len(imported) == 2, "the stage no longer imports the pricing engine"

    names = imported[1].split(")", 1)[0]

    for name in (
        "_marginal_pmf_for_row",
        "_pmf_over_under_push",
        "empirical_over_under_push",
        "independent_combo_pmf",
        "simulate_combo_values",
    ):
        assert name in names, name

    for forbidden in ("nbinom", "poisson", "scipy.stats", "def _pmf(", "logpmf"):
        assert forbidden not in source, forbidden

    # Loading the pricing entry point for its seed function must not price
    # anything: it is imported for one function and the module body has to
    # stay inert.
    module = stage.load_pricing_entry_point(PROJECT)

    assert callable(module.stable_event_seed)

    assert not hasattr(module, "__priced__")
