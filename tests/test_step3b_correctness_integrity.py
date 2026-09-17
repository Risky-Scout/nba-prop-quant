"""Step 3B: prediction-time correctness and rolling-state determinism.

Covers the prediction cutoff that keeps live inference on the same estimator
training produced, the explicit NBA slate-date convention, deterministic
rolling parquet ordering, and the semantic fingerprint used to track mutable
rolling state.

Nothing here trains a model, touches the network or writes outside a pytest
temporary directory. The repository is only ever read from.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import socket
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from nba_prop_quant.slate import (
    NBA_SLATE_TIMEZONE,
    NBA_SLATE_ZONE,
    HistoryLeakageError,
    SlateDateError,
    assert_history_precedes_slate,
    assert_slate_inputs_precede_slate,
    build_upcoming_slate_features,
    resolve_slate_date,
    slate_date_from_schedule_date,
    slate_date_from_tip_timestamp,
    slate_date_of_upcoming_games,
)
from nba_prop_quant.storage import sort_by_keys, upsert_parquet


PROJECT = Path(__file__).resolve().parents[1]

# Historical stats run 2025-10-20 .. 2025-10-26 on a two-day cadence.
HISTORY_END = pd.Timestamp("2025-10-26")


def load_refresh_wrapper():
    spec = importlib.util.spec_from_file_location(
        "refresh_wrapper_for_step3b",
        PROJECT / "ops" / "refresh_current_season_state.py",
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the refresh wrapper")

    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


refresh = load_refresh_wrapper()


@pytest.fixture(autouse=True)
def block_all_network(monkeypatch):
    def deny(*args, **kwargs):
        raise RuntimeError(
            f"network access is forbidden in this test module: {args!r}"
        )

    monkeypatch.setattr(socket, "socket", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)

    yield


# ----------------------------------------------------------------------
# synthetic slate inputs
# ----------------------------------------------------------------------


def synthetic_history() -> pd.DataFrame:
    rows = []
    game_id = 1

    for index, moment in enumerate(
        pd.date_range("2025-10-20", periods=4, freq="2D")
    ):
        for player_id, team_id in [(1, 10), (2, 10), (3, 20), (4, 20)]:
            home = 10 if index % 2 == 0 else 20

            rows.append(
                {
                    "stat_id": len(rows) + 1,
                    "player_id": player_id,
                    "team_id": team_id,
                    "game_id": game_id,
                    "date": moment,
                    "season": 2025,
                    "postseason": False,
                    "home_team_id": home,
                    "visitor_team_id": 20 if home == 10 else 10,
                    "position": "G" if player_id % 2 else "F",
                    "draft_year": 2020,
                    "min": "30:00",
                    "minutes": 30.0,
                    "pts": 10 + player_id + index,
                    "reb": 3 + player_id,
                    "ast": 2 + index,
                    "stl": player_id % 2,
                    "blk": 0,
                    "fg3m": 1,
                    "fga": 10,
                    "fg3a": 4,
                    "fta": 2,
                    "oreb": 1,
                    "dreb": 3,
                    "turnover": 1,
                    "pf": 2,
                }
            )

        game_id += 1

    return pd.DataFrame(rows)


def synthetic_games(slate_date: str, game_id: int = 99) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": game_id,
                "date": pd.Timestamp(slate_date),
                "season": 2025,
                "home_team_id": 10,
                "visitor_team_id": 20,
            }
        ]
    )


def synthetic_active() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": player_id,
                "current_team_id": team_id,
                "position": "G" if player_id % 2 else "F",
                "draft_year": 2020,
                "first_name": f"P{player_id}",
                "last_name": "Test",
            }
            for player_id, team_id in [(1, 10), (2, 10), (3, 20), (4, 20)]
        ]
    )


def synthetic_advanced(dates: list[str]) -> pd.DataFrame:
    if not dates:
        return pd.DataFrame(columns=["player_id", "game_id", "date"])

    return pd.DataFrame(
        [
            {
                "player_id": 1,
                "game_id": 1,
                "date": pd.Timestamp(moment),
                "offensive_rating": 110.0,
                "usage_percentage": 24.0,
            }
            for moment in dates
        ]
    )


@pytest.fixture(scope="module")
def dynamic_params() -> dict:
    return json.loads(
        (PROJECT / "models" / "dynamic_params.json").read_text(
            encoding="utf-8"
        )
    )


def build_slate(slate_date: str, dynamic_params: dict, **overrides):
    return build_upcoming_slate_features(
        history_stats=overrides.get("history", synthetic_history()),
        upcoming_games=overrides.get("games", synthetic_games(slate_date)),
        active_players=overrides.get("active", synthetic_active()),
        advanced=overrides.get("advanced", synthetic_advanced([])),
        dynamic_params=dynamic_params,
    )


# ----------------------------------------------------------------------
# Section 6 / 7: prediction-time cutoff and train/serve skew guard
# ----------------------------------------------------------------------


def test_history_ending_the_day_before_the_slate_is_accepted(dynamic_params):
    """D-1 history is the only case where serve state matches train state."""
    slate = build_slate("2025-10-27", dynamic_params)

    assert not slate.empty
    assert len(slate) == 4


@pytest.mark.parametrize(
    "slate_date,label",
    [
        ("2025-10-26", "same day as the last historical game"),
        ("2025-10-25", "a day already covered by history"),
    ],
)
def test_history_reaching_the_slate_date_is_rejected(
    dynamic_params, slate_date, label
):
    """D and D+1 history both mean live inference saw the outcome."""
    with pytest.raises(HistoryLeakageError) as caught:
        build_slate(slate_date, dynamic_params)

    message = str(caught.value)

    assert "historical stats" in message
    assert "not strictly before slate date" in message
    assert slate_date in message


def test_failure_names_dataset_observed_max_and_required_slate_date(
    dynamic_params,
):
    with pytest.raises(HistoryLeakageError) as caught:
        build_slate("2025-10-26", dynamic_params)

    message = str(caught.value)

    assert "historical stats" in message
    assert "2025-10-26" in message
    assert "max date" in message


def test_advanced_history_is_checked_independently(dynamic_params):
    """Clean box scores must not excuse a leaking advanced table."""
    with pytest.raises(HistoryLeakageError) as caught:
        build_slate(
            "2025-10-27",
            dynamic_params,
            advanced=synthetic_advanced(["2025-10-27"]),
        )

    assert "historical advanced" in str(caught.value)


def test_advanced_history_one_day_before_is_accepted(dynamic_params):
    slate = build_slate(
        "2025-10-27",
        dynamic_params,
        advanced=synthetic_advanced(["2025-10-25", "2025-10-26"]),
    )

    assert len(slate) == 4


def test_guard_runs_before_any_feature_is_derived(dynamic_params):
    """Unusable downstream inputs must not be reached when history leaks.

    active_players is emptied, which makes the builder produce no rows at all.
    A leaking slate still fails with the cutoff error, proving the guard runs
    before feature derivation and therefore before all model inference.
    """
    with pytest.raises(HistoryLeakageError):
        build_slate(
            "2025-10-26",
            dynamic_params,
            active=pd.DataFrame(
                columns=["id", "current_team_id", "position", "draft_year"]
            ),
        )


def test_future_scheduled_games_are_legitimate(dynamic_params):
    """games.parquet may hold future rows; only history is constrained."""
    games = pd.concat(
        [
            synthetic_games("2025-10-27", game_id=99),
            synthetic_games("2025-11-05", game_id=100),
        ],
        ignore_index=True,
    )

    slate = build_slate("2025-10-27", dynamic_params, games=games)

    assert not slate.empty


def test_multi_date_slate_uses_the_earliest_game(dynamic_params):
    """History must be complete before the first tip, not the last."""
    games = pd.concat(
        [
            synthetic_games("2025-10-26", game_id=99),
            synthetic_games("2025-10-30", game_id=100),
        ],
        ignore_index=True,
    )

    assert slate_date_of_upcoming_games(games) == pd.Timestamp("2025-10-26")

    with pytest.raises(HistoryLeakageError):
        build_slate("2025-10-26", dynamic_params, games=games)


def test_unparseable_history_dates_fail_closed():
    frame = pd.DataFrame({"date": ["2025-10-20", "not-a-date"]})

    with pytest.raises(HistoryLeakageError, match="unparseable"):
        assert_history_precedes_slate(frame, "historical stats", "2025-10-27")


def test_history_with_no_usable_date_fails_closed():
    frame = pd.DataFrame({"date": pd.Series([], dtype="datetime64[ns]")})

    with pytest.raises(HistoryLeakageError, match="no usable date"):
        assert_history_precedes_slate(frame, "historical stats", "2025-10-27")


def test_missing_date_column_fails_closed():
    with pytest.raises(HistoryLeakageError, match="no date column"):
        assert_history_precedes_slate(
            pd.DataFrame({"player_id": [1]}),
            "historical stats",
            "2025-10-27",
        )


def test_guard_returns_the_observed_maximum_when_safe():
    observed = assert_slate_inputs_precede_slate(
        synthetic_history(),
        synthetic_advanced([]),
        synthetic_games("2025-10-27"),
    )

    assert observed == pd.Timestamp("2025-10-27")


def test_refresh_wrapper_and_prediction_share_one_cutoff_rule():
    """The staging gate must not be able to drift from the prediction gate."""
    leaking = pd.DataFrame(
        {
            "date": ["2026-11-15"],
            "player_id": [1],
            "game_id": [1],
        }
    )

    with pytest.raises(refresh.RefreshFailure) as staged:
        refresh.assert_no_leakage(
            leaking,
            "stats",
            Path("/tmp/staged/stats.parquet"),
            pd.Timestamp("2026-11-15"),
        )

    with pytest.raises(HistoryLeakageError) as predicted:
        assert_history_precedes_slate(
            leaking,
            "staged stats",
            pd.Timestamp("2026-11-15"),
        )

    assert str(staged.value) == str(predicted.value)
    assert staged.value.code == refresh.EXIT_VALIDATION_FAILED


# ----------------------------------------------------------------------
# Section 8: NBA slate date and timezone semantics
# ----------------------------------------------------------------------


def test_canonical_timezone_is_eastern():
    assert NBA_SLATE_TIMEZONE == "America/New_York"
    assert NBA_SLATE_ZONE == ZoneInfo("America/New_York")


def test_eastern_evening_game_keeps_its_own_date():
    # 7:30pm ET on 2025-10-22.
    assert slate_date_from_tip_timestamp(
        "2025-10-22T19:30:00-04:00"
    ) == date(2025, 10, 22)


def test_late_west_coast_game_stays_on_its_nba_slate_date():
    """A 7:30pm PT tip is still that evening's slate, not the next day's.

    Its UTC timestamp has already rolled over to 2025-10-23, so taking the date
    in UTC would move the game onto the following slate. In Eastern it is
    10:30pm on 2025-10-22, which is the slate the NBA scheduled it on.
    """
    tip = "2025-10-23T02:30:00Z"

    assert datetime.fromisoformat(
        tip.replace("Z", "+00:00")
    ).date() == date(2025, 10, 23)

    assert slate_date_from_tip_timestamp(tip) == date(2025, 10, 22)


def test_tip_after_midnight_eastern_belongs_to_the_next_eastern_date():
    """Eastern conversion cannot rescue a tip that is genuinely past midnight.

    A 10:30pm PT tip is 1:30am Eastern the following day, so a timestamp alone
    resolves to 2025-10-23. This is exactly why a BALLDONTLIE date field is
    authoritative for such games and is preserved rather than recomputed.
    """
    assert slate_date_from_tip_timestamp(
        "2025-10-23T05:30:00Z"
    ) == date(2025, 10, 23)

    assert slate_date_from_schedule_date("2025-10-22") == date(2025, 10, 22)


@pytest.mark.parametrize(
    "tip,expected",
    [
        ("2025-10-23T03:59:59Z", date(2025, 10, 22)),
        ("2025-10-23T04:00:00Z", date(2025, 10, 23)),
        ("2025-10-22T23:59:59-04:00", date(2025, 10, 22)),
        ("2025-10-23T00:00:00-04:00", date(2025, 10, 23)),
    ],
)
def test_timestamps_around_midnight_eastern(tip, expected):
    assert slate_date_from_tip_timestamp(tip) == expected


def test_winter_game_uses_standard_time_offset():
    """January tips fall under EST, so the UTC boundary moves to 05:00Z."""
    assert slate_date_from_tip_timestamp(
        "2026-01-15T04:59:59Z"
    ) == date(2026, 1, 14)

    assert slate_date_from_tip_timestamp(
        "2026-01-15T05:00:00Z"
    ) == date(2026, 1, 15)


def test_aware_datetime_objects_are_accepted():
    moment = datetime(2025, 10, 23, 2, 30, tzinfo=timezone.utc)

    assert slate_date_from_tip_timestamp(moment) == date(2025, 10, 22)
    assert slate_date_from_tip_timestamp(
        pd.Timestamp(moment)
    ) == date(2025, 10, 22)


def test_naive_tip_timestamp_is_refused_not_guessed():
    with pytest.raises(SlateDateError, match="timezone information"):
        slate_date_from_tip_timestamp("2025-10-23T05:30:00")


@pytest.mark.parametrize(
    "value",
    ["2025-10-22", date(2025, 10, 22), pd.Timestamp("2025-10-22")],
)
def test_date_only_schedule_values_are_preserved(value):
    """A BALLDONTLIE date field is authoritative and is never shifted."""
    assert slate_date_from_schedule_date(value) == date(2025, 10, 22)
    assert resolve_slate_date(value) == date(2025, 10, 22)


def test_date_only_value_is_not_reinterpreted_as_utc_midnight():
    """Reinterpreting midnight UTC in Eastern would move the slate back a day."""
    reinterpreted = (
        datetime(2025, 10, 22, tzinfo=timezone.utc)
        .astimezone(NBA_SLATE_ZONE)
        .date()
    )

    assert reinterpreted == date(2025, 10, 21)
    assert slate_date_from_schedule_date("2025-10-22") == date(2025, 10, 22)


def test_naive_timestamp_with_a_time_of_day_is_ambiguous():
    with pytest.raises(SlateDateError, match="ambiguous"):
        slate_date_from_schedule_date(pd.Timestamp("2025-10-22 19:30"))


def test_aware_timestamp_is_not_treated_as_a_schedule_date():
    with pytest.raises(SlateDateError, match="offset-aware"):
        slate_date_from_schedule_date(
            pd.Timestamp("2025-10-23T05:30:00+00:00")
        )


def test_resolve_slate_date_dispatches_on_offset_awareness():
    assert resolve_slate_date("2025-10-23T02:30:00Z") == date(2025, 10, 22)
    assert resolve_slate_date("2025-10-23") == date(2025, 10, 23)


def test_malformed_schedule_date_is_refused():
    with pytest.raises(SlateDateError, match="YYYY-MM-DD"):
        slate_date_from_schedule_date("10/22/2025")


# ----------------------------------------------------------------------
# Section 9: deterministic rolling parquet order
# ----------------------------------------------------------------------


def sample_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [3, 1, 2, 5, 4],
            "player_id": [30, 10, 20, 50, 40],
            "value": [0.3, 0.1, 0.2, 0.5, 0.4],
            "label": ["c", "a", "b", "e", "d"],
        }
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_upsert_sorts_by_declared_key_columns(tmp_path):
    path = tmp_path / "rolling.parquet"

    upsert_parquet(sample_rows(), path, ["id"])

    stored = pd.read_parquet(path)

    assert stored["id"].tolist() == [1, 2, 3, 4, 5]
    assert stored.index.tolist() == list(range(5))


def test_same_semantic_input_reruns_to_identical_order(tmp_path):
    first = tmp_path / "a.parquet"
    second = tmp_path / "b.parquet"

    upsert_parquet(sample_rows(), first, ["id"])
    upsert_parquet(sample_rows(), second, ["id"])

    assert (
        pd.read_parquet(first)["id"].tolist()
        == pd.read_parquet(second)["id"].tolist()
    )


def test_arrival_order_does_not_change_the_result(tmp_path):
    """Equivalent records assembled in any order produce the same file."""
    sorted_write = tmp_path / "sorted.parquet"
    shuffled_write = tmp_path / "shuffled.parquet"

    upsert_parquet(sample_rows(), sorted_write, ["id"])
    upsert_parquet(
        sample_rows().sample(frac=1.0, random_state=7).reset_index(drop=True),
        shuffled_write,
        ["id"],
    )

    assert (
        pd.read_parquet(sorted_write)["id"].tolist()
        == pd.read_parquet(shuffled_write)["id"].tolist()
    )


def test_incremental_upsert_matches_a_single_write(tmp_path):
    one_shot = tmp_path / "one.parquet"
    batched = tmp_path / "many.parquet"

    rows = sample_rows()

    upsert_parquet(rows, one_shot, ["id"])
    upsert_parquet(rows.iloc[:3], batched, ["id"])
    upsert_parquet(rows.iloc[3:], batched, ["id"])

    assert pd.read_parquet(one_shot).equals(pd.read_parquet(batched))


def test_parquet_output_is_byte_identical(tmp_path):
    """Byte reproducibility holds under the pinned pyarrow, so it is asserted.

    Semantic determinism is asserted separately above, so a future writer that
    changes incidental metadata will fail here without hiding that the logical
    ordering guarantee still holds.
    """
    one_shot = tmp_path / "one.parquet"
    batched = tmp_path / "many.parquet"
    shuffled = tmp_path / "shuffled.parquet"

    rows = sample_rows()

    upsert_parquet(rows, one_shot, ["id"])

    upsert_parquet(rows.iloc[:3], batched, ["id"])
    upsert_parquet(rows.iloc[3:], batched, ["id"])

    upsert_parquet(
        rows.sample(frac=1.0, random_state=3).reset_index(drop=True),
        shuffled,
        ["id"],
    )

    assert digest(one_shot) == digest(batched) == digest(shuffled)


def test_replacement_record_still_wins(tmp_path):
    path = tmp_path / "rolling.parquet"

    upsert_parquet(sample_rows(), path, ["id"])

    upsert_parquet(
        pd.DataFrame(
            {
                "id": [2],
                "player_id": [20],
                "value": [99.0],
                "label": ["REPLACED"],
            }
        ),
        path,
        ["id"],
    )

    stored = pd.read_parquet(path)

    assert len(stored) == 5
    assert stored["id"].tolist() == [1, 2, 3, 4, 5]
    assert stored.loc[stored["id"].eq(2), "label"].item() == "REPLACED"
    assert stored.loc[stored["id"].eq(2), "value"].item() == 99.0


def test_key_uniqueness_is_preserved(tmp_path):
    path = tmp_path / "rolling.parquet"

    rows = sample_rows()

    upsert_parquet(rows, path, ["id"])
    upsert_parquet(rows, path, ["id"])

    stored = pd.read_parquet(path)

    assert len(stored) == 5
    assert stored["id"].is_unique


def test_multi_column_keys_sort_lexicographically(tmp_path):
    path = tmp_path / "rolling.parquet"

    frame = pd.DataFrame(
        {
            "game_id": [2, 1, 2, 1],
            "player_id": [9, 9, 8, 8],
            "value": [1, 2, 3, 4],
        }
    )

    upsert_parquet(
        frame.sample(frac=1.0, random_state=5),
        path,
        ["game_id", "player_id"],
    )

    stored = pd.read_parquet(path)

    assert stored[["game_id", "player_id"]].values.tolist() == [
        [1, 8],
        [1, 9],
        [2, 8],
        [2, 9],
    ]


def test_sorting_by_an_absent_key_column_fails_closed():
    with pytest.raises(KeyError, match="absent key column"):
        sort_by_keys(pd.DataFrame({"a": [1]}), ["missing"])


def test_null_keys_sort_last_deterministically():
    frame = pd.DataFrame({"id": [2, None, 1], "value": ["b", "n", "a"]})

    ordered = sort_by_keys(frame, ["id"])

    assert ordered["value"].tolist() == ["a", "b", "n"]


# ----------------------------------------------------------------------
# Section 10: semantic rolling-state fingerprint
# ----------------------------------------------------------------------


def rolling_stats() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [1, 1, 2, 2],
            "player_id": [10, 11, 10, 11],
            "date": pd.to_datetime(
                ["2026-11-10", "2026-11-10", "2026-11-12", "2026-11-12"]
            ),
            "pts": [10.0, 12.0, 14.0, 16.0],
        }
    )


def test_semantic_fingerprint_ignores_row_order():
    """A rewrite that preserves the records must preserve the fingerprint."""
    frame = rolling_stats()

    shuffled = frame.sample(frac=1.0, random_state=11).reset_index(drop=True)

    assert refresh.semantic_fingerprint(
        frame, ("game_id", "player_id")
    ) == refresh.semantic_fingerprint(shuffled, ("game_id", "player_id"))


def test_semantic_fingerprint_ignores_column_order():
    frame = rolling_stats()

    reordered = frame[["pts", "date", "player_id", "game_id"]]

    assert refresh.semantic_fingerprint(
        frame, ("game_id", "player_id")
    ) == refresh.semantic_fingerprint(reordered, ("game_id", "player_id"))


def test_semantic_fingerprint_detects_a_changed_value():
    frame = rolling_stats()

    changed = frame.copy()
    changed.loc[0, "pts"] = 11.0

    assert refresh.semantic_fingerprint(
        frame, ("game_id", "player_id")
    ) != refresh.semantic_fingerprint(changed, ("game_id", "player_id"))


def test_semantic_fingerprint_detects_an_added_record():
    frame = rolling_stats()

    extra = pd.concat(
        [
            frame,
            pd.DataFrame(
                {
                    "game_id": [3],
                    "player_id": [10],
                    "date": pd.to_datetime(["2026-11-13"]),
                    "pts": [20.0],
                }
            ),
        ],
        ignore_index=True,
    )

    assert refresh.semantic_fingerprint(
        frame, ("game_id", "player_id")
    ) != refresh.semantic_fingerprint(extra, ("game_id", "player_id"))


def test_semantic_fingerprint_detects_a_removed_record():
    frame = rolling_stats()

    assert refresh.semantic_fingerprint(
        frame, ("game_id", "player_id")
    ) != refresh.semantic_fingerprint(
        frame.iloc[1:], ("game_id", "player_id")
    )


def test_dataset_state_record_fields(tmp_path):
    data_root = tmp_path / "data"
    path = data_root / "raw" / "seasons" / "season=2026" / "stats.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)

    frame = rolling_stats()
    frame.to_parquet(path, index=False)

    record = refresh.dataset_state_record(frame, "stats", path, data_root)

    assert record["relative_path"] == (
        "raw/seasons/season=2026/stats.parquet"
    )
    assert record["row_count"] == 4
    assert record["key_columns"] == ["game_id", "player_id"]
    assert record["unique_key_count"] == 4
    assert record["game_count"] == 2
    assert record["min_date"] == "2026-11-10"
    assert record["max_date"] == "2026-11-12"
    assert record["columns"] == ["date", "game_id", "player_id", "pts"]
    assert len(record["semantic_fingerprint"]) == 64
    assert len(record["schema_fingerprint"]) == 64


def test_state_record_holds_no_absolute_paths(tmp_path):
    data_root = tmp_path / "data"
    path = data_root / "raw" / "seasons" / "season=2026" / "stats.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)

    rolling_stats().to_parquet(path, index=False)

    record = refresh.dataset_state_record(
        rolling_stats(), "stats", path, data_root
    )

    text = json.dumps(record)

    assert str(tmp_path) not in text
    assert not text.count('"/')


def test_schema_fingerprint_reacts_to_a_dtype_change():
    frame = rolling_stats()

    retyped = frame.copy()
    retyped["pts"] = retyped["pts"].astype("int64")

    assert refresh.dataset_state_record(
        frame, "stats", Path("/d/x.parquet"), Path("/d")
    )["schema_fingerprint"] != refresh.dataset_state_record(
        retyped, "stats", Path("/d/x.parquet"), Path("/d")
    )["schema_fingerprint"]


def test_declared_keys_cover_every_rolling_dataset():
    assert set(refresh.ROLLING_KEY_COLUMNS) == {"stats", "games", "advanced"}
