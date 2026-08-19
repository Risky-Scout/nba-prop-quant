import pandas as pd

from nba_prop_quant.features import build_base_frame, feature_columns


def synthetic_stats():
    rows = []
    game_id = 1
    dates = pd.date_range("2025-10-20", periods=4, freq="2D")
    for date_idx, date in enumerate(dates):
        for player_id, team_id, opponent_id in [
            (1, 10, 20),
            (2, 10, 20),
            (3, 20, 10),
            (4, 20, 10),
        ]:
            home = 10 if date_idx % 2 == 0 else 20
            visitor = 20 if home == 10 else 10
            rows.append(
                {
                    "stat_id": len(rows) + 1,
                    "player_id": player_id,
                    "team_id": team_id,
                    "game_id": game_id,
                    "date": date,
                    "season": 2025,
                    "postseason": False,
                    "home_team_id": home,
                    "visitor_team_id": visitor,
                    "position": "G" if player_id % 2 else "F",
                    "draft_year": 2020,
                    "min": "30:00",
                    "minutes": 30.0,
                    "pts": 10 + player_id + date_idx,
                    "reb": 3 + player_id,
                    "ast": 2 + date_idx,
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


def test_feature_allowlist_excludes_current_game_targets():
    frame = build_base_frame(synthetic_stats())
    features = feature_columns(frame)

    forbidden = {"pts", "reb", "ast", "stl", "blk", "fg3m", "minutes"}
    assert forbidden.isdisjoint(features)
    assert "prior_pts_rate10" in features
    assert "days_rest" in features


def test_first_player_game_has_no_rolling_current_game_leak():
    frame = build_base_frame(synthetic_stats())
    first = frame.sort_values(["player_id", "date"]).groupby("player_id").head(1)
    assert first["prior_pts_rate10"].isna().all()
