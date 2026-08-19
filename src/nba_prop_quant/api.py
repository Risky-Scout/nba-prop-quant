from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


class BDLHTTPError(RuntimeError):
    pass


class RateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self.interval = 60.0 / max(requests_per_minute, 1)
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            sleep_for = self._next_allowed - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._next_allowed = max(now, self._next_allowed) + self.interval


class BDLClient:
    """Thin client that follows the current BALLDONTLIE NBA OpenAPI paths."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.balldontlie.io",
        requests_per_minute: int = 600,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.limiter = RateLimiter(requests_per_minute)
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": api_key, "Accept": "application/json"},
            timeout=timeout_seconds,
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "BDLClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.TransportError, BDLHTTPError)),
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=1.0, min=1.0, max=30.0),
        reraise=True,
    )
    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.limiter.wait()
        response = self.client.get(path, params=params)
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                time.sleep(float(retry_after))
            raise BDLHTTPError("BDL rate limit reached")
        if response.status_code >= 500:
            raise BDLHTTPError(f"BDL server error {response.status_code}: {response.text[:500]}")
        if response.status_code >= 400:
            raise RuntimeError(
                f"BDL request failed {response.status_code} for {path}: {response.text[:1000]}"
            )
        return response.json()

    def paginate(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        per_page: int = 100,
    ) -> Iterator[dict[str, Any]]:
        query = dict(params or {})
        query["per_page"] = min(per_page, 100)
        cursor = None

        while True:
            if cursor is not None:
                query["cursor"] = cursor
            payload = self.get(path, query)
            for row in payload.get("data", []):
                yield row
            cursor = payload.get("meta", {}).get("next_cursor")
            if cursor is None:
                break

    def teams(self) -> list[dict[str, Any]]:
        return self.get("/nba/v1/teams").get("data", [])

    def players(self) -> Iterator[dict[str, Any]]:
        yield from self.paginate("/nba/v1/players")

    def active_players(
        self,
        team_ids: list[int] | None = None,
    ) -> Iterator[dict[str, Any]]:
        params: dict[str, Any] = {}
        if team_ids:
            params["team_ids[]"] = team_ids
        yield from self.paginate("/nba/v1/players/active", params)

    def games(
        self,
        seasons: list[int] | None = None,
        dates: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        season_type: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        params: dict[str, Any] = {}
        if seasons:
            params["seasons[]"] = seasons
        if dates:
            params["dates[]"] = dates
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if season_type:
            params["season_type"] = season_type
        yield from self.paginate("/nba/v1/games", params)

    def stats(
        self,
        seasons: list[int] | None = None,
        game_ids: list[int] | None = None,
        player_ids: list[int] | None = None,
        period: int = 0,
        season_type: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        params: dict[str, Any] = {"period": period}
        if seasons:
            params["seasons[]"] = seasons
        if game_ids:
            params["game_ids[]"] = game_ids
        if player_ids:
            params["player_ids[]"] = player_ids
        if season_type:
            params["season_type"] = season_type
        yield from self.paginate("/nba/v1/stats", params)

    def advanced_stats(
        self,
        seasons: list[int] | None = None,
        game_ids: list[int] | None = None,
        player_ids: list[int] | None = None,
        period: int = 0,
    ) -> Iterator[dict[str, Any]]:
        params: dict[str, Any] = {"period": period}
        if seasons:
            params["seasons[]"] = seasons
        if game_ids:
            params["game_ids[]"] = game_ids
        if player_ids:
            params["player_ids[]"] = player_ids
        yield from self.paginate("/nba/v2/stats/advanced", params)

    def injuries(
        self,
        team_ids: list[int] | None = None,
        player_ids: list[int] | None = None,
    ) -> Iterator[dict[str, Any]]:
        params: dict[str, Any] = {}
        if team_ids:
            params["team_ids[]"] = team_ids
        if player_ids:
            params["player_ids[]"] = player_ids
        yield from self.paginate("/nba/v1/player_injuries", params)

    def box_scores(self, date: str, season_type: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"date": date}
        if season_type:
            params["season_type"] = season_type
        return self.get("/nba/v1/box_scores", params).get("data", [])

    def live_box_scores(self, season_type: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if season_type:
            params["season_type"] = season_type
        return self.get("/nba/v1/box_scores/live", params).get("data", [])

    def lineups(self, game_ids: list[int]) -> Iterator[dict[str, Any]]:
        params = {"game_ids": game_ids}
        yield from self.paginate("/nba/v1/lineups", params)

    def plays(self, game_id: int) -> list[dict[str, Any]]:
        return self.get("/nba/v1/plays", {"game_id": game_id}).get("data", [])

    def opening_odds(
        self,
        game_ids: list[int] | None = None,
        dates: list[str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        params: dict[str, Any] = {}
        if game_ids:
            params["game_ids"] = game_ids
        if dates:
            params["dates"] = dates
        yield from self.paginate("/nba/v2/odds/opening", params)

    def opening_player_props(
        self,
        game_id: int,
        player_id: int | None = None,
        prop_type: str | None = None,
        vendors: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"game_id": game_id}
        if player_id is not None:
            params["player_id"] = player_id
        if prop_type:
            params["prop_type"] = prop_type
        if vendors:
            params["vendors"] = vendors
        return self.get("/nba/v2/odds/player_props/opening", params).get("data", [])

    def live_player_props(
        self,
        game_id: int,
        player_id: int | None = None,
        prop_type: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"game_id": game_id}
        if player_id is not None:
            params["player_id"] = player_id
        if prop_type:
            params["prop_type"] = prop_type
        return self.get("/nba/v2/odds/player_props", params).get("data", [])
