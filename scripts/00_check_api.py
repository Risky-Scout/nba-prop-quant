from __future__ import annotations

from rich.console import Console

from nba_prop_quant.api import BDLClient
from nba_prop_quant.settings import get_settings

console = Console()


def main() -> None:
    settings = get_settings()
    with BDLClient(
        api_key=settings.bdl_api_key,
        base_url=settings.bdl_base_url,
        requests_per_minute=settings.bdl_requests_per_minute,
    ) as client:
        teams = client.teams()
        one_game = client.get(
            "/nba/v1/games",
            {"seasons[]": [2025], "per_page": 1},
        ).get("data", [])

        console.print(f"[green]Authentication OK[/green]. Teams returned: {len(teams)}")
        console.print(f"2025 game probe rows: {len(one_game)}")

        try:
            advanced = client.get(
                "/nba/v2/stats/advanced",
                {"seasons[]": [2025], "period": 0, "per_page": 1},
            ).get("data", [])
            console.print(f"[green]GOAT advanced endpoint OK[/green]: {len(advanced)} probe row")
        except RuntimeError as exc:
            console.print(f"[yellow]Advanced endpoint unavailable[/yellow]: {exc}")

        try:
            injuries = client.get(
                "/nba/v1/player_injuries",
                {"per_page": 1},
            ).get("data", [])
            console.print(f"[green]Injury endpoint OK[/green]: {len(injuries)} probe row")
        except RuntimeError as exc:
            console.print(f"[yellow]Injury endpoint unavailable[/yellow]: {exc}")


if __name__ == "__main__":
    main()
