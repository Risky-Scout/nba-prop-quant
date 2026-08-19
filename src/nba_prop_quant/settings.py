from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    bdl_api_key: str
    bdl_base_url: str = "https://api.balldontlie.io"
    bdl_requests_per_minute: int = 600

    nba_prop_data_dir: Path = Path("./data")
    nba_prop_model_dir: Path = Path("./models")

    history_start_season: int = 2001
    advanced_start_season: int = 2015
    play_by_play_start_season: int = 2025

    @property
    def raw_dir(self) -> Path:
        return self.nba_prop_data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.nba_prop_data_dir / "processed"

    @property
    def snapshot_dir(self) -> Path:
        return self.nba_prop_data_dir / "snapshots"

    def ensure_dirs(self) -> None:
        for path in (
            self.nba_prop_data_dir,
            self.raw_dir,
            self.processed_dir,
            self.snapshot_dir,
            self.nba_prop_model_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
