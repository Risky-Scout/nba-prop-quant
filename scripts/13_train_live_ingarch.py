from __future__ import annotations

import json

import joblib
import pandas as pd
from rich.console import Console

from nba_prop_quant.features import TARGETS
from nba_prop_quant.live import INGARCH11, archived_snapshots_to_bins
from nba_prop_quant.settings import get_settings

console = Console()


def load_snapshot_archive() -> pd.DataFrame:
    settings = get_settings()
    rows = []
    for path in sorted((settings.snapshot_dir / "live_box").rglob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                envelope = json.loads(line)
                payload = dict(envelope["payload"])
                payload["captured_at"] = envelope["captured_at"]
                rows.append(payload)
    return pd.DataFrame(rows)


def main() -> None:
    settings = get_settings()
    snapshots = load_snapshot_archive()
    if snapshots.empty:
        raise RuntimeError("No live box snapshot archive exists yet")

    models = {}
    for target in TARGETS:
        binned = archived_snapshots_to_bins(snapshots, target=target, bin_seconds=60)
        sequences = [
            group.sort_values("captured_at")["count"].to_numpy(dtype=float)
            for _, group in binned.groupby(["game_id", "player_id"])
        ]
        usable = [seq for seq in sequences if len(seq) >= 5]
        if len(usable) < 10:
            console.print(f"[yellow]{target}: insufficient sequences; skipped[/yellow]")
            continue
        model = INGARCH11().fit_sequences(usable)
        models[target] = model
        console.print(
            f"{target}: omega={model.omega:.4f}, alpha={model.alpha:.4f}, beta={model.beta:.4f}"
        )

    joblib.dump(models, settings.nba_prop_model_dir / "live_ingarch.joblib")
    console.print("[green]Saved live INGARCH models[/green]")


if __name__ == "__main__":
    main()
