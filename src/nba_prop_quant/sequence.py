from __future__ import annotations

"""
Optional LSTM anchor.

This module is intentionally not part of the default dependency set. Install with:

    pip install -e ".[sequence]"

Use it only after the live snapshot archive is large enough to justify a sequence
model. The primary production model remains interpretable without it.
"""

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        'Install the optional sequence dependency with: pip install -e ".[sequence]"'
    ) from exc


class RateResidualLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last = output[:, -1, :]
        return self.head(last).squeeze(-1)
