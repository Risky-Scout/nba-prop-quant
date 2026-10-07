"""Game-level latent-state dependence shadow model, v1.

SHADOW / RESEARCH ONLY.

This package adds a hierarchical Gaussian latent-factor layer *after* the
production conditional means and calibrated marginals are available. It never
replaces, promotes or reconfigures the incumbent model:

* it does not write to ``models/``,
* it does not touch ``promotion_state.json`` or ``current_good_fit_id``,
* it does not publish to WizardOfOdds,
* it has no promotion entry point of any kind.

The pipeline it implements is::

    existing game state
      -> existing conditional means (season walk-forward OOF)
      -> existing calibrated marginals (walk-forward refit)
      -> randomized PIT -> Gaussian residuals            (pit.py)
      -> hierarchical latent-factor loadings             (factors.py)
      -> PSD block covariance, same-player block pinned  (covariance.py)
      -> correlated latent normals -> Phi -> uniforms
      -> existing marginal inverse CDFs                  (simulator.py)
      -> coherent whole-game realizations
      -> joint prop-event probabilities                  (query.py)
"""

from __future__ import annotations

DEPENDENCE_MODEL_VERSION = "game-latent-state-shadow-v1"

__all__ = ["DEPENDENCE_MODEL_VERSION"]
