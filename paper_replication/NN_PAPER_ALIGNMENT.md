# Neural-network alignment with Insight 177

Reference: local `Insight-177-Money-Positioning-in-Energy.pdf`, Sections 2–3,
footnotes 3–5, 9–10, and Tables 1–2.

## Implemented equations

- MOM: cumulative held-contract price P&L minus its n-day average, divided by
  annualized exponentially weighted root mean square P&L. Decay is 60/61;
  only prior daily observations enter volatility. The recursive calculation
  avoids underflow and does not depend on how much future data is loaded.
- Inputs: horizons 5, 10, ..., 250; divide each MOM by its 252-day sample std.
- Target: `(MM - lagged_mean) / lagged_sigma`. Each market's one-year window
  contains reporting observations in its last 252 trading dates. At each
  historical report i, calculate `e_i = MM_i - mean_i`; sigma is
  `sqrt(sum(e_i**2)/(N-1))`, without subtracting the mean of residuals.
  Both statistics are lagged to the preceding report. Full history is required.
- Loss: mean squared error over all training weeks and six markets, plus
  `0.04 * sum(abs(shared_weights))`, plus
  `0.01 * sum((bias[:,1:] - bias[:,:-1])**2)`. Penalties are sums, not means.
- Three shared factors, activation `u*exp((1-u*u)/2)`, market-specific weights,
  and market/date biases. Initial shared weights and biases are zero; market
  weights are (0.4, 0.2, 0.1).
- Full-batch Adam: learning rate .01, betas (.9,.999), epsilon 1e-7;
  1024 epochs initially and 36 subsequently. `PaperAdam` uses TensorFlow's
  epsilon-hat placement, rather than PyTorch Adam's corrected-moment placement.
  Reference: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
- Score predicted changes as `sigma_lag * (yhat_current - yhat_previous)`.
  Do not difference reconstructed position levels. Actual changes are computed
  from raw consecutive reports, including the first evaluated week.

## Explicit conventions where the paper is incomplete

- Futures-only positions; roll at settlement on the final observed trading day
  before the held contract's expiration month. Bloomberg generic-to-contract
  mapping and occasional nearby-3 holdings follow the shared rolling builder.
- Actual report as-of dates determine feature sampling; the earliest report date
  across markets assigns each reporting week to the evaluation year.
- A training window contains 104 complete reporting weeks, interpreting two years
  as 104 observations. The target's one-year window uses 252 market trading dates.
  Its one-week lag means the prior report, including holiday weeks.
- At each evaluation date, the current fitted model evaluates both the prior
  training endpoint and the current input. The current bias equals the last
  training bias, so their difference isolates changing signals. The paper does
  not explicitly specify the model vintage for the prior fitted endpoint.
- Warm-start weights; retain overlapping bias parameters by actual week and
  initialize new dates with the last fitted bias. Reset Adam moments per fit;
  the paper specifies parameter warm starts but not optimizer-state persistence.
- PyTorch float32 arithmetic remains; identical TensorFlow floating-point paths
  or exact author data are not guaranteed.

## Validation

`PYTHONPATH=paper_replication/src python -m unittest discover -s paper_replication/tests -v`
checks the target against a direct footnote-5 calculation, target lag causality,
loss values and gradients, Adam updates, bias date alignment, and volatility
prefix invariance. The notebook records run coverage and compares cumulative
metrics with the paper; numerical differences must remain visible.

## Full-run check (2026-09-15)

Six tests passed. A full paper-window run produced 574 weeks per market
(2015-01-06 to 2025-12-30), or 3,444 finite scored observations.

Pooled R²: 45.0318% (paper 44.59%). Directional accuracy: 71.8351% (paper 72.50%).
These remaining gaps mean equation alignment is not an exact numerical replication.
Validation used the configured full epoch counts; no hyperparameters were tuned to close the gap.
