from dataclasses import replace
import numpy as np
import pandas as pd
import torch

from .config import HORIZONS, MARKETS, ReplicationConfig
from .data import load_inputs, build_rolling_contract, build_daily_features, build_weekly_panel
from .model import fit_window, set_deterministic


def _arrays(window, markets=MARKETS):
    """Reshape a long weekly panel into tensors: x [weeks, markets, horizons] and y [weeks, markets].

    Markets follow the order of `markets`, and a market missing in a week becomes NaN.
    """
    feature_cols = [f"x_{h}" for h in HORIZONS]
    dates = sorted(window["report_week_tuesday"].unique())
    x = np.stack([window.loc[window["report_week_tuesday"].eq(date)].set_index("market").reindex(markets)[feature_cols].to_numpy(float) for date in dates])
    y = np.stack([window.loc[window["report_week_tuesday"].eq(date)].set_index("market").reindex(markets)["y"].to_numpy(float) for date in dates])
    return dates, torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def _metrics(result):
    """Out-of-sample R^2 (uncentred: 1 - SSE / sum of actual_change^2) and directional accuracy.

    Both are pooled over all markets and weeks in `result`.
    """
    result = result.dropna(subset=["actual_change", "predicted_change"])
    error = result["actual_change"] - result["predicted_change"]
    denom = np.sum(result["actual_change"] ** 2)
    r2 = np.nan if denom == 0 else 1 - np.sum(error ** 2) / denom
    accuracy = np.mean(np.sign(result["actual_change"]) == np.sign(result["predicted_change"]))
    return pd.Series({"r2_oos": r2, "directional_accuracy": accuracy, "observations": len(result)})


def run_replication(config=ReplicationConfig(), expanded=False):
    """Run the rolling out-of-sample evaluation of the shared momentum network.

    Builds the weekly panel, then walks forward one reporting week at a time. For each week,
    the model is fit on the preceding `train_weeks` weeks: a long first fit, then short fits
    warm-started from the previous window. It then predicts y for the current week. Every
    week is fit so the warm start stays continuous, but predictions are kept only for the
    evaluation window (config.evaluation_start to evaluation_end), or for all weeks if
    `expanded`. Position changes are then scored as
        actual_change    = mm_net_t - mm_net_{t-1}
        predicted_change = (y_hat_t - y_hat_{t-1}) * mm_std_lag_t
    Returns the intermediate tables, the predictions and the pooled metrics.
    """
    set_deterministic(config.seed)
    prices, meta, mm = load_inputs(config)
    rolling = build_rolling_contract(prices, meta)
    daily = build_daily_features(rolling)
    panel = build_weekly_panel(daily, mm)
    panel["year"] = panel["report_week_tuesday"].dt.year
    if expanded:
        evaluation_dates = sorted(panel["report_week_tuesday"].unique())
    else:
        evaluation_dates = sorted(
            panel.loc[
                panel["report_week_tuesday"].between(
                    config.evaluation_start, config.evaluation_end
                ),
                "report_week_tuesday",
            ].unique()
        )
    evaluation_dates = set(pd.to_datetime(evaluation_dates))
    dates = sorted(panel["report_week_tuesday"].unique())
    previous_state = None
    predictions = []
    for evaluation_date in dates:
        eligible = panel.loc[panel["report_week_tuesday"] < evaluation_date]
        train_dates = sorted(eligible["report_week_tuesday"].unique())[-config.train_weeks:]
        if len(train_dates) < config.train_weeks:
            continue
        train = eligible.loc[eligible["report_week_tuesday"].isin(train_dates)]
        if train["market"].nunique() != len(MARKETS):
            continue
        _, x_train, y_train = _arrays(train)
        model = fit_window(x_train, y_train, config, previous_state)
        previous_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if pd.Timestamp(evaluation_date) not in evaluation_dates:
            continue
        current = panel.loc[panel["report_week_tuesday"].eq(evaluation_date)]
        if current["market"].nunique() != len(MARKETS):
            continue
        _, x_current, _ = _arrays(current)
        with torch.no_grad():
            # A single week does not match the training bias length, so forward() uses each market's last training bias
            current_prediction, _ = model.forward(x_current)
        prediction = current[["market", "report_week_tuesday", "mm_net", "mm_std_lag"]].copy().rename(columns={"mm_net": "actual_position"})
        prediction["predicted_y"] = current_prediction.numpy()[0]
        prediction["y"] = current.set_index("market").reindex(MARKETS)["y"].to_numpy()
        prediction["predicted_position_change"] = np.nan
        prediction["actual_position_change"] = np.nan
        predictions.append(prediction)
    result = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    if not result.empty:
        result = result.sort_values(["market", "report_week_tuesday"])
        result["actual_change"] = result.groupby("market")["actual_position"].diff()
        result["predicted_change"] = result.groupby("market")["predicted_y"].diff() * result["mm_std_lag"]
    return {"rolling_contract": rolling, "daily_features": daily, "weekly_panel": panel, "predictions": result, "metrics": _metrics(result) if not result.empty else pd.Series(dtype=float)}
