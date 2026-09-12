from pathlib import Path
import numpy as np
import pandas as pd

from .config import HORIZONS, MARKETS, ReplicationConfig


def load_inputs(config: ReplicationConfig):
    root = Path(config.data_dir)
    prices = pd.read_csv(root / "futures_generic_prices.csv", parse_dates=["date"])
    meta = pd.read_csv(root / "contract_meta.csv", parse_dates=["LAST_TRADEABLE_DT"])
    mm = pd.read_csv(root / "managed_money.csv", parse_dates=["report_week_tuesday"])
    mm = mm.loc[mm["basis"].eq(config.basis)].copy()
    mm = mm.loc[mm["market"].isin(MARKETS)]
    mm = mm.sort_values(["market", "report_week_tuesday"])
    if mm.duplicated(["market", "report_week_tuesday"]).any():
        raise ValueError("Duplicate Managed Money market/week rows after basis selection")
    return prices, meta, mm


def _last_business_day(year, month):
    end = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    return end if end.weekday() < 5 else end - pd.offsets.BDay(1)


def build_rolling_contract(prices, meta):
    prices = prices.loc[prices["market"].isin(MARKETS)].copy()
    prices["settle"] = prices["PX_SETTLE"].fillna(prices["PX_LAST"])
    prices = prices.dropna(subset=["settle"])
    outputs = []
    for market, group in prices.groupby("market", sort=False):
        group = group.sort_values(["date", "nearby"])
        market_meta = meta.loc[meta["market"].eq(market)].sort_values("LAST_TRADEABLE_DT")
        expiries = market_meta["LAST_TRADEABLE_DT"].to_numpy(dtype="datetime64[ns]")
        roll_dates = np.array([
            _last_business_day(
                (pd.Timestamp(ts).replace(day=1) - pd.offsets.Day(1)).year,
                (pd.Timestamp(ts).replace(day=1) - pd.offsets.Day(1)).month,
            )
            for ts in expiries
        ], dtype="datetime64[ns]")
        # For each date, the first metadata expiry not yet passed is the front contract.
        dates = group["date"].drop_duplicates().sort_values()
        front_index = np.searchsorted(expiries, dates.to_numpy(dtype="datetime64[ns]"), side="left")
        front_index = np.clip(front_index, 0, len(expiries) - 1)
        front = pd.DataFrame({"date": dates.to_numpy(), "front_index": front_index})
        front["roll_date"] = roll_dates[front_index]
        front["selected_nearby"] = np.where(front["date"] <= front["roll_date"], 1, 2)
        selected = group.merge(front[["date", "front_index", "roll_date", "selected_nearby"]], on="date", how="inner")
        selected = selected.loc[selected["nearby"].eq(selected["selected_nearby"])]
        selected = selected.rename(columns={"settle": "price"})
        selected["contract_key"] = selected["front_index"].astype(int)
        selected = selected.sort_values("date")
        selected["daily_pnl"] = selected["price"].diff().where(selected["contract_key"].eq(selected["contract_key"].shift(1)), 0.0)
        selected["daily_pnl"] = selected["daily_pnl"].fillna(0.0)
        selected["cum_pnl"] = selected["daily_pnl"].cumsum()
        selected["market"] = market
        outputs.append(selected[["market", "date", "price", "selected_nearby", "contract_key", "roll_date", "daily_pnl", "cum_pnl"]])
    return pd.concat(outputs, ignore_index=True).sort_values(["market", "date"]).reset_index(drop=True)


def _ewm_volatility(values, decay=60 / 61):
    squared = pd.Series(values, dtype=float).shift(1).pow(2)
    weights = decay ** np.arange(len(squared))[::-1]
    weighted = squared.fillna(0).to_numpy() * weights
    denom = np.where(np.isnan(squared.to_numpy()), 0.0, weights).cumsum()
    return np.sqrt(252 * np.divide(np.cumsum(weighted), np.maximum(denom, 1e-12)))


def build_daily_features(rolling):
    frames = []
    for market, group in rolling.groupby("market", sort=False):
        group = group.sort_values("date").copy()
        group["pnl_vol"] = _ewm_volatility(group["daily_pnl"])
        for horizon in HORIZONS:
            mom = (group["cum_pnl"] - group["cum_pnl"].rolling(horizon, min_periods=horizon).mean()) / group["pnl_vol"].replace(0, np.nan)
            group[f"mom_{horizon}"] = mom
            group[f"x_{horizon}"] = mom / mom.rolling(252, min_periods=252).std().replace(0, np.nan)
        frames.append(group)
    return pd.concat(frames, ignore_index=True).sort_values(["market", "date"]).reset_index(drop=True)


def build_weekly_panel(daily, mm):
    feature_cols = [f"x_{h}" for h in HORIZONS]
    daily = daily.sort_values("date")
    weekly = []
    for market, group in daily.groupby("market", sort=False):
        target_dates = mm.loc[mm["market"].eq(market), "report_week_tuesday"].sort_values().unique()
        sampled = pd.merge_asof(
            pd.DataFrame({"report_week_tuesday": target_dates}).sort_values("report_week_tuesday"),
            group[["date"] + feature_cols].rename(columns={"date": "feature_date"}).sort_values("feature_date"),
            left_on="report_week_tuesday", right_on="feature_date", direction="backward")
        sampled["market"] = market
        weekly.append(sampled)
    features = pd.concat(weekly, ignore_index=True)
    panel = mm.merge(features, on=["market", "report_week_tuesday"], how="inner")
    panel = panel.sort_values(["market", "report_week_tuesday"])
    panel["mm_mean_lag"] = panel.groupby("market")["mm_net"].transform(lambda s: s.rolling(52, min_periods=52).mean().shift(1))
    panel["mm_std_lag"] = panel.groupby("market").apply(lambda g: (g["mm_net"] - g["mm_net"].rolling(52, min_periods=52).mean()).rolling(52, min_periods=52).std().shift(1), include_groups=False).reset_index(level=0, drop=True)
    panel["y"] = (panel["mm_net"] - panel["mm_mean_lag"]) / panel["mm_std_lag"].replace(0, np.nan)
    return panel.dropna(subset=feature_cols + ["y"]).reset_index(drop=True)
