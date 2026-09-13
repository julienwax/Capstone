from pathlib import Path
import numpy as np
import pandas as pd

from .config import HORIZONS, MARKETS, ReplicationConfig


def load_inputs(config: ReplicationConfig):
    """Read generic futures prices, contract expiry metadata and Managed Money positions.

    Managed Money is filtered to `config.basis` (futures_only or combined) and the six
    markets, sorted by market and week, and checked for one row per market per week.
    """
    root = Path(config.data_dir)
    prices = pd.read_csv(root / "futures_generic_prices.csv", parse_dates=["date"])
    meta = pd.read_csv(root / "contract_meta.csv", parse_dates=["LAST_TRADEABLE_DT"])
    mm = pd.read_csv(root / "managed_money.csv", parse_dates=["report_date", "report_week_tuesday"])
    mm = mm.loc[mm["basis"].eq(config.basis)].copy()
    mm = mm.loc[mm["market"].isin(MARKETS)]
    mm = mm.sort_values(["market", "report_week_tuesday"])
    if mm.duplicated(["market", "report_week_tuesday"]).any():
        raise ValueError("Duplicate Managed Money market/week rows after basis selection")
    return prices, meta, mm


def _held_contract_calendar(dates, expiries):
    """For each trading date, the front contract and the contract held at the close.

    Contracts are identified by their position in the sorted `expiries`. Paper footnote 1: each
    contract is rolled out of on the last business day of the month preceding its expiration,
    trading at that day's settlement. Business days are the market's own trading `dates`, so
    exchange holidays are respected. The contract held at the close of a date is therefore the
    first one whose roll date is after that date.

    front_index is the first contract whose last tradeable date has not passed (generic nearby 1)
    and nearby_offset = contract_key - front_index. The offset is usually 0 or 1, and 2 on the
    expiry day of contracts that expire on the last business day of a month (Heating Oil, RBOB,
    Brent since 2016): that day is also the next contract's roll date, while the expiring contract
    is still nearby 1. roll_date is the held contract's roll date. Dates after the last known
    contract's roll date are dropped.
    """
    expiry_month_start = expiries.astype("datetime64[M]").astype("datetime64[ns]")
    last_date_before = np.searchsorted(dates, expiry_month_start, side="left") - 1
    # Contracts whose roll month precedes the data get a roll date just before the first date
    roll_dates = np.where(last_date_before >= 0, dates[np.clip(last_date_before, 0, None)], dates[0] - np.timedelta64(1, "D"))
    contract_key = np.searchsorted(roll_dates, dates, side="right")
    valid = contract_key < len(expiries)
    dates, contract_key = dates[valid], contract_key[valid]
    front_index = np.searchsorted(expiries, dates, side="left")
    return pd.DataFrame({
        "date": dates,
        "front_index": front_index,
        "roll_date": roll_dates[contract_key],
        "nearby_offset": contract_key - front_index,
        "contract_key": contract_key,
    })


def build_rolling_contract(prices, meta):
    """Build one continuous rolling-contract series per market (paper footnote 1).

    `contract_key` is the contract held at the close and `price` its settlement price. Returns
    on day t are measured on the contract held at the close of t-1, using that contract's
    prices on both days, so a roll or an expiry never mixes two contracts:
    `daily_pnl` is the price change (price units, the dP_{m,t} used for momentum) and
    `daily_log_return` the log return (the benchmark regressor). `cum_pnl` and
    `cum_log_return` are their running sums.
    """
    # The held contract is at most nearby 3, and yesterday's held contract at most nearby 3 today
    prices = prices.loc[prices["market"].isin(MARKETS) & prices["nearby"].le(3)].copy()
    prices["settle"] = prices["PX_SETTLE"].fillna(prices["PX_LAST"])
    prices = prices.dropna(subset=["settle"])
    outputs = []
    for market, group in prices.groupby("market", sort=False):
        expiries = meta.loc[meta["market"].eq(market), "LAST_TRADEABLE_DT"].sort_values().to_numpy(dtype="datetime64[ns]")
        calendar = _held_contract_calendar(np.sort(group["date"].unique()), expiries)

        # Generic nearby n on a date is contract front_index + n - 1
        quotes = group.merge(calendar[["date", "front_index"]], on="date")
        quotes["contract_key"] = quotes["front_index"] + quotes["nearby"] - 1
        contract_price = quotes.set_index(["contract_key", "date"])["settle"]

        calendar["price"] = contract_price.reindex(pd.MultiIndex.from_arrays([calendar["contract_key"], calendar["date"]])).to_numpy()
        calendar = calendar.dropna(subset=["price"]).reset_index(drop=True)
        previous_contract = calendar["contract_key"].shift(1, fill_value=-1)
        previous_contract_today = contract_price.reindex(pd.MultiIndex.from_arrays([previous_contract, calendar["date"]])).to_numpy()
        previous_price = calendar["price"].shift(1).to_numpy()
        if (previous_contract_today <= 0).any() or (previous_price <= 0).any():
            raise ValueError(f"{market}: non-positive settlement price on a held contract")

        calendar["daily_pnl"] = np.nan_to_num(previous_contract_today - previous_price)
        calendar["daily_log_return"] = np.nan_to_num(np.log(previous_contract_today / previous_price))
        calendar["cum_pnl"] = calendar["daily_pnl"].cumsum()
        calendar["cum_log_return"] = calendar["daily_log_return"].cumsum()
        calendar["market"] = market
        outputs.append(calendar[[
            "market", "date", "price", "nearby_offset", "contract_key", "roll_date",
            "daily_pnl", "cum_pnl", "daily_log_return", "cum_log_return",
        ]])
    return pd.concat(outputs, ignore_index=True).sort_values(["market", "date"]).reset_index(drop=True)


def _ewm_volatility(values, decay=60 / 61):
    """Annualised exponentially weighted volatility, sigma_{m,t}(P) (paper footnote 3).

    On day t, squared P&L from days strictly before t is weighted by decay**(t - 1 - i),
    divided by the sum of weights, multiplied by 252 and square-rooted. Squares are not
    demeaned, and the average expands from the first observation.
    """
    squared = pd.Series(values, dtype=float).shift(1).pow(2)
    weights = decay ** np.arange(len(squared))[::-1]
    weighted = squared.fillna(0).to_numpy() * weights
    denom = np.where(np.isnan(squared.to_numpy()), 0.0, weights).cumsum()
    return np.sqrt(252 * np.divide(np.cumsum(weighted), np.maximum(denom, 1e-12)))


def build_daily_features(rolling):
    """Compute the normalised daily momentum inputs x_{m,t}(n) for n in HORIZONS.

    mom_n = (cum_pnl - n-day moving average of cum_pnl) / price volatility, and
    x_n = mom_n / its 252-day rolling standard deviation, so inputs are comparable across
    time and markets. Values stay NaN until each rolling window is full.
    """
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
    """Join weekly Managed Money to momentum inputs and build the dependent variable y.

    Features are taken from the last trading day on or before each report_week_tuesday.
    mm_mean_lag is the 52-week rolling mean of net positions, and mm_std_lag the 52-week
    rolling std of deviations from that mean; both are lagged one week. Then
    y = (mm_net - mm_mean_lag) / mm_std_lag (paper Section 2.2). Rows missing any
    feature or y are dropped.
    """
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
