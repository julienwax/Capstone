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
    # Recursive numerator/denominator avoid underflow and dependence on the
    # length of future data (a global weight floor distorted early observations).
    values = np.asarray(values, dtype=float)
    result = np.zeros(len(values))
    numerator = denominator = 0.0
    for t in range(1, len(values)):
        numerator *= decay
        denominator *= decay
        if np.isfinite(values[t - 1]):
            numerator += values[t - 1] ** 2
            denominator += 1.0
        result[t] = np.sqrt(252 * numerator / denominator) if denominator else np.nan
    return result



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
        feature_columns = {}
        for horizon in HORIZONS:
            mom = (group["cum_pnl"] - group["cum_pnl"].rolling(horizon, min_periods=horizon).mean()) / group["pnl_vol"].replace(0, np.nan)
            feature_columns[f"mom_{horizon}"] = mom
            feature_columns[f"x_{horizon}"] = mom / mom.rolling(252, min_periods=252).std().replace(0, np.nan)
        group = pd.concat([group, pd.DataFrame(feature_columns, index=group.index)], axis=1)
        frames.append(group)
    return pd.concat(frames, ignore_index=True).sort_values(["market", "date"]).reset_index(drop=True)


def build_weekly_panel(daily, mm):
    """Paper §2.2 / footnote 5 on each market's trading calendar.

    A year contains 252 trading dates. Residual squares are NOT demeaned.
    Statistics are lagged to the preceding report; holidays use actual as-of dates.
    Full windows are required, including historical means for every residual.
    """
    feature_cols = [f"x_{h}" for h in HORIZONS]
    frames = []
    for market, group in daily.groupby("market", sort=False):
        group = group.sort_values("date")
        w = mm.loc[mm["market"].eq(market)].sort_values("report_date").copy()
        w = pd.merge_asof(w, group[["date"] + feature_cols].rename(
            columns={"date": "feature_date"}), left_on="report_date",
            right_on="feature_date", direction="backward")
        dates = group["date"].to_numpy()
        indices = np.searchsorted(dates, w["report_date"].to_numpy(), side="right") - 1
        values = w["mm_net"].to_numpy(float)
        means = np.full(len(w), np.nan)
        sigma = np.full(len(w), np.nan)
        for i, day in enumerate(indices):
            if day < 251:
                continue
            left = np.searchsorted(indices, day - 251, side="left")
            # Require position history covering the entire trading-year window.
            if indices[0] > day - 251:
                continue
            means[i] = values[left:i + 1].mean()
            residuals = values[left:i + 1] - means[left:i + 1]
            if len(residuals) > 1 and np.isfinite(residuals).all():
                sigma[i] = np.sqrt(np.sum(residuals ** 2) / (len(residuals) - 1))
        w["mm_mean_lag"] = pd.Series(means).shift(1).to_numpy()
        w["mm_std_lag"] = pd.Series(sigma).shift(1).to_numpy()
        w["actual_change"] = w["mm_net"].diff()
        w["previous_report_week"] = w["report_week_tuesday"].shift(1)
        w["y"] = (w["mm_net"] - w["mm_mean_lag"]) / w["mm_std_lag"].replace(0, np.nan)
        frames.append(w)
    panel = pd.concat(frames, ignore_index=True)
    panel["panel_date"] = panel.groupby("report_week_tuesday")["report_date"].transform("min")
    return panel.dropna(subset=feature_cols + ["y"]).reset_index(drop=True)
