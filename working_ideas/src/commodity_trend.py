"""Helpers for `brent_crude_analysis.ipynb`: trend-follower positions implied by the 16-futures SG Trend
replication (Kestner, 2020; `kestner.py`) in every commodity with Managed Money data, and their link to
weekly Managed Money changes.

The replication holds, in each of its 16 markets, the notional

    w = k * mean_L(k_L * s_L) / (16 * annualized volatility),    L = 16, 32, 52 weeks,

with s_L Kestner's capped momentum signal and the leverages k_L and k fitted on the 16 markets (`K.ensemble`).
Applying the same rule to a market outside the 16 gives the position the replication would hold there if
the market were one of its 16, with the same risk budget.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import norm

import kestner as K

ENERGY = ["BRENT", "WTI", "GASOIL", "HEATOIL", "RBOB", "NATGAS"]
OTHER = ["GC", "SI", "PL", "PA", "HG", "C", "W", "S", "KC", "CC", "SB", "CT"]
NAMES = {"BRENT": "ICE Brent", "WTI": "CME WTI", "GASOIL": "ICE Gasoil", "HEATOIL": "CME Heating oil",
         "RBOB": "CME RBOB gasoline", "NATGAS": "CME Natural gas", "GC": "Gold", "SI": "Silver", "PL": "Platinum",
         "PA": "Palladium", "HG": "Copper", "C": "Corn", "W": "Wheat (SRW)", "S": "Soybeans", "KC": "Coffee",
         "CC": "Cocoa", "SB": "Sugar No. 11", "CT": "Cotton"}
EXCHANGE = {"BRENT": "ICE Futures Europe", "GASOIL": "ICE Futures Europe", "WTI": "NYMEX", "HEATOIL": "NYMEX",
            "RBOB": "NYMEX", "NATGAS": "NYMEX", "PL": "NYMEX", "PA": "NYMEX", "GC": "COMEX", "SI": "COMEX",
            "HG": "COMEX", "C": "CBOT", "W": "CBOT", "S": "CBOT", "KC": "ICE US", "CC": "ICE US", "SB": "ICE US",
            "CT": "ICE US"}
# Energy futures that only `paper_replication`'s data has: USD per point (heating oil and RBOB settle in cents/gallon).
PAPER_ENERGY = {"BRENT": 1000, "GASOIL": 100, "HEATOIL": 420, "RBOB": 420}
RENAME = {"CL": "WTI", "NG": "NATGAS"}


def load_commodities(data_dir, paper_data_dir, calendar, roll_days=5):
    """Daily held-contract returns and prices of the 18 commodities, on the trading days in `calendar`.

    The 14 commodities of `working_ideas/data_required/` and Brent, gasoil, heating oil and RBOB from
    `paper_replication/data_required/`, all rolled `roll_days` before the last trade date as in `kestner.py`.
    Returns (returns, prices, USD per point). Raises if a return falls outside `calendar`.
    """
    markets = pd.read_csv(data_dir / "futures_markets.csv").set_index("market")
    markets = markets[markets["sector"] == "Commodities"]
    returns = [K.daily_returns(data_dir, markets, roll_days).rename(columns=RENAME)]
    prices = [K.daily_prices(data_dir, markets, roll_days).rename(columns=RENAME)]
    generic = pd.read_csv(paper_data_dir / "futures_generic_prices.csv", parse_dates=["date"])
    generic["px"] = generic["PX_SETTLE"].fillna(generic["PX_LAST"])
    meta = pd.read_csv(paper_data_dir / "contract_meta.csv", parse_dates=["LAST_TRADEABLE_DT"])
    for market in PAPER_ENERGY:
        px = (generic[(generic["market"] == market) & generic["nearby"].isin([1, 2])]
              .pivot_table(index="date", columns="nearby", values="px"))
        last_trade = meta.loc[meta["market"] == market, "LAST_TRADEABLE_DT"]
        returns.append(K.held_contract_returns(px, last_trade, roll_days).rename(market))
        prices.append(K.held_contract_prices(px, last_trade, roll_days).rename(market))
    returns, prices = pd.concat(returns, axis=1, sort=True), pd.concat(prices, axis=1, sort=True)
    outside = returns.dropna(how="all").index.difference(calendar)
    if len(outside):
        raise ValueError(f"{len(outside)} return dates are not in the calendar, e.g. {outside[0]:%Y-%m-%d}")
    point_value = markets["point_value"].rename(RENAME).to_dict() | PAPER_ENERGY
    return returns.reindex(calendar)[ENERGY + OTHER], prices.reindex(calendar)[ENERGY + OTHER], pd.Series(point_value)


def ensemble_scales(weekly, vol, sg_weekly):
    """The replication's fit on its markets, as in `K.ensemble`.

    Returns each lookback's leverage, the leverage of their average, and each lookback model's number
    of markets by week (16 once every market has enough history).
    """
    models = {L: K.model(weekly, {90: vol}, L) for L in K.ENSEMBLE}
    scales = {L: K.leverage(ret, sg_weekly) for L, (ret, _) in models.items()}
    average = sum(w * scales[L] for L, (_, w) in models.items()) / len(models)
    counts = {L: w.notna().sum(axis=1) for L, (_, w) in models.items()}
    return scales, K.leverage(K.portfolio(weekly, average), sg_weekly), counts


def implied_weights(weekly, vol, scales, k, counts, cap=K.CHOSEN["cap"]):
    """Notional per $1 of capital the replication would hold in each market, as one of its markets.

    Uses `ensemble_scales`' output, so a market of the replication gets exactly its replication weight.
    """
    per_lookback = [k_L * (K.signals(weekly, vol, L, cap) / (vol * np.sqrt(52))).div(counts[L], axis=0)
                    for L, k_L in scales.items()]
    return k * sum(per_lookback) / len(per_lookback)


def capped(weekly, vol, cap=K.CHOSEN["cap"]):
    """Weeks in which the signals of all three lookbacks sit at the cap: positions then move only with sizing."""
    return sum((K.signals(weekly, vol, L, cap).abs() >= cap).astype(int) for L in K.ENSEMBLE) == len(K.ENSEMBLE)


def n_capped(weekly, vol, cap=K.CHOSEN["cap"]):
    """Number of lookbacks (0 to 3) whose signal sits at the cap, by week and market."""
    return sum((K.signals(weekly, vol, L, cap).abs() >= cap).astype(int) for L in K.ENSEMBLE)


def load_managed_money(paper_data_dir, data_dir):
    """Weekly futures-only Managed Money net positions (contracts), one column per market, by report week."""
    energy = pd.read_csv(paper_data_dir / "managed_money.csv", parse_dates=["report_week_tuesday"])
    other = pd.read_csv(data_dir / "cot_managed_money_commodities.csv", parse_dates=["report_week_tuesday"])
    mm = pd.concat([energy[energy["basis"] == "futures_only"], other])
    return mm.pivot(index="report_week_tuesday", columns="market", values="mm_net")[ENERGY + OTHER]


def trend_panel(data_dir, paper_data_dir, end, roll_days=5, week_end="W-TUE"):
    """Weekly Managed Money and SG-implied trend positions of the 18 commodities, on COT report weeks.

    The implied position in contracts is P = u * h, with u the replication's risk weight (its signals and
    leverages) and h = capital / (annualized volatility * price * point value) the sizing. Its weekly change
    splits exactly into signal flow, (u_t - u_{t-1}) * h_{t-1}, and sizing flow, u_t * (h_t - h_{t-1}).
    When signals sit at the cap only the sizing flow moves. All frames have one column per market:

    mm, d_mm: net position and its weekly change; ret: the week's return on the held contract;
    contracts, d_pos: implied position and its weekly change; signal_flow, sizing_flow: the two parts of d_pos;
    n_capped: number of lookbacks (0 to 3) at the cap.
    """
    markets = K.load_markets(data_dir)
    daily16 = K.daily_returns(data_dir, markets, roll_days=roll_days)
    sg_weekly = K.weekly_level_returns(K.load_sg_trend(paper_data_dir / "sg_cta_indices.csv"), week_end)
    weekly16, vol16 = K.weekly_returns(daily16, week_end), K.weekly_vol(daily16, 90, week_end)
    scales, k, counts = ensemble_scales(weekly16, vol16, sg_weekly)
    returns, prices, point_value = load_commodities(data_dir, paper_data_dir, daily16.index, roll_days)
    weekly, vol = K.weekly_returns(returns, week_end), K.weekly_vol(returns, 90, week_end)

    mm = load_managed_money(paper_data_dir, data_dir).loc[:end]
    annual_vol = vol * np.sqrt(52)
    u = (implied_weights(weekly, vol, scales, k, counts) * annual_vol).reindex(mm.index)
    h = (K.AUM_USD / (annual_vol * prices.resample(week_end).last() * point_value)).reindex(mm.index)
    contracts = u * h
    return SimpleNamespace(mm=mm, d_mm=mm.diff(), ret=weekly.reindex(mm.index), contracts=contracts,
                           d_pos=contracts.diff(), signal_flow=u.diff() * h.shift(1), sizing_flow=u * h.diff(),
                           n_capped=n_capped(weekly, vol).reindex(mm.index))


# ---------------------------------------------------------------- statistics

def fisher_ci(r, n, level=0.95):
    """Confidence interval of a correlation from the Fisher transformation."""
    half = norm.ppf(0.5 + level / 2) / np.sqrt(n - 3)
    return np.tanh(np.arctanh(r) - half), np.tanh(np.arctanh(r) + half)


def fisher_p(r1, n1, r2, n2):
    """Two-sided p-value that two correlations from independent samples are equal."""
    z = (np.arctanh(r1) - np.arctanh(r2)) / np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    p = 2 * norm.sf(np.abs(z))
    return pd.Series(p, index=z.index) if isinstance(z, pd.Series) else p


def link(x, y, periods):
    """Correlation (with 95% interval), OLS slope of y on x and weeks, per market and period.

    `x` and `y` have one column per market; `periods` maps a label to (start, end).
    """
    rows = []
    for label, (start, end) in periods.items():
        for market in x.columns:
            pair = pd.concat([x[market].loc[start:end], y[market].loc[start:end]], axis=1).dropna()
            r, n = pair.iloc[:, 0].corr(pair.iloc[:, 1]), len(pair)
            low, high = fisher_ci(r, n)
            rows.append({"period": label, "market": market, "corr": r, "low": low, "high": high,
                         "slope": np.polyfit(pair.iloc[:, 0], pair.iloc[:, 1], 1)[0], "weeks": n})
    return pd.DataFrame(rows)


def bootstrap_gap(x, y, a, b, n_draws=10_000, seed=7):
    """Bootstrap (resampling weeks) of corr(x[a], y[a]) - corr(x[b], y[b]) on the same weeks: estimate and 95% interval."""
    pair = pd.concat([x[a], y[a], x[b], y[b]], axis=1).dropna().to_numpy()
    rng = np.random.default_rng(seed)

    def gap(rows):
        return np.corrcoef(rows[:, 0], rows[:, 1])[0, 1] - np.corrcoef(rows[:, 2], rows[:, 3])[0, 1]

    draws = np.array([gap(pair[rng.integers(0, len(pair), len(pair))]) for _ in range(n_draws)])
    return gap(pair), *np.nanpercentile(draws, [2.5, 97.5])
