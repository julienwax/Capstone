"""Helpers for `Replicating_CTA.ipynb`: Kestner (2020), "Replicating CTA Positioning: An Improved Method".

Kestner explains weekly SG Trend Index returns with a volatility-scaled momentum portfolio on 16
liquid futures. Every week, at the last trading day's close, each market gets

    signal  s = clip(average weekly return over L weeks / weekly volatility * sqrt(L), -cap, +cap)
    weight  w = s / (N * annualized volatility)          (equal risk per market, no correlations)

and the portfolio holds those weights over the following week. Futures data come from
`working_ideas/data_required/` (Bloomberg generic curves and contract chain); the SG Trend Index
from `paper_replication/data_required/sg_cta_indices.csv`.
"""
from itertools import product

import numpy as np
import pandas as pd

SECTORS = ["Equities", "Rates", "Currencies", "Commodities"]

# The paper's 125 parameter combinations (page 4).
LOOKBACKS = (4, 8, 16, 32, 52)        # momentum lookback, weeks
CAPS = (0.01, 0.5, 1.0, 1.5, 2.0)     # signal cap; listed as 0.0 but plotted as 0.01, since 0 holds nothing
VOL_WINDOWS = (20, 30, 60, 90, 180)   # realized volatility lookback, trading days
CHOSEN = {"cap": 1.0, "vol_window": 90}
ENSEMBLE = (16, 32, 52)               # the three lookbacks averaged in the final model

PAPER_WINDOW = ("2015", "2019")       # the paper's model charts
FULL_WINDOW = ("2007", "2019")        # the paper's SG Trend chart; also where leverage is fitted
OOS_WINDOW = ("2020", "2026")         # after the paper
AUM_USD = 300e9                       # trend-follower assets, Wall Street Journal estimate quoted by Kestner

# R² read off the paper's bar charts (pages 6-7); approximate to about 0.01.
PAPER_R2 = {
    "lookback": {4: 0.12, 8: 0.40, 16: 0.61, 32: 0.67, 52: 0.62},
    "cap": {0.01: 0.62, 0.5: 0.65, 1.0: 0.67, 1.5: 0.67, 2.0: 0.66},
    "vol_window": {20: 0.60, 30: 0.64, 60: 0.66, 90: 0.67, 180: 0.67},
}


# ---------------------------------------------------------------- data

def load_markets(data_dir, paper="kestner"):
    """Futures markets used by `paper`, one row each, indexed by market key, in sector order."""
    markets = pd.read_csv(data_dir / "futures_markets.csv", parse_dates=["first_date", "last_date"])
    markets = markets[markets["papers"].str.split("+").apply(lambda p: paper in p)]
    order = markets["sector"].map({s: i for i, s in enumerate(SECTORS)})
    return markets.assign(_order=order).sort_values("_order", kind="stable").drop(columns="_order").set_index("market")


def held_contract_returns(px, last_trade, roll_days=5):
    """Daily returns of the contract a trader holds, from generics 1 and 2 and the last trade dates.

    `px` has the generic prices by date, columns 1 and 2. `last_trade` holds the last trade dates of
    the contracts the generic steps through. The generic keeps the front contract through its last
    trade date and moves to the next one the following day. A trader holds the front contract until
    `roll_days` trading days before its last trade date and the next contract after that
    (`roll_days=0` follows the generic). Each day's return is measured on the contract held at the
    previous close, so no return ever spans two contracts. Non-positive prices are treated as missing,
    and a day without a valid generic 1 price is skipped: the next return then covers both days
    (WTI's −37.63 settlement on 2020-04-20 is the one case in Kestner's markets).
    """
    px = px.where(px > 0)
    g1 = px[1].dropna()
    dates = g1.index
    g1 = g1.to_numpy()
    g2 = (px[2].reindex(dates) if 2 in px else pd.Series(np.nan, index=dates)).to_numpy()
    ltd = np.sort(pd.to_datetime(pd.Series(last_trade)).unique())

    front = np.searchsorted(ltd, dates.to_numpy(), side="left")      # contract number of generic 1
    has_ltd = front < len(ltd)
    expiry_pos = dates.searchsorted(ltd[np.minimum(front, len(ltd) - 1)])
    days_left = np.where(has_ltd, expiry_pos - np.arange(len(dates)), np.iinfo(int).max)
    held = front + (days_left < roll_days)                            # contract held at each close

    def price(contract, i):
        """Price at date i of a contract that is generic 1 or 2 that day."""
        return np.where(contract == front[i], g1[i], np.where(contract == front[i] + 1, g2[i], np.nan))

    i = np.arange(1, len(dates))
    # A contract that expired overnight (only with roll_days=0) is replaced by the next one at the
    # previous close, as the generic does.
    contract = np.maximum(held[i - 1], front[i])
    ret = price(contract, i) / price(contract, i - 1) - 1
    return pd.Series(np.r_[np.nan, ret], index=dates, name="return")


def daily_returns(data_dir, markets, roll_days=5):
    """Daily held-contract returns, one column per market (local currency, excess of cash)."""
    gen = pd.read_csv(data_dir / "futures_generic_prices.csv.gz", parse_dates=["date"])
    gen = gen[gen["market"].isin(markets.index) & gen["nearby"].isin([1, 2])]
    gen["px"] = gen["PX_SETTLE"].fillna(gen["PX_LAST"])
    chain = pd.read_csv(data_dir / "futures_contract_chain.csv", parse_dates=["last_trade_date"])
    chain = chain[chain["in_generic_cycle"]]
    out = {}
    for m in markets.index:
        px = gen[gen["market"] == m].pivot_table(index="date", columns="nearby", values="px")
        out[m] = held_contract_returns(px, chain.loc[chain["market"] == m, "last_trade_date"], roll_days)
    return pd.DataFrame(out).sort_index()


def load_sg_trend(path):
    """Daily SG Trend Index level (rebased to 100 on 2000-01-01)."""
    sg = pd.read_csv(path, parse_dates=["trade_date"]).set_index("trade_date")
    return sg["SG Trend Index"].dropna()


# ---------------------------------------------------------------- weekly model

def weekly_returns(daily):
    """Compounded returns between consecutive weeks' last trading days (weeks end on Friday)."""
    weekly = (1 + daily.fillna(0)).resample("W-FRI").prod() - 1
    return weekly.where(daily.notna().resample("W-FRI").sum() > 0)


def weekly_level_returns(level):
    """Weekly returns of a daily index level, from one week's last value to the next."""
    return level.resample("W-FRI").last().pct_change(fill_method=None)


def weekly_vol(daily, window):
    """Weekly volatility: standard deviation of the last `window` daily returns times sqrt(5), at week end."""
    vol = daily.rolling(window, min_periods=int(0.75 * window)).std()
    return vol.resample("W-FRI").last() * np.sqrt(5)


def signals(weekly, vol, lookback, cap):
    """Kestner's normalized momentum: average weekly return / weekly vol * sqrt(lookback), capped."""
    return (weekly.rolling(lookback).mean() / vol * np.sqrt(lookback)).clip(-cap, cap)


def weights(sig, vol):
    """Notional weights per unit of capital: signal / annualized vol, divided by the number of markets."""
    w = sig / (vol * np.sqrt(52))
    return w.div(w.notna().sum(axis=1), axis=0)


def portfolio(weekly, w):
    """Weekly portfolio returns: weights set at one week's close, earned over the next week."""
    return (w.shift(1) * weekly).sum(axis=1, min_count=1)


def model(weekly, vols, lookback, cap=CHOSEN["cap"], vol_window=CHOSEN["vol_window"]):
    """One parameter set: (weekly returns, weights), unscaled. `vols` maps vol windows to `weekly_vol`."""
    w = weights(signals(weekly, vols[vol_window], lookback, cap), vols[vol_window])
    return portfolio(weekly, w), w


def window(x, span):
    return x.loc[span[0]:span[1]]


def r2(x, y, span):
    """Coefficient of determination of a one-variable regression: squared correlation over `span`."""
    joint = window(pd.concat([x, y], axis=1), span).dropna()
    return joint.corr().iloc[0, 1] ** 2


def leverage(x, y, span=FULL_WINDOW):
    """Scale that gives `x` the volatility of the benchmark `y` over `span` (in sample, as in the paper)."""
    joint = window(pd.concat([x, y], axis=1), span).dropna()
    return joint.iloc[:, 1].std() / joint.iloc[:, 0].std()


def r2_grid(weekly, vols, sg_weekly, spans):
    """R² of all 125 parameter combinations over each span in `spans` (name -> (start, end))."""
    rows = []
    for lookback, cap, vol_window in product(LOOKBACKS, CAPS, VOL_WINDOWS):
        ret, _ = model(weekly, vols, lookback, cap, vol_window)
        row = {"lookback": lookback, "cap": cap, "vol_window": vol_window,
               "label": f"{lookback}/{cap:g}/{vol_window}"}
        row.update({name: r2(ret, sg_weekly, span) for name, span in spans.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def ensemble(weekly, vols, sg_weekly, lookbacks=ENSEMBLE, span=FULL_WINDOW):
    """The paper's final model: the three lookbacks, each scaled to SG Trend volatility, averaged.

    Returns the scaled weekly returns of each model and of the ensemble, and the ensemble's weights
    (notional per unit of capital), rescaled so that the ensemble also has SG Trend's volatility.
    """
    returns, scaled = {}, []
    for lookback in lookbacks:
        ret, w = model(weekly, vols, lookback)
        k = leverage(ret, sg_weekly, span)
        returns[f"{lookback}/1/90"] = ret * k
        scaled.append(w * k)
    w = sum(scaled) / len(scaled)
    ret = portfolio(weekly, w)
    k = leverage(ret, sg_weekly, span)
    returns["Ensemble"] = ret * k
    return pd.DataFrame(returns), w * k


# ---------------------------------------------------------------- exposures and regression

def sector_exposure(w, markets):
    """Notional exposure per $1 of capital, summed by sector."""
    return w.T.groupby(markets["sector"]).sum().T[SECTORS]


def rolling_beta(y, x, weeks=20):
    """Rolling OLS slope of `y` on `x` over `weeks` weekly returns."""
    joint = pd.concat([y, x], axis=1).dropna()
    cov = joint.iloc[:, 0].rolling(weeks).cov(joint.iloc[:, 1])
    return cov / joint.iloc[:, 1].rolling(weeks).var()


def lagged_correlation(a, b, lags):
    """corr(a_t, b_{t-k}) for each k in `lags`: a peak at k > 0 means `a` follows `b` by k weeks."""
    return pd.Series({k: a.corr(b.shift(k)) for k in lags}, name="correlation")


def yearly_r2(x, y, years):
    """R² within each calendar year."""
    return pd.Series({year: r2(x, y, (str(year), str(year))) for year in years}, name="R2")
