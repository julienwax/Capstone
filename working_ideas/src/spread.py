"""Mean-reversion portfolio on inter-commodity spreads, for `spread_mean_reversion.ipynb`.

A spread is long leg A and short leg B in equal barrels, both in the same delivery month. The rule uses prices
only: it sells the spread when it is above its recent average and buys it when below, sized by the spread's
volatility. Its weekly position change in each leg is the implied spread flow, to set against Managed Money
changes, or with the errors of the Insight 177 network, in the same way as the trend flow of
`commodity_trend.trend_panel`.

Nothing calibrates the capital of spread traders, so `RISK_USD` is arbitrary: correlations, signs and R2 are
identified, the number of contracts is not.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

# Leg A, leg B. Leg A is the one whose contract expires first for a given delivery month: it sets the roll.
PAIRS = {"Brent-WTI": ("BRENT", "WTI"), "Heating oil-Gasoil": ("HEATOIL", "GASOIL")}
USD_PER_BBL = {"BRENT": 1.0, "WTI": 1.0, "HEATOIL": 0.42, "RBOB": 0.42, "GASOIL": 1 / 7.45}  # quote -> USD/bbl
BBL_PER_CONTRACT = {"BRENT": 1000, "WTI": 1000, "HEATOIL": 1000, "RBOB": 1000, "GASOIL": 745}
LOOKBACKS = (4, 13, 26)   # weeks
CAP = 1.0
RISK_USD = 1e8            # annualized dollar volatility of a spread position at full signal (arbitrary)


def load_energy(paper_data_dir):
    """Generic prices (with `px`: settle, else last) and contract metadata of `paper_replication`."""
    generic = pd.read_csv(paper_data_dir / "futures_generic_prices.csv", parse_dates=["date"])
    generic["px"] = generic["PX_SETTLE"].fillna(generic["PX_LAST"])
    meta = pd.read_csv(paper_data_dir / "contract_meta.csv", parse_dates=["LAST_TRADEABLE_DT"])
    return generic, meta


def contract_prices(generic, meta, market, nearbys=4):
    """Prices in USD/bbl by date and delivery month (year * 12 + month - 1), from the first `nearbys` generics.

    Returns the price table and the last trade date of each delivery month.
    """
    m = meta[meta["market"] == market].sort_values("LAST_TRADEABLE_DT")
    ltd = m["LAST_TRADEABLE_DT"].to_numpy()
    month = (m["maturity_year"] * 12 + m["maturity_month"] - 1).to_numpy()
    g = generic[(generic["market"] == market) & (generic["nearby"] <= nearbys)].dropna(subset=["px"])
    number = np.searchsorted(ltd, g["date"].to_numpy(), side="left") + g["nearby"].to_numpy() - 1
    table = pd.DataFrame({"date": g["date"].to_numpy(), "month": month[number], "px": g["px"].to_numpy()})
    return table.pivot(index="date", columns="month", values="px") * USD_PER_BBL[market], pd.Series(ltd, index=month)


def held_month(dates, last_trade, roll_days=5):
    """Delivery month held after each close: the front contract until `roll_days` trading days before its
    last trade date, then the next one (as `kestner.held_contract_returns`)."""
    ltd = last_trade.to_numpy()
    front = np.searchsorted(ltd, dates.to_numpy(), side="left")
    days_left = dates.searchsorted(ltd[front]) - np.arange(len(dates))
    return pd.Series(last_trade.index[front + (days_left < roll_days)], index=dates)


def daily_spread(generic, meta, leg_a, leg_b, roll_days=5):
    """Daily level and P&L of long `leg_a`, short `leg_b`, in USD/bbl, both legs in the delivery month leg A holds.

    The P&L of a day is measured on the delivery month held at the previous close, so it never spans two
    contracts in either leg. Days on which either exchange is closed are skipped: the next P&L covers them.
    """
    pa, ltd_a = contract_prices(generic, meta, leg_a)
    pb, _ = contract_prices(generic, meta, leg_b)
    held = held_month(pa.index, ltd_a, roll_days)
    months = pa.columns.intersection(pb.columns)
    value = (pa[months] - pb[months]).dropna(how="all")
    held = held.reindex(value.index)
    v = value.to_numpy()
    col = months.get_indexer(held.to_numpy())
    row = np.arange(len(v))
    level = v[row, col]
    pnl = np.r_[np.nan, v[row[1:], col[:-1]] - v[row[:-1], col[:-1]]]
    return pd.DataFrame({"level": level, "pnl": pnl, "month": held.to_numpy()}, index=value.index)


def weekly_vol(daily_pnl, window=90, week_end="W-TUE"):
    """Weekly volatility in USD/bbl: standard deviation of the last `window` overlapping five-day P&L sums.

    Five-day sums, not daily P&L times sqrt(5): gasoil settles in London about three hours before heating oil
    in New York, so the daily spread P&L reverses the next day (autocorrelation -0.44) and overstates weekly risk.
    """
    vol = daily_pnl.rolling(5).sum().rolling(window, min_periods=int(0.75 * window)).std()
    return vol.resample(week_end).last()


def reversion_signal(cum, vol, lookbacks=LOOKBACKS, cap=CAP):
    """Minus the capped distance of the cumulated spread P&L from its `n`-week average, by lookback.

    The distance is divided by its standard deviation under a random walk with weekly volatility `vol`,
    vol * sqrt((n - 1)(2n - 1) / (6n)), so each signal has unit variance before the cap, as Kestner's momentum.
    """
    out = {}
    for n in lookbacks:
        z = (cum - cum.rolling(n).mean()) / (vol * np.sqrt((n - 1) * (2 * n - 1) / (6 * n)))
        out[n] = -z.clip(-cap, cap)
    return pd.DataFrame(out)


def spread_panel(generic, meta, pair, weeks, roll_days=5, week_end="W-TUE", vol_days=90,
                 lookbacks=LOOKBACKS, cap=CAP, risk_usd=RISK_USD):
    """Weekly mean-reversion position in one spread, on the report weeks `weeks`.

    The position, in units of 1,000 barrels of spread, is N = u * h, with u the average signal and
    h = risk / (annualized volatility * 1,000) the sizing. Its weekly change splits exactly into signal flow,
    (u_t - u_{t-1}) * h_{t-1}, and sizing flow, u_t * (h_t - h_{t-1}). `flow` has the change in contracts of
    each leg: positive in leg A and negative in leg B when the rule buys the spread.
    """
    leg_a, leg_b = PAIRS[pair]
    daily = daily_spread(generic, meta, leg_a, leg_b, roll_days)
    pnl = daily["pnl"].resample(week_end).sum().where(daily["pnl"].notna().resample(week_end).sum() > 0)
    vol = weekly_vol(daily["pnl"], vol_days, week_end)
    signals = reversion_signal(pnl.cumsum(), vol, lookbacks, cap)
    u = signals.mean(axis=1, skipna=False).reindex(weeks)
    h = (risk_usd / (vol * np.sqrt(52) * 1000)).reindex(weeks)
    units = u * h
    d_units = units.diff()
    flow = pd.DataFrame({leg_a: d_units * 1000 / BBL_PER_CONTRACT[leg_a],
                         leg_b: -d_units * 1000 / BBL_PER_CONTRACT[leg_b]})
    return SimpleNamespace(pair=pair, legs=(leg_a, leg_b), daily=daily, pnl=pnl.reindex(weeks),
                           level=daily["level"].resample(week_end).last().reindex(weeks), vol=vol.reindex(weeks),
                           signals=signals.reindex(weeks), signal=u, units=units, d_units=d_units,
                           signal_flow=u.diff() * h.shift(1), sizing_flow=u * h.diff(), flow=flow)
