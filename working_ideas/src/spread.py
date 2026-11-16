"""Mean-reversion portfolio on inter-commodity spreads, for `spread_mean_reversion.ipynb`.

A spread is one dollar long leg A and one dollar short leg B, both in the same delivery month. Its return is
the return of A minus the return of B, so a move common to both legs cancels. The rule uses prices only: it
sells the spread when its cumulated return is above its recent average and buys it when below, sized by the
spread's volatility. Its weekly position change in each leg is the implied spread flow, to set against Managed
Money changes, or the errors of the Insight 177 network, in the same way as the trend flow of
`commodity_trend.trend_panel`.

`basket_panel` applies the same rule to several markets at once: each market against the equal-weighted basket
of all of them. With two markets it is the pair.

Nothing calibrates the capital of spread traders, so `RISK_USD` is arbitrary: correlations, signs and R2 are
identified, the number of contracts is not.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

# Leg A, leg B. Leg A is the one whose contract expires first for a given delivery month: it sets the roll.
PAIRS = {"Brent-WTI": ("BRENT", "WTI"), "Heating oil-Gasoil": ("HEATOIL", "GASOIL")}
BASKET = ("BRENT", "WTI", "HEATOIL", "GASOIL")   # Brent first: its contract expires first, so it sets the roll
POINT_VALUE = {"BRENT": 1000, "WTI": 1000, "HEATOIL": 420, "RBOB": 420, "GASOIL": 100}   # USD per point, one contract
USD_PER_BBL = {"BRENT": 1.0, "WTI": 1.0, "HEATOIL": 0.42, "RBOB": 0.42, "GASOIL": 1 / 7.45}  # quote -> USD/bbl
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
    """Prices by date and delivery month (year * 12 + month - 1), from the first `nearbys` generics.

    Returns the price table and the last trade date of each delivery month.
    """
    m = meta[meta["market"] == market].sort_values("LAST_TRADEABLE_DT")
    ltd = m["LAST_TRADEABLE_DT"].to_numpy()
    month = (m["maturity_year"] * 12 + m["maturity_month"] - 1).to_numpy()
    g = generic[(generic["market"] == market) & (generic["nearby"] <= nearbys)].dropna(subset=["px"])
    number = np.searchsorted(ltd, g["date"].to_numpy(), side="left") + g["nearby"].to_numpy() - 1
    table = pd.DataFrame({"date": g["date"].to_numpy(), "month": month[number], "px": g["px"].to_numpy()})
    return table.pivot(index="date", columns="month", values="px"), pd.Series(ltd, index=month)


def held_month(dates, last_trade, roll_days=5):
    """Delivery month held after each close: the front contract until `roll_days` trading days before its
    last trade date, then the next one (as `kestner.held_contract_returns`)."""
    ltd = last_trade.to_numpy()
    front = np.searchsorted(ltd, dates.to_numpy(), side="left")
    days_left = dates.searchsorted(ltd[front]) - np.arange(len(dates))
    return pd.Series(last_trade.index[front + (days_left < roll_days)], index=dates)


def matched_returns(generic, meta, markets, roll_days=5):
    """Daily returns and prices of `markets`, all in the delivery month the first market holds, on their common days.

    The first market must be the one whose contract expires first for a given delivery month: it sets the roll.
    Each day's return is measured on the delivery month held at the previous close, so it never spans two
    contracts in any market. Days on which one of the exchanges is closed are skipped: the next return covers
    them. Prices are those of the month held after the close. Returns (returns, prices, held month).
    """
    tables = {m: contract_prices(generic, meta, m) for m in markets}
    dates = tables[markets[0]][0].index
    months = tables[markets[0]][0].columns
    for m in markets[1:]:
        dates, months = dates.intersection(tables[m][0].index), months.intersection(tables[m][0].columns)
    held = held_month(tables[markets[0]][0].index, tables[markets[0]][1], roll_days).reindex(dates)
    col, row = months.get_indexer(held.to_numpy()), np.arange(len(dates))
    returns, prices = {}, {}
    for m in markets:
        p = tables[m][0].loc[dates, months].to_numpy()
        returns[m] = np.r_[np.nan, p[row[1:], col[:-1]] / p[row[:-1], col[:-1]] - 1]
        prices[m] = p[row, col]
    return pd.DataFrame(returns, index=dates), pd.DataFrame(prices, index=dates), held


def daily_spread(generic, meta, leg_a, leg_b, roll_days=5):
    """Daily returns and prices of the two legs, both in the delivery month leg A holds (`matched_returns`).

    `ret` is the return of one dollar long `leg_a` and one dollar short `leg_b`. `price_a` and `price_b` are the
    prices of the month held after the close, and `ratio` is their ratio in USD per barrel.
    """
    returns, prices, held = matched_returns(generic, meta, [leg_a, leg_b], roll_days)
    return pd.DataFrame({"ret": returns[leg_a] - returns[leg_b], "ret_a": returns[leg_a], "ret_b": returns[leg_b],
                         "price_a": prices[leg_a], "price_b": prices[leg_b],
                         "ratio": prices[leg_a] * USD_PER_BBL[leg_a] / (prices[leg_b] * USD_PER_BBL[leg_b]),
                         "month": held.to_numpy()})


def weekly_vol(daily_ret, window=90, week_end="W-TUE"):
    """Weekly volatility: standard deviation of the last `window` overlapping five-day sums of the spread return.

    Five-day sums, not daily returns times sqrt(5): gasoil settles in London about three hours before heating
    oil in New York, so the daily spread return reverses the next day and overstates weekly risk.
    """
    vol = daily_ret.rolling(5).sum().rolling(window, min_periods=int(0.75 * window)).std()
    return vol.resample(week_end).last()


def reversion_signal(cum, vol, lookbacks=LOOKBACKS, cap=CAP):
    """Minus the capped distance of the cumulated spread return from its `n`-week average, by lookback.

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

    The rule holds D = u * risk / annualized volatility dollars of leg A and -D dollars of leg B, with u the
    average signal. In contracts, leg m holds +/- u * h_m with h_m = risk / (annualized volatility * price *
    point value). The weekly change splits exactly into signal flow, +/- (u_t - u_{t-1}) * h_{m,t-1}, and
    sizing flow, +/- u_t * (h_{m,t} - h_{m,t-1}), as in `commodity_trend.trend_panel`. `contracts`, `flow`,
    `signal_flow` and `sizing_flow` have one column per leg.
    """
    leg_a, leg_b = PAIRS[pair]
    daily = daily_spread(generic, meta, leg_a, leg_b, roll_days)
    count = daily["ret"].notna().resample(week_end).sum()
    ret = daily["ret"].resample(week_end).sum().where(count > 0)
    vol = weekly_vol(daily["ret"], vol_days, week_end)
    signals = reversion_signal(ret.cumsum(), vol, lookbacks, cap).reindex(weeks)
    u = signals.mean(axis=1, skipna=False)
    annual_vol = vol.reindex(weeks) * np.sqrt(52)
    price = daily[["price_a", "price_b"]].resample(week_end).last().reindex(weeks)
    h = pd.DataFrame({leg_a: risk_usd / (annual_vol * price["price_a"] * POINT_VALUE[leg_a]),
                      leg_b: -risk_usd / (annual_vol * price["price_b"] * POINT_VALUE[leg_b])})
    contracts = h.mul(u, axis=0)
    return SimpleNamespace(pair=pair, legs=(leg_a, leg_b), daily=daily, ret=ret.reindex(weeks), vol=vol.reindex(weeks),
                           ratio=daily["ratio"].resample(week_end).last().reindex(weeks), signals=signals, signal=u,
                           dollars=u * risk_usd / annual_vol, contracts=contracts, flow=contracts.diff(),
                           signal_flow=h.shift(1).mul(u.diff(), axis=0), sizing_flow=h.diff().mul(u, axis=0))


def basket_panel(generic, meta, weeks, markets=BASKET, roll_days=5, week_end="W-TUE", vol_days=90,
                 lookbacks=LOOKBACKS, cap=CAP, risk_usd=RISK_USD):
    """Weekly mean-reversion positions in several markets at once, on the report weeks `weeks`.

    Each market is traded against the equal-weighted basket of all `markets`: its relative return is its
    return minus the basket's, and the signal u_m is `reversion_signal` on the cumulated relative return. One
    dollar of the spread "market m minus basket" is one dollar long m and 1/N dollars short every market, so
    with G_m = u_m * (risk / N) / annualized volatility of the relative return, the dollars held in market m are

        D_m = G_m - mean over the markets of G,

    which sum to zero. Each of the N spreads gets risk / N, so that with two markets this is exactly
    `spread_panel`: D_A = -D_B = u * risk / annualized volatility of r_A - r_B.
    All frames have one column per market: relative (weekly relative returns), vol, signal, dollars, contracts,
    flow. `rule_return` is the portfolio's weekly P&L in dollars.
    """
    markets = list(markets)
    returns, prices, _ = matched_returns(generic, meta, markets, roll_days)
    daily = returns.sub(returns.mean(axis=1), axis=0)
    count = daily.notna().resample(week_end).sum()
    relative = daily.resample(week_end).sum().where(count > 0)
    vol = pd.DataFrame({m: weekly_vol(daily[m], vol_days, week_end) for m in markets})
    signal = pd.DataFrame({m: reversion_signal(relative[m].cumsum(), vol[m], lookbacks, cap).mean(axis=1, skipna=False)
                           for m in markets}).reindex(weeks)
    gross = signal * (risk_usd / len(markets)) / (vol.reindex(weeks) * np.sqrt(52))
    dollars = gross.sub(gross.mean(axis=1, skipna=False), axis=0)
    contracts = dollars / (prices.resample(week_end).last().reindex(weeks) * pd.Series(POINT_VALUE)[markets])
    weekly = returns.resample(week_end).sum().reindex(weeks)
    return SimpleNamespace(markets=markets, relative=relative.reindex(weeks), vol=vol.reindex(weeks), signal=signal,
                           dollars=dollars, contracts=contracts, flow=contracts.diff(), returns=weekly,
                           rule_return=(dollars.shift(1) * weekly).sum(axis=1, min_count=len(markets)))
