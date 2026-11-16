"""Inputs of the network on the weeks in which all six markets report.

Price momentum is the replication's own x_5 ... x_250. Curve momentum is the same construction on a roll-free index
of the calendar spread. Both are kept on the report date and on the ten trading days before it (for the delay layer).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LAGS = 10
CURVE_HORIZONS = (5, 10, 20, 40, 80, 160)
CURVE_GAP = {"BRENT": 1, "WTI": 1, "GASOIL": 12, "HEATOIL": 12, "RBOB": 12, "NATGAS": 12}   # months between the two contracts


def find_repo(start="."):
    """The Capstone checkout: the first parent folder that holds paper_replication/."""
    p = Path(start).absolute()
    for q in [p, *p.parents]:
        if (q / "paper_replication" / "src").exists():
            return q
    raise FileNotFoundError("put this folder inside the Capstone repository (or pass its path)")


def spread_index(prices, meta, held, market, gap, calendar_fn):
    """Cumulated daily change of log(P_held / P_held+gap). Each day compares the same pair of contracts, so no roll jump."""
    g = prices[prices.market.eq(market)].copy()
    g["settle"] = g.PX_SETTLE.fillna(g.PX_LAST)
    g = g[g.settle > 0]
    expiries = meta.loc[meta.market.eq(market), "LAST_TRADEABLE_DT"].sort_values().to_numpy(dtype="datetime64[ns]")
    cal = calendar_fn(np.sort(g.date.unique()), expiries)
    q = g.merge(cal[["date", "front_index"]], on="date")
    q["contract"] = q.front_index + q.nearby - 1
    price = q.drop_duplicates(["contract", "date"]).set_index(["contract", "date"])["settle"]
    h = held.reset_index()[["date", "contract_key"]]
    k0, d0 = h.contract_key.shift(1).fillna(-1).astype(int), h.date.shift(1)        # contract held yesterday
    px = lambda k, d: price.reindex(pd.MultiIndex.from_arrays([k, d])).to_numpy()
    change = (np.log(px(k0, h.date)) - np.log(px(k0 + gap, h.date))) - (np.log(px(k0, d0)) - np.log(px(k0 + gap, d0)))
    change = pd.Series(np.nan_to_num(change), index=pd.DatetimeIndex(h.date))
    return change.cumsum(), change


def spread_level(prices, meta, held, market, gap, calendar_fn):
    """100 * ln(P_held / P_held+gap), daily: positive in backwardation, negative in contango."""
    g = prices[prices.market.eq(market)].copy()
    g["settle"] = g.PX_SETTLE.fillna(g.PX_LAST)
    g = g[g.settle > 0]
    expiries = meta.loc[meta.market.eq(market), "LAST_TRADEABLE_DT"].sort_values().to_numpy(dtype="datetime64[ns]")
    cal = calendar_fn(np.sort(g.date.unique()), expiries)
    q = g.merge(cal[["date", "front_index"]], on="date")
    q["contract"] = q.front_index + q.nearby - 1
    price = q.drop_duplicates(["contract", "date"]).set_index(["contract", "date"])["settle"]
    h = held.reset_index()[["date", "contract_key"]]
    px = lambda k: price.reindex(pd.MultiIndex.from_arrays([k, h.date])).to_numpy()
    return pd.Series(100 * (np.log(px(h.contract_key)) - np.log(px(h.contract_key + gap))), index=pd.DatetimeIndex(h.date)).ffill()


def momentum(level, change, horizons=CURVE_HORIZONS):
    """(level - n-day mean) / volatility, divided by its own 252-day std: the paper's input construction."""
    vol = np.sqrt((change ** 2).ewm(alpha=1 / 61).mean().shift(1)).replace(0, np.nan)
    out = {}
    for n in horizons:
        m = (level - level.rolling(n, min_periods=n).mean()) / vol
        out[n] = (m / m.rolling(252, min_periods=252).std()).clip(-5, 5)
    return pd.DataFrame(out)


def build(repo=None):
    repo = find_repo() if repo is None else Path(repo)
    sys.path.insert(0, str(repo / "paper_replication" / "src"))
    from paper_replication.config import HORIZONS, MARKETS, ReplicationConfig
    from paper_replication.data import (_held_contract_calendar, build_daily_features, build_rolling_contract,
                                        build_weekly_panel, load_inputs)
    prices, meta, mm = load_inputs(ReplicationConfig(data_dir=repo / "paper_replication" / "data_required"))
    daily = build_daily_features(build_rolling_contract(prices, meta))
    panel = build_weekly_panel(daily, mm).sort_values(["market", "report_week_tuesday"]).reset_index(drop=True)
    g = panel.groupby("market")
    panel["consecutive"] = g["report_week_tuesday"].shift(1).eq(panel["previous_report_week"])
    panel["flow_vol"] = g["actual_change"].transform(lambda s: s.rolling(52, min_periods=52).std().shift(1))

    markets, x_cols = list(MARKETS), [f"x_{h}" for h in HORIZONS]
    n = panel.groupby("report_week_tuesday")["market"].nunique()
    weeks = np.array(sorted(n[n.eq(len(markets))].index))
    rows = {m: panel[panel.market.eq(m)].set_index("report_week_tuesday").reindex(weeks) for m in markets}
    by_market = {m: daily[daily.market.eq(m)].sort_values("date").set_index("date") for m in markets}
    col = lambda c: np.stack([rows[m][c].to_numpy() for m in markets], axis=1)

    def lagged(frames):   # [weeks, markets, LAGS + 1, inputs]
        out = np.full((len(weeks), len(markets), LAGS + 1, next(iter(frames.values())).shape[1]), np.nan, dtype=np.float32)
        for j, m in enumerate(markets):
            pos = frames[m].index.get_indexer(pd.DatetimeIndex(rows[m].feature_date))
            v = frames[m].to_numpy()
            for lag in range(LAGS + 1):
                out[:, j, lag, :] = v[np.clip(pos - lag, 0, None)]
        return out

    spread, curve, slope = {}, {}, {}
    for m in markets:
        level, change = spread_index(prices, meta, by_market[m], m, CURVE_GAP[m], _held_contract_calendar)
        spread[m], curve[m] = level, momentum(level, change)
        slope[m] = pd.DataFrame({g_: spread_level(prices, meta, by_market[m], m, g_, _held_contract_calendar) for g_ in (1, 12)})
    return dict(repo=repo, weeks=weeks, markets=markets, horizons=list(HORIZONS),
                date=rows[markets[0]].panel_date.to_numpy(dtype="datetime64[ns]"),
                x=lagged({m: by_market[m][x_cols] for m in markets}), xc=np.nan_to_num(lagged(curve)),
                y=col("y"), sigma=col("mm_std_lag"), flow_vol=col("flow_vol"), change=col("actual_change"),
                consecutive=col("consecutive").astype(bool).all(1), spread=spread, curve=curve, daily=by_market,
                net=col("mm_net"), slope=slope, feature_date={m: pd.DatetimeIndex(rows[m].feature_date) for m in markets})
