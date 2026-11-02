"""Helpers for `position_analysis.ipynb`: co-movement of Managed Money across the six energy futures.

Weekly changes in COT Managed Money net positions (futures only) for WTI, Brent, Gasoil, Heating oil,
RBOB and Natural gas: their correlations over time, clusters of markets and of periods, and the part of
each change that the replicated Insight 177 network does not attribute to trend following.
Positions come from `paper_replication/data_required/managed_money.csv`, prices from the same folder.
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist, squareform

from paper_replication.data import build_rolling_contract

MARKETS = ["WTI", "BRENT", "GASOIL", "HEATOIL", "RBOB", "NATGAS"]
LABELS = {"WTI": "WTI", "BRENT": "Brent", "GASOIL": "Gasoil", "HEATOIL": "Heating oil", "RBOB": "RBOB",
          "NATGAS": "Nat gas"}
PAIRS = [f"{a}-{b}" for a, b in combinations(MARKETS, 2)]
# Barrels per contract; gasoil is 100 tonnes, converted at 7.45 barrels a tonne as in the OIES comment.
BARRELS = {"WTI": 1000, "BRENT": 1000, "GASOIL": 745, "HEATOIL": 1000, "RBOB": 1000}
START = "2011-01-04"   # first week with all six markets (ICE Brent and Gasoil COT start in 2011)
# 2026 Iran war (Wikipedia "2026 Iran war", hormuz.now timeline, CRS R45281); sources differ by a day.
EVENTS = {"2026-02-28": "War begins, Hormuz closed", "2026-04-07": "Ceasefire",
          "2026-07-08": "Ceasefire collapses"}
# COT weeks between the events: a report on Tuesday t covers the change from t - 7 days to t.
PHASES = {"Pre-war": ("2025-09-01", "2026-02-24"), "War to ceasefire": ("2026-03-03", "2026-04-07"),
          "Ceasefire": ("2026-04-14", "2026-07-07"), "After collapse": ("2026-07-14", "2026-12-31")}


# ---------------------------------------------------------------- data

def load_positions(path, basis="futures_only", start=START):
    """Weekly Managed Money net, long and short positions (contracts) and as-of dates, one column per market.

    Indexed by the report week's Tuesday; `report_date` is the as-of date (Monday when Tuesday is a holiday).
    """
    mm = pd.read_csv(path, parse_dates=["report_date", "report_week_tuesday"])
    mm = mm[(mm["basis"] == basis) & mm["market"].isin(MARKETS) & (mm["report_week_tuesday"] >= start)]
    wide = lambda column: mm.pivot(index="report_week_tuesday", columns="market", values=column)[MARKETS]
    return {"net": wide("mm_net"), "long": wide("mm_long"), "short": wide("mm_short"),
            "report_date": wide("report_date")}


def _at_report_dates(series, report_dates):
    """Last value of a daily series on or before each as-of date."""
    values = series.sort_index().reindex(pd.DatetimeIndex(report_dates), method="ffill")
    return pd.Series(values.to_numpy(), index=report_dates.index)


def weekly_returns(prices, meta, report_dates):
    """Log return of each market's held contract between consecutive COT as-of dates (Tuesday to Tuesday).

    The held contract is the paper's rolling contract (`build_rolling_contract`); its cumulative log return
    is read at each market's as-of date, or the last trading day before it.
    """
    rolling = build_rolling_contract(prices, meta)
    cum = {m: _at_report_dates(rolling.loc[rolling["market"] == m].set_index("date")["cum_log_return"],
                               report_dates[m]) for m in MARKETS}
    return pd.DataFrame(cum).diff()


def matched_spread(prices, meta, report_dates, first="BRENT", second="WTI"):
    """`first` minus `second` settlement price for the same delivery month, at each of `second`'s as-of dates.

    Front months cannot be compared directly: Brent's contract for a month expires about three weeks
    before WTI's, so for part of each month the two fronts are different months, and an expiring
    contract can trade far from the rest of the curve (Brent May 2026 settled at 118.35 on its last day,
    31 March, against 103.97 for June). The month used is that of `second`'s second contract, which
    both markets trade on every date. Generic n on a date is the n-th contract whose last trade date
    has not passed, as in `build_rolling_contract`.
    """
    px = prices.assign(settle=prices["PX_SETTLE"].fillna(prices["PX_LAST"])).dropna(subset=["settle"])
    curves = {m: px[px["market"] == m].pivot(index="date", columns="nearby", values="settle").sort_index()
              for m in (first, second)}
    chains = {m: meta[meta["market"] == m].sort_values("LAST_TRADEABLE_DT") for m in (first, second)}
    out = {}
    for week, date in report_dates[second].items():
        expiries = {m: chains[m]["LAST_TRADEABLE_DT"].to_numpy() for m in chains}
        front = {m: np.searchsorted(expiries[m], np.datetime64(date), side="left") for m in chains}
        month = chains[second].iloc[front[second] + 1][["maturity_year", "maturity_month"]].tolist()
        later = chains[first].iloc[front[first]:]
        n = np.flatnonzero((later["maturity_year"] == month[0]).to_numpy()
                           & (later["maturity_month"] == month[1]).to_numpy())[0] + 1
        row = {m: curves[m].loc[:date].iloc[-1] for m in chains}
        out[week] = row[first].get(n, np.nan) - row[second].get(2, np.nan)
    return pd.Series(out, name=f"{first} - {second}")


def nn_discretionary(path):
    """Weekly change in Managed Money split into the replicated network's trend part and the rest (contracts).

    In the replication the bias is held at last week's value in both weeks, so the predicted change is
    mm_std_lag x (trend_cur - trend_prev): the move implied by the momentum inputs alone. The rest,
    `actual_change - predicted_change`, is the OIES comment's "discretionary" change plus model error.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run the expanded-history part of "
                                "paper_replication/notebooks/nn_replicate_insight_177.ipynb first")
    d = pd.read_csv(path, parse_dates=["report_week_tuesday"])
    wide = lambda column: d.pivot(index="report_week_tuesday", columns="market", values=column)[MARKETS]
    actual, trend = wide("actual_change"), wide("predicted_change")
    return {"actual": actual, "trend": trend, "discretionary": actual - trend,
            "identity_gap": (trend - wide("mm_std_lag") * (wide("trend_cur") - wide("trend_prev"))).abs()}


# ---------------------------------------------------------------- correlations

def pairs_to_matrix(row, markets=MARKETS):
    """6 x 6 correlation matrix from one row of pairwise correlations named "A-B"."""
    m = pd.DataFrame(np.eye(len(markets)), index=markets, columns=markets)
    for a, b in combinations(markets, 2):
        m.loc[a, b] = m.loc[b, a] = row[f"{a}-{b}"]
    return m


def rolling_corr(x, window):
    """Rolling correlation of every pair of columns over `window` weeks, one column per pair ("A-B")."""
    return pd.DataFrame({f"{a}-{b}": x[a].rolling(window).corr(x[b]) for a, b in combinations(x.columns, 2)})


def absorption_ratio(x, window):
    """Rolling share of the variance of the standardized changes explained by their first principal component."""
    out = {}
    for end in range(window, len(x) + 1):
        c = x.iloc[end - window:end].corr().to_numpy()
        if np.isfinite(c).all():
            out[x.index[end - 1]] = np.linalg.eigvalsh(c)[-1] / len(c)
    return pd.Series(out)


def market_average(rolling, markets):
    """Rolling average correlation of each market with the other `markets`, from `rolling_corr` output."""
    pairs = [f"{a}-{b}" for a, b in combinations(markets, 2)]
    return pd.DataFrame({m: rolling[[p for p in pairs if m in p.split("-")]].mean(axis=1) for m in markets})


def trend_slopes(changes, trend, window):
    """Rolling slope of each market's weekly change on its own trend part: how closely positions follow the signal."""
    return changes.rolling(window).cov(trend) / trend.rolling(window).var()


def correlation_split(changes, trend, window, markets):
    """Rolling average pairwise correlation of weekly changes, split into a trend-following term and the rest.

    In each window a market's change is x = beta t + e, with t its trend part and beta = `trend_slopes`. Then
    cov(x_a, x_b) = beta_a beta_b cov(t_a, t_b) + cov(e_a, e_b) + beta_b cov(e_a, t_b) + beta_a cov(t_a, e_b).
    Each term over sd(x_a) sd(x_b), averaged over the pairs of `markets`, gives the columns "trend", "other"
    and "cross", which add up to the average correlation. The raw split t + (x - t) is less useful: the
    network's trend part often swings more than the change itself, which leaves two large offsetting terms.
    """
    beta = trend_slopes(changes, trend, window)
    cov = lambda u, v: u.rolling(window).cov(v)
    terms = []
    for a, b in combinations(markets, 2):
        scale = changes[a].rolling(window).std() * changes[b].rolling(window).std()
        tt, xt, tx = cov(trend[a], trend[b]), cov(changes[a], trend[b]), cov(trend[a], changes[b])
        terms.append(pd.DataFrame({
            "trend": beta[a] * beta[b] * tt / scale,
            "other": (cov(changes[a], changes[b]) - beta[b] * xt - beta[a] * tx + beta[a] * beta[b] * tt) / scale,
            "cross": (beta[b] * (xt - beta[a] * tt) + beta[a] * (tx - beta[b] * tt)) / scale}))
    return sum(terms) / len(terms)


def low_spells(series, threshold, gap=13):
    """Spells of `series` below `threshold`, joining spells less than `gap` weeks apart.

    One row per spell: first and last week, number of weeks below, and the date and value of the low.
    """
    low = series.dropna()
    low = low[low < threshold]
    spell = (np.diff(low.index.to_numpy(), prepend=low.index.to_numpy()[:1]) > np.timedelta64(7 * gap, "D")).cumsum()
    return pd.DataFrame([{"start": s.index[0], "end": s.index[-1], "weeks": len(s), "low on": s.idxmin(), "low": s.min()}
                         for _, s in low.groupby(spell)])


def block_bootstrap_corr(x, y, window, n=1000, block=4, q=(0.05, 0.95), seed=0):
    """Rolling correlation of `x` and `y` with a moving-block bootstrap band for each window.

    Each resample of a window joins randomly placed blocks of `block` consecutive weeks from it, which
    keeps short-run dependence. Returns the correlation and the `q` quantiles of the resampled ones.
    """
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, window - block + 1, size=(n, -(-window // block)))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n, -1)[:, :window]
    joint = pd.concat([x, y], axis=1).dropna()
    a, b = joint.iloc[:, 0].to_numpy(float), joint.iloc[:, 1].to_numpy(float)
    rows = {}
    for end in range(window, len(joint) + 1):
        wa, wb = a[end - window:end], b[end - window:end]
        ra, rb = wa[idx], wb[idx]
        ra, rb = ra - ra.mean(1, keepdims=True), rb - rb.mean(1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
        rows[joint.index[end - 1]] = (np.corrcoef(wa, wb)[0, 1], *np.nanquantile(r, q))
    return pd.DataFrame.from_dict(rows, orient="index", columns=["corr", "low", "high"])


# ---------------------------------------------------------------- clusters

def market_clusters(corr, k=3):
    """Average-linkage clusters of markets from a correlation matrix, with distance sqrt(2 (1 - rho)).

    Returns the cluster label of each market and the linkage matrix (for a dendrogram).
    """
    dist = np.sqrt(np.clip(2 * (1 - corr.to_numpy()), 0, None))
    np.fill_diagonal(dist, 0)
    z = linkage(squareform(dist, checks=False), method="average")
    return pd.Series(fcluster(z, k, criterion="maxclust"), index=corr.index), z


def co_clustering(rolling, k=3):
    """For each window and pair, 1 if the two markets fall in the same of `k` clusters, else 0."""
    out = {}
    for date, row in rolling.dropna().iterrows():
        labels, _ = market_clusters(pairs_to_matrix(row), k)
        out[date] = [int(labels[a] == labels[b]) for a, b in combinations(MARKETS, 2)]
    return pd.DataFrame.from_dict(out, orient="index", columns=PAIRS)


def silhouette(x, labels):
    """Mean silhouette width of a clustering (Euclidean distance)."""
    d = cdist(x, x)
    widths = []
    for i, label in enumerate(labels):
        same = labels == label
        if same.sum() < 2:
            widths.append(0.0)
            continue
        a = d[i, same].sum() / (same.sum() - 1)
        b = min(d[i, labels == other].mean() for other in set(labels) if other != label)
        widths.append((b - a) / max(a, b))
    return float(np.mean(widths))


def correlation_regimes(rolling, ks=range(2, 6), seed=0):
    """Group weeks into regimes by k-means on their rolling pairwise correlations, k chosen by silhouette.

    Returns the regime of each week (1 = highest average correlation), the regimes' average correlation
    by pair, and the silhouette score of each k tried.
    """
    x = rolling.dropna()
    scores, fits = {}, {}
    for k in ks:
        _, labels = kmeans2(x.to_numpy(), k, minit="++", rng=seed)
        fits[k], scores[k] = labels, silhouette(x.to_numpy(), labels)
    best = max(scores, key=scores.get)
    labels = pd.Series(fits[best], index=x.index)
    means = x.groupby(labels).mean()
    order = means.mean(axis=1).sort_values(ascending=False).index
    relabel = {old: new for new, old in enumerate(order, start=1)}
    return labels.map(relabel), means.rename(index=relabel).sort_index(), pd.Series(scores, name="silhouette")


# ---------------------------------------------------------------- phases

def phase_corr(x, phases=PHASES):
    """Correlation matrix of the columns of `x` within each phase (name -> (start, end))."""
    return {name: x.loc[start:end].corr() for name, (start, end) in phases.items()}


def own_return_corr(changes, returns, phases=PHASES):
    """Correlation of each market's change in positions with its own weekly return, by phase."""
    return pd.DataFrame({name: changes.loc[s:e].corrwith(returns.loc[s:e]) for name, (s, e) in phases.items()})
