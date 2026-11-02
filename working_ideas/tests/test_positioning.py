"""Tests for working_ideas/src/positioning.py on small synthetic inputs."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "paper_replication" / "src"), str(ROOT / "working_ideas" / "src")]
import positioning as P  # noqa: E402


def _blocks(n=200, seed=0):
    """Six markets in two blocks: oil (first five, common factor) and gas (independent)."""
    rng = np.random.default_rng(seed)
    common = rng.normal(size=n)
    data = {m: common + 0.5 * rng.normal(size=n) for m in P.MARKETS[:5]}
    data["NATGAS"] = rng.normal(size=n)
    return pd.DataFrame(data, index=pd.date_range("2015-01-06", periods=n, freq="W-TUE"))


def test_rolling_corr_matches_pandas_on_each_window():
    x = _blocks()
    r = P.rolling_corr(x, 26)
    assert list(r.columns) == P.PAIRS
    assert r.iloc[:25].isna().all().all()
    np.testing.assert_allclose(r["WTI-BRENT"].iloc[60], x.iloc[35:61]["WTI"].corr(x.iloc[35:61]["BRENT"]))
    m = P.pairs_to_matrix(r.iloc[60])
    np.testing.assert_allclose(m.to_numpy(), x.iloc[35:61].corr().to_numpy())


def test_market_average_is_the_mean_over_its_pairs():
    r = P.rolling_corr(_blocks(), 26)
    avg = P.market_average(r, P.MARKETS[:5])
    assert list(avg.columns) == P.MARKETS[:5]
    np.testing.assert_allclose(avg["WTI"], r[["WTI-BRENT", "WTI-GASOIL", "WTI-HEATOIL", "WTI-RBOB"]].mean(axis=1))


def test_correlation_split_adds_up_and_drops_markets_that_ignore_the_trend():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2015-01-06", periods=120, freq="W-TUE")
    trend = _blocks(120, 4).set_index(idx)
    noise = pd.DataFrame(rng.normal(size=trend.shape), index=idx, columns=trend.columns)
    oil = P.MARKETS[:5]
    # Every market follows its trend part for 60 weeks; afterwards only WTI and Brent do.
    follow = pd.DataFrame(1.0, index=idx, columns=trend.columns)
    follow.iloc[60:, 2:] = 0.0
    changes = follow * trend + 0.5 * noise
    split = P.correlation_split(changes, trend, 26, oil)
    average = P.rolling_corr(changes[oil], 26).mean(axis=1)
    np.testing.assert_allclose(split.sum(axis=1, min_count=3).dropna(), average.dropna())
    np.testing.assert_allclose(P.trend_slopes(changes, trend, 26).iloc[-1, 2:5], 0, atol=0.3)
    assert split["trend"].iloc[-1] < 0.2 < 0.5 < split["trend"].iloc[59]
    assert split["cross"].abs().max() < 0.1


def test_low_spells_join_close_dips():
    idx = pd.date_range("2015-01-06", periods=60, freq="W-TUE")
    s = pd.Series(1.0, index=idx)
    s.iloc[[5, 6, 10]] = [0.1, 0.0, 0.2]               # two dips 4 weeks apart: one spell
    s.iloc[40] = -0.3                                  # far away: a second spell
    spells = P.low_spells(s, 0.5)
    assert list(spells["weeks"]) == [3, 1]
    assert list(spells["low on"]) == [idx[6], idx[40]]
    np.testing.assert_allclose(spells["low"], [0.0, -0.3])


def test_two_blocks_give_two_market_clusters():
    labels, _ = P.market_clusters(_blocks().corr(), k=2)
    assert labels[P.MARKETS[:5]].nunique() == 1
    assert labels["NATGAS"] != labels["WTI"]


def test_two_correlation_regimes_are_found():
    # 150 weeks of the two-block structure, then 150 weeks of independent markets.
    noise = np.random.default_rng(2).normal(size=(150, len(P.MARKETS)))
    x = pd.concat([_blocks(150, 1), pd.DataFrame(noise, columns=P.MARKETS)], ignore_index=True)
    x.index = pd.date_range("2010-01-05", periods=len(x), freq="W-TUE")
    labels, means, scores = P.correlation_regimes(P.rolling_corr(x, 26), ks=[2])
    first, last = labels.loc[:x.index[140]], labels.loc[x.index[200]:]
    assert first.mode()[0] != last.mode()[0]
    assert means.mean(axis=1).is_monotonic_decreasing


def test_bootstrap_band_contains_the_correlation():
    x = _blocks()
    band = P.block_bootstrap_corr(x["WTI"], x["BRENT"], 26, n=200)
    np.testing.assert_allclose(band["corr"], P.rolling_corr(x, 26)["WTI-BRENT"].dropna())
    assert (band["low"] <= band["high"]).all()
    assert ((band["low"] <= band["corr"] + 0.05) & (band["corr"] - 0.05 <= band["high"])).mean() > 0.95


def test_spread_compares_the_same_delivery_month():
    dates = pd.bdate_range("2026-03-02", periods=3)          # Mon 2, Tue 3, Wed 4 March
    meta = pd.DataFrame({
        "market": ["BRENT"] * 3 + ["WTI"] * 3,
        "maturity_year": 2026, "maturity_month": [5, 6, 7, 5, 6, 7],
        "LAST_TRADEABLE_DT": pd.to_datetime(["2026-03-02", "2026-04-30", "2026-05-29",
                                             "2026-04-21", "2026-05-19", "2026-06-22"])})
    rows = []
    for d in dates:
        brent_front = 0 if d == dates[0] else 1               # Brent May expires on Monday
        for n in (1, 2, 3):
            rows.append({"market": "BRENT", "date": d, "nearby": n, "PX_SETTLE": 100 + 10 * (brent_front + n)})
            rows.append({"market": "WTI", "date": d, "nearby": n, "PX_SETTLE": 50 + 10 * n})
    report = pd.DataFrame({"WTI": dates, "BRENT": dates}, index=dates)
    s = P.matched_spread(pd.DataFrame(rows).assign(PX_LAST=np.nan), meta, report)
    # Month = WTI's second contract (June). Monday: Brent June is generic 2; afterwards generic 1.
    # Brent June is always contract 1 (price 120) and WTI generic 2 is 70.
    np.testing.assert_allclose(s, [50, 50, 50])


def test_values_are_read_on_or_before_each_report_date():
    daily = pd.Series(np.arange(10.0), index=pd.bdate_range("2026-03-02", periods=10))   # Mon 2 to Fri 13
    report = pd.Series(pd.to_datetime(["2026-03-03", "2026-03-09", "2026-03-15"]),
                       index=pd.to_datetime(["2026-03-03", "2026-03-10", "2026-03-17"]))
    out = P._at_report_dates(daily, report)
    # Tue 3 -> 1; Mon 9 (Tuesday holiday) -> 5; Sun 15 -> Fri 13 -> 9
    np.testing.assert_allclose(out, [1, 5, 9])
    assert out.index.equals(report.index)
