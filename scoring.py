"""Scores for the nowcast: uncentred out-of-sample R2 (against predicting no change), direction, paired block bootstrap."""
from pathlib import Path

import numpy as np
import pandas as pd

PERIODS = {"2015-25": ("2015-01-01", "2025-12-31"), "2015-16": ("2015-01-01", "2016-12-31"),
           "2017-25": ("2017-01-01", "2025-12-31"), "2026": ("2026-01-01", "2026-12-31")}


def _mask(data, period, *preds):
    a, b = PERIODS[period] if isinstance(period, str) else period
    m = (data["date"] >= np.datetime64(a)) & (data["date"] <= np.datetime64(b))
    for p in preds:
        m &= np.isfinite(p).all(1)
    return m


def r2(data, pred, period="2015-25", market=None):
    m = _mask(data, period, pred)
    a, p = data["change"][m], pred[m]
    if market is not None:
        j = data["markets"].index(market)
        a, p = a[:, j], p[:, j]
    return 100 * (1 - ((a - p) ** 2).sum() / (a ** 2).sum())


def hit(data, pred, period="2015-25"):
    m = _mask(data, period, pred)
    return 100 * np.mean(np.sign(data["change"][m]) == np.sign(pred[m]))


def gain(data, pred, ref, period="2015-25", draws=4000, block=8, seed=0):
    """R2 of `pred` minus R2 of `ref`, with a 95% interval from resampling 8-week blocks of weeks (paired)."""
    m = _mask(data, period, pred, ref)
    a = data["change"][m]
    s0, s1, s2 = (a ** 2).sum(1), ((a - pred[m]) ** 2).sum(1), ((a - ref[m]) ** 2).sum(1)
    n = len(s0)
    block = min(block, n)
    st = np.random.default_rng(seed).integers(0, n - block + 1, size=(draws, int(np.ceil(n / block))))
    idx = (st[:, :, None] + np.arange(block)[None, None, :]).reshape(draws, -1)[:, :n]
    d = 100 * (s2[idx].sum(1) - s1[idx].sum(1)) / s0[idx].sum(1)
    return 100 * (s2.sum() - s1.sum()) / s0.sum(), np.percentile(d, 2.5), np.percentile(d, 97.5)


def summary(data, runs):
    """runs: {name: [nowcast of each refit]}. Mean over the refits by period, their spread, and the average nowcast."""
    out = {}
    for name, preds in runs.items():
        avg = np.mean(preds, axis=0)
        row = {f"R2 {k}": np.mean([r2(data, p, k) for p in preds]) for k in PERIODS}
        v = [r2(data, p) for p in preds]
        row.update({"sd of refits": np.std(v, ddof=1), "worst refit": min(v), "best refit": max(v),
                    "average nowcast, R2 2015-25": r2(data, avg), "direction, %": hit(data, avg)})
        out[name] = row
    return pd.DataFrame(out).T


def by_year(data, preds):
    return pd.DataFrame({name: {y: r2(data, p, (f"{y}-01-01", f"{y}-12-31")) for y in range(2015, 2027)} for name, p in preds.items()})


def by_market(data, preds, period="2015-25"):
    t = pd.DataFrame({name: {m: r2(data, p, period, m) for m in data["markets"]} for name, p in preds.items()})
    t.loc["six markets"] = [r2(data, p, period) for p in preds.values()]
    return t


def repo_baseline(data, cache="runs/run_replication.csv"):
    """paper_replication.evaluate.run_replication itself (about three minutes), cached. Returns [weeks, markets]."""
    if not Path(cache).exists():
        from dataclasses import replace
        from paper_replication.config import ReplicationConfig
        from paper_replication.evaluate import run_replication
        cfg = replace(ReplicationConfig(data_dir=data["repo"] / "paper_replication" / "data_required"), evaluation_end="2026-12-31")
        run_replication(cfg)["predictions"][["report_week_tuesday", "market", "predicted_change"]].to_csv(cache, index=False)
    p = pd.read_csv(cache, parse_dates=["report_week_tuesday"]).pivot(index="report_week_tuesday", columns="market", values="predicted_change")
    return p.reindex(index=data["weeks"], columns=data["markets"]).to_numpy()
