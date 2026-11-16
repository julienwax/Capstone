"""Delivery-month matching, signal sign and flow split of `spread.py`."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import spread as sp


def _inputs():
    """Two markets with flat curves. A expires on the 10th of the month before delivery, B on the 20th.
    A trades at 100 plus one per delivery month, B at 50 plus one per delivery month."""
    dates = pd.bdate_range("2024-01-01", "2024-06-28")
    meta = pd.DataFrame([{"market": m, "LAST_TRADEABLE_DT": pd.Timestamp(2024, k, day), "maturity_year": 2024,
                          "maturity_month": k + 1} for m, day in [("BRENT", 10), ("WTI", 20)] for k in range(1, 12)])
    rows = []
    for market, base in [("BRENT", 100.0), ("WTI", 50.0)]:
        ltd = meta.loc[meta["market"] == market, "LAST_TRADEABLE_DT"].to_numpy()
        for date in dates:
            front = np.searchsorted(ltd, date.to_datetime64(), side="left")
            rows += [{"market": market, "date": date, "nearby": n, "px": base + front + n} for n in (1, 2, 3)]
    return pd.DataFrame(rows), meta


def test_both_legs_hold_the_same_delivery_month():
    generic, meta = _inputs()
    daily = sp.daily_spread(generic, meta, "BRENT", "WTI")
    assert np.allclose(daily["level"], 50.0)             # same month in both legs: 100 + m - (50 + m)
    assert np.allclose(daily["pnl"].dropna(), 0.0)       # prices by contract never move, rolls included
    assert daily["month"].nunique() > 3


def test_roll_is_five_trading_days_before_leg_a_expires():
    generic, meta = _inputs()
    month = sp.daily_spread(generic, meta, "BRENT", "WTI")["month"]
    first_new = month[month != month.iloc[0]].index[0]
    assert len(pd.bdate_range(first_new, "2024-01-10")) == 5   # held from the close four trading days before expiry


def test_signal_sells_a_wide_spread_and_is_capped():
    cum = pd.Series(np.r_[np.zeros(30), 10.0, 10.0])
    vol = pd.Series(1.0, index=cum.index)
    signal = sp.reversion_signal(cum, vol, lookbacks=(4,))[4]
    assert signal.iloc[30] == -1.0 and signal.iloc[29] == 0.0
    small = sp.reversion_signal(pd.Series(np.r_[np.zeros(30), 0.4]), pd.Series(1.0, index=range(31)), lookbacks=(4,))[4]
    assert np.isclose(small.iloc[-1], -0.3 / np.sqrt(3 * 7 / 24))


def test_flow_splits_into_signal_and_sizing_and_legs_offset():
    generic, meta = _inputs()
    rng = np.random.default_rng(0)
    generic.loc[generic["market"] == "BRENT", "px"] += rng.normal(0, 1, (generic["market"] == "BRENT").sum())
    weeks = pd.date_range("2024-03-05", "2024-06-25", freq="W-TUE")
    panel = sp.spread_panel(generic, meta, "Brent-WTI", weeks, vol_days=20, lookbacks=(4,))
    ok = panel.d_units.notna()
    assert ok.sum() > 5
    assert np.allclose((panel.signal_flow + panel.sizing_flow)[ok], panel.d_units[ok])
    assert np.allclose(panel.flow["BRENT"][ok], -panel.flow["WTI"][ok])
