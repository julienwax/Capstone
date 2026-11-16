"""Delivery-month matching, dollar neutrality, signal sign, flow split and the basket of `spread.py`."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import spread as sp


def _inputs(daily_growth=0.0):
    """Two markets with flat curves. A expires on the 10th of the month before delivery, B on the 20th.
    A trades at 100 plus one per delivery month, B at 50 plus one per delivery month, both growing by
    `daily_growth` a day."""
    dates = pd.bdate_range("2024-01-01", "2024-06-28")
    meta = pd.DataFrame([{"market": m, "LAST_TRADEABLE_DT": pd.Timestamp(2024, k, day), "maturity_year": 2024,
                          "maturity_month": k + 1} for m, day in [("BRENT", 10), ("WTI", 20)] for k in range(1, 12)])
    rows = []
    for market, base in [("BRENT", 100.0), ("WTI", 50.0)]:
        ltd = meta.loc[meta["market"] == market, "LAST_TRADEABLE_DT"].to_numpy()
        for i, date in enumerate(dates):
            front = np.searchsorted(ltd, date.to_datetime64(), side="left")
            rows += [{"market": market, "date": date, "nearby": n, "px": (base + front + n) * (1 + daily_growth) ** i}
                     for n in (1, 2, 3)]
    return pd.DataFrame(rows), meta


def test_both_legs_hold_the_same_delivery_month():
    generic, meta = _inputs()
    daily = sp.daily_spread(generic, meta, "BRENT", "WTI")
    month = daily["month"] - 2024 * 12 + 1                              # delivery month, 1 = January
    assert np.allclose(daily["price_a"], 100 + month - 1) and np.allclose(daily["price_b"], 50 + month - 1)
    assert np.allclose(daily["ret"].dropna(), 0.0)                       # prices by contract never move, rolls included
    assert daily["month"].nunique() > 3


def test_roll_is_five_trading_days_before_leg_a_expires():
    generic, meta = _inputs()
    month = sp.daily_spread(generic, meta, "BRENT", "WTI")["month"]
    first_new = month[month != month.iloc[0]].index[0]
    assert len(pd.bdate_range(first_new, "2024-01-10")) == 5   # held from the close four trading days before expiry


def test_a_common_move_gives_no_spread_return():
    generic, meta = _inputs(daily_growth=0.01)
    daily = sp.daily_spread(generic, meta, "BRENT", "WTI")
    assert np.allclose(daily["ret_a"].dropna(), 0.01) and np.allclose(daily["ret"].dropna(), 0.0)


def test_signal_sells_a_wide_spread_and_is_capped():
    cum = pd.Series(np.r_[np.zeros(30), 10.0, 10.0])
    vol = pd.Series(1.0, index=cum.index)
    signal = sp.reversion_signal(cum, vol, lookbacks=(4,))[4]
    assert signal.iloc[30] == -1.0 and signal.iloc[29] == 0.0
    small = sp.reversion_signal(pd.Series(np.r_[np.zeros(30), 0.4]), pd.Series(1.0, index=range(31)), lookbacks=(4,))[4]
    assert np.isclose(small.iloc[-1], -0.3 / np.sqrt(3 * 7 / 24))


def test_legs_are_dollar_neutral_and_flow_splits_into_signal_and_sizing():
    generic, meta = _inputs()
    rng = np.random.default_rng(0)
    generic.loc[generic["market"] == "BRENT", "px"] *= 1 + rng.normal(0, 0.01, (generic["market"] == "BRENT").sum())
    weeks = pd.date_range("2024-03-05", "2024-06-25", freq="W-TUE")
    panel = sp.spread_panel(generic, meta, "Brent-WTI", weeks, vol_days=20, lookbacks=(4,))
    ok = panel.flow["BRENT"].notna()
    assert ok.sum() > 5
    price = panel.daily[["price_a", "price_b"]].resample("W-TUE").last().reindex(weeks)
    dollars_a = panel.contracts["BRENT"] * price["price_a"] * sp.POINT_VALUE["BRENT"]
    dollars_b = panel.contracts["WTI"] * price["price_b"] * sp.POINT_VALUE["WTI"]
    assert np.allclose(dollars_a[ok], -dollars_b[ok]) and np.allclose(dollars_a[ok], panel.dollars[ok])
    assert np.allclose((panel.signal_flow + panel.sizing_flow)[ok], panel.flow[ok])


def _noisy_inputs():
    generic, meta = _inputs()
    rng = np.random.default_rng(1)
    generic["px"] *= 1 + rng.normal(0, 0.01, len(generic))
    extra = generic[generic["market"] == "WTI"].assign(market="HEATOIL")
    extra["px"] *= 1 + rng.normal(0, 0.01, len(extra))
    return pd.concat([generic, extra]), pd.concat([meta, meta[meta["market"] == "WTI"].assign(market="HEATOIL")])


def test_basket_of_two_markets_is_the_pair():
    generic, meta = _noisy_inputs()
    weeks = pd.date_range("2024-03-05", "2024-06-25", freq="W-TUE")
    pair = sp.spread_panel(generic, meta, "Brent-WTI", weeks, vol_days=20, lookbacks=(4,))
    basket = sp.basket_panel(generic, meta, weeks, markets=("BRENT", "WTI"), vol_days=20, lookbacks=(4,))
    ok = pair.flow["BRENT"].notna()
    assert ok.sum() > 5
    assert np.allclose(basket.dollars["BRENT"][ok], pair.dollars[ok])
    assert np.allclose(basket.flow[ok], pair.flow[ok])


def test_basket_is_dollar_neutral_and_ignores_a_common_move():
    generic, meta = _noisy_inputs()
    weeks = pd.date_range("2024-03-05", "2024-06-25", freq="W-TUE")
    markets = ("BRENT", "WTI", "HEATOIL")
    basket = sp.basket_panel(generic, meta, weeks, markets=markets, vol_days=20, lookbacks=(4,))
    ok = basket.dollars.notna().all(axis=1)
    assert ok.sum() > 5 and np.allclose(basket.dollars[ok].sum(axis=1), 0.0, atol=1e-6)
    shifted = generic.copy()
    days = shifted["date"].rank(method="dense") - 1
    shifted["px"] *= 1.001 ** days                                  # every market gains 0.1% more each trading day
    moved = sp.basket_panel(shifted, meta, weeks, markets=markets, vol_days=20, lookbacks=(4,))
    assert np.allclose(moved.signal[ok], basket.signal[ok], atol=1e-6)
