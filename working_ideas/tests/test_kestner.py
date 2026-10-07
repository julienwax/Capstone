"""Tests for working_ideas/src/kestner.py on small synthetic inputs."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import kestner as K  # noqa: E402


def _generics():
    """Ten trading days. Contract A expires on day 5 (last trade date), B follows, C after B.

    Generic 1 is A through day 5 and B from day 6; generic 2 is B through day 5 and C after.
    """
    dates = pd.bdate_range("2020-01-06", periods=10)
    a = 100 + np.arange(10.0)            # A rises by 1 a day
    b = 200 + 2 * np.arange(10.0)        # B rises by 2 a day
    c = 300 + 3 * np.arange(10.0)
    g1 = np.where(np.arange(10) <= 5, a, b)
    g2 = np.where(np.arange(10) <= 5, b, c)
    px = pd.DataFrame({1: g1, 2: g2}, index=dates)
    return px, [dates[5], dates[9] + pd.offsets.BDay(20)], a, b


def test_roll_at_expiry_never_spans_two_contracts():
    px, ltd, a, b = _generics()
    r = K.held_contract_returns(px, ltd, roll_days=0)
    # Days 1-5 earn A, day 6 is the first day on B: B(6) / B(5), not B(6) / A(5).
    np.testing.assert_allclose(r.iloc[1:6], a[1:6] / a[0:5] - 1)
    np.testing.assert_allclose(r.iloc[6], b[6] / b[5] - 1)
    np.testing.assert_allclose(r.iloc[7:], b[7:] / b[6:9] - 1)


def test_early_roll_switches_contract_before_last_trade_date():
    px, ltd, a, b = _generics()
    r = K.held_contract_returns(px, ltd, roll_days=2)
    # Days left to A's expiry at each close: 5, 4, 3, 2, 1, 0. B is held from the close with
    # fewer than 2 days left (day 4), so day 5 already earns B.
    np.testing.assert_allclose(r.iloc[1:5], a[1:5] / a[0:4] - 1)
    np.testing.assert_allclose(r.iloc[5:], b[5:] / b[4:9] - 1)


def test_held_price_switches_with_the_position():
    px, ltd, a, b = _generics()
    p = K.held_contract_prices(px, ltd, roll_days=2)
    # A is held through day 3's close, B from day 4's (see the early-roll test). Day 9 is left out:
    # B expires after the data ends, and the roll logic then counts days to the end of the data.
    np.testing.assert_allclose(p.iloc[:4], a[:4])
    np.testing.assert_allclose(p.iloc[4:9], b[4:9])


def test_weeks_can_end_on_tuesday():
    daily = pd.Series(0.01, index=pd.bdate_range("2020-01-06", periods=10))   # Mon 6 to Fri 17 Jan
    weekly = K.weekly_returns(daily, week_end="W-TUE")
    assert list(weekly.index.dayofweek) == [1, 1, 1]
    np.testing.assert_allclose(weekly, [1.01 ** 2 - 1, 1.01 ** 5 - 1, 1.01 ** 3 - 1])


def test_non_positive_prices_are_skipped():
    px, ltd, a, _ = _generics()
    px.iloc[2, 0] = -5.0
    r = K.held_contract_returns(px, ltd, roll_days=0)
    assert px.index[2] not in r.index
    np.testing.assert_allclose(r.loc[px.index[3]], a[3] / a[1] - 1)


def test_signal_formula_and_cap():
    idx = pd.date_range("2020-01-03", periods=6, freq="W-FRI")
    weekly = pd.DataFrame({"X": [0.01] * 6, "Y": [-0.02] * 6}, index=idx)
    vol = pd.DataFrame({"X": [0.02] * 6, "Y": [0.02] * 6}, index=idx)
    s = K.signals(weekly, vol, lookback=4, cap=1.0)
    assert s["X"].iloc[:3].isna().all()
    np.testing.assert_allclose(s["X"].iloc[3:], 0.01 / 0.02 * 2)  # 1.0 exactly at the cap
    np.testing.assert_allclose(s["Y"].iloc[3:], -1.0)             # -2.0 capped


def test_weights_earn_the_following_week():
    idx = pd.date_range("2020-01-03", periods=3, freq="W-FRI")
    weekly = pd.DataFrame({"X": [0.0, 0.10, -0.05]}, index=idx)
    w = pd.DataFrame({"X": [2.0, 1.0, 0.0]}, index=idx)
    ret = K.portfolio(weekly, w)
    assert np.isnan(ret.iloc[0])
    np.testing.assert_allclose(ret.iloc[1:], [2.0 * 0.10, 1.0 * -0.05])


def test_r2_is_squared_correlation_and_leverage_matches_vol():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2015-01-02", periods=300, freq="W-FRI")
    y = pd.Series(rng.normal(size=300), index=idx)
    x = 0.5 * y + pd.Series(rng.normal(size=300), index=idx)
    span = ("2015", "2020")
    np.testing.assert_allclose(K.r2(x, y, span), x.corr(y) ** 2)
    np.testing.assert_allclose((x * K.leverage(x, y, span)).std(), y.std())
