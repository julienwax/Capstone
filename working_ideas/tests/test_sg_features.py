"""Behavioral checks: temporal isolation, carry parity, and evaluation alignment.

Run: PYTHONPATH=paper_replication/src:working_ideas/src python -m unittest discover -s working_ideas/tests
"""
import copy
import unittest
import numpy as np
import pandas as pd
import torch

from paper_replication.config import ReplicationConfig, HORIZONS, MARKETS
from carry_extensions import calibrate_carry_design
import sg_features as sf


def design():
    rng = np.random.default_rng(17)
    dates = pd.date_range('2018-01-02', periods=18, freq='W-TUE')
    index = pd.MultiIndex.from_product([dates, MARKETS], names=sf.KEYS)
    d = index.to_frame(index=False)
    d['panel_date'] = d.report_week_tuesday
    d['mm_std_lag'] = rng.uniform(2, 4, len(d))
    d['predicted_change'] = rng.normal(size=len(d))
    d['carry_delta_z'] = rng.normal(size=len(d))
    d['state'] = rng.normal(size=len(d))
    d['actual_change'] = d.predicted_change + .8 * d.carry_delta_z + .3 * d.state + rng.normal(size=len(d))
    return d


class TemporalChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.settings = sf.ExperimentSettings(overlay_weeks=8, validation_weeks=2)

    def test_null_overlay_reproduces_existing_carry(self):
        d = design()
        expected = calibrate_carry_design(d, train_weeks=8).set_index(sf.KEYS).sort_index()
        result = sf.rolling_overlay(d, [], self.settings).set_index(sf.KEYS).sort_index()
        np.testing.assert_allclose(result.predicted_change, expected.predicted_change, atol=1e-10)

    def test_overlay_current_and_future_targets_do_not_change_prediction(self):
        d = design()
        date = sorted(d.report_week_tuesday.unique())[10]
        first = sf.rolling_overlay(d, ['state'], self.settings, interaction=True)
        mutated = d.copy()
        mutated.loc[mutated.report_week_tuesday.ge(date), 'actual_change'] += 1e6
        second = sf.rolling_overlay(mutated, ['state'], self.settings, interaction=True)
        a = first.loc[first.report_week_tuesday.le(date)]
        b = second.loc[second.report_week_tuesday.le(date)]
        np.testing.assert_allclose(a.predicted_change, b.predicted_change)
        np.testing.assert_allclose(a.selected_penalty, b.selected_penalty)

    def test_comparison_uses_common_keys(self):
        d = design()
        result, _, _, aligned = sf.compare({'all': d, 'short': d.iloc[6:].copy()})
        self.assertEqual(result.loc['all', 'Observations'], len(d)-6)
        self.assertEqual(aligned['all'][sf.KEYS].to_dict('list'), aligned['short'][sf.KEYS].to_dict('list'))
        duplicated = pd.concat([d, d.iloc[:1]])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            sf.compare({'duplicate': duplicated})

    def test_holiday_uses_last_close_not_one_close_earlier(self):
        dates = pd.bdate_range('2020-01-01', periods=5)
        frames = []
        for market in MARKETS:
            # Market closed on date[2], index open. At date[2] use date[1].
            g = pd.DataFrame({'date': dates.delete(2), 'market': market,
                              'daily_pnl': 1., 'pnl_vol': 10., 'x_5': [1., 2., 4., 5.]})
            frames.append(g)
        data = {'daily': pd.concat(frames), 'sg_return': pd.Series(.1, index=dates)}
        arrays = sf.daily_arrays(data, horizons=(5,))
        row = arrays['dates'].get_loc(dates[2])
        np.testing.assert_equal(arrays['x'][row, :, 0], 2.)
        np.testing.assert_equal(arrays['u'][row], 0.)

    def test_joint_ignores_current_mm_and_unavailable_sg_targets(self):
        rng = np.random.default_rng(4)
        dates = pd.bdate_range('2017-01-02', periods=230)
        weeks = dates[-40::5]
        daily = []
        for market in MARKETS:
            frame = pd.DataFrame({'date': dates, 'market': market, 'pnl_vol': 10., 'daily_pnl': rng.normal(size=len(dates))})
            frame = pd.concat([frame, pd.DataFrame({f'x_{h}': rng.normal(size=len(dates)) for h in HORIZONS})], axis=1)
            daily.append(frame)
        rows = pd.MultiIndex.from_product([weeks, MARKETS], names=sf.KEYS).to_frame(index=False)
        rows['report_date'] = rows.report_week_tuesday
        rows['panel_date'] = rows.report_week_tuesday
        rows['previous_report_week'] = rows.report_week_tuesday.map(dict(zip(weeks[1:], weeks[:-1])))
        rows['actual_change'] = rng.normal(size=len(rows))
        rows['mm_std_lag'] = 3.
        data = dict(daily=pd.concat(daily), sg_return=pd.Series(rng.normal(size=len(dates)), index=dates),
                    weeks=weeks, rows=rows.set_index(sf.KEYS), vol=np.full((len(weeks), 6), 10.),
                    x=rng.normal(size=(len(weeks), 6, len(HORIZONS))).astype('float32'),
                    y=rng.normal(size=(len(weeks), 6)).astype('float32'))
        config = ReplicationConfig(train_weeks=4, first_epochs=2, rolling_epochs=1,
                                   evaluation_start=str(weeks[-1].date()), evaluation_end=str(weeks[-1].date()))
        a = sf.run_network(data, config, self.settings, joint_weight=.1, progress=False)
        changed = copy.deepcopy(data)
        changed['y'][-1] += 1000
        cutoff = weeks[-2] - pd.offsets.BDay(self.settings.sg_lag_bdays)
        changed['sg_return'].loc[lambda s: s.index > cutoff] += 1000
        b = sf.run_network(changed, config, self.settings, joint_weight=.1, progress=False)
        np.testing.assert_allclose(a.predicted_change, b.predicted_change, atol=1e-7)
        self.assertTrue((a.sg_train_end <= cutoff).all())

    def test_annual_horizon_blend_does_not_fit_test_year_sg(self):
        rng = np.random.default_rng(22)
        dates = pd.bdate_range('2010-01-01', '2017-12-31')
        weeks = dates[::5]
        daily, panel = [], []
        for market in MARKETS:
            g = pd.DataFrame({'date': dates, 'market': market, 'pnl_vol': 10.,
                              'daily_pnl': rng.normal(size=len(dates))})
            for horizon in sf.LIBRARY_HORIZONS:
                g[f'x_{horizon}'] = rng.normal(size=len(dates))
            daily.append(g)
            w = pd.DataFrame({'report_week_tuesday': weeks, 'market': market, 'feature_date': weeks})
            w['previous_report_week'] = w.report_week_tuesday.shift(1)
            panel.append(w)
        data = dict(daily=pd.concat(daily), panel=pd.concat(panel),
                    sg_return=pd.Series(rng.normal(size=len(dates)), index=dates))
        first, weights = sf.horizon_flows(data, self.settings)
        changed = copy.deepcopy(data)
        changed['sg_return'].loc['2016':] += rng.normal(0, 100, len(changed['sg_return'].loc['2016':]))
        second, changed_weights = sf.horizon_flows(changed, self.settings)
        np.testing.assert_allclose(first.loc[first.report_week_tuesday.dt.year.le(2016), 'sg_flow'],
                                   second.loc[second.report_week_tuesday.dt.year.le(2016), 'sg_flow'], equal_nan=True)
        self.assertTrue((weights.train_end.dt.year < weights.year).all())


if __name__ == '__main__':
    unittest.main()
