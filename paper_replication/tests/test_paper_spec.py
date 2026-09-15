import unittest
import numpy as np
import pandas as pd
import torch
from paper_replication.config import HORIZONS
from paper_replication.data import build_weekly_panel, _ewm_volatility
from paper_replication.model import SharedMomentumNetwork, PaperAdam, align_bias_state

class PaperSpecTests(unittest.TestCase):
    def test_target_against_direct_footnote_sum_and_no_future_data(self):
        dates = pd.bdate_range('2010-01-01', periods=900)
        daily = pd.DataFrame({'date': dates, 'market': 'WTI', **{f'x_{h}': 1. for h in HORIZONS}})
        report = dates[::5]
        mm = pd.DataFrame({'market':'WTI','report_date':report,'report_week_tuesday':report,
                           'mm_net':np.arange(len(report)) ** 2 + 10.})
        panel = build_weekly_panel(daily, mm)
        row = panel.iloc[-1]
        i = len(mm)-2
        indices = np.arange(len(report))*5
        window = np.flatnonzero((indices >= indices[i]-251)&(indices <= indices[i]))
        means = [mm.mm_net.iloc[np.flatnonzero((indices >= indices[j]-251)&(indices <= indices[j]))].mean() for j in window]
        sigma = np.sqrt(np.sum((mm.mm_net.iloc[window].to_numpy()-means)**2)/(len(window)-1))
        self.assertAlmostEqual(row.mm_std_lag,sigma)
        self.assertAlmostEqual(row.y,(mm.mm_net.iloc[-1]-mm.mm_net.iloc[window].mean())/sigma)
        mm.loc[len(mm)-1,'mm_net'] += 10000
        changed = build_weekly_panel(daily,mm).iloc[-1]
        self.assertEqual(row.mm_std_lag,changed.mm_std_lag)
        self.assertEqual(row.mm_mean_lag,changed.mm_mean_lag)

    def test_loss_and_gradient(self):
        model=SharedMomentumNetwork(['WTI'],[5],2)
        with torch.no_grad():
            model.shared_weights.fill_(.2);model.bias.copy_(torch.tensor([[1.,3.]]))
        x=torch.zeros(2,1,1);y=torch.zeros(2,1)
        loss,_,_=model.objective(x,y)
        self.assertAlmostEqual(loss.item(),5+.04*.6+.01*4,places=6)
        loss.backward()
        torch.testing.assert_close(model.bias.grad,torch.tensor([[.96,3.04]]))

    def test_tensorflow_adam_equation(self):
        p=torch.nn.Parameter(torch.tensor([1.],dtype=torch.float64)); opt=PaperAdam([p])
        m=v=0.; expected=1.
        for t,g in enumerate([.000001,.2,-.3],1):
            p.grad=torch.tensor([g],dtype=torch.float64); opt.step()
            m=.9*m+.1*g;v=.999*v+.001*g*g
            expected-=.01*np.sqrt(1-.999**t)/(1-.9**t)*m/(np.sqrt(v)+1e-7)
            self.assertAlmostEqual(p.item(),expected,places=12)

    def test_volatility_is_causal_and_matches_weighted_sum(self):
        values=np.arange(1.,101.)
        result=_ewm_volatility(values)
        weights=(60/61)**np.arange(98,-1,-1)
        self.assertAlmostEqual(result[-1],np.sqrt(252*np.sum(weights*values[:-1]**2)/weights.sum()))
        np.testing.assert_allclose(result,_ewm_volatility(np.r_[values,np.ones(5000)])[:100])

    def test_defaults_and_first_fit_are_finite(self):
        from paper_replication.config import ReplicationConfig, MARKETS
        from paper_replication.model import fit_window
        from dataclasses import replace
        c=ReplicationConfig()
        self.assertEqual((c.lambda_w,c.lambda_bias,c.learning_rate,c.adam_epsilon),(.04,.01,.01,1e-7))
        self.assertEqual((c.first_epochs,c.rolling_epochs,c.train_weeks),(1024,36,104))
        x=torch.ones(3,len(MARKETS),len(HORIZONS))*.01
        y=torch.ones(3,len(MARKETS))
        model=fit_window(x,y,replace(c,first_epochs=2))
        self.assertTrue(torch.isfinite(model.forward(x)[0]).all())

    def test_bias_dates(self):
        state={'bias':torch.tensor([[1.,2.,3.]])}
        result=align_bias_state(state,[1,2,3],[2,3,4])
        torch.testing.assert_close(result['bias'],torch.tensor([[2.,3.,3.]]))
        torch.testing.assert_close(state['bias'],torch.tensor([[1.,2.,3.]]))

if __name__=='__main__': unittest.main()
