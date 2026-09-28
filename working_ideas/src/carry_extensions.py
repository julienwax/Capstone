"""Carry helpers preserved from nn_carry_volatility.ipynb for reproducible comparisons.

No notebook execution or global fitted state is needed. The risk network is in
sg_features.py; these functions implement the existing carry definition and head.
"""
import numpy as np
import pandas as pd
from paper_replication.config import MARKETS
KEYS = ["report_week_tuesday", "market"]
DEFAULT_FEATURE = "annual_carry_smooth4"

def standardize_window(raw, train_weeks=104, clip=3.0):
    """Scale train+current rows using only the preceding training rows."""
    raw = np.asarray(raw, dtype=float)
    missing = ~np.isfinite(raw)
    median = np.nanmedian(np.where(missing[:-1], np.nan, raw[:-1]), axis=0)
    filled = np.where(missing, median[None, :], raw)
    mean, std = filled[:-1].mean(axis=0), filled[:-1].std(axis=0)
    z = np.clip((filled-mean)/np.maximum(std, 1e-6), -clip, clip)
    return dict(z=z, mean=mean, std=std, median=median, current_missing=missing[-1])


def build_carry_features(daily, panel, prices, meta, smoothing_weeks=4):
    """Map actual annual contract pairs, then take a trailing four-report mean."""
    frames = []
    for market, group in daily.groupby('market', sort=True):
        g = group.sort_values('date').reset_index(drop=True)
        m = meta.loc[meta.market.eq(market)].sort_values('LAST_TRADEABLE_DT').reset_index(drop=True)
        front = np.searchsorted(m.LAST_TRADEABLE_DT.to_numpy(), g.date.to_numpy(), side='left')
        held_key = g.contract_key.to_numpy(dtype=int)
        held = m.iloc[held_key].reset_index(drop=True)
        lookup = {(int(r.maturity_year), int(r.maturity_month)): i for i,r in m.iterrows()}
        far_key = np.array([lookup[(int(r.maturity_year)+1,int(r.maturity_month))] for _,r in held.iterrows()])
        far = m.iloc[far_key].reset_index(drop=True)
        near_generic, far_generic = held_key-front+1, far_key-front+1
        q = prices.loc[prices.market.eq(market)].set_index(['date','nearby'])
        # Strict settlement quotes: missing carry is imputed only inside a later training window.
        near_px = q.PX_SETTLE.reindex(pd.MultiIndex.from_arrays([g.date,near_generic])).to_numpy()
        far_px = q.PX_SETTLE.reindex(pd.MultiIndex.from_arrays([g.date,far_generic])).to_numpy()
        gap = (far.LAST_TRADEABLE_DT-held.LAST_TRADEABLE_DT).dt.days.to_numpy()/365.25
        vol = g.pnl_vol.to_numpy(dtype=float)
        carry = np.divide(near_px-far_px,gap*vol,out=np.full(len(g),np.nan),where=np.isfinite(vol)&(vol>0))
        frames.append(pd.DataFrame(dict(date=g.date, market=market, annual_carry=carry)))
    curve = pd.concat(frames,ignore_index=True)
    rows = []
    for market,group in panel.groupby('market',sort=True):
        left = group[['report_week_tuesday','report_date','market']].sort_values('report_date')
        right = curve.loc[curve.market.eq(market),['date','annual_carry']].sort_values('date')
        w = pd.merge_asof(left,right,left_on='report_date',right_on='date',direction='backward')
        w = w.rename(columns={'date':'carry_feature_date'})
        w[DEFAULT_FEATURE] = w.annual_carry.rolling(smoothing_weeks,min_periods=smoothing_weeks).mean()
        rows.append(w)
    weekly = pd.concat(rows,ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    return dict(weekly=weekly)


def build_carry_design(predictions,features,complete_dates,feature_col=DEFAULT_FEATURE,train_weeks=104,clip=3.,markets=MARKETS):
    """Freeze each carry change using its own date's historical endpoint scaler."""
    dates = pd.DatetimeIndex(complete_dates).sort_values()
    raw = features.pivot(index='report_week_tuesday',columns='market',values=feature_col).reindex(index=dates,columns=markets)
    rows = []
    for date,group in predictions.groupby('report_week_tuesday',sort=True):
        i = dates.get_loc(date)
        if i < train_weeks:
            continue
        g = group.set_index('market').reindex(markets)
        s = standardize_window(raw.iloc[i-train_weeks:i+1].to_numpy(),train_weeks,clip)
        out = g.reset_index().copy()
        out['carry_delta_z'] = s['z'][-1]-s['z'][-2]
        rows.append(out)
    design = pd.concat(rows,ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    return design


def calibrate_carry_design(design,train_weeks=104,penalty=.1):
    """Fit market-specific ridge coefficients on prior genuine OOS errors only."""
    d = design.sort_values(KEYS).copy()
    d['normalized_oos_residual'] = (d.actual_change-d.predicted_change)/d.mm_std_lag
    weeks = pd.DatetimeIndex(sorted(d.report_week_tuesday.unique()))
    rows = []
    for i,date in enumerate(weeks):
        if i < train_weeks:
            continue
        hist = d.loc[d.report_week_tuesday.isin(weeks[i-train_weeks:i])]
        cur = d.loc[d.report_week_tuesday.eq(date)].copy()
        coefs = {}
        for market,g in hist.groupby('market'):
            x,r = g.carry_delta_z.to_numpy(),g.normalized_oos_residual.to_numpy()
            coefs[market] = float(np.dot(x,r)/(np.dot(x,x)+len(x)*penalty))
        cur['carry_head_coef'] = cur.market.map(coefs)
        cur['base_prediction'] = cur.predicted_change
        cur['carry_correction'] = cur.mm_std_lag*cur.carry_head_coef*cur.carry_delta_z
        cur['predicted_change'] = cur.base_prediction+cur.carry_correction
        rows.append(cur)
    return pd.concat(rows,ignore_index=True)


def carry_residual_head(predictions,features,complete_dates,feature_col=DEFAULT_FEATURE,train_weeks=104,penalty=.1,clip=3.,markets=MARKETS):
    """Calibrate against the supplied model: use risk predictions for the combination."""
    design = build_carry_design(predictions,features,complete_dates,feature_col,train_weeks,clip,markets)
    return calibrate_carry_design(design,train_weeks,penalty),design
