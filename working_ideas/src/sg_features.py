"""Walk-forward SG-conditioned MM experiments. All fitted transforms use history.

The prediction task is a same-week position nowcast, not a return forecast.
SG dates have no publication timestamps: a configurable business-day buffer is
an explicit availability assumption, not a guarantee of point-in-time data.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import nnls
import torch
from torch import nn
from tqdm.auto import tqdm

from paper_replication.config import HORIZONS, MARKETS
from paper_replication.model import SharedMomentumNetwork, PaperAdam, align_bias_state, reaction
from carry_extensions import build_carry_features, build_carry_design, calibrate_carry_design

KEYS = ["report_week_tuesday", "market"]
LIBRARY_HORIZONS = (20, 60, 125, 250, 500)


@dataclass(frozen=True)
class ExperimentSettings:
    sg_lag_bdays: int = 2
    sg_train_years: int = 4
    joint_weight: float = 0.1
    joint_days: int = 504
    overlay_weeks: int = 104
    validation_weeks: int = 26
    penalties: tuple = (1.0, 10.0, 100.0)
    carry_penalty: float = 0.1


def prepare(reference, prices, meta, data_dir):
    """Align complete weekly tensors and causal daily momentum library."""
    daily = reference["daily_features"].copy()
    panel = reference["weekly_panel"].merge(
        daily[["market", "date", "pnl_vol"]],
        left_on=["market", "feature_date"], right_on=["market", "date"],
        validate="one_to_one")
    counts = panel.groupby("report_week_tuesday")["market"].nunique()
    weeks = pd.DatetimeIndex(counts[counts.eq(len(MARKETS))].index).sort_values()
    rows = panel.set_index(KEYS).reindex(pd.MultiIndex.from_product([weeks, MARKETS], names=KEYS))
    x = rows[[f"x_{h}" for h in HORIZONS]].to_numpy().reshape(-1, len(MARKETS), len(HORIZONS))
    y = rows["y"].to_numpy().reshape(-1, len(MARKETS))
    vol = rows["pnl_vol"].to_numpy().reshape(-1, len(MARKETS))
    if not all(np.isfinite(a).all() for a in (x, y, vol)) or (vol <= 0).any():
        raise ValueError("Incomplete/nonfinite weekly data or nonpositive volatility")
    levels = pd.read_csv(data_dir / "sg_cta_indices.csv", parse_dates=["trade_date"]).set_index("trade_date")
    sg_return = 100 * np.log(levels["SG Trend Index"].where(levels["SG Trend Index"] > 0)).diff()
    sg_return = sg_return.dropna().sort_index()
    carry = build_carry_features(daily, panel, prices, meta)["weekly"]
    return dict(daily=daily, panel=panel, weeks=weeks, rows=rows,
                x=x.astype("float32"), y=y.astype("float32"), vol=vol,
                sg_return=sg_return, carry=carry)


def daily_arrays(data, horizons=HORIZONS):
    """Prior-close signals, contemporaneous realized risk returns, and SG target.

    Missing market dates have zero P&L; their position uses the last actual close.
    This is a daily return-replication training target, never a future SG forecast.
    """
    dates = data["sg_return"].index
    xs, us, vs = [], [], []
    for market in MARKETS:
        g = data["daily"].loc[lambda d: d.market.eq(market)].set_index("date").sort_index()
        cols = []
        for horizon in horizons:
            if f"x_{horizon}" in g:
                value = g[f"x_{horizon}"]
            else:
                mom = (g.cum_pnl - g.cum_pnl.rolling(horizon).mean()) / g.pnl_vol.replace(0, np.nan)
                value = mom / mom.rolling(252).std().replace(0, np.nan)
            cols.append(value.rename(str(horizon)))
        signals = pd.concat(cols, axis=1)
        # asof strictly BEFORE each SG date, including market-specific holidays.
        union = signals.index.union(dates).sort_values()
        prior = signals.reindex(union).ffill().shift(1).reindex(dates)
        xs.append(prior.to_numpy())
        us.append((g.daily_pnl / (g.pnl_vol / np.sqrt(252))).reindex(dates).fillna(0).to_numpy())
        vs.append(g.pnl_vol.reindex(union).ffill().reindex(dates).to_numpy())
    x, u, vol = np.stack(xs, 1), np.stack(us, 1), np.stack(vs, 1)
    keep = np.isfinite(x).all((1, 2)) & np.isfinite(u).all(1) & np.isfinite(vol).all(1) & (vol > 0).all(1)
    return dict(dates=dates[keep], x=x[keep].astype("float32"), u=u[keep].astype("float32"),
                vol=vol[keep].astype("float32"), r=data["sg_return"].reindex(dates[keep]).to_numpy("float32"))


class SGConditionedNetwork(SharedMomentumNetwork):
    """Original momentum factors + risk gate + optional separate SG return head."""
    def __init__(self, config, joint=False):
        super().__init__(MARKETS, HORIZONS, config.train_weeks, config.seed)
        self.alpha = nn.Parameter(torch.zeros(1))
        if joint:
            self.sg_weights = nn.Parameter(torch.tensor([0.4, 0.2, 0.1]))
            self.sg_intercept = nn.Parameter(torch.zeros(1))

    def trend(self, x):
        return reaction(torch.einsum("wmh,kh->wmk", x, self.shared_weights))

    def predict(self, x, log_vol_ratio, training=False):
        trend = torch.einsum("wmk,mk->wm", self.trend(x), self.market_weights)
        bias = self.bias.T if training else self.bias[:, -1].expand(len(x), -1)
        return bias + torch.exp(-self.alpha * log_vol_ratio) * trend

    def mm_loss(self, x, y, log_vol_ratio, config):
        return ((self.predict(x, log_vol_ratio, True) - y).square().mean()
                + config.lambda_w * self.shared_weights.abs().sum()
                + config.lambda_bias * self.bias.diff(dim=1).square().sum()
                + 0.1 * self.alpha.square().mean())

    def sg_prediction(self, x, u, log_vol_ratio):
        position = torch.einsum("dmk,k->dm", self.trend(x), self.sg_weights)
        return self.sg_intercept + (position * torch.exp(-self.alpha * log_vol_ratio) * u).sum(1)


def run_network(data, config, settings=ExperimentSettings(), joint_weight=0.0, progress=True):
    """Weekly risk/joint fits; MM and SG training end before the target week.

    joint_weight=0 exactly uses the risk-only objective. Both prediction endpoints
    share the fitted weights, volatility reference and final training bias.
    """
    joint = joint_weight > 0
    ds = daily_arrays(data) if joint else None
    x, y = torch.from_numpy(data["x"]), torch.from_numpy(data["y"])
    weeks, vol, rows = data["weeks"], data["vol"], data["rows"]
    state, old_dates, out = None, None, []
    iterator = tqdm(range(config.train_weeks, len(weeks)), disable=not progress,
                    desc="Joint MM/SG" if joint else "Volatility NN", unit="week")
    for i in iterator:
        week = weeks[i]
        cur = rows.loc[[week]].reset_index()
        if not cur.panel_date.between(config.evaluation_start, config.evaluation_end).all():
            continue
        if not cur.previous_report_week.eq(weeks[i - 1]).all():
            raise ValueError("A multi-week gap cannot be scored as a weekly change")
        start = i - config.train_weeks
        train_dates = weeks[start:i]
        ref_vol = np.median(vol[start:i], axis=0)
        features = torch.tensor(np.log(vol[start:i + 1] / ref_vol), dtype=torch.float32)
        model = SGConditionedNetwork(config, joint)
        if state is not None:
            model.load_state_dict(align_bias_state(state, old_dates, train_dates))
        optimizer = PaperAdam(model.parameters(), lr=config.learning_rate,
                              betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_epsilon)
        sg_end = pd.NaT
        if joint:
            # Conservative: even the last training week's SG has an availability buffer.
            cutoff = rows.loc[[weeks[i - 1]], "report_date"].min() - pd.offsets.BDay(settings.sg_lag_bdays)
            take = np.flatnonzero(ds["dates"] <= cutoff)[-settings.joint_days:]
            if len(take) < 126:
                raise ValueError("Joint fit needs at least 126 historical SG days")
            sg_end = ds["dates"][take[-1]]
            sx, su = torch.from_numpy(ds["x"][take]), torch.from_numpy(ds["u"][take])
            sr = torch.from_numpy(ds["r"][take] / max(float(ds["r"][take].std()), 1e-6))
            sv = torch.tensor(np.log(ds["vol"][take] / ref_vol), dtype=torch.float32)
        epochs = config.first_epochs if state is None else config.rolling_epochs
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = model.mm_loss(x[start:i], y[start:i], features[:-1], config)
            if joint:
                loss = loss + joint_weight * (model.sg_prediction(sx, su, sv) - sr).square().mean()
                loss = loss + joint_weight * 0.01 * model.sg_weights.square().mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.alpha.clamp_(0, 1)
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        old_dates = train_dates
        with torch.no_grad():
            pair = model.predict(x[i - 1:i + 1], features[-2:]).numpy()
        cur["predicted_change"] = cur.mm_std_lag * (pair[1] - pair[0])
        cur["alpha"] = model.alpha.item()
        cur["mm_train_end"] = weeks[i - 1]
        cur["sg_train_end"] = sg_end
        out.append(cur[KEYS + ["report_date", "panel_date", "previous_report_week", "actual_change",
                              "mm_std_lag", "predicted_change", "alpha", "mm_train_end", "sg_train_end"]])
    if not out:
        raise ValueError("No network predictions: check evaluation dates")
    return pd.concat(out, ignore_index=True)


def carry_design(predictions, data, train_weeks=104):
    return build_carry_design(predictions, data["carry"], data["weeks"], train_weeks=train_weeks)


def with_carry(predictions, data, train_weeks=104):
    return calibrate_carry_design(carry_design(predictions, data, train_weeks), train_weeks=train_weeks)


def conditioning_features(data, settings=ExperimentSettings()):
    """Two SG states and matched energy-only controls, sampled by previous report.

    States are lagged by a whole reporting week PLUS the publication buffer.
    The energy comparator is an equal-weight 20/120 moving-average rule.
    """
    frames = []
    for market, g in data["daily"].groupby("market"):
        g = g.sort_values("date").set_index("date")
        signal = np.sign(g.cum_pnl.rolling(20).mean() - g.cum_pnl.rolling(120).mean()).shift(1)
        frames.append((signal * g.daily_pnl / (g.pnl_vol / np.sqrt(252))).rename(market))
    energy = pd.concat(frames, axis=1).sum(axis=1, min_count=1)
    frame = pd.concat([data["sg_return"].rename("sg"), energy.rename("energy")], axis=1).dropna()
    states = pd.DataFrame(index=frame.index)
    states["sg_vol"] = frame.sg.rolling(60).std()
    states["sg_alignment"] = frame.sg.rolling(60).corr(frame.energy)
    states["energy_vol"] = frame.energy.rolling(60).std()
    states["energy_strength"] = frame.energy.rolling(60).mean() / states.energy_vol.replace(0, np.nan)
    dates = data["panel"].groupby("report_week_tuesday").report_date.min().sort_index()
    target = pd.DataFrame({"report_week_tuesday": dates.index,
                           "cutoff": (dates.shift(1) - pd.offsets.BDay(settings.sg_lag_bdays)).to_numpy()})
    target = target.dropna().sort_values("cutoff")
    right = states.rename_axis("state_date").reset_index()
    return pd.merge_asof(target, right, left_on="cutoff", right_on="state_date", direction="backward")


def _ridge_predict(hist, cur, columns, penalty, carry, interaction, settings):
    """Historical feature scaling; market-specific carry plus pooled new terms."""
    def base_matrix(frame):
        if not carry:
            return np.empty((len(frame), 0))
        return np.column_stack([frame.carry_delta_z.to_numpy() * frame.market.eq(m) for m in MARKETS])
    a, b = base_matrix(hist), base_matrix(cur)
    penalties = [settings.carry_penalty] * a.shape[1]
    if columns and penalty is not None:
        means = hist[columns].mean()
        std = hist[columns].std().replace(0, 1).fillna(1)
        ha = ((hist[columns] - means) / std).to_numpy()
        cb = ((cur[columns] - means) / std).to_numpy()
        if interaction:
            selected = list(range(len(columns))) if interaction is True else [columns.index(c) for c in interaction]
            ha[:, selected] *= (hist.predicted_change / hist.mm_std_lag).to_numpy()[:, None]
            cb[:, selected] *= (cur.predicted_change / cur.mm_std_lag).to_numpy()[:, None]
        a, b = np.column_stack([a, ha]), np.column_stack([b, cb])
        # There are M times as many pooled rows as rows per carry coefficient.
        penalties += [penalty * len(MARKETS)] * len(columns)
    if a.shape[1] == 0:
        return cur.predicted_change.to_numpy()
    target = ((hist.actual_change - hist.predicted_change) / hist.mm_std_lag).to_numpy()
    per_market_n = len(hist) / len(MARKETS)
    beta = np.linalg.solve(a.T @ a + per_market_n * np.diag(penalties), a.T @ target)
    return cur.predicted_change.to_numpy() + cur.mm_std_lag.to_numpy() * (b @ beta)


def rolling_overlay(design, columns, settings=ExperimentSettings(), carry=True, interaction=False):
    """Select shrinkage on a trailing validation block, never on the scored week.

    The no-new-feature candidate is always available. Historical base predictions
    must themselves be walk-forward OOS predictions. Validation targets are used
    only to choose a penalty; refitting then uses the entire historical window.
    """
    required = columns + ["actual_change", "predicted_change", "mm_std_lag"] + (["carry_delta_z"] if carry else [])
    design = design.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    counts = design.groupby("report_week_tuesday").market.nunique()
    dates = sorted(counts[counts.eq(len(MARKETS))].index)
    out = []
    for i in range(settings.overlay_weeks, len(dates)):
        past = dates[i - settings.overlay_weeks:i]
        h = design.loc[design.report_week_tuesday.isin(past)]
        cur = design.loc[design.report_week_tuesday.eq(dates[i])].copy()
        nvalid = settings.validation_weeks
        if not 0 < nvalid < len(past):
            raise ValueError("Validation length must be inside the historical training window")
        inner = h.loc[h.report_week_tuesday.isin(past[:-nvalid])]
        valid = h.loc[h.report_week_tuesday.isin(past[-nvalid:])]
        choices = (None,) + settings.penalties if columns else (None,)
        losses = []
        for penalty in choices:
            p = _ridge_predict(inner, valid, columns, penalty, carry, interaction, settings)
            losses.append(float(np.square(valid.actual_change.to_numpy() - p).sum()))
        chosen = choices[int(np.argmin(losses))]
        cur["base_prediction"] = cur.predicted_change
        cur["predicted_change"] = _ridge_predict(h, cur, columns, chosen, carry, interaction, settings)
        cur["selected_penalty"] = np.nan if chosen is None else chosen
        cur["new_feature_active"] = chosen is not None
        cur["overlay_train_end"] = past[-1]
        cur["validation_sse"] = min(losses)
        out.append(cur)
    if not out:
        raise ValueError("Insufficient common history for rolling overlay")
    return pd.concat(out, ignore_index=True)


def _fit_blend(a, y, penalty):
    """Nonnegative regularized SG replication weights, scales fitted on history."""
    scale = np.maximum(a.std(axis=0), 1e-6)
    target_scale = max(float(y.std()), 1e-6)
    weights = nnls(np.vstack([a / scale, np.sqrt(len(a) * penalty) * np.eye(a.shape[1])]),
                   np.r_[y / target_scale, np.zeros(a.shape[1])])[0]
    return weights / scale * target_scale


def horizon_flows(data, settings=ExperimentSettings(), horizons=LIBRARY_HORIZONS):
    """Annual SG-fitted nonnegative horizon blend plus fixed no-SG comparator.

    Fits stop before each test year with an availability buffer. The previous
    year is validation only for penalty selection. Endpoint flows use the SAME
    year's blend, avoiding spurious turnover at annual refits.
    """
    ds = daily_arrays(data, horizons)
    positions = ds["x"] * np.exp((1 - ds["x"] ** 2) / 2)
    library_pnl = (positions * ds["u"][:, :, None]).sum(axis=1)
    # Recompute close-of-day signals for weekly endpoint positions (not shifted).
    weekly = []
    for market, g in data["daily"].groupby("market"):
        g = g.sort_values("date").copy()
        cols = []
        for horizon in horizons:
            if f"x_{horizon}" in g:
                z = g[f"x_{horizon}"]
            else:
                mom = (g.cum_pnl - g.cum_pnl.rolling(horizon).mean()) / g.pnl_vol.replace(0, np.nan)
                z = mom / mom.rolling(252).std().replace(0, np.nan)
            col = f"position_{horizon}"
            g[col] = z * np.exp((1 - z ** 2) / 2) / g.pnl_vol
            cols.append(col)
        left = data["panel"].loc[lambda d: d.market.eq(market), KEYS + ["feature_date", "previous_report_week"]]
        w = left.merge(g[["date"] + cols], left_on="feature_date", right_on="date", validate="one_to_one")
        prev = w[["report_week_tuesday"] + cols].rename(columns={"report_week_tuesday": "previous_report_week", **{c: c + "_prev" for c in cols}})
        w = w.merge(prev, on="previous_report_week", how="left", validate="many_to_one")
        for col in cols:
            w[col] -= w[col + "_prev"]
        weekly.append(w[KEYS + cols])
    flows = pd.concat(weekly, ignore_index=True)
    flow_cols = [f"position_{h}" for h in horizons]
    flows["fixed_flow"] = flows[flow_cols].mean(axis=1, skipna=False)
    flows["sg_flow"] = np.nan
    audit = []
    for year in sorted(flows.report_week_tuesday.dt.year.unique()):
        cutoff = pd.Timestamp(year=year, month=1, day=1) - pd.offsets.BDay(settings.sg_lag_bdays)
        train = (ds["dates"] >= pd.Timestamp(year=year - settings.sg_train_years, month=1, day=1)) & (ds["dates"] < cutoff)
        inner = train & (ds["dates"].year < year - 1)
        valid = train & (ds["dates"].year == year - 1)
        if inner.sum() < 252 or valid.sum() < 126:
            continue
        choices = (0.1, 1.0, 10.0)
        loss = [np.square(ds["r"][valid] - library_pnl[valid] @ _fit_blend(library_pnl[inner], ds["r"][inner], p)).sum() for p in choices]
        penalty = choices[int(np.argmin(loss))]
        weights = _fit_blend(library_pnl[train], ds["r"][train], penalty)
        take = flows.report_week_tuesday.dt.year.eq(year)
        flows.loc[take, "sg_flow"] = flows.loc[take, flow_cols].to_numpy() @ weights
        audit.append(dict(year=year, train_end=ds["dates"][train][-1], penalty=penalty,
                          **{f"weight_{h}": w for h, w in zip(horizons, weights)}))
    return flows[KEYS + ["fixed_flow", "sg_flow"]], pd.DataFrame(audit)


def compare(predictions, start="2017-01-01", end="2025-12-31"):
    """Identical finite keys for every model; uncentred R² matches the notebooks."""
    indexed = {}
    for name, frame in predictions.items():
        frame = frame.loc[frame.panel_date.between(start, end)].dropna(subset=["predicted_change", "actual_change"])
        if frame.duplicated(KEYS).any():
            raise ValueError(f"Duplicate keys in {name}")
        indexed[name] = frame.set_index(KEYS).sort_index()
    keys = None
    for frame in indexed.values():
        keys = frame.index if keys is None else keys.intersection(frame.index)
    if keys is None or not len(keys):
        raise ValueError("Models have no common evaluation observations")
    aligned = {n: f.loc[keys].reset_index() for n, f in indexed.items()}
    first = next(iter(aligned.values())).actual_change.to_numpy()
    for frame in aligned.values():
        np.testing.assert_allclose(frame.actual_change, first)
    def score(f):
        a, p = f.actual_change.to_numpy(), f.predicted_change.to_numpy()
        return {"R² (%)": 100 * (1 - np.square(a - p).sum() / np.square(a).sum()),
                "Direction (%)": 100 * np.mean(np.sign(a) == np.sign(p)),
                "MAE (contracts)": np.abs(a - p).mean(), "Observations": len(a)}
    summary = pd.DataFrame({n: score(f) for n, f in aligned.items()}).T
    annual, markets = [], []
    for name, frame in aligned.items():
        for year, g in frame.groupby(frame.panel_date.dt.year):
            annual.append(dict(model=name, year=year, **score(g)))
        for market, g in frame.groupby("market"):
            markets.append(dict(model=name, market=market, **score(g)))
    return summary, pd.DataFrame(annual), pd.DataFrame(markets), aligned
