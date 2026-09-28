"""Helpers for the SG CTA working-ideas notebook (`sg_cta_decomposition.ipynb`).

Compares the replicated Insight 177 network with SG Prime Services index returns:
position decomposition, energy P&L attribution, a momentum network fitted to daily
SG returns, and error corrections of the Managed Money nowcast.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from torch import nn

from paper_replication.config import HORIZONS, MARKETS
from paper_replication.data import build_daily_features, build_rolling_contract, build_weekly_panel, load_inputs
from paper_replication.model import PaperAdam, align_bias_state, fit_window, reaction

MARKETS = list(MARKETS)
FEATURES = [f"x_{h}" for h in HORIZONS]
KEYS = ["report_week_tuesday", "market"]
# USD per unit of settlement price; heating oil and RBOB settle in cents per gallon.
MULTIPLIER = {"WTI": 1000, "BRENT": 1000, "GASOIL": 100, "HEATOIL": 420, "RBOB": 420, "NATGAS": 10000}
SG_SERIES = {"SG Trend Index": "SG Trend", "SG CTA Index": "SG CTA", "SG Trend Indicator": "SG Trend Indicator"}
PNL_LABELS = {"pnl_trend": "Trend positions", "pnl_bias": "Bias positions", "pnl_residual": "Fit residual",
              "pnl_mm": "Total MM", "pnl_rule": "20/120 rule"}


def _tensor(a):
    return torch.tensor(np.asarray(a), dtype=torch.float32)


# ---------------------------------------------------------------- data

def load_weekly_data(config):
    """Weekly panel and daily features as in `run_replication`, without refitting the baseline."""
    prices, meta, mm = load_inputs(config)
    daily = build_daily_features(build_rolling_contract(prices, meta))
    return build_weekly_panel(daily, mm), daily


def load_sg_log_levels(data_dir):
    """Daily log levels of the three SG series, renamed to short labels."""
    sg = pd.read_csv(data_dir / "sg_cta_indices.csv", parse_dates=["trade_date"]).set_index("trade_date")
    return np.log(sg[list(SG_SERIES)]).dropna().rename(columns=SG_SERIES).sort_index()


def add_trend_rule(daily, fast=20, slow=120):
    """SG Trend Indicator rule on the held contract: sign of the fast minus slow moving average."""
    daily = daily.sort_values(["market", "date"]).reset_index(drop=True)
    cum = daily.groupby("market")["cum_pnl"]
    daily["rule"] = np.sign(cum.transform(lambda s: s.rolling(fast).mean())
                            - cum.transform(lambda s: s.rolling(slow).mean()))
    daily["risk_return"] = daily["daily_pnl"] / (daily["pnl_vol"] / np.sqrt(252))
    return daily


def weekly_tensors(panel, config):
    """Complete reporting weeks and the network tensors x [weeks, markets, horizons], y [weeks, markets]."""
    complete = panel.groupby("report_week_tuesday")["market"].nunique().eq(len(MARKETS))
    weeks = sorted(complete[complete].index)
    rows = panel.set_index(KEYS).reindex(pd.MultiIndex.from_product([weeks, MARKETS], names=KEYS))
    x = _tensor(rows[FEATURES].to_numpy().reshape(-1, len(MARKETS), len(FEATURES)))
    y = _tensor(rows["y"].to_numpy().reshape(-1, len(MARKETS)))
    evaluation_weeks = set(panel.loc[panel["panel_date"].between(
        config.evaluation_start, config.evaluation_end), "report_week_tuesday"])
    return weeks, x, y, evaluation_weeks


# ------------------------------------------------ 0. trend / bias decomposition

def run_decomposition(panel, config):
    """Refit the replication network weekly and keep its trend and bias components.

    Uses the same windows, warm starts and seed as `run_replication`, so
    mm_std_lag * (trend_cur - trend_prev) equals the baseline predicted change.
    Also returns each fit's linearized horizon loading, sum_k W_m(k) w_k(n) averaged over markets.
    """
    weeks, x, y, evaluation_weeks = weekly_tensors(panel, config)
    records, loadings, state, state_dates = [], [], None, None
    for i, week in enumerate(weeks):
        if i < config.train_weeks or week not in evaluation_weeks:
            continue
        train_dates = weeks[i - config.train_weeks:i]
        initial = None if state is None else align_bias_state(state, state_dates, train_dates)
        model = fit_window(x[i - config.train_weeks:i], y[i - config.train_weeks:i], config, initial)
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        state_dates = train_dates
        with torch.no_grad():
            latent = reaction(torch.einsum("wmh,kh->wmk", x[i - 1:i + 1], model.shared_weights))
            trend = torch.einsum("wmk,mk->wm", latent, model.market_weights).numpy()
            loadings.append((model.market_weights.mean(0) @ model.shared_weights).numpy())
        records.append(pd.DataFrame({"report_week_tuesday": week, "market": MARKETS,
                                     "trend_prev": trend[0], "trend_cur": trend[1],
                                     "bias_prev": model.bias[:, -1].detach().numpy()}))
    return pd.concat(records, ignore_index=True), np.array(loadings)


# ------------------------------------------------ 1. energy P&L attribution

def weekly_energy_pnl(panel, daily, decomposition, sg_log_levels):
    """Weekly energy P&L of last week's position components, with SG returns on the same dates.

    Last week's MM position is split into trend (sigma*T), bias (mu + sigma*b) and fit residual,
    in contracts. P&L is contracts x multiplier x price change on the held contract, in USD
    millions, summed over markets. The 20/120 rule P&L is in volatility units.
    """
    weekly = (panel.merge(daily[["market", "date", "cum_pnl", "pnl_vol", "rule"]],
                          left_on=["market", "feature_date"], right_on=["market", "date"],
                          how="left", validate="one_to_one")
              .sort_values(["market", "report_week_tuesday"]).reset_index(drop=True))
    lag_cols = ["cum_pnl", "pnl_vol", "rule", "mm_net", "mm_mean_lag", "mm_std_lag",
                "report_week_tuesday", "panel_date"]
    weekly = pd.concat([weekly, weekly.groupby("market")[lag_cols].shift(1).add_suffix("_prev")], axis=1)
    weekly = weekly.merge(decomposition, on=KEYS, validate="one_to_one")
    if not weekly["report_week_tuesday_prev"].eq(weekly["previous_report_week"]).all():
        raise ValueError("A P&L week does not start at the previous report")

    weekly["price_change"] = weekly["cum_pnl"] - weekly["cum_pnl_prev"]
    weekly["q_trend"] = weekly["mm_std_lag_prev"] * weekly["trend_prev"]
    weekly["q_bias"] = weekly["mm_mean_lag_prev"] + weekly["mm_std_lag_prev"] * weekly["bias_prev"]
    weekly["q_residual"] = weekly["mm_net_prev"] - weekly["q_trend"] - weekly["q_bias"]
    weekly["q_mm"] = weekly["mm_net_prev"]
    usd_millions = weekly["market"].map(MULTIPLIER) * weekly["price_change"] / 1e6
    for part in ["trend", "bias", "residual", "mm"]:
        weekly[f"pnl_{part}"] = weekly[f"q_{part}"] * usd_millions
    weekly["pnl_rule"] = weekly["rule_prev"] * weekly["price_change"] / (weekly["pnl_vol_prev"] / np.sqrt(52))

    pnl = weekly.groupby("report_week_tuesday").agg(
        panel_date=("panel_date", "min"), panel_date_prev=("panel_date_prev", "min"),
        **{c: (c, "sum") for c in PNL_LABELS})
    level_at = lambda dates: sg_log_levels.reindex(pd.DatetimeIndex(dates), method="ffill").to_numpy()
    pnl[list(sg_log_levels.columns)] = 100 * (level_at(pnl["panel_date"]) - level_at(pnl["panel_date_prev"]))
    return pnl


def pnl_correlations(pnl, target="SG Trend"):
    """Pearson, rank and excluding-2020 correlations of each P&L series with an SG return."""
    cols, y = list(PNL_LABELS), pnl[target]
    no_2020 = pnl.index.year != 2020
    return pd.DataFrame({
        "Pearson": pnl[cols].corrwith(y),
        "Spearman (rank)": pnl[cols].corrwith(y, method="spearman"),
        "Pearson, excl. 2020": pnl.loc[no_2020, cols].corrwith(y[no_2020]),
    }).rename(index=PNL_LABELS)


def attribution_table(pnl, periods):
    """Joint regression of SG returns on standardized trend, bias and residual P&L (Newey-West, 4 lags).

    `periods` maps a row label to (SG series, first year, last year).
    """
    cols = ["pnl_trend", "pnl_bias", "pnl_residual"]
    rows = {}
    for label, (series, first, last) in periods.items():
        frame = pnl.loc[pnl.index.year.isin(range(first, last + 1))]
        X = sm.add_constant((frame[cols] - frame[cols].mean()) / frame[cols].std())
        fit = sm.OLS(frame[series], X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
        rows[label] = {**{f"t: {PNL_LABELS[c]}": fit.tvalues[c] for c in cols}, "R² (%)": 100 * fit.rsquared}
    return pd.DataFrame(rows).T


# ------------------------------------------------ 2. momentum network fitted to SG returns

def daily_sg_arrays(daily, sg_log_levels, series="SG Trend", start="2011", end="2025"):
    """Daily inputs for fitting SG returns.

    Signals and rule positions come from each market's previous close; a market that is
    closed on an SG date contributes a zero return.
    """
    sg_return = 100 * sg_log_levels[series].diff()
    dates = sg_return.loc[start:end].dropna().index
    xs, us, rules = [], [], []
    for market in MARKETS:
        g = daily.loc[daily["market"].eq(market)].set_index("date")
        xs.append(g[FEATURES].shift(1).reindex(dates, method="ffill").to_numpy())
        rules.append(g["rule"].shift(1).reindex(dates, method="ffill").to_numpy())
        us.append(g["risk_return"].reindex(dates).fillna(0).to_numpy())
    X, U, R, r = np.stack(xs, 1), np.stack(us, 1), np.stack(rules, 1), sg_return.reindex(dates).to_numpy()
    keep = np.isfinite(X).all((1, 2)) & np.isfinite(R).all(1) & np.isfinite(r)
    return {"dates": dates[keep], "x": X[keep], "u": U[keep], "rule": R[keep], "r": r[keep]}


class SGMomentumNetwork(nn.Module):
    """The paper's shared trend layer with one set of trend weights for all markets.

    position(x) = sum_k v_k R(sum_n w_k(n) x(n)); the SG return is c + sum_m position * risk return.
    """

    def __init__(self, seed=7):
        super().__init__()
        torch.manual_seed(seed)
        self.shared_weights = nn.Parameter(torch.zeros(3, len(HORIZONS)))
        self.trend_weights = nn.Parameter(torch.tensor([0.4, 0.2, 0.1]))
        self.intercept = nn.Parameter(torch.zeros(1))

    def position(self, x):
        latent = reaction(torch.einsum("dmh,kh->dmk", x, self.shared_weights))
        return torch.einsum("dmk,k->dm", latent, self.trend_weights)

    def forward(self, x, u):
        return self.intercept + (self.position(x) * u).sum(1)


def fit_sg(data, mask, penalty, config, epochs=1500):
    """Full-batch fit on the masked days; the target is scaled by its training standard deviation."""
    model = SGMomentumNetwork(config.seed)
    optimizer = PaperAdam(model.parameters(), lr=config.learning_rate,
                          betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_epsilon)
    scale = float(data["r"][mask].std())
    x, u, y = _tensor(data["x"][mask]), _tensor(data["u"][mask]), _tensor(data["r"][mask] / scale)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = ((y - model(x, u)) ** 2).mean() + penalty * model.shared_weights.abs().sum()
        loss.backward()
        optimizer.step()
    return model, scale


def predict_sg(model, scale, data, mask):
    with torch.no_grad():
        return scale * model(_tensor(data["x"][mask]), _tensor(data["u"][mask])).numpy()


def rolling_sg_fit(data, config, years=range(2015, 2026), train_years=4,
                   penalties=(0.0005, 0.002, 0.01, 0.04)):
    """Fit each year on the previous `train_years` years and predict it out of sample.

    The L1 penalty is chosen by correlation on the last training year after fitting on the
    years before it. The 20/120 rule benchmark gets a linear scale fitted on the same window.
    """
    year = data["dates"].year
    rule_pnl = (data["rule"] * data["u"]).sum(1)
    models, chosen, sg_fit, rule_fit = {}, {}, [], []
    for test_year in years:
        inner = (year >= test_year - train_years) & (year < test_year - 1)
        valid = year == test_year - 1
        chosen[test_year] = max(penalties, key=lambda p: np.corrcoef(
            predict_sg(*fit_sg(data, inner, p, config), data, valid), data["r"][valid])[0, 1])
        train, test = (year >= test_year - train_years) & (year < test_year), year == test_year
        models[test_year] = fit_sg(data, train, chosen[test_year], config)
        sg_fit.append(pd.Series(predict_sg(*models[test_year], data, test), index=data["dates"][test]))
        slope, intercept = np.polyfit(rule_pnl[train], data["r"][train], 1)
        rule_fit.append(pd.Series(intercept + slope * rule_pnl[test], index=data["dates"][test]))
    sg_fit, rule_fit = pd.concat(sg_fit), pd.concat(rule_fit)
    actual = pd.Series(data["r"], index=data["dates"]).loc[sg_fit.index]
    return {"models": models, "penalty": chosen, "fitted": sg_fit, "rule_fitted": rule_fit, "actual": actual}


def fit_scores(fitted, actual):
    weekly = pd.concat([fitted, actual], axis=1).resample("W-TUE").sum()
    return pd.Series({
        "Daily correlation": fitted.corr(actual),
        "Daily R² (%)": 100 * (1 - ((actual - fitted) ** 2).sum() / ((actual - actual.mean()) ** 2).sum()),
        "Weekly correlation": weekly.corr().iloc[0, 1]})


def yearly_correlations(fits, actual):
    return pd.DataFrame({name: pd.concat([f, actual], axis=1).groupby(f.index.year)
                         .apply(lambda d: d.corr().iloc[0, 1]) for name, f in fits.items()}).T


def sg_horizon_loadings(models):
    """Linearized horizon loading sum_k v_k w_k(n) of each yearly SG fit."""
    return np.array([(m.trend_weights @ m.shared_weights).detach().numpy() for m, _ in models.values()])


# ------------------------------------------------ 3. SG-implied flow

def sg_implied_flow(panel, daily, models):
    """Weekly change in the SG-implied contract proxy h/vol, both endpoints from the current year's model."""
    report = panel[KEYS + ["feature_date", "previous_report_week"] + FEATURES].merge(
        daily[["market", "date", "pnl_vol"]], left_on=["market", "feature_date"],
        right_on=["market", "date"], how="left", validate="one_to_one")
    previous = report[KEYS + FEATURES + ["pnl_vol"]].rename(columns={"report_week_tuesday": "previous_report_week"})
    report = report.merge(previous, on=["previous_report_week", "market"], suffixes=("", "_prev"), how="left")
    report = report.loc[report["report_week_tuesday"].dt.year.isin(list(models))
                        & report["pnl_vol_prev"].notna()].reset_index(drop=True)
    report["sg_flow"] = np.nan
    for year, (model, scale) in models.items():
        rows = report["report_week_tuesday"].dt.year.eq(year).to_numpy()
        with torch.no_grad():
            now = model.position(_tensor(report.loc[rows, FEATURES].to_numpy()[:, None, :])).numpy()[:, 0]
            before = model.position(_tensor(
                report.loc[rows, [f + "_prev" for f in FEATURES]].to_numpy()[:, None, :])).numpy()[:, 0]
        report.loc[rows, "sg_flow"] = scale * (now / report.loc[rows, "pnl_vol"]
                                               - before / report.loc[rows, "pnl_vol_prev"])
    return report[KEYS + ["sg_flow"]]


def flow_correlations(flows):
    """Per-market correlations between actual MM changes, NN trend flow and SG-implied flow."""
    corr = lambda a, b: flows.groupby("market").apply(lambda d: d[a].corr(d[b]))
    return pd.DataFrame({
        "NN trend flow": corr("actual_change", "nn_trend_flow"),
        "SG-implied flow": corr("actual_change", "sg_flow"),
        "NN trend flow vs SG-implied flow": corr("nn_trend_flow", "sg_flow"),
    }).loc[MARKETS]


# ------------------------------------------------ 4. daily SG residual features

def residual_features(residual, risk_returns, as_of):
    """Weekly features of the daily SG residual over (previous as-of date, as-of date].

    resid_week: summed residual; resid_x_return: residual times the market's risk return;
    resid_beta20: 20-day regression slope of the residual on the market's risk return.
    """
    beta20 = {m: (residual * risk_returns[m]).rolling(20).sum() / (risk_returns[m] ** 2).rolling(20).sum()
              for m in MARKETS}
    records = []
    for (week, end), start in zip(as_of.iloc[1:].items(), as_of.iloc[:-1]):
        window = (residual.index > start) & (residual.index <= end)
        if not window.any():
            continue
        for m in MARKETS:
            records.append({"report_week_tuesday": week, "market": m,
                            "resid_week": residual[window].sum(),
                            "resid_x_return": (residual[window] * risk_returns.loc[window, m]).sum(),
                            "resid_beta20": beta20[m].loc[:end].iloc[-1]})
    return pd.DataFrame(records)


def daily_residual_table(residual, risk_returns, as_of, lags=5):
    """The last `lags` daily residuals up to each as-of date, one column per day (lag 0 = as-of day).

    resid_lag{k}: SG residual k trading days before the as-of date;
    resid_x_return_lag{k}: that residual times the market's risk return on the same day.
    Weeks whose as-of date is outside the residual sample are skipped.
    """
    index = residual.index
    records = []
    for week, end in as_of.items():
        pos = index.searchsorted(end, side="right") - 1
        if pos < lags - 1 or (end - index[pos]).days > 4:
            continue
        rows = slice(pos - lags + 1, pos + 1)
        e = residual.iloc[rows].to_numpy()[::-1]
        for m in MARKETS:
            u = risk_returns[m].iloc[rows].to_numpy()[::-1]
            record = {"report_week_tuesday": week, "market": m}
            record.update({f"resid_lag{k}": e[k] for k in range(lags)})
            record.update({f"resid_x_return_lag{k}": e[k] * u[k] for k in range(lags)})
            records.append(record)
    return pd.DataFrame(records)


def residual_columns(lags):
    return [f"resid_lag{k}" for k in range(lags)] + [f"resid_x_return_lag{k}" for k in range(lags)]


# ------------------------------------------------ nowcast scoring

def score(frame, column="predicted_change"):
    actual, predicted = frame["actual_change"].to_numpy(), frame[column].to_numpy()
    return pd.Series({"R² (%)": 100 * (1 - np.sum((actual - predicted) ** 2) / np.sum(actual ** 2)),
                      "Direction (%)": 100 * np.mean(np.sign(actual) == np.sign(predicted)),
                      "MAE (contracts)": np.mean(np.abs(actual - predicted)),
                      "Observations": len(frame)})


def common_sample(frame, start="2017-01-01", end="2025-12-31"):
    return frame.loc[frame["panel_date"].between(start, end)].copy()


def rolling_ridge(design, columns, target, base=None, pooled=True, train_weeks=104, penalty=0.1):
    """Weekly out-of-sample ridge fitted on the previous `train_weeks` weeks.

    `target` is a column already divided by mm_std_lag; features are standardized on the
    training window and the fit has no intercept. The prediction, in contracts, is
    base + mm_std_lag * fit, where `base` is a column of contracts or None for zero.
    Returns the predictions and each week's standardized coefficients.
    """
    d = design.sort_values(KEYS).copy()
    dates = sorted(d["report_week_tuesday"].unique())
    groups = [MARKETS] if pooled else [[m] for m in MARKETS]
    out, coefs = [], []
    for i in range(train_weeks, len(dates)):
        hist = d.loc[d["report_week_tuesday"].isin(dates[i - train_weeks:i])].dropna(subset=columns + [target])
        cur = d.loc[d["report_week_tuesday"].eq(dates[i])].copy()
        cur["fit"] = 0.0
        for group in groups:
            h, rows = hist.loc[hist["market"].isin(group)], cur["market"].isin(group)
            mean, std = h[columns].mean(), h[columns].std().replace(0, 1)
            Xh = ((h[columns] - mean) / std).to_numpy()
            Xc = ((cur.loc[rows, columns] - mean) / std).fillna(0).to_numpy()
            beta = np.linalg.solve(Xh.T @ Xh + len(Xh) * penalty * np.eye(len(columns)),
                                   Xh.T @ h[target].to_numpy())
            cur.loc[rows, "fit"] = cur.loc[rows, "mm_std_lag"] * (Xc @ beta)
            coefs.append(pd.Series(beta, index=columns, name=(dates[i], "+".join(group))))
        cur["prediction"] = (0.0 if base is None else cur[base]) + cur["fit"]
        out.append(cur)
    return pd.concat(out, ignore_index=True), pd.DataFrame(coefs)


def error_correction(design, columns, pooled, train_weeks=104, penalty=0.1):
    """Correct NN predicted changes with a ridge fit to the prior genuine out-of-sample errors.

    Errors are divided by mm_std_lag; features are standardized on the same training window.
    One regression per market, or one pooled across markets.
    """
    d = design.copy()
    d["normalized_error"] = (d["actual_change"] - d["predicted_change"]) / d["mm_std_lag"]
    out, _ = rolling_ridge(d, columns, "normalized_error", "predicted_change", pooled, train_weeks, penalty)
    return out.rename(columns={"fit": "correction", "prediction": "corrected_change"})
