"""Lead-lag networks of Managed Money flows for `leadlag_network.ipynb` (Bennett, Cucuringu and Reinert, 2022).

Convention throughout: S[i, j] > 0 means market i leads market j, and C[l-1, i, j] = corr(Y^i_{t-l}, Y^j_t).
"""
import warnings

import networkx as nx
import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy.cluster.vq import kmeans2

SECTOR = {"BRENT": "oil", "WTI": "oil", "GASOIL": "oil", "HEATOIL": "oil", "RBOB": "oil", "NATGAS": "gas",
          "GC": "metals", "SI": "metals", "PL": "metals", "PA": "metals", "HG": "metals",
          "C": "grains", "W": "grains", "S": "grains", "KC": "softs", "CC": "softs", "SB": "softs", "CT": "softs"}
MARKETS = list(SECTOR)
OIL = ["BRENT", "WTI", "GASOIL", "HEATOIL", "RBOB"]
OILGAS = OIL + ["NATGAS"]
START = "2011-07-12"   # first week all 18 scaled flows exist (ICE COT starts 2011-01, 26-week minimum for the scale)
END = "2026-08-25"
PERIODS = {"2011-2015": ("2011-07-12", "2015-12-31"), "2016-2020": ("2016-01-01", "2020-12-31"),
           "2021-2025": ("2021-01-01", "2025-12-31")}


def scale_lag(frame, window=52, min_periods=26):
    """Divide by the rolling standard deviation of the previous `window` weeks (known before the week starts)."""
    return frame / frame.rolling(window, min_periods=min_periods).std().shift(1)


def _ols_resid(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta, beta


def disc_window(raw, sf, sret, start=START, end=END):
    """Window OLS: residual of scaled d_mm on constant, own scaled signal flow and own scaled return."""
    out, rows = {}, []
    idx = raw.loc[start:end].index
    for m in raw.columns:
        y = raw[m].reindex(idx).to_numpy()
        X = np.c_[np.ones(len(idx)), sf[m].reindex(idx).to_numpy(), sret[m].reindex(idx).to_numpy()]
        ok = np.isfinite(y) & np.isfinite(X).all(1)
        e = np.full(len(idx), np.nan)
        e[ok], beta = _ols_resid(y[ok], X[ok])
        r2 = 1 - np.sum(e[ok] ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2)
        rows.append({"market": m, "const": beta[0], "b_signal": beta[1], "b_ret": beta[2], "R2": r2, "n": int(ok.sum())})
        out[m] = e
    return pd.DataFrame(out, index=idx), pd.DataFrame(rows).set_index("market")


def build_series(T):
    """Weekly series on the common 18-market panel. Returns a dict of DataFrames and the table of `disc_window`."""
    raw, sf, sret = scale_lag(T.d_mm), scale_lag(T.signal_flow), scale_lag(T.ret)
    disc, disc_fit = disc_window(raw, sf, sret)
    idx = disc.index
    return {"ret": T.ret.reindex(idx), "sret": sret.reindex(idx), "raw": raw.reindex(idx), "sf": sf.reindex(idx),
            "disc": disc}, disc_fit


# ---------------------------------------------------------------- AR prewhitening (Haugh 1976)

def fit_ar_bic(y, pmax=8):
    """Own AR(p) with constant, p in 0..pmax chosen by BIC on a common estimation sample. Returns (p, c, phi, resid)."""
    y = np.asarray(y, float)
    n = len(y) - pmax
    best = None
    target = y[pmax:]
    for p in range(pmax + 1):
        X = np.ones((n, 1 + p))
        for k in range(1, p + 1):
            X[:, k] = y[pmax - k:-k]
        beta, *_ = np.linalg.lstsq(X, target, rcond=None)
        e = target - X @ beta
        bic = n * np.log(e @ e / n) + (p + 1) * np.log(n)
        if best is None or bic < best[0]:
            best = (bic, p, beta[0], beta[1:], e)
    return best[1], best[2], best[3], best[4]


def prewhiten(frame, pmax=8):
    """Own-AR residuals of each column, same index (the first pmax weeks dropped). Returns (residuals, table)."""
    res, table = {}, []
    for m in frame.columns:
        p, c, phi, e = fit_ar_bic(frame[m].to_numpy(), pmax)
        res[m] = e
        table.append({"market": m, "p": p, "c": c, "phi": phi})
    return pd.DataFrame(res, index=frame.index[pmax:]), pd.DataFrame(table).set_index("market")


def sieve_draw(frame_values, table, rng, burn=60):
    """Own-AR recursions driven by jointly row-shuffled residuals: keeps autocorrelation and contemporaneous
    co-movement, destroys all cross-lag dynamics. `frame_values` holds the residuals (T x p)."""
    T, p = frame_values.shape
    draw = rng.integers(0, T, burn)
    e = np.r_[frame_values[draw], frame_values[rng.permutation(T)]]
    out = np.empty((T, p))
    for j in range(p):
        c, phi = table[j]
        a = np.r_[1.0, -phi] if len(phi) else np.array([1.0])
        out[:, j] = sps.lfilter([1.0], a, e[:, j] + c)[burn:]
    return out


# ---------------------------------------------------------------- lead-lag matrices

def ccf_pearson(Y, L):
    """C[l-1, i, j] = Pearson corr(Y^i_{t-l}, Y^j_t), l = 1..L."""
    T, p = Y.shape
    C = np.empty((L, p, p))
    for l in range(1, L + 1):
        X, Z = Y[:-l], Y[l:]
        X = X - X.mean(0)
        Z = Z - Z.mean(0)
        C[l - 1] = (X.T @ Z) / np.sqrt((X * X).sum(0)[:, None] * (Z * Z).sum(0)[None, :])
    return C


def signature(Y):
    """Levy-area lead-lag on the cumulated standardized series, per week: S_ij = sum_{u<v}(y^i_u y^j_v - y^j_u y^i_v)/T."""
    y = (Y - Y.mean(0)) / Y.std(0)
    X = np.zeros_like(y)
    X[1:] = np.cumsum(y, axis=0)[:-1]
    M = X.T @ y
    return (M - M.T) / len(y)


def from_ccf(C):
    """ccf-auc (paper), its centred variant, ccf-lag1 and the signed sum over all lags, from a CCF array."""
    I = np.abs(C).sum(0)
    np.fill_diagonal(I, 0.0)
    den = I + I.T
    den[den == 0] = 1.0
    diff = I - I.T
    C1 = C[0].copy()
    Cs = C.sum(0)
    out = {"auc": np.sign(diff) * np.maximum(I, I.T) / den, "auc_c": diff / den, "lag1": C1 - C1.T, "sum": Cs - Cs.T}
    return out


def all_metrics(Y, L=4):
    """Dict of skew-symmetric lead-lag matrices: auc, auc_c, lag1, sum, sig."""
    return from_ccf(ccf_pearson(Y, L)) | {"sig": signature(Y)}


def lam1(S):
    """Largest eigenvalue of the Hermitian matrix i(A - A^T) = iS."""
    return np.linalg.eigvalsh(1j * S)[-1]


def upper(S):
    return S[np.triu_indices(S.shape[0], 1)]


# ---------------------------------------------------------------- resampling of rows

def block_perm(T, b, rng):
    """Joint block permutation: consecutive blocks of b weeks (random offset) in random order; keeps within-block order."""
    offset = rng.integers(b)
    idx = np.roll(np.arange(T), -offset)
    blocks = [idx[i:i + b] for i in range(0, T, b)]
    return np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])


def local_perm(T, b, rng):
    """Weeks shuffled only inside consecutive blocks of b weeks: keeps the slow volatility path, kills short lags."""
    offset = int(rng.integers(b))
    edges = np.unique(np.r_[0, np.arange(offset, T, b), T])
    out = np.arange(T)
    for lo, hi in zip(edges[:-1], edges[1:]):
        out[lo:hi] = rng.permutation(np.arange(lo, hi))
    return out


# ---------------------------------------------------------------- Hermitian RW clustering

def hermitian_rw(S, k, restarts=64, seed=0):
    """Hermitian RW clustering of the directed graph A = max(S, 0). Labels: 0 = most leading cluster.

    Eigenvalues of the Hermitian matrix come in +/- pairs with conjugate eigenvectors, so the embedding uses
    ceil(k/2) complete pairs (real and imaginary parts): Euclidean distances there equal those between rows of
    the paper's projector P, and the result does not depend on an arbitrary pick inside a pair.
    Returns (labels, meta-flow F, cluster leadingness L, eigenvalue magnitudes).
    """
    A = np.maximum(S, 0.0)
    At = 1j * (A - A.T)
    d = np.abs(At).sum(1)
    di = np.where(d > 0, 1.0 / np.sqrt(np.where(d > 0, d, 1.0)), 0.0)
    H = di[:, None] * At * di[None, :]
    w, V = np.linalg.eigh(H)
    pos = np.argsort(-w)[:int(np.ceil(k / 2))]
    U = di[:, None] * V[:, pos]
    U = U / np.linalg.norm(U, axis=0)
    emb = np.c_[U.real, U.imag]
    best, best_cost = None, np.inf
    rng = np.random.default_rng(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(restarts):
            cent, lab = kmeans2(emb, k, minit="++", iter=50, seed=rng)
            if len(np.unique(lab)) < k:
                continue
            cost = ((emb - cent[lab]) ** 2).sum()
            if cost < best_cost - 1e-12:
                best, best_cost = lab, cost
    if best is None:
        best = np.arange(len(S)) % k
    lead = np.array([S[best == c].sum(1).mean() for c in range(k)])
    order = np.argsort(-lead)
    remap = np.empty(k, int)
    remap[order] = np.arange(k)
    labels = remap[best]
    return labels, meta_flow(S, labels, k), lead[order], np.sort(np.abs(w))[::-1]


def meta_flow(S, labels, k=None):
    k = k or labels.max() + 1
    F = np.zeros((k, k))
    for a in range(k):
        for b in range(k):
            if a != b:
                F[a, b] = S[np.ix_(labels == a, labels == b)].sum() / ((labels == a).sum() * (labels == b).sum())
    return F


def ari(a, b):
    """Adjusted Rand index of two labelings (Hubert and Arabie 1985)."""
    a, b = np.asarray(a), np.asarray(b)
    ua, ai = np.unique(a, return_inverse=True)
    ub, bi = np.unique(b, return_inverse=True)
    n = np.zeros((len(ua), len(ub)))
    np.add.at(n, (ai, bi), 1)
    comb = lambda x: x * (x - 1) / 2
    sum_ij = comb(n).sum()
    sum_a, sum_b = comb(n.sum(1)).sum(), comb(n.sum(0)).sum()
    total = comb(len(a))
    expected = sum_a * sum_b / total
    max_index = (sum_a + sum_b) / 2
    return 0.0 if max_index == expected else (sum_ij - expected) / (max_index - expected)


# ---------------------------------------------------------------- multiple testing, evaluation

def bh(p, level):
    """Benjamini-Hochberg rejections; returns a boolean array."""
    p = np.asarray(p)
    m = len(p)
    order = np.argsort(p)
    thresh = level * np.arange(1, m + 1) / m
    below = p[order] <= thresh
    keep = np.zeros(m, bool)
    if below.any():
        keep[order[:np.max(np.flatnonzero(below)) + 1]] = True
    return keep


def bh_qvalues(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def r2_unc(y, f):
    """Uncentred R2 in percent: 100 * (1 - SSE / sum y^2)."""
    y, f = np.asarray(y, float), np.asarray(f, float)
    return 100 * (1 - np.sum((y - f) ** 2) / np.sum(y ** 2))


def nw_t(x, lags=4):
    """t statistic of the mean of x with Newey-West variance."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    u = x - x.mean()
    v = u @ u / n
    for l in range(1, lags + 1):
        v += 2 * (1 - l / (lags + 1)) * (u[l:] @ u[:-l]) / n
    return x.mean() / np.sqrt(v / n)


def clark_west(y, f_base, f_model, lags=4):
    """Clark-West (2007) MSPE-adjusted statistic; one-sided p for: the model adds to the nested base model.

    Inputs may be 2-d (weeks x markets): the loss differential is summed across markets each week, so the
    t statistic respects cross-sectional correlation. NaNs must be aligned already.
    """
    from scipy.stats import norm
    y, fb, fm = np.asarray(y, float), np.asarray(f_base, float), np.asarray(f_model, float)
    adj = (y - fb) ** 2 - ((y - fm) ** 2 - (fb - fm) ** 2)
    if adj.ndim == 2:
        adj = np.nansum(adj, axis=1)
    t = nw_t(adj, lags)
    return t, norm.sf(t)


# ---------------------------------------------------------------- permutation tests and the network

def permutation_draws(Y, draws, null="iid", seed=7, L=4, residuals=None, ar_table=None):
    """Lead-lag matrices under a null, as a dict of arrays (draws x p x p), one per metric of `from_ccf`.

    iid: weeks shuffled jointly. block4: blocks of 4 weeks in random order. local13: weeks shuffled within 13-week blocks.
    sieve: own-AR recursions driven by jointly shuffled residuals (`residuals`, `ar_table`), which keeps each market's
    autocorrelation and the co-movement across markets and destroys only the cross-lag dynamics.
    """
    rng = np.random.default_rng(seed)
    T = len(Y)
    out = {key: np.empty((draws, Y.shape[1], Y.shape[1]), np.float32) for key in ("auc_c", "lag1", "sum")}
    for b in range(draws):
        if null == "sieve":
            sample = sieve_draw(residuals, ar_table, rng)
        else:
            order = {"iid": lambda: rng.permutation(T), "block4": lambda: block_perm(T, 4, rng),
                     "local13": lambda: local_perm(T, 13, rng)}[null]()
            sample = Y[order]
        metrics = from_ccf(ccf_pearson(sample, L))
        for key in out:
            out[key][b] = metrics[key]
    return out


def global_p(S, draws):
    """Permutation p-value of the largest eigenvalue of iS, the global test of Bennett et al. (2022)."""
    observed = lam1(S)
    return (1 + sum(lam1(d.astype(np.float64)) >= observed for d in draws)) / (len(draws) + 1)


def pair_p(S, draws):
    """Two-sided permutation p-value of each pair, on the upper triangle."""
    i, j = np.triu_indices(S.shape[0], 1)
    return (1 + (np.abs(draws[:, i, j]) >= np.abs(S[i, j])).sum(0)) / (len(draws) + 1)


def leadlag_graph(S, markets, p=None, level=0.10, threshold=None):
    """Directed network: an edge i -> j when i leads j.

    Keeps the pairs surviving Benjamini-Hochberg at `level` if `p` is given, or those with |S| above `threshold`.
    Nodes carry their sector and leadingness (row mean of S); edges carry |S| and, with `p`, the p- and q-values.
    """
    G = nx.DiGraph()
    for k, market in enumerate(markets):
        G.add_node(market, sector=SECTOR[market], leadingness=S[k].sum() / (len(markets) - 1))
    i, j = np.triu_indices(len(markets), 1)
    values = S[i, j]
    keep = bh(p, level) if p is not None else np.abs(values) > threshold
    q = bh_qvalues(p) if p is not None else np.full(len(values), np.nan)
    for n in np.flatnonzero(keep):
        leader, lagger = (i[n], j[n]) if values[n] > 0 else (j[n], i[n])
        G.add_edge(markets[leader], markets[lagger], weight=abs(values[n]), p=np.nan if p is None else p[n], q=q[n])
    return G


def edge_table(G):
    rows = [{"Leader": a, "Lagger": b, "|S|": d["weight"], "p": d["p"], "BH q": d["q"]} for a, b, d in G.edges(data=True)]
    return pd.DataFrame(rows, columns=["Leader", "Lagger", "|S|", "p", "BH q"]).sort_values("|S|", ascending=False)


def sup_break(x, trim=5):
    """Largest standardized difference in means over one break at an unknown date: statistic and position."""
    n, s, best, where = len(x), x.std(ddof=1), 0.0, None
    for k in range(trim, n - trim + 1):
        v = (x[k:].mean() - x[:k].mean()) * np.sqrt(k * (n - k) / n) / s
        if abs(v) > abs(best):
            best, where = v, k
    return best, where


# ---------------------------------------------------------------- past-only networks and forecasts

def top_k_graph(S, k):
    """Leader weights for each target j: its top k positive S_ij, summing to one. Returns weights and the mask."""
    A = np.maximum(S, 0.0)
    mask = np.zeros_like(A)
    for j in range(A.shape[1]):
        order = np.argsort(-A[:, j])[:k]
        mask[order[A[order, j] > 0], j] = 1.0
    weights = A * mask
    total = weights.sum(0, keepdims=True)
    return weights / np.where(total > 0, total, 1.0), mask


def past_only_weights(y, step=13, window=260, minimum=104, k=3, cluster=False):
    """For each week t, weights from rows up to t only, refreshed every `step` weeks, on own-AR(1) innovations.

    Returns weights (T x p x p), AR coefficients (T x p) and, with `cluster`, the labels of a two-cluster Hermitian
    clustering (T x p; 0 = the leading cluster).
    """
    T, p = y.shape
    first = np.flatnonzero(np.isfinite(y).all(1))[0]
    W, phi_path, labels = np.full((T, p, p), np.nan), np.full((T, p), np.nan), np.full((T, p), np.nan)
    for t in range(first + minimum - 1, T):
        if (t - first - minimum + 1) % step == 0:
            Y = y[max(first, t - window + 1):t + 1]
            phi = (Y[:-1] * Y[1:]).sum(0) / (Y[:-1] ** 2).sum(0)
            S = from_ccf(ccf_pearson(Y[1:] - phi * Y[:-1], 1))["lag1"]
            w = top_k_graph(S, k)[0]
            lab = hermitian_rw(S, 2, restarts=8)[0] if cluster else None
        W[t], phi_path[t] = w, phi
        if cluster:
            labels[t] = lab
    return W, phi_path, labels


def innovations(y, phi):
    e = np.full_like(y, np.nan)
    e[1:] = y[1:] - phi[1:] * y[:-1]
    return e


def message(y, W, phi, permutation=None):
    """n_{j,t} = sum_i w_ij e_{i,t}, known at the end of week t. Optionally on a graph with shuffled node labels."""
    if permutation is not None:
        W = W[:, permutation][:, :, permutation]
    return np.einsum("ti,tij->tj", innovations(y, phi), W)


def cluster_feature(y, labels, alpha=0.4, lags=5):
    """The paper's signal: for markets of the lagging cluster, an EWMA of the leading cluster's mean flow; zero otherwise."""
    weight = alpha * (1 - alpha) ** np.arange(lags)
    out = np.full_like(y, np.nan)
    for t in range(lags - 1, len(y)):
        if np.isfinite(labels[t]).all():
            lead = labels[t] == 0
            out[t] = np.where(lead, 0.0, sum(weight[l] * y[t - l, lead].mean() for l in range(lags)))
    return out


def pooled_ridge(y, features, train=104, penalty=0.1):
    """Weekly out-of-sample pooled ridge, as `sg_cta.rolling_ridge`, on arrays.

    y: T x p. features: list of T x p arrays, each row already known before that row's y. No intercept; features are
    standardized on the training window. Returns the forecasts, T x p.
    """
    X = np.stack(features, axis=2)
    k = X.shape[2]
    out = np.full(y.shape, np.nan)
    for t in range(train, len(y)):
        Xh, yh = X[t - train:t].reshape(-1, k), y[t - train:t].ravel()
        ok = np.isfinite(yh) & np.isfinite(Xh).all(1)
        if ok.sum() < 50:
            continue
        Xh, yh = Xh[ok], yh[ok]
        mean, std = Xh.mean(0), Xh.std(0, ddof=1)
        std[std == 0] = 1.0
        Z = (Xh - mean) / std
        beta = np.linalg.solve(Z.T @ Z + len(Z) * penalty * np.eye(k), Z.T @ yh)
        out[t] = np.nan_to_num((X[t] - mean) / std) @ beta
    return out
