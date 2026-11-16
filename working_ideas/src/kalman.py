"""Linear-Gaussian state-space tools for `kalman_filter.ipynb`.

Convention (Durbin-Koopman, same as statsmodels): alpha_0 ~ N(a1, P1) is the prediction for the first observation;
alpha_{t+1} = T alpha_t + eta_t, eta ~ N(0, Q); y_t = Z_t alpha_t + e_t, e ~ N(0, R_t).

kf / smooth: general multivariate filter and RTS smoother (time-varying Z and R, element-wise missing values).
uni_*: scalar random-coefficient model y_t = x_t beta_t + e_t in R units, with R concentrated out.
grid_ll: profile log-likelihood over a q grid and over many y columns at once.
bi_*: bivariate model in pure Python scalars, about 30 times faster than the general filter.

Diffuse treatment: weak prior on beta, and the first informative observation is dropped from the log-likelihood.
That equals the exact diffuse log-likelihood up to a constant that does not depend on (R, Q).
"""
import math
import numpy as np
import pandas as pd
from scipy import optimize, stats

LOG2PI = math.log(2 * math.pi)
KAPPA = 1e6


# ---------------------------------------------------------------------------------- general filter
def kf(y, Z, R, Q, a1, P1, Tm=None):
    """y: n x p with NaN for missing elements. Z: p x m or n x p x m. R: p x p or n x p x p. Q: m x m or n x m x m."""
    y = np.asarray(y, float)
    if y.ndim == 1:
        y = y[:, None]
    n, p = y.shape
    a1 = np.asarray(a1, float).ravel()
    m = a1.size
    Z = np.asarray(Z, float)
    Z = np.broadcast_to(Z, (n,) + Z.shape) if Z.ndim == 2 else Z
    R = np.asarray(R, float)
    R = np.broadcast_to(R, (n,) + R.shape) if R.ndim == 2 else R
    Q = np.asarray(Q, float)
    Q = np.broadcast_to(Q, (n,) + Q.shape) if Q.ndim == 2 else Q
    Tm = np.eye(m) if Tm is None else np.asarray(Tm, float)

    a_pred, P_pred = np.empty((n, m)), np.empty((n, m, m))
    a_filt, P_filt = np.empty((n, m)), np.empty((n, m, m))
    v, Fd = np.full((n, p), np.nan), np.full((n, p), np.nan)
    Fs = [None] * n
    ll = np.zeros(n)
    nobs = np.zeros(n, int)
    a, P = a1.copy(), np.asarray(P1, float).copy()
    for t in range(n):
        a_pred[t], P_pred[t] = a, P
        obs = np.isfinite(y[t])
        if obs.any():
            Zt = Z[t][obs]
            Rt = R[t][np.ix_(obs, obs)]
            PZ = P @ Zt.T
            F = Zt @ PZ + Rt
            vt = y[t][obs] - Zt @ a
            L = np.linalg.cholesky(F)
            w = np.linalg.solve(L, vt)
            Kt = np.linalg.solve(L.T, np.linalg.solve(L, PZ.T)).T
            a = a + Kt @ vt
            P = P - Kt @ PZ.T
            P = 0.5 * (P + P.T)
            ll[t] = -0.5 * (obs.sum() * LOG2PI + 2 * np.log(np.diag(L)).sum() + w @ w)
            nobs[t] = obs.sum()
            v[t, obs] = vt
            Fd[t, obs] = np.diag(F)
            Fs[t] = F
        a_filt[t], P_filt[t] = a, P
        a = Tm @ a
        P = Tm @ P @ Tm.T + Q[t]
    return dict(a_pred=a_pred, P_pred=P_pred, a_filt=a_filt, P_filt=P_filt, v=v, Fd=Fd, F=Fs, ll=ll, nobs=nobs,
                Tm=Tm, a_next=a, P_next=P, y=y, Z=Z, R=R, Q=Q)


def smooth(f):
    """RTS smoother. Returns a_s, P_s and J with Cov(alpha_t, alpha_{t+1} | Y) = J[t] P_s[t+1]."""
    a_f, P_f, a_p, P_p, Tm = f['a_filt'], f['P_filt'], f['a_pred'], f['P_pred'], f['Tm']
    n, m = a_f.shape
    a_s, P_s, J = a_f.copy(), P_f.copy(), np.zeros((n, m, m))
    for t in range(n - 2, -1, -1):
        J[t] = np.linalg.solve(P_p[t + 1].T, (P_f[t] @ Tm.T).T).T
        a_s[t] = a_f[t] + J[t] @ (a_s[t + 1] - a_p[t + 1])
        P_s[t] = P_f[t] + J[t] @ (P_s[t + 1] - P_p[t + 1]) @ J[t].T
    return dict(a_s=a_s, P_s=P_s, J=J)


def joint_cov(P_s, J, s, t):
    """Cov(alpha_s, alpha_t | Y_n), s <= t, from the smoother output."""
    C = P_s[t]
    for k in range(t - 1, s - 1, -1):
        C = J[k] @ C
    return C


# ---------------------------------------------------------------------------------- scalar model, R concentrated
def clean_xy(y, x):
    y, x = np.asarray(y, float).copy(), np.asarray(x, float).copy()
    bad = ~np.isfinite(x)
    y[bad] = np.nan
    x[bad] = 0.0
    return y, x


def first_informative(y, x, kappa=KAPPA):
    ok = np.isfinite(y) & (x * x * kappa > 1e3)
    return int(np.argmax(ok)) if ok.any() else -1


def uni_ll(y, x, q, s=None, kappa=KAPPA):
    """Concentrated diffuse log-likelihood of y_t = x_t beta_t + e_t, e ~ N(0, R s_t), Q = q R. Returns (ll, R-hat)."""
    d = first_informative(y, x, kappa)
    a, P, S, logF, n_eff = 0.0, kappa, 0.0, 0.0, 0
    yl, xl = y.tolist(), x.tolist()
    sl = [1.0] * len(yl) if s is None else list(s)
    for t in range(len(yl)):
        yt, xt = yl[t], xl[t]
        if yt == yt:
            st = sl[t]
            F = xt * xt * P + st
            v = yt - xt * a
            Kk = P * xt / F
            a += Kk * v
            P = P * st / F
            if t != d:
                S += v * v / F
                logF += math.log(F)
                n_eff += 1
        P += q
    sig2 = S / n_eff
    return -0.5 * (n_eff * (LOG2PI + 1 + math.log(sig2)) + logF), sig2


def uni_filter(y, x, q, s=None, kappa=KAPPA):
    """Scalar filter in R units. Arrays: a_p, P_p (prediction for t), a_f, P_f (after t), v, F. Multiply P/F by R-hat."""
    n = len(y)
    a_p, P_p, a_f, P_f, v_, F_ = (np.zeros(n) for _ in range(6))
    a, P = 0.0, kappa
    yl, xl = y.tolist(), x.tolist()
    sl = [1.0] * n if s is None else list(s)
    for t in range(n):
        a_p[t], P_p[t] = a, P
        yt, xt = yl[t], xl[t]
        if yt == yt:
            st = sl[t]
            F = xt * xt * P + st
            v = yt - xt * a
            Kk = P * xt / F
            a += Kk * v
            P = P * st / F
            v_[t], F_[t] = v, F
        else:
            v_[t], F_[t] = np.nan, np.nan
        a_f[t], P_f[t] = a, P
        P += q
    return dict(a_p=a_p, P_p=P_p, a_f=a_f, P_f=P_f, v=v_, F=F_)


def uni_smooth(f, q):
    """Scalar RTS smoother in R units; J[t] = P_f[t]/P_p[t+1], Cov(beta_t, beta_{t+1}|Y) = J[t] P_s[t+1]."""
    a_f, P_f, a_p, P_p = f['a_f'], f['P_f'], f['a_p'], f['P_p']
    n = len(a_f)
    a_s, P_s, J = a_f.copy(), P_f.copy(), np.zeros(n)
    for t in range(n - 2, -1, -1):
        J[t] = P_f[t] / P_p[t + 1]
        a_s[t] = a_f[t] + J[t] * (a_s[t + 1] - a_p[t + 1])
        P_s[t] = P_f[t] + J[t] ** 2 * (P_s[t + 1] - P_p[t + 1])
    return dict(a_s=a_s, P_s=P_s, J=J)


def qn_grid(lo=-8.0, hi=1.5, step=0.1):
    """q_n = Q mean(x^2) / R is the dimensionless signal-to-noise; grid is log10-spaced, with 0 as first element."""
    return np.r_[0.0, 10.0 ** np.arange(lo, hi + 1e-9, step)]


def grid_ll(Y, x, q, s=None, kappa=KAPPA):
    """Profile log-likelihood for many y columns and a grid of q at once.

    Y: n x B (same missing pattern in all columns). x: n. q: G. s: n or None.
    Returns ll (B x G), sig2 (B x G). Gains depend on x and q only, so this is a single loop over time.
    """
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    n, B = Y.shape
    G = len(q)
    obs = np.isfinite(Y[:, 0])
    d = first_informative(Y[:, 0], x, kappa)
    s_ = np.ones(n) if s is None else np.asarray(s, float)
    a = np.zeros((B, G))
    P = np.full(G, kappa, float)
    S = np.zeros((B, G))
    logF = np.zeros(G)
    n_eff = 0
    for t in range(n):
        if obs[t]:
            xt, st = x[t], s_[t]
            F = xt * xt * P + st
            Kk = P * xt / F
            v = Y[t][:, None] - xt * a
            a = a + v * Kk
            if t != d:
                S += v * v / F
                logF += np.log(F)
                n_eff += 1
            P = P * st / F
        P = P + q
    sig2 = S / n_eff
    return -0.5 * (n_eff * (LOG2PI + 1 + np.log(sig2)) + logF), sig2


def uni_mle(y, x, s=None, grid=None):
    """MLE of q by grid then bounded Brent on log10 q_n. q_hat = 0 when the likelihood is flat or falls off zero."""
    y, x = clean_xy(y, x)
    obs = np.isfinite(y)
    scale = float(np.mean(x[obs] ** 2))
    grid = qn_grid() if grid is None else grid
    q = grid / scale
    ll, sig2 = grid_ll(y, x, q, s)
    ll, sig2 = ll[0], sig2[0]
    i = int(np.argmax(ll))
    ll0 = float(ll[0])
    out = dict(ll0=ll0, R0=float(sig2[0]), scale=scale, n=int(obs.sum()), grid=grid, profile=ll.copy())
    if i == 0 or ll[i] - ll0 < 1e-6:
        out.update(q=0.0, qn=0.0, R=float(sig2[0]), ll=ll0, LR=0.0, boundary=True, Q=0.0)
    else:
        lo, hi = math.log10(grid[max(i - 1, 1)]), math.log10(grid[min(i + 1, len(grid) - 1)])
        f = lambda lqn: -uni_ll(y, x, 10 ** lqn / scale, s)[0]
        r = optimize.minimize_scalar(f, bounds=(lo, hi), method='bounded', options=dict(xatol=1e-4))
        qn = 10 ** r.x
        ll_, R = uni_ll(y, x, qn / scale, s)
        if ll_ < ll[i]:
            qn, ll_, R = grid[i], float(ll[i]), float(sig2[i])
        out.update(q=qn / scale, qn=qn, R=R, ll=ll_, LR=2 * (ll_ - ll0), boundary=False, Q=R * qn / scale)
    # 95% profile interval for q_n (LR inverted with the chi2(1) cut-off, conservative at the boundary)
    keep = 2 * (out['ll'] - ll) <= 3.8415
    gpos = grid[keep]
    out['qn_ci'] = (float(gpos.min()), float(gpos.max()))
    return out


def uni_fit(y, x, q=None, s=None):
    """Convenience: MLE (unless q given), filter, smoother, all in natural units."""
    y, x = clean_xy(y, x)
    m = uni_mle(y, x, s) if q is None else None
    q_ = m['q'] if q is None else q
    R = m['R'] if q is None else uni_ll(y, x, q_, s)[1]
    f = uni_filter(y, x, q_, s)
    sm = uni_smooth(f, q_)
    return dict(mle=m, q=q_, R=R, Q=q_ * R, a_f=f['a_f'], P_f=f['P_f'] * R, a_p=f['a_p'], P_p=f['P_p'] * R,
                a_s=sm['a_s'], P_s=sm['P_s'] * R, J=sm['J'], v=f['v'], F=f['F'] * R, y=y, x=x)


# ---------------------------------------------------------------------------------- stability statistics
def nyblom(y, x, B=0, rng=None, block=1):
    """Nyblom (1989) / Hansen (1992) L for constancy of beta in y = x beta + e (no intercept).

    L = sum_t S_t^2 / (n V), S_t = partial sums of x_t e_t. L_h: V = sigma^2 sum x^2. L_r: V = sum (x e)^2 (robust).
    With B > 0: fixed-regressor wild bootstrap (Hansen 2000) of both, with Rademacher weights, optionally in blocks.
    """
    y, x = clean_xy(y, x)
    obs = np.isfinite(y)
    y_, x_ = y[obs], x[obs]
    n = len(y_)
    b = x_ @ y_ / (x_ @ x_)
    e = y_ - b * x_
    f = x_ * e
    S = np.cumsum(f)
    Lh = (S @ S) / (n * (e @ e / n) * (x_ @ x_))
    Lr = (S @ S) / (n * (f @ f))
    out = dict(b=b, Lh=Lh, Lr=Lr, n=n, e=e, sig2=e @ e / n)
    if B:
        rng = np.random.default_rng(7) if rng is None else rng
        if block == 1:
            w = rng.choice([-1.0, 1.0], size=(B, n))
        else:
            k = n // block + 1
            w = np.repeat(rng.choice([-1.0, 1.0], size=(B, k)), block, axis=1)[:, :n]
        ys = e[None, :] * w
        bs = (ys @ x_) / (x_ @ x_)
        es = ys - bs[:, None] * x_[None, :]
        fs = es * x_[None, :]
        Ss = np.cumsum(fs, axis=1)
        SS = (Ss * Ss).sum(1)
        Lh_b = SS / (n * ((es * es).sum(1) / n) * (x_ @ x_))
        Lr_b = SS / (n * (fs * fs).sum(1))
        out.update(p_h=(1 + (Lh_b >= Lh).sum()) / (B + 1), p_r=(1 + (Lr_b >= Lr).sum()) / (B + 1),
                   q95_h=float(np.quantile(Lh_b, .95)), q95_r=float(np.quantile(Lr_b, .95)))
    return out


# ---------------------------------------------------------------------------------- bivariate, diagonal Z, pure Python
def bi_pack(R, Q):
    """Unconstrained 6-vector <- (R, Q): log variances and atanh correlations."""
    return np.array([math.log(R[0, 0]), math.log(R[1, 1]), math.atanh(R[0, 1] / math.sqrt(R[0, 0] * R[1, 1])),
                     math.log(Q[0, 0]), math.log(Q[1, 1]), math.atanh(Q[0, 1] / math.sqrt(Q[0, 0] * Q[1, 1]))])


def bi_unpack(th):
    r1, r2 = math.exp(th[0]), math.exp(th[1])
    rho_r = math.tanh(th[2])
    q1, q2 = math.exp(th[3]), math.exp(th[4])
    rho_q = math.tanh(th[5])
    return r1, r2, rho_r * math.sqrt(r1 * r2), q1, q2, rho_q * math.sqrt(q1 * q2)


def bi_ll(y1, y2, x1, x2, r11, r22, r12, q11, q22, q12, s1=None, s2=None, kappa_a=1e4):
    """Bivariate log-likelihood; states (beta_1, beta_2), Z_t = diag(x1_t, x2_t). Weak prior variance kappa_a.

    The first time-step at which each state gets informative is dropped from the likelihood, so the prior does not
    enter. s1, s2 scale the observation noise: R_t = D_t R D_t, D_t = diag(sqrt s1_t, sqrt s2_t).
    """
    a1 = a2 = 0.0
    p11 = p22 = kappa_a
    p12 = 0.0
    ll = 0.0
    seen1 = seen2 = False
    n = len(y1)
    for t in range(n):
        yy1, yy2, xx1, xx2 = y1[t], y2[t], x1[t], x2[t]
        o1, o2 = yy1 == yy1, yy2 == yy2
        d1 = 1.0 if s1 is None else s1[t]
        d2 = 1.0 if s2 is None else s2[t]
        if o1 and o2:
            r11t, r22t, r12t = r11 * d1, r22 * d2, r12 * math.sqrt(d1 * d2)
            pz11, pz12 = p11 * xx1, p12 * xx2         # P Z'
            pz21, pz22 = p12 * xx1, p22 * xx2
            f11 = xx1 * pz11 + r11t
            f12 = xx1 * pz12 + r12t
            f22 = xx2 * pz22 + r22t
            det = f11 * f22 - f12 * f12
            v1, v2 = yy1 - xx1 * a1, yy2 - xx2 * a2
            i11, i12, i22 = f22 / det, -f12 / det, f11 / det
            w1, w2 = i11 * v1 + i12 * v2, i12 * v1 + i22 * v2
            a1 += pz11 * w1 + pz12 * w2
            a2 += pz21 * w1 + pz22 * w2
            k11, k12 = pz11 * i11 + pz12 * i12, pz11 * i12 + pz12 * i22
            k21, k22 = pz21 * i11 + pz22 * i12, pz21 * i12 + pz22 * i22
            p11_new = p11 - (k11 * pz11 + k12 * pz12)
            p12_new = p12 - (k11 * pz21 + k12 * pz22)
            p22_new = p22 - (k21 * pz21 + k22 * pz22)
            p11, p12, p22 = p11_new, p12_new, p22_new
            inf1, inf2 = abs(xx1) > 1e-2, abs(xx2) > 1e-2
            if (inf1 and not seen1) or (inf2 and not seen2):
                seen1, seen2 = seen1 or inf1, seen2 or inf2
            else:
                ll += -LOG2PI - 0.5 * (math.log(det) + v1 * w1 + v2 * w2)
        elif o1 or o2:
            yy, xx, rr = (yy1, xx1, r11 * d1) if o1 else (yy2, xx2, r22 * d2)
            pz1, pz2 = (p11 * xx, p12 * xx) if o1 else (p12 * xx, p22 * xx)
            f = xx * (pz1 if o1 else pz2) + rr
            v = yy - xx * (a1 if o1 else a2)
            a1 += pz1 * v / f
            a2 += pz2 * v / f
            p11, p12, p22 = p11 - pz1 * pz1 / f, p12 - pz1 * pz2 / f, p22 - pz2 * pz2 / f
            inf = abs(xx) > 1e-2
            already = seen1 if o1 else seen2
            if inf and not already:
                if o1:
                    seen1 = True
                else:
                    seen2 = True
            else:
                ll += -0.5 * (LOG2PI + math.log(f) + v * v / f)
        p11 += q11
        p12 += q12
        p22 += q22
    return ll


def bi_mle(y1, y2, x1, x2, starts, s1=None, s2=None, maxiter=4000, verbose=False):
    """Multi-start Nelder-Mead + BFGS polish on the unconstrained 6-vector. Returns dict with theta, ll, R, Q, Hessian-based se."""
    y1l, y2l, x1l, x2l = [np.asarray(v, float).tolist() for v in (y1, y2, x1, x2)]
    s1l = None if s1 is None else list(s1)
    s2l = None if s2 is None else list(s2)

    def nll(th):
        # keep the search away from numerically absurd regions
        if max(abs(th[2]), abs(th[5])) > 6 or min(th[3], th[4]) < -40 or max(th[0], th[1]) > 25:
            return 1e12
        r1, r2, r12, q1, q2, q12 = bi_unpack(th)
        try:
            return -bi_ll(y1l, y2l, x1l, x2l, r1, r2, r12, q1, q2, q12, s1l, s2l)
        except (ValueError, ZeroDivisionError):
            return 1e12

    best = None
    for th0 in starts:
        r = optimize.minimize(nll, th0, method='Nelder-Mead',
                              options=dict(maxiter=maxiter, maxfev=maxiter, xatol=1e-3, fatol=1e-4, adaptive=True))
        r2_ = optimize.minimize(nll, r.x, method='BFGS', options=dict(maxiter=200, gtol=1e-3))
        cand = r2_ if r2_.fun <= r.fun else r
        if verbose:
            print('   start', np.round(th0, 2), '->', round(-cand.fun, 3), np.round(cand.x, 2), flush=True)
        if best is None or cand.fun < best.fun:
            best = cand
    th = best.x
    r1, r2, r12, q1, q2, q12 = bi_unpack(th)
    return dict(theta=th, ll=-best.fun, R=np.array([[r1, r12], [r12, r2]]), Q=np.array([[q1, q12], [q12, q2]]),
                rho_R=math.tanh(th[2]), rho_Q=math.tanh(th[5]), nll=nll)


def bi_run(y1, y2, x1, x2, R, Q, Rt=None, kappa_a=1e4):
    """General filter + smoother for the bivariate model at given (R, Q). Rt: optional n x 2 x 2 path."""
    y = np.column_stack([y1, y2])
    x1, x2 = np.asarray(x1, float), np.asarray(x2, float)
    n = len(y)
    Z = np.zeros((n, 2, 2))
    Z[:, 0, 0], Z[:, 1, 1] = x1, x2
    f = kf(y, Z, R if Rt is None else Rt, Q, np.zeros(2), np.eye(2) * kappa_a)
    s = smooth(f)
    return f, s


def tilt(a, P):
    """P(alpha_0 > alpha_1), mean and sd of the spread, from a state mean a and covariance P."""
    d = a[0] - a[1]
    sd = math.sqrt(max(P[0, 0] + P[1, 1] - 2 * P[0, 1], 1e-300))
    return float(stats.norm.cdf(d / sd)), float(d), float(sd)


# ---------------------------------------------------------------------------------- tests and evaluation
def wild_lr(y, x, B=999, seed=7):
    """LR of constant beta against a random walk, with a fixed-regressor wild bootstrap of the boundary test.

    Likelihoods are maximized on the `qn_grid` for the data and the draws alike. Returns (LR, p-value, 95% quantile).
    """
    y, x = clean_xy(y, x)
    obs = np.isfinite(y)
    q = qn_grid() / float(np.mean(x[obs] ** 2))
    ll, _ = grid_ll(y, x, q)
    stat = 2 * (ll[0].max() - ll[0, 0])
    b = x[obs] @ y[obs] / (x[obs] @ x[obs])
    weights = np.random.default_rng(seed).choice([-1.0, 1.0], size=(len(y), B))
    draws, _ = grid_ll(b * x[:, None] + (y - b * x)[:, None] * weights, x, q)
    boot = 2 * (draws.max(1) - draws[:, 0])
    return float(stat), float((1 + (boot >= stat).sum()) / (B + 1)), float(np.quantile(boot, .95))


def end_of_sample(Y, x, q, kappa=KAPPA):
    """Filtered mean (one per column of Y) and variance in R units at the last week, for fixed q. Gains depend on x only."""
    obs = np.isfinite(Y[:, 0])
    a, P = np.zeros(Y.shape[1]), kappa
    for t in range(len(x)):
        if obs[t]:
            F = x[t] * x[t] * P + 1
            a = a + (P * x[t] / F) * (Y[t] - x[t] * a)
            P = P / F
        P += q
    return a, P - q


def stays_above(path, threshold):
    """First week from which `path` (a Series) stays above `threshold` through its last week, or None."""
    below = path <= threshold
    if not below.any():
        return path.index[0]
    last = below[below].index[-1]
    return None if last == path.index[-1] else path.index[path.index.get_loc(last) + 1]


def ewma_scale(errors, decay=0.94):
    """Lagged EWMA of squared errors with mean one, to scale the observation noise: R_t = R * s_t."""
    s = (pd.Series(errors) ** 2).ewm(alpha=1 - decay, ignore_na=True).mean().shift(1)
    s = s.fillna(s.median())
    return (s / s.mean()).to_numpy()


def past_slope_prediction(y, x, window=None):
    """b_{t-1} * x_t, with b from OLS through the origin on the previous weeks (all of them, or the last `window`)."""
    y, x = clean_xy(y, x)
    ok = np.isfinite(y)
    sum_xy, sum_xx = np.r_[0.0, np.cumsum(np.where(ok, x * y, 0.0))], np.r_[0.0, np.cumsum(np.where(ok, x * x, 0.0))]
    t = np.arange(len(y))
    low = np.zeros(len(y), int) if window is None else np.maximum(t - window, 0)
    sxy, sxx = sum_xy[t] - sum_xy[low], sum_xx[t] - sum_xx[low]
    return np.where(sxx > 1e-9, sxy / np.where(sxx > 1e-9, sxx, 1.0), 0.0) * x


def realtime_prediction(y, x, dates, first_year):
    """beta_{t|t-1} * x_t, with q re-estimated each January on the data before that January only."""
    y, x = clean_xy(y, x)
    prediction = np.full(len(y), np.nan)
    for year in range(first_year, dates[-1].year + 1):
        cut = dates.searchsorted(pd.Timestamp(year, 1, 1))
        if np.isfinite(y[:cut]).sum() >= 52:
            f = uni_filter(y, x, uni_mle(y[:cut], x[:cut])["q"])
            rows = dates.year == year
            prediction[rows] = (f["a_p"] * x)[rows]
    return prediction


def r2_gain(frame, base, model, block=13, draws=4000, seed=7):
    """R2 gain (percentage points) of `model` over `base` with a circular block bootstrap of weeks: estimate and 95% interval."""
    weekly = (frame.assign(e0=(frame["actual_change"] - frame[base]) ** 2, e1=(frame["actual_change"] - frame[model]) ** 2,
                           y2=frame["actual_change"] ** 2)
              .groupby("report_week_tuesday")[["e0", "e1", "y2"]].sum().to_numpy())
    n, k = len(weekly), len(weekly) // block + 1
    starts = np.random.default_rng(seed).integers(0, n, (draws, k))
    rows = ((starts[:, :, None] + np.arange(block)) % n).reshape(draws, -1)[:, :n]
    total = weekly[rows].sum(1)
    boot = 100 * (total[:, 0] - total[:, 1]) / total[:, 2]
    return 100 * (weekly[:, 0].sum() - weekly[:, 1].sum()) / weekly[:, 2].sum(), *np.percentile(boot, [2.5, 97.5])
