"""The Insight 177 network and its extended version, refitted every week.

With every addition off this is paper_replication (SharedMomentumNetwork, PaperAdam, run_replication): same start
values, same loss, same warm start, same nowcast sigma * (T_t - T_{t-1}).
"""
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
PAPER = dict(curve=False, recency=0, change_loss=False, delay_days=0, lambda_w=0.04,
             lambda_bias=0.01, train_weeks=104, first_epochs=1024, rolling_epochs=36, lr=0.01)
FINAL = dict(curve=True, recency=26, change_loss=True, delay_days=2, lambda_w=0.02)
_DATA = None


def _init(repo):   # each worker builds its own copy of the inputs
    global _DATA
    import inputs
    _DATA = inputs.build(repo)


class Network(torch.nn.Module):
    def __init__(self, n_inputs, train_weeks, n_markets, delay_days, rep):
        super().__init__()
        self.shared_weights = torch.nn.Parameter(torch.zeros(3, n_inputs))
        if rep:   # replicate: the zero start is nudged by 1e-3 noise
            self.shared_weights.data = 1e-3 * torch.randn(3, n_inputs, generator=torch.Generator().manual_seed(int(rep)))
        self.market_weights = torch.nn.Parameter(torch.tensor([[0.4, 0.2, 0.1]] * n_markets, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros(n_markets, train_weeks))
        self.a = None if not delay_days else torch.sigmoid(torch.tensor(float(np.log((1 - 0.5 ** (1 / delay_days)) / 0.5 ** (1 / delay_days)))))

    def trend(self, x):
        """x: [weeks, markets, inputs], or [weeks, markets, lags, inputs] with the delay layer."""
        if x.dim() == 4:
            x = x.transpose(1, 2)
        u = torch.einsum("...mh,kh->...mk", x, self.shared_weights)
        t = torch.einsum("...mk,mk->...m", u * torch.exp((1.0 - u.square()) / 2.0), self.market_weights)   # R(u)
        if self.a is not None:   # execution delay: weights a (1 - a)^j on the last days
            w = self.a * (1 - self.a) ** torch.arange(t.shape[1], dtype=torch.float32)
            t = torch.einsum("wjm,j->wm", t, w / w.sum())
        return t


def _adam(params, state, lr, b1=0.9, b2=0.999, eps=1e-7):   # TensorFlow-style, as PaperAdam
    with torch.no_grad():
        for p in params:
            if p.grad is None:
                continue
            s = state.setdefault(id(p), {"t": 0, "m": torch.zeros_like(p), "v": torch.zeros_like(p)})
            s["t"] += 1
            s["m"].mul_(b1).add_(p.grad, alpha=1 - b1)
            s["v"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
            p.addcdiv_(s["m"], s["v"].sqrt().add_(eps), value=-lr * (1 - b2 ** s["t"]) ** 0.5 / (1 - b1 ** s["t"]))


def run(data, rep=0, start="2015-01-01", end="2026-12-31", **version):
    """One rolling pass over the evaluation weeks. Returns the nowcast of the weekly change, [weeks, markets]."""
    c = {**PAPER, **version}
    T, W = c["train_weeks"], len(data["weeks"])
    x = data["x"] if c["delay_days"] else data["x"][:, :, 0, :]
    if c["curve"]:
        x = np.concatenate([x, data["xc"] if c["delay_days"] else data["xc"][:, :, 0, :]], axis=-1)
    x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(data["y"], dtype=torch.float32)
    dy = data["change"] / data["sigma"]
    ok = torch.tensor(np.isfinite(dy) & data["consecutive"][:, None], dtype=torch.float32)
    dy = torch.tensor(np.nan_to_num(dy), dtype=torch.float32)
    w = torch.ones(T) if not c["recency"] else 0.5 ** ((T - 1 - torch.arange(T, dtype=torch.float32)) / c["recency"])
    w = w / w.mean()
    pred = np.full((W, y.shape[1]), np.nan)
    state = last = None
    for i in range(T, W):
        if not (np.datetime64(start) <= data["date"][i] <= np.datetime64(end)):
            continue
        win = slice(i - T, i)
        net = Network(x.shape[-1], T, y.shape[1], c["delay_days"], rep if state is None else 0)
        if state is not None:   # warm start; a new week's bias starts at the last fitted one
            k, b = i - last, state["bias"]
            state["bias"] = torch.cat([b[:, k:], b[:, -1:].expand(-1, min(k, T))], dim=1)[:, -T:].clone()
            net.load_state_dict(state)
        params, adam = list(net.parameters()), {}
        for _ in range(c["first_epochs"] if state is None else c["rolling_epochs"]):
            for p in params:
                p.grad = None
            t = net.trend(x[win])
            loss = c["lambda_w"] * torch.sum(torch.abs(net.shared_weights))
            if not c["change_loss"]:   # paper: level of the position, with the time-varying bias
                fit = t + net.bias.T
                mse = torch.mean((y[win] - fit) ** 2) if not c["recency"] else torch.mean(w[:, None] * (y[win] - fit) ** 2)
                loss = mse + loss + c["lambda_bias"] * torch.sum((net.bias[:, 1:] - net.bias[:, :-1]) ** 2)
            else:                      # weekly change of the trend output against the weekly change of the position
                e = (dy[win][1:] - (t[1:] - t[:-1])) ** 2 * ok[win][1:] * w[1:, None]
                loss = loss + e.sum() / ok[win][1:].sum()
            loss.backward()
            _adam(params, adam, c["lr"])
        state, last = {k: v.detach().clone() for k, v in net.state_dict().items()}, i
        with torch.no_grad():
            t = net.trend(x[i - 1:i + 1]).numpy()
        pred[i] = data["sigma"][i] * (t[1] - t[0])
    return pred


def _job(args):
    path, rep, version = args
    if not path.exists():
        np.savez_compressed(path, pred=run(_DATA, rep=rep, **version).astype(np.float32))
    return np.load(path)["pred"].astype(float)


def fit(data, version, reps=16, cache="runs", workers=2):
    """Replicates 0 .. reps-1 of one version, read from `cache` when already fitted."""
    global _DATA
    _DATA = data
    Path(cache).mkdir(exist_ok=True)
    key = hashlib.md5(json.dumps(version, sort_keys=True).encode()).hexdigest()[:10]
    jobs = [(Path(cache) / f"{key}_r{r}.npz", r, version) for r in range(reps)]
    if all(p.exists() for p, _, _ in jobs) or workers == 1:
        return [_job(j) for j in jobs]
    with Pool(workers, initializer=_init, initargs=(str(data["repo"]),)) as pool:
        return pool.map(_job, jobs)
