import random
import numpy as np
import torch
from torch import nn


def set_deterministic(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def reaction(signal):
    return signal * torch.exp((1.0 - signal.square()) / 2.0)


class SharedMomentumNetwork(nn.Module):
    def __init__(self, markets, horizons, train_weeks, seed=7):
        super().__init__()
        set_deterministic(seed)
        self.markets = list(markets)
        self.horizons = list(horizons)
        self.market_index = {name: i for i, name in enumerate(self.markets)}
        self.shared_weights = nn.Parameter(torch.zeros(3, len(self.horizons)))
        self.market_weights = nn.Parameter(torch.tensor([[0.4, 0.2, 0.1]] * len(self.markets), dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(len(self.markets), train_weeks))

    def forward(self, x, bias_indices=None):
        # x: [weeks, markets, horizons]
        latent_input = torch.einsum("wmh,kh->wmk", x, self.shared_weights)
        latent = reaction(latent_input)
        prediction = torch.einsum("wmk,mk->wm", latent, self.market_weights)
        if bias_indices is None and x.shape[0] == self.bias.shape[1]:
            prediction = prediction + self.bias.T
        elif bias_indices is None:
            prediction = prediction + self.bias[:, -1].expand(x.shape[0], -1)
        else:
            prediction = prediction + self.bias[:, bias_indices].T
        return prediction, latent

    def objective(self, x, y, bias_indices=None, lambda_w=0.04, lambda_bias=0.01):
        prediction, latent = self.forward(x, bias_indices)
        mse = torch.mean((y - prediction) ** 2)
        l1 = lambda_w * torch.sum(torch.abs(self.shared_weights))
        if self.bias.shape[1] > 1:
            bias_penalty = lambda_bias * torch.sum((self.bias[:, 1:] - self.bias[:, :-1]) ** 2)
        else:
            bias_penalty = torch.zeros((), dtype=mse.dtype)
        return mse + l1 + bias_penalty, prediction, latent


def align_bias_state(state, old_dates, new_dates):
    """Retain biases by date; initialize new dates with the last fitted bias."""
    aligned = {key: value.detach().clone() for key, value in state.items()}
    lookup = {date: i for i, date in enumerate(old_dates)}
    aligned["bias"] = torch.stack([
        state["bias"][:, lookup[date]] if date in lookup else state["bias"][:, -1]
        for date in new_dates
    ], dim=1).clone()
    return aligned


class PaperAdam(torch.optim.Optimizer):
    """TensorFlow-style Adam: epsilon is added to the uncorrected second moment.

    A new optimizer is used per fit; optimizer-state persistence is unspecified
    in the paper. Hyperparameters follow footnote 10.
    """
    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-7):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state.update(step=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
                state["step"] += 1
                state["m"].mul_(b1).add_(p.grad, alpha=1 - b1)
                state["v"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                t = state["step"]
                alpha = group["lr"] * (1 - b2 ** t) ** 0.5 / (1 - b1 ** t)
                p.addcdiv_(state["m"], state["v"].sqrt().add_(group["eps"]), value=-alpha)


def fit_window(x, y, config, initial_state=None):
    from .config import HORIZONS, MARKETS

    if config.hidden_factors != 3:
        raise ValueError("The paper specification requires exactly three hidden factors")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Training inputs and targets must be finite")
    model = SharedMomentumNetwork(MARKETS, HORIZONS, x.shape[0], config.seed)
    if initial_state is not None:
        model.load_state_dict(initial_state)
    optimizer = PaperAdam(model.parameters(), lr=config.learning_rate,
                          betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_epsilon)
    epochs = config.first_epochs if initial_state is None else config.rolling_epochs
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = model.objective(x, y, lambda_w=config.lambda_w, lambda_bias=config.lambda_bias)
        loss.backward()
        optimizer.step()
    return model
