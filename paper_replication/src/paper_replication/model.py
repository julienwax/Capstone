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


def fit_window(x, y, config, initial_state=None):
    from .config import HORIZONS, MARKETS

    model = SharedMomentumNetwork(MARKETS, HORIZONS, x.shape[0], config.seed)
    if initial_state is not None:
        model.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    epochs = config.first_epochs if initial_state is None else config.rolling_epochs
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = model.objective(x, y, lambda_w=config.lambda_w, lambda_bias=config.lambda_bias)
        loss.backward()
        optimizer.step()
    return model
