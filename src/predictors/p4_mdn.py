"""P4 — mixture density network on y (spec §7.6)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.eval.metrics import TAUS
from src.predictors.base import Predictor

torch.set_num_threads(4)


class MDN(nn.Module):
    def __init__(self, d_in, K, hidden=(128, 128), dropout=0.1):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, 3 * K)
        self.K = K

    def forward(self, x):
        o = self.head(self.body(x))
        logit, mu, s = o.split(self.K, dim=1)
        return torch.log_softmax(logit, 1), mu, nn.functional.softplus(s) + 0.05


def nll(logw, mu, sd, y):
    lp = -0.5 * ((y[:, None] - mu) / sd) ** 2 - torch.log(sd) - 0.5 * np.log(2 * np.pi)
    return -torch.logsumexp(logw + lp, 1).mean()


def mixture_quantiles(w, mu, sd, tol=1e-4):
    """Invert mixture CDF per row by bisection, vectorised over rows and the 99 levels."""
    from scipy.special import ndtr
    n = len(w)
    lo = (mu - 8 * sd).min(1)[:, None].repeat(99, 1)
    hi = (mu + 8 * sd).max(1)[:, None].repeat(99, 1)
    tau = TAUS[None, :]
    while np.max(hi - lo) > tol:
        mid = 0.5 * (lo + hi)
        F = (w[:, None, :] * ndtr((mid[:, :, None] - mu[:, None, :]) / sd[:, None, :])).sum(-1)
        below = F < tau
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


class P4MDN(Predictor):
    name = "P4"

    def __init__(self, K=3, hidden=(128, 128), dropout=0.1, lr=1e-3, wd=1e-4, batch=512, max_epochs=100,
                 patience=10, seed=0):
        self.K, self.hidden, self.dropout = K, hidden, dropout
        self.lr, self.wd, self.batch, self.max_epochs, self.patience, self.seed = lr, wd, batch, max_epochs, patience, seed

    def fit(self, train, val):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X = torch.tensor(train.Xd)
        y = torch.tensor(train.y, dtype=torch.float32)
        self.ymu, self.ysd = float(y.mean()), float(y.std())
        y = (y - self.ymu) / self.ysd
        Xv = torch.tensor(val.Xd)
        yv = (torch.tensor(val.y, dtype=torch.float32) - self.ymu) / self.ysd
        self.net = MDN(X.shape[1], self.K, self.hidden, self.dropout)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        best, best_state, bad = np.inf, None, 0
        n = len(y)
        for ep in range(self.max_epochs):
            self.net.train()
            perm = torch.tensor(rng.permutation(n))
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]
                opt.zero_grad()
                loss = nll(*self.net(X[idx]), y[idx])
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                v = float(nll(*self.net(Xv), yv))
            if v < best - 1e-5:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.val_nll, self.epochs = best, ep + 1
        return self

    def mixture(self, b):
        self.net.eval()
        with torch.no_grad():
            lw, mu, sd = self.net(torch.tensor(b.Xd))
        w = lw.exp().numpy().astype(float)
        mu = mu.numpy().astype(float) * self.ysd + self.ymu
        sd = sd.numpy().astype(float) * self.ysd
        return w, mu, sd

    def predict_quantiles(self, b):
        w, mu, sd = self.mixture(b)
        out = np.empty((len(w), 99))
        for i in range(0, len(w), 4096):
            out[i:i + 4096] = mixture_quantiles(w[i:i + 4096], mu[i:i + 4096], sd[i:i + 4096])
        return np.sort(out, axis=1)
