"""DP-SGD (Abadi et al., 2016) implemented by hand for logistic
regression: per-example gradient clipping, then Gaussian noise added to
the *averaged* clipped gradient before each weight update.

The per-example clipping step is the whole point and the whole cost of
DP-SGD relative to ordinary SGD: an ordinary minibatch gradient is a
single averaged vector, so one outlier example can't be bounded
individually. Here every example's own gradient is computed and clipped
*before* averaging, which is what makes the sensitivity of each update
bounded and the Gaussian-noise privacy guarantee valid.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


@dataclass
class DPLogisticRegression:
    n_features: int
    seed: int = 0
    w: np.ndarray = field(init=False)
    b: float = field(default=0.0, init=False)
    grad_norm_history: list = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.w = rng.normal(0, 0.01, size=self.n_features)
        self.b = 0.0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(X @ self.w + self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def _per_example_gradients(self, Xb: np.ndarray, yb: np.ndarray) -> np.ndarray:
        """Returns an (batch_size, n_features + 1) matrix: one gradient row
        per example (weights, then bias), *before* any clipping."""
        p = self.predict_proba(Xb)
        err = (p - yb)[:, None]
        grad_w = err * Xb
        grad_b = err
        return np.concatenate([grad_w, grad_b], axis=1)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 0.5,
        clip_norm: float | None = 1.0,
        noise_multiplier: float = 0.0,
        seed: int = 0,
    ) -> "DPLogisticRegression":
        """``clip_norm=None`` and ``noise_multiplier=0`` reproduces ordinary
        (non-private) minibatch SGD exactly -- the private and non-private
        baselines in the experiment share this one code path, so any
        accuracy difference is attributable only to clipping and noise.
        """
        rng = np.random.default_rng(seed)
        n = len(X)
        self.grad_norm_history = []

        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                batch = order[start:start + batch_size]
                Xb, yb = X[batch], y[batch]

                grads = self._per_example_gradients(Xb, yb)  # (b, d+1)
                norms = np.linalg.norm(grads, axis=1, keepdims=True)
                self.grad_norm_history.append(float(norms.mean()))

                if clip_norm is not None:
                    scale = np.minimum(1.0, clip_norm / (norms + 1e-12))
                    grads = grads * scale

                avg_grad = grads.mean(axis=0)

                if noise_multiplier > 0 and clip_norm is not None:
                    noise_std = noise_multiplier * clip_norm / len(batch)
                    avg_grad = avg_grad + rng.normal(0, noise_std, size=avg_grad.shape)

                self.w -= lr * avg_grad[:-1]
                self.b -= lr * avg_grad[-1]

        return self

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == y).mean())
