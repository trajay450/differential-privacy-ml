"""Privacy accounting for DP-SGD.

This implements the simplified, closed-form epsilon bound from Abadi et
al. (2016), "Deep Learning with Differential Privacy", Theorem 1: for a
subsampled Gaussian mechanism run for ``steps`` iterations with sampling
rate ``q`` and noise multiplier ``sigma``, there is a constant c such
that the mechanism is (epsilon, delta)-DP for

    epsilon ~= c * q * sqrt(steps * log(1/delta)) / sigma

This is the standard illustrative bound used to build intuition for how
epsilon scales with noise, steps and sampling rate -- it is **not** the
tight numerical moments-accountant / RDP accountant that production
libraries (Opacus, TensorFlow Privacy) use, and is noted as such. We take
c=1, matching the commonly cited simplified form of the theorem.
"""
from __future__ import annotations

import numpy as np


def compute_epsilon(
    steps: int, noise_multiplier: float, sampling_rate: float, delta: float, c: float = 1.0
) -> float:
    if noise_multiplier <= 0:
        return float("inf")
    return float(c * sampling_rate * np.sqrt(steps * np.log(1.0 / delta)) / noise_multiplier)


def required_noise_multiplier(
    steps: int, target_epsilon: float, sampling_rate: float, delta: float, c: float = 1.0
) -> float:
    """Invert :func:`compute_epsilon`: how much noise is needed to hit a
    target epsilon for a given training length and sampling rate."""
    return float(c * sampling_rate * np.sqrt(steps * np.log(1.0 / delta)) / target_epsilon)
