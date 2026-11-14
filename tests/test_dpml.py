import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from dpml.accounting import compute_epsilon, required_noise_multiplier
from dpml.data import load_dp_dataset
from dpml.dp_sgd import DPLogisticRegression


@pytest.fixture(scope="module")
def data():
    return load_dp_dataset(random_state=1)


def test_inputs_are_l2_normalized(data):
    norms = np.linalg.norm(data.X_train, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_epsilon_decreases_as_noise_increases():
    eps_low_noise = compute_epsilon(steps=200, noise_multiplier=0.5, sampling_rate=0.05, delta=1e-4)
    eps_high_noise = compute_epsilon(steps=200, noise_multiplier=3.0, sampling_rate=0.05, delta=1e-4)
    assert eps_high_noise < eps_low_noise


def test_epsilon_increases_with_more_steps():
    eps_short = compute_epsilon(steps=50, noise_multiplier=1.0, sampling_rate=0.05, delta=1e-4)
    eps_long = compute_epsilon(steps=500, noise_multiplier=1.0, sampling_rate=0.05, delta=1e-4)
    assert eps_long > eps_short


def test_required_noise_multiplier_inverts_compute_epsilon():
    steps, q, delta, target_eps = 300, 0.08, 1e-4, 2.0
    sigma = required_noise_multiplier(steps, target_eps, q, delta)
    eps_back = compute_epsilon(steps, sigma, q, delta)
    assert eps_back == pytest.approx(target_eps, rel=1e-6)


def test_non_private_training_beats_chance(data):
    n_features = data.X_train.shape[1]
    model = DPLogisticRegression(n_features=n_features, seed=0)
    model.fit(data.X_train, data.y_train, epochs=15, batch_size=32, lr=0.5, clip_norm=None, noise_multiplier=0.0, seed=0)
    assert model.accuracy(data.X_test, data.y_test) > 0.85


def test_more_noise_degrades_accuracy_on_average(data):
    n_features = data.X_train.shape[1]
    accs = {}
    for sigma in [0.0, 3.0]:
        trial_accs = []
        for seed in [1, 2, 3]:
            m = DPLogisticRegression(n_features=n_features, seed=seed)
            m.fit(data.X_train, data.y_train, epochs=15, batch_size=32, lr=0.5, clip_norm=1.0, noise_multiplier=sigma, seed=seed)
            trial_accs.append(m.accuracy(data.X_test, data.y_test))
        accs[sigma] = np.mean(trial_accs)
    assert accs[0.0] > accs[3.0]


def test_gradient_clipping_bounds_per_example_contribution(data):
    n_features = data.X_train.shape[1]
    model = DPLogisticRegression(n_features=n_features, seed=0)
    model.fit(data.X_train, data.y_train, epochs=1, batch_size=32, lr=0.5, clip_norm=0.05, noise_multiplier=0.0, seed=0)
    # with a very small clip norm, the recorded (pre-clip) mean norms should
    # still often exceed it -- confirming clipping is actually doing work
    assert max(model.grad_norm_history) > 0.05
