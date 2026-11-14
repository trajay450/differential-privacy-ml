#!/usr/bin/env python3
"""Reproduce every number and figure in the README.

    python experiments/run_experiment.py   (~15s on a laptop CPU)
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dpml.accounting import compute_epsilon
from dpml.data import load_dp_dataset
from dpml.dp_sgd import DPLogisticRegression

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
EPOCHS = 20
BATCH_SIZE = 32
LR = 0.5
CLIP_NORM = 1.0
NOISE_MULTIPLIERS = [0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
SEEDS = [1, 2, 3, 4, 5]


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    data = load_dp_dataset(random_state=7)
    n, d = data.X_train.shape
    steps_per_epoch = n // BATCH_SIZE + 1
    total_steps = EPOCHS * steps_per_epoch
    sampling_rate = BATCH_SIZE / n
    delta = 1.0 / (n * 10)  # standard choice: delta << 1/n

    tradeoff = []
    grad_norms_by_sigma = {}
    for sigma in NOISE_MULTIPLIERS:
        accs = []
        last_grad_norms = None
        for seed in SEEDS:
            model = DPLogisticRegression(n_features=d, seed=seed)
            # clipping is applied even at sigma=0, isolating noise's effect
            # from clipping's effect on accuracy (see the true non-private
            # baseline below, which has neither)
            model.fit(
                data.X_train, data.y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                clip_norm=CLIP_NORM, noise_multiplier=sigma, seed=seed,
            )
            accs.append(model.accuracy(data.X_test, data.y_test))
            last_grad_norms = model.grad_norm_history
        epsilon = compute_epsilon(total_steps, sigma, sampling_rate, delta) if sigma > 0 else float("inf")
        tradeoff.append({
            "noise_multiplier": sigma,
            "epsilon": epsilon,
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
        })
        grad_norms_by_sigma[sigma] = last_grad_norms

    # true non-private baseline: no clipping, no noise at all
    accs_nonprivate = []
    for seed in SEEDS:
        m = DPLogisticRegression(n_features=d, seed=seed)
        m.fit(data.X_train, data.y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
              clip_norm=None, noise_multiplier=0.0, seed=seed)
        accs_nonprivate.append(m.accuracy(data.X_test, data.y_test))

    metrics = {
        "config": {
            "dataset": "Breast Cancer Wisconsin (Diagnostic), 569 samples, 30 features",
            "n_train": n, "n_test": len(data.X_test),
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "clip_norm": CLIP_NORM,
            "sampling_rate": sampling_rate, "total_steps": total_steps, "delta": delta,
            "accountant": "Abadi et al. (2016) Theorem 1 simplified bound (c=1); illustrative, not a numerical RDP accountant",
            "seeds_per_point": SEEDS,
        },
        "true_nonprivate_baseline_accuracy": {
            "mean": float(np.mean(accs_nonprivate)), "std": float(np.std(accs_nonprivate)),
            "note": "no clipping, no noise -- ordinary logistic regression SGD",
        },
        "privacy_utility_tradeoff": tradeoff,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # plot 1: privacy-utility tradeoff -- THE canonical DP-ML plot
    finite = [t for t in tradeoff if t["epsilon"] != float("inf")]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    eps_vals = [t["epsilon"] for t in finite]
    acc_vals = [t["accuracy_mean"] for t in finite]
    acc_stds = [t["accuracy_std"] for t in finite]
    ax.errorbar(eps_vals, acc_vals, yerr=acc_stds, marker="o", color="#D85A30", capsize=3)
    ax.axhline(np.mean(accs_nonprivate), color="#639922", linestyle="--", linewidth=1.3,
               label=f"Non-private baseline ({np.mean(accs_nonprivate):.3f})")
    ax.set_xscale("log")
    ax.invert_xaxis()  # smaller epsilon (more private) on the right, matching "cost of privacy" reading
    ax.set_xlabel("Privacy budget ε (log scale, smaller = more private)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("The privacy-utility tradeoff curve")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "privacy_utility_tradeoff.png", dpi=160)
    plt.close(fig)

    # plot 2: accuracy vs. noise multiplier directly (the knob you actually turn)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    sigmas = [t["noise_multiplier"] for t in tradeoff]
    accs = [t["accuracy_mean"] for t in tradeoff]
    stds = [t["accuracy_std"] for t in tradeoff]
    ax.errorbar(sigmas, accs, yerr=stds, marker="s", color="#378ADD", capsize=3)
    ax.set_xlabel("Noise multiplier σ")
    ax.set_ylabel("Test accuracy")
    ax.set_title("More injected noise -> lower accuracy")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "accuracy_vs_noise.png", dpi=160)
    plt.close(fig)

    # plot 3: per-example gradient norm distribution (why clipping matters)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    unclipped_norms = grad_norms_by_sigma[0.0]
    ax.hist(unclipped_norms, bins=40, color="#888780", alpha=0.85)
    ax.axvline(CLIP_NORM, color="#D85A30", linestyle="--", linewidth=1.5, label=f"Clip norm C={CLIP_NORM}")
    ax.set_xlabel("Mean per-example gradient L2 norm (per batch)")
    ax.set_ylabel("Batches")
    ax.set_title("Gradient norms, pre-clipping\n(inputs are L2-normalized, so C=1.0 rarely binds here)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "gradient_norms.png", dpi=160)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print(f"\nWrote results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
