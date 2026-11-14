"""The public Breast Cancer Wisconsin (Diagnostic) dataset -- a sensitive,
patient-level binary classification task, which is exactly the kind of
data differential privacy is meant to protect training on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class DPData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def load_dp_dataset(test_size: float = 0.25, random_state: int = 7) -> DPData:
    raw = load_breast_cancer()
    X, y = raw.data, raw.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    # bound each row's L2 norm to 1 -- a common, simple preprocessing step
    # for DP-SGD so the per-example gradient scale is well-controlled
    # independent of any single feature's raw magnitude
    X_train = X_train / np.maximum(np.linalg.norm(X_train, axis=1, keepdims=True), 1e-8)
    X_test = X_test / np.maximum(np.linalg.norm(X_test, axis=1, keepdims=True), 1e-8)
    return DPData(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
