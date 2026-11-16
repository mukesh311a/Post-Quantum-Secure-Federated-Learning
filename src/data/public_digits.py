from typing import Dict, Tuple
import torch
import numpy as np
from sklearn.datasets import load_digits

from src.data.synthetic import dirichlet_partition

DIGITS_DIMS = (64, 64, 10)


def load_federated_digits(
    seed: int,
    num_clients: int = 50,
    alpha: float = 0.5,
    test_fraction: float = 0.2,
    min_client_size: int = 4,
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    digits = load_digits()
    X_all = digits.data.astype(np.float64)
    y_all = digits.target.astype(np.int64)
    n_classes = int(y_all.max()) + 1

    rng = np.random.default_rng(seed)
    is_test = np.zeros(len(y_all), dtype=bool)
    for c in range(n_classes):
        idx = np.where(y_all == c)[0]
        n_te = int(round(test_fraction * len(idx)))
        te_idx = rng.choice(idx, size=n_te, replace=False)
        is_test[te_idx] = True

    tr_idx = np.where(~is_test)[0]
    te_idx = np.where(is_test)[0]

    mu = X_all[tr_idx].mean(axis=0)
    raw_sd = X_all[tr_idx].std(axis=0)
    sd = np.where(raw_sd > 1e-6, raw_sd, 1.0)

    X_tr = ((X_all[tr_idx] - mu) / sd).astype(np.float32)
    y_tr = y_all[tr_idx]
    X_te = ((X_all[te_idx] - mu) / sd).astype(np.float32)
    y_te = y_all[te_idx]

    parts = dirichlet_partition(y_tr, num_clients=num_clients, alpha=alpha, rng=rng, min_size=min_client_size)
    clients = {k: (X_tr[p], y_tr[p]) for k, p in enumerate(parts)}
    return clients, X_te, y_te
