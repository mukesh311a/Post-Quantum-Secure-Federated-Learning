import hashlib
import os
import tarfile
import urllib.request
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import h5py

FEMNIST_URL = "https://storage.googleapis.com/tff-datasets-public/fed_emnist.tar.bz2"
FEMNIST_SHA256 = "fe1ed5a502cea3a952eb105920bff8cffb32836b5173cb18a57a32c3606f3ea0"
FEMNIST_DIMS = (196, 64, 62)
FEMNIST_NUM_CLASSES = 62

_DEFAULT_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "dataset", "femnist_cache"
)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_femnist(cache_dir: Optional[str] = None) -> Tuple[str, str]:
    cache_dir = cache_dir or _DEFAULT_CACHE
    os.makedirs(cache_dir, exist_ok=True)
    tr = os.path.join(cache_dir, "fed_emnist_train.h5")
    te = os.path.join(cache_dir, "fed_emnist_test.h5")
    if os.path.exists(tr) and os.path.exists(te):
        return tr, te
    archive = os.path.join(cache_dir, "fed_emnist.tar.bz2")
    if not os.path.exists(archive):
        print(f"[LEAF FEMNIST] Downloading {FEMNIST_URL} ...")
        urllib.request.urlretrieve(FEMNIST_URL, archive)
    digest = _sha256(archive)
    if digest != FEMNIST_SHA256:
        raise RuntimeError(f"FEMNIST archive checksum mismatch: {digest} != {FEMNIST_SHA256}")
    with tarfile.open(archive, "r:bz2") as tf:
        tf.extractall(cache_dir)
    return tr, te


def _prep(pixels: np.ndarray, pool: int) -> np.ndarray:
    x = 1.0 - pixels.astype(np.float32)
    if pool > 1:
        x = F.avg_pool2d(torch.from_numpy(x[:, None, :, :]), pool).numpy()
    return x.reshape(len(x), -1)


def load_federated_femnist(
    seed: int,
    num_writers: int = 100,
    pool: int = 2,
    cache_dir: Optional[str] = None,
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray, Dict]:
    tr_path, te_path = ensure_femnist(cache_dir)
    raw_tr, te_x, te_y = {}, [], []
    with h5py.File(tr_path, "r") as ftr, h5py.File(te_path, "r") as fte:
        all_writers = sorted(ftr["examples"].keys())
        rng = np.random.default_rng(seed)
        chosen = sorted(rng.choice(all_writers, size=num_writers, replace=False).tolist())
        for k, wid in enumerate(chosen):
            g_tr, g_te = ftr["examples"][wid], fte["examples"][wid]
            raw_tr[k] = (_prep(g_tr["pixels"][:], pool), g_tr["label"][:].astype(np.int64))
            te_x.append(_prep(g_te["pixels"][:], pool))
            te_y.append(g_te["label"][:].astype(np.int64))

    X_cat = np.concatenate([v[0] for v in raw_tr.values()], axis=0)
    mu = X_cat.mean(axis=0)
    raw_sd = X_cat.std(axis=0)
    sd = np.where(raw_sd > 1e-6, raw_sd, 1.0)
    clients = {k: (((x - mu) / sd).astype(np.float32), y) for k, (x, y) in raw_tr.items()}
    X_te = ((np.concatenate(te_x, axis=0) - mu) / sd).astype(np.float32)
    y_te = np.concatenate(te_y, axis=0)

    sizes = np.array([len(v[1]) for v in clients.values()])
    labels_per_client = np.array([len(np.unique(v[1])) for v in clients.values()])
    meta = {
        "total_writers_available": len(all_writers),
        "num_writers": num_writers,
        "train_examples": int(sizes.sum()),
        "test_examples": int(len(y_te)),
        "test_classes_present": int(len(np.unique(y_te))),
        "client_size_min": int(sizes.min()),
        "client_size_median": float(np.median(sizes)),
        "client_size_max": int(sizes.max()),
        "classes_per_client_mean": float(labels_per_client.mean()),
        "feature_dim": int(X_te.shape[1]),
    }
    return clients, X_te, y_te, meta
