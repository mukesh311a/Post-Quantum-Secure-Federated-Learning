import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

from src.config import ExperimentConfig
from src.data.public_digits import load_federated_digits
from src.data.synthetic import FEATURE_RANGES, FEATURES, build_dataset, load_federated


def main():
    cfg = ExperimentConfig()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ds_dir = os.path.join(root, cfg.dataset_dir)

    print("=== 1. Synthetic 6G-IoT Federated Dataset Integrity Across 5 Seeds ===")
    qda_accs, mlp_accs = [], []
    for seed in cfg.seeds:
        path = os.path.join(ds_dir, f"6G_IoT_Federated_Data_seed{seed}.csv")
        if not os.path.exists(path):
            build_dataset(path, seed, cfg.num_records, cfg.num_clients, cfg.dirichlet_alpha)
        df = pd.read_csv(path)
        assert list(df.columns) == ["Record_ID"] + FEATURES + ["Device_Type", "Label", "Split", "Client_ID"]
        for feat, (lo, hi) in FEATURE_RANGES.items():
            assert df[feat].min() >= lo - 1e-6 and df[feat].max() <= hi + 1e-6, f"Range violation in {feat}"
        tr_ids = set(df[df["Split"] == "train"]["Record_ID"])
        te_ids = set(df[df["Split"] == "test"]["Record_ID"])
        assert len(tr_ids & te_ids) == 0 and len(tr_ids) == 8000 and len(te_ids) == 2000

        clients, Xte, yte = load_federated(path)
        Xtr = np.concatenate([c[0] for c in clients.values()])
        ytr = np.concatenate([c[1] for c in clients.values()])
        sizes = np.array([len(c[1]) for c in clients.values()])
        qda = QuadraticDiscriminantAnalysis().fit(Xtr, ytr).score(Xte, yte) * 100.0
        mlp = MLPClassifier((128, 64), max_iter=250, random_state=seed).fit(Xtr, ytr).score(Xte, yte) * 100.0
        qda_accs.append(qda)
        mlp_accs.append(mlp)
        print(
            f"  seed={seed:4d} | train={len(ytr)} test={len(yte)} clients={len(clients)} "
            f"(min={sizes.min():2d}, med={int(np.median(sizes)):2d}, max={sizes.max():3d}) "
            f"| maj={np.bincount(yte).max() / len(yte) * 100:.2f}% | QDA={qda:.2f}% | MLP={mlp:.2f}%"
        )
    print(f"  Mean centralised QDA: {np.mean(qda_accs):.2f}% +/- {np.std(qda_accs, ddof=1):.2f}% | "
          f"MLP(128,64): {np.mean(mlp_accs):.2f}% +/- {np.std(mlp_accs, ddof=1):.2f}%")

    print("\n=== 2. Public UCI Optical Handwritten Digits Federated Benchmark (Dirichlet alpha=0.5) ===")
    dig_mlp = []
    for seed in cfg.seeds:
        clients, Xte, yte = load_federated_digits(seed, num_clients=50, alpha=0.5, min_client_size=4)
        Xtr = np.concatenate([c[0] for c in clients.values()])
        ytr = np.concatenate([c[1] for c in clients.values()])
        sizes = np.array([len(c[1]) for c in clients.values()])
        mlp = MLPClassifier((64,), max_iter=300, random_state=seed).fit(Xtr, ytr).score(Xte, yte) * 100.0
        dig_mlp.append(mlp)
        print(
            f"  seed={seed:4d} | train={len(ytr)} test={len(yte)} clients={len(clients)} "
            f"(min={sizes.min():2d}, med={int(np.median(sizes)):2d}, max={sizes.max():2d}) "
            f"| centralised MLP(64): {mlp:.2f}%"
        )
    print(f"  Mean centralised UCI Digits MLP(64): {np.mean(dig_mlp):.2f}% +/- {np.std(dig_mlp, ddof=1):.2f}%")

    print("\n=== 3. Public LEAF FEMNIST (writer-partitioned, 62 classes, N=100 writers per seed) ===")
    from src.data.leaf_femnist import FEMNIST_SHA256, load_federated_femnist
    print(f"  Archive SHA-256 expected: {FEMNIST_SHA256}")
    for seed in cfg.seeds:
        clients, Xte, yte, meta = load_federated_femnist(seed, num_writers=100)
        Xtr = np.concatenate([c[0] for c in clients.values()])
        assert Xtr.shape[1] == 196 and Xte.shape[1] == 196
        assert abs(float(Xtr.mean())) < 1e-3, "train-only standardisation violated"
        print(
            f"  seed={seed:4d} | writers={meta['num_writers']}/{meta['total_writers_available']} "
            f"train={meta['train_examples']} test={meta['test_examples']} "
            f"(client size min={meta['client_size_min']}, med={meta['client_size_median']:.0f}, "
            f"max={meta['client_size_max']}) | test classes={meta['test_classes_present']} "
            f"| mean classes/client={meta['classes_per_client_mean']:.1f}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
