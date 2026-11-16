import argparse
import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import torch
import numpy as np
from src.config import ExperimentConfig
from src.data.synthetic import build_dataset, load_federated
from src.fl.methods import METHODS
from src.fl.trainer import run_method, establish_pairwise_secrets
from src.crypto.rlwe import RLWE, RLWEParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    a = ap.parse_args()
    cfg = ExperimentConfig(num_rounds=a.rounds)
    path = os.path.join(cfg.dataset_dir, f"6G_IoT_Federated_Data_seed{a.seed}.csv")
    if not os.path.exists(path):
        build_dataset(path, a.seed, cfg.num_records, cfg.num_clients, cfg.dirichlet_alpha)
    clients, Xte, yte = load_federated(path)
    sigma = float(a.sigma)
    print(f"sigma = {sigma:.4f}")
    secrets = None
    if any(METHODS[m]["secagg"] for m in a.methods):
        rl = RLWE(RLWEParams(cfg.rlwe_n, cfg.rlwe_q, cfg.rlwe_eta, cfg.rlwe_t))
        secrets, secs, mism = establish_pairwise_secrets(rl, cfg.num_clients, np.random.default_rng(a.seed))
        print(f"pairwise setup: {secs:.2f}s, mismatches={mism}")
    for m in a.methods:
        t0 = time.time()
        r = run_method(m, METHODS[m], a.seed, cfg, clients, Xte, yte, sigma, secrets)
        acc = r["metrics"]["accuracy"]
        dg = dict(r["diagnostics"])
        ct, un = dg.pop("clip_trajectory"), dg.pop("update_norm_trajectory")
        print(f"{m:28s} final acc={acc[-1]:6.2f} best={max(acc):6.2f}  time={time.time() - t0:5.1f}s")
        print(f"    diag={dg}\n    timing={ {k: round(v, 4) for k, v in r['timing_s'].items()} }")
        print(f"    acc[::{max(1, len(acc) // 10)}]={[round(x, 1) for x in acc[::max(1, len(acc) // 10)]]}")
        if ct:
            print(f"    C_t[:10]={[round(x, 3) for x in ct[:10]]}  C_t[-3:]={[round(x, 3) for x in ct[-3:]]}")
        print(f"    mean raw update norm per round[:10]={[round(x, 3) for x in un[:10]]}")


if __name__ == "__main__":
    main()
