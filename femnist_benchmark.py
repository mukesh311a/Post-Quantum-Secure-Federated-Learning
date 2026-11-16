import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import numpy as np
import pandas as pd

from src.config import ExperimentConfig
from src.data.leaf_femnist import FEMNIST_DIMS, ensure_femnist, load_federated_femnist
from src.fl.methods import METHODS
from src.fl.trainer import establish_pairwise_secrets, run_method
from src.crypto.rlwe import RLWE, RLWEParams
from src.analysis.stats import SeedAlignedStatsAnalyzer
from src.privacy.accounting import epsilon_pld, epsilon_rdp_with_order

SQRT10 = math.sqrt(10.0)
FEMNIST_METHODS = [
    ("Vanilla FedAvg", "Vanilla FedAvg", 1.0),
    ("FedProx", "FedProx", 1.0),
    ("SCAFFOLD", "SCAFFOLD", 1.0),
    ("DP-FedAvg", "DP-FedAvg", 1.0),
    ("FL + Distributed Noise", "FL + Distributed Noise", 1.0),
    ("PQ-FL (Proposed)", "PQ-FL (Proposed)", 1.0),
    ("PQ-FL (sigma=sqrt(10))", "PQ-FL (Proposed)", SQRT10),
]
METRICS = ("accuracy", "f1", "auc", "mae")


def femnist_config(num_rounds: int = 200, seeds=None) -> ExperimentConfig:
    return ExperimentConfig(
        num_clients=100,
        clients_per_round=10,
        num_rounds=num_rounds,
        local_epochs=3,
        lr=0.05,
        batch_size=32,
        fedprox_mu=0.01,
        seeds=list(seeds) if seeds is not None else [42, 123, 456, 789, 1024],
        model_dims=FEMNIST_DIMS,
        clip_init=0.30,
        clip_min=0.05,
        clip_max=0.60,
    )


def _run_seed(args):
    seed, num_rounds, output_dir = args
    torch.set_num_threads(1)
    cfg = femnist_config(num_rounds)
    cache_dir = os.path.join(output_dir, ".femnist_cache")
    os.makedirs(cache_dir, exist_ok=True)
    clients, X_te, y_te, meta = load_federated_femnist(seed, num_writers=cfg.num_clients)
    rl = RLWE(RLWEParams(cfg.rlwe_n, cfg.rlwe_q, cfg.rlwe_eta, cfg.rlwe_t))
    secrets, t_sec, mism_setup = establish_pairwise_secrets(rl, cfg.num_clients, np.random.default_rng(seed))
    out = {"seed": seed, "meta": meta, "setup_s": t_sec, "setup_mismatches": mism_setup, "runs": {}}
    for label, key, sig in FEMNIST_METHODS:
        spec = METHODS[key]
        safe_name = "".join(c if c.isalnum() else "_" for c in label)
        ckpt_path = os.path.join(cache_dir, f"ckpt_s{seed}_{safe_name}.json")
        if os.path.exists(ckpt_path):
            try:
                with open(ckpt_path, "r", encoding="utf-8") as fh:
                    cached_data = json.load(fh)
                if "metrics" in cached_data and len(cached_data["metrics"].get("accuracy", [])) == num_rounds:
                    out["runs"][label] = cached_data
                    print(
                        f"  [FEMNIST seed={seed:4d}] {label:<24} [LOADED CHECKPOINT] "
                        f"acc={cached_data['metrics']['accuracy'][-1]:6.2f}% f1={cached_data['metrics']['f1'][-1]:.4f}",
                        flush=True,
                    )
                    continue
            except Exception as e:
                print(f"  [FEMNIST seed={seed:4d}] Corrupted checkpoint {ckpt_path}: {e}, re-running...", flush=True)

        t0 = time.time()
        res = run_method(label, spec, seed, cfg, clients, X_te, y_te, sig, secrets if spec["secagg"] else None)
        dg = res["diagnostics"]
        run_record = {
            "metrics": res["metrics"],
            "payload": int(res["payload_bytes_per_client"]),
            "mismatches": int(dg["mismatched_coords"]),
            "clamped": int(dg["clamped_coords"]),
            "coords_decoded": int(cfg.num_rounds * dg["coords_per_round"]),
            "mean_clip": dg["mean_clip_threshold"],
            "timing_s": res["timing_s"],
            "wall_s": time.time() - t0,
        }
        out["runs"][label] = run_record
        with open(ckpt_path, "w", encoding="utf-8") as fh:
            json.dump(run_record, fh)
        print(
            f"  [FEMNIST seed={seed:4d}] {label:<24} acc={res['metrics']['accuracy'][-1]:6.2f}% "
            f"f1={res['metrics']['f1'][-1]:.4f} mism={dg['mismatched_coords']} ({time.time() - t0:6.1f}s)",
            flush=True,
        )
    return out


def _budgets(spec, sig, q, T, delta):
    if spec["dp"] is None:
        return ("Inf",) * 4
    n_sel = 10
    sig_agg = sig * math.sqrt(n_sel) if spec["dp"] == "local" else sig
    sig_srv = sig if (spec["dp"] == "local" or spec["secagg"]) else sig / math.sqrt(n_sel)
    ea, _ = epsilon_rdp_with_order(sig_agg, q, T, delta)
    es, _ = epsilon_rdp_with_order(sig_srv, q, T, delta)
    return (
        round(ea, 2),
        round(epsilon_pld(sig_agg, q, T, delta), 2),
        round(es, 2),
        round(epsilon_pld(sig_srv, q, T, delta), 2),
    )


def run_femnist_benchmark(seeds, num_rounds: int = 200, output_dir: str = "results", workers: int = None):
    ensure_femnist()
    seeds = list(seeds)
    cfg = femnist_config(num_rounds, seeds)
    workers = workers or min(len(seeds), max(1, (os.cpu_count() or 2) - 1))
    print(
        f"\n[PHASE 2D] LEAF FEMNIST (62 classes, writer-partitioned, N=100, |S_t|=10, T={num_rounds}, "
        f"dims={FEMNIST_DIMS}) - {len(seeds)} seeds on {workers} worker processes...",
        flush=True,
    )
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_run_seed, [(s, num_rounds, output_dir) for s in seeds]))
    by_seed = {r["seed"]: r for r in results}
    ordered = [by_seed[s] for s in seeds]
    os.makedirs(output_dir, exist_ok=True)

    final = {lbl: [] for lbl, _, _ in FEMNIST_METHODS}
    final_m = {lbl: {k: [] for k in METRICS} for lbl, _, _ in FEMNIST_METHODS}
    curves = {lbl: {k: [] for k in METRICS} for lbl, _, _ in FEMNIST_METHODS}
    raw_rows, total_mism, total_coords = [], 0, 0
    for r in ordered:
        for lbl, _, _ in FEMNIST_METHODS:
            run = r["runs"][lbl]
            total_mism += run["mismatches"]
            total_coords += run["coords_decoded"]
            for k in METRICS:
                curves[lbl][k].append(run["metrics"][k])
                final_m[lbl][k].append(run["metrics"][k][-1])
            final[lbl].append(run["metrics"]["accuracy"][-1])
            for t_idx in range(num_rounds):
                raw_rows.append(
                    {
                        "Algorithm": lbl,
                        "Seed": r["seed"],
                        "Round": t_idx + 1,
                        "Accuracy": run["metrics"]["accuracy"][t_idx],
                        "F1_Score": run["metrics"]["f1"][t_idx],
                        "AUC": run["metrics"]["auc"][t_idx],
                        "MAE": run["metrics"]["mae"][t_idx],
                    }
                )
    supp = os.path.join(ROOT_DIR, "Supplementary_Data_Sheet") if os.path.normpath(output_dir) == "results" else os.path.join(output_dir, "Supplementary_Data_Sheet")
    os.makedirs(supp, exist_ok=True)
    pd.DataFrame(raw_rows).to_csv(os.path.join(supp, "FEMNIST_Raw_Metrics.csv"), index=False)

    summary, rows = {}, []
    q, T, delta = cfg.sampling_rate, cfg.num_rounds, cfg.delta
    for lbl, key, sig in FEMNIST_METHODS:
        spec = METHODS[key]
        summary[lbl] = {"rounds": list(range(1, num_rounds + 1))}
        for k in METRICS:
            mat = np.array(curves[lbl][k], dtype=np.float64)
            summary[lbl][k] = {
                "mean": mat.mean(axis=0).tolist(),
                "std": (mat.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros(num_rounds)).tolist(),
            }
        ea, ep, es, esp = _budgets(spec, sig, q, T, delta)
        accs = np.array(final[lbl])
        rows.append(
            {
                "Algorithm": lbl,
                "Nominal_Sigma": round(sig, 4) if spec["dp"] is not None else "None",
                "Eps_Agg_RDP": ea,
                "Eps_Agg_PLD": ep,
                "Eps_ServerView_RDP": es,
                "Eps_ServerView_PLD": esp,
                "Payload_Bytes": ordered[0]["runs"][lbl]["payload"],
                "Payload_KiB": round(ordered[0]["runs"][lbl]["payload"] / 1024.0, 2),
                "Accuracy_Mean": round(float(accs.mean()), 2),
                "Accuracy_SD": round(float(accs.std(ddof=1)) if len(accs) > 1 else 0.0, 2),
                "F1_Mean": round(float(np.mean(final_m[lbl]["f1"])), 4),
                "F1_SD": round(float(np.std(final_m[lbl]["f1"], ddof=1)) if len(accs) > 1 else 0.0, 4),
                "AUC_Mean": round(float(np.mean(final_m[lbl]["auc"])), 4),
                "MAE_Mean": round(float(np.mean(final_m[lbl]["mae"])), 4),
                "Mean_Client_Train_DP_ms": round(
                    float(np.mean([r["runs"][lbl]["timing_s"]["client_train"] + r["runs"][lbl]["timing_s"]["client_dpops"] for r in ordered])) * 1000.0, 2
                ),
                "Mean_Server_Agg_ms": round(
                    float(np.mean([r["runs"][lbl]["timing_s"]["server_aggregate"] for r in ordered])) * 1000.0, 2
                ),
                "Run_Wall_Clock_s_Mean": round(float(np.mean([r["runs"][lbl]["wall_s"] for r in ordered])), 1),
                "Crypto_Mismatches": int(sum(r["runs"][lbl]["mismatches"] for r in ordered)),
                "Per_Seed_Acc": str([round(x, 2) for x in final[lbl]]),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "Public_Benchmark_LEAF_FEMNIST.csv"), index=False)
    with open(os.path.join(output_dir, "femnist_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    stats_txt = ""
    if len(seeds) > 1:
        stats_txt = SeedAlignedStatsAnalyzer(final).generate_report(
            [
                ("PQ-FL (Proposed)", "DP-FedAvg"),
                ("PQ-FL (Proposed)", "Vanilla FedAvg"),
                ("PQ-FL (Proposed)", "FL + Distributed Noise"),
                ("PQ-FL (sigma=sqrt(10))", "DP-FedAvg"),
            ]
        )
    meta_rows = pd.DataFrame([dict(seed=r["seed"], **r["meta"], setup_mismatches=r["setup_mismatches"]) for r in ordered])
    header = (
        f"LEAF FEMNIST (writer-partitioned, 62 classes; N=100 writers/seed, |S_t|=10, q=0.10, T={num_rounds}, "
        f"dims={FEMNIST_DIMS}, d=16638, E=3, lr=0.05, B=32, C_1=0.30 in [0.05,0.60])\n"
        f"Total Ring-LWE/mask decoding mismatches: {total_mism} over {total_coords:,d} decoded coordinates\n"
        f"Parallel wall-clock for all seeds: {(time.time() - t_start) / 60.0:.2f} min\n\n"
    )
    with open(os.path.join(output_dir, "femnist_stats.txt"), "w", encoding="utf-8") as fh:
        fh.write(header + "Per-seed data summary:\n" + meta_rows.to_string(index=False) + "\n\n"
                 + df.drop(columns=["Per_Seed_Acc"]).to_string(index=False) + "\n\nPer-seed accuracies:\n"
                 + "\n".join(f"  {lbl:<24}: {final[lbl]}" for lbl, _, _ in FEMNIST_METHODS)
                 + "\n\n" + stats_txt + "\n")
    print(header + df[["Algorithm", "Accuracy_Mean", "Accuracy_SD", "F1_Mean", "Eps_Agg_RDP", "Eps_ServerView_RDP", "Payload_KiB"]].to_string(index=False))
    print(stats_txt)

    try:
        from src.visualization.plots import Visualizer
        Visualizer(output_dir).plot_femnist(summary, n_seeds=len(seeds))
    except Exception as exc:
        print(f"[WARN] FEMNIST plot failed: {exc}")
    return df, summary, final


def main():
    ap = argparse.ArgumentParser(description="LEAF FEMNIST benchmark for PQ-FL")
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 123, 456, 789, 1024])
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--output_dir", default="results")
    a = ap.parse_args()
    os.chdir(ROOT_DIR)
    run_femnist_benchmark(a.seeds, a.rounds, a.output_dir, a.workers)


if __name__ == "__main__":
    main()
