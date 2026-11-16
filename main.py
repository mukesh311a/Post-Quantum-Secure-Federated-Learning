import os
import sys
import json
import math
import time
import platform

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import torch
import numpy as np
import pandas as pd
import scipy
import sklearn

from src.config import ExperimentConfig
from src.data.synthetic import build_dataset, load_federated, write_description
from src.data.public_digits import load_federated_digits
from src.fl.methods import METHODS, privacy_score
from src.fl.trainer import run_method, establish_pairwise_secrets
from src.crypto.rlwe import RLWE, RLWEParams
from src.analysis.stats import SeedAlignedStatsAnalyzer
from src.analysis.rdp_accountant import run_privacy_accounting
from src.analysis import payload_audit, decryption_failure
from src.privacy.accounting import (
    dp_accounting_version,
    epsilon_pld,
    epsilon_rdp_with_order,
    gradient_sigma_for_adaptive_clipping,
)
from src.visualization.plots import Visualizer
from scaling_analysis import run_scaling_analysis
from femnist_benchmark import run_femnist_benchmark


def _first_round_at_or_above(mean_curve, threshold: float = 85.0):
    for idx, val in enumerate(mean_curve):
        if val >= threshold:
            return idx + 1
    return None


def main():
    import argparse
    ap = argparse.ArgumentParser(description="PQ-FL End-to-End Reproducibility Pipeline")
    ap.add_argument("--smoke", action="store_true", help="Quick 2-seed, 2-round verification of all pipeline phases")
    args = ap.parse_args()

    t_pipeline_start = time.time()
    print("=" * 84)
    print("      PQ-FL Framework - End-to-End Empirical Reproducibility Pipeline      ")
    print("=" * 84)

    cfg = ExperimentConfig()
    if args.smoke:
        cfg.seeds = [42, 123]
        cfg.num_rounds = 2
        cfg.output_dir = "results_smoke"
        print("[SMOKE MODE] Running 2 seeds x 2 rounds to verify all pipeline phases end-to-end.")
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.dataset_dir, exist_ok=True)

    desc_path = os.path.join(cfg.dataset_dir, "Dataset_Description.txt")
    write_description(desc_path)

    rl = RLWE(RLWEParams(cfg.rlwe_n, cfg.rlwe_q, cfg.rlwe_eta, cfg.rlwe_t))
    pairwise_secrets_by_seed = {}
    setup_times_s = []
    setup_mismatches = 0
    print(f"\n[PHASE 1] Pre-establishing Ring-LWE pairwise shared secrets across {cfg.num_clients} clients...")
    for seed in cfg.seeds:
        rng = np.random.default_rng(seed)
        sec, t_sec, mism = establish_pairwise_secrets(rl, cfg.num_clients, rng)
        pairwise_secrets_by_seed[seed] = sec
        setup_times_s.append(t_sec)
        setup_mismatches += mism
        print(f"  > Seed {seed:4d}: Established {len(sec)} pairwise secrets in {t_sec:.2f}s (Mismatches: {mism})")

    methods_to_run = list(METHODS.keys())
    metrics_to_track = ["accuracy", "f1", "auc", "mae"]

    raw_runs = {
        m: {k: np.zeros((len(cfg.seeds), cfg.num_rounds), dtype=np.float64) for k in metrics_to_track}
        for m in methods_to_run
    }
    final_acc_by_seed = {m: [] for m in methods_to_run}
    final_metrics_by_seed = {m: {k: [] for k in metrics_to_track} for m in methods_to_run}
    raw_records = []
    timing_records = {m: [] for m in methods_to_run}
    payload_records = {m: 0 for m in methods_to_run}
    wall_records = []
    diag_records = []

    sigma_primary = 1.0
    datasets_by_seed = {}

    print(
        f"\n[PHASE 2A] Primary 6G-IoT Study: {len(cfg.seeds)} seeds x {len(methods_to_run)} methods "
        f"({cfg.num_rounds} rounds each, sigma={sigma_primary})..."
    )
    for s_idx, seed in enumerate(cfg.seeds):
        print(f"\n==================== SEED {seed} ({s_idx + 1}/{len(cfg.seeds)}) ====================")
        ds_path = os.path.join(cfg.dataset_dir, f"6G_IoT_Federated_Data_seed{seed}.csv")
        build_dataset(ds_path, seed, cfg.num_records, cfg.num_clients, cfg.dirichlet_alpha)
        clients, X_test, y_test = load_federated(ds_path)
        datasets_by_seed[seed] = (clients, X_test, y_test)

        for m_name in methods_to_run:
            spec = METHODS[m_name]
            t0 = time.time()
            pair_sec = pairwise_secrets_by_seed[seed] if spec["secagg"] else None
            res = run_method(m_name, spec, seed, cfg, clients, X_test, y_test, sigma_primary, pair_sec, quiet=True)
            elapsed = time.time() - t0

            final_acc = res["metrics"]["accuracy"][-1]
            final_acc_by_seed[m_name].append(final_acc)
            for k in metrics_to_track:
                final_metrics_by_seed[m_name][k].append(res["metrics"][k][-1])
                raw_runs[m_name][k][s_idx, :] = res["metrics"][k]

            timing_records[m_name].append(res["timing_s"])
            payload_records[m_name] = res["payload_bytes_per_client"]
            wall_records.append(
                {
                    "Algorithm": m_name,
                    "Seed": seed,
                    "Run_Wall_Clock_s": elapsed,
                    "Per_Round_Wall_Clock_s": elapsed / cfg.num_rounds,
                    **{f"{k}_ms": v * 1000.0 for k, v in res["timing_s"].items()},
                }
            )
            dg = res["diagnostics"]
            sig_b_diag = (cfg.sigma_b_factor * sigma_primary) if spec["dp"] is not None else None
            sig_g_diag = (
                gradient_sigma_for_adaptive_clipping(sigma_primary, sig_b_diag) if spec["dp"] is not None else None
            )
            diag_records.append(
                {
                    "Algorithm": m_name,
                    "Seed": seed,
                    "Mismatched_Coords": dg["mismatched_coords"],
                    "Clamped_Coords": dg["clamped_coords"],
                    "Coords_Decoded": cfg.num_rounds * dg["coords_per_round"],
                    "Mean_Clip_Threshold": dg["mean_clip_threshold"],
                    "Mean_Clipped_Fraction": dg["mean_clipped_fraction"],
                    "Mean_Update_Norm": dg["mean_update_norm"],
                    "Sigma_Grad": sig_g_diag,
                    "Sigma_Quantile": sig_b_diag,
                }
            )

            for r_idx in range(cfg.num_rounds):
                raw_records.append(
                    {
                        "Algorithm": m_name,
                        "Seed": seed,
                        "Round": r_idx + 1,
                        "Accuracy": res["metrics"]["accuracy"][r_idx],
                        "F1_Score": res["metrics"]["f1"][r_idx],
                        "AUC": res["metrics"]["auc"][r_idx],
                        "MAE": res["metrics"]["mae"][r_idx],
                    }
                )

            print(
                f"  [{m_name:<26}] Final Acc: {final_acc:6.2f}% | "
                f"Elapsed: {elapsed:5.1f}s | "
                f"Client Crypto: {(res['timing_s']['client_encode_mask'] + res['timing_s']['client_encrypt'])*1000:5.2f}ms | "
                f"Server Agg: {res['timing_s']['server_aggregate']*1000:5.2f}ms"
            )

    supp_dir = os.path.join(cfg.output_dir, "Supplementary_Data_Sheet") if args.smoke else "Supplementary_Data_Sheet"
    os.makedirs(supp_dir, exist_ok=True)
    raw_df = pd.DataFrame(raw_records)
    supp_csv_path = os.path.join(supp_dir, "Training_Raw_Metrics.csv")
    raw_df.to_csv(supp_csv_path, index=False)
    print(f"\n[SUCCESS] Exported raw seed-aligned metrics to {supp_csv_path}")

    summary_metrics = {}
    for m_name in methods_to_run:
        summary_metrics[m_name] = {"rounds": list(range(1, cfg.num_rounds + 1))}
        for k in metrics_to_track:
            mat = raw_runs[m_name][k]
            summary_metrics[m_name][k] = {
                "mean": np.mean(mat, axis=0).tolist(),
                "std": np.std(mat, axis=0, ddof=1).tolist(),
            }

    metrics_json_path = os.path.join(cfg.output_dir, "comprehensive_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=4)
    print(f"[SUCCESS] Saved aggregated metrics JSON to {metrics_json_path}")

    print("\n[PHASE 2A-STATS] Computing Statistical Significance (Seed-Aligned Paired t-tests)...")
    stats_analyzer = SeedAlignedStatsAnalyzer(final_acc_by_seed)
    comparisons = [
        ("PQ-FL (Proposed)", "Vanilla FedAvg"),
        ("PQ-FL (Proposed)", "FedProx"),
        ("PQ-FL (Proposed)", "SCAFFOLD"),
        ("PQ-FL (Proposed)", "DP-FedAvg"),
        ("PQ-FL (Proposed)", "FL + Distributed Noise"),
        ("PQ-FL (Proposed)", "FL + Secure Aggregation"),
        ("PQ-FL (Proposed)", "FL + PQ Encryption Only"),
    ]
    stat_report = stats_analyzer.generate_report(comparisons)
    print(stat_report)

    stat_report_path = os.path.join(cfg.output_dir, "statistical_report.txt")
    with open(stat_report_path, "w", encoding="utf-8") as f:
        f.write(stat_report)
        f.write("\n\nFinal Metrics per Method (Mean +/- Sample SD across 5 seeds):\n")
        for m_name in methods_to_run:
            accs = final_acc_by_seed[m_name]
            f1s = final_metrics_by_seed[m_name]["f1"]
            aucs = final_metrics_by_seed[m_name]["auc"]
            maes = final_metrics_by_seed[m_name]["mae"]
            r85 = _first_round_at_or_above(summary_metrics[m_name]["accuracy"]["mean"], 85.0)
            f.write(
                f"  {m_name:<26}: Acc={[round(x, 2) for x in accs]} "
                f"(Mean={np.mean(accs):.2f}% +/- {np.std(accs, ddof=1):.2f}%, "
                f"F1={np.mean(f1s):.4f}+/-{np.std(f1s, ddof=1):.4f}, "
                f"AUC={np.mean(aucs):.4f}+/-{np.std(aucs, ddof=1):.4f}, "
                f"MAE={np.mean(maes):.4f}+/-{np.std(maes, ddof=1):.4f}, "
                f"FirstRound>=85%={r85})\n"
            )

    stds = {m: float(np.std(final_acc_by_seed[m], ddof=1)) for m in methods_to_run}
    max_s = max(stds.values()) if max(stds.values()) > 0 else 1.0
    stability_scores = {m: round(float(1.0 - stds[m] / max_s) * 9.0 + 1.0, 2) for m in methods_to_run}

    def client_time(m):
        return float(
            np.mean(
                [
                    t["client_train"] + t["client_dpops"] + t["client_encode_mask"] + t["client_encrypt"]
                    for t in timing_records[m]
                ]
            )
        )

    ref_payload = payload_records["Vanilla FedAvg"]
    ref_client = client_time("Vanilla FedAvg")

    radar_data = {
        "labels": ["Test Accuracy", "Privacy Security", "Seed Stability", "Comm. Efficiency", "Client Comp. Speed"],
        "data": {},
    }
    for m in methods_to_run:
        acc_score = round(float(np.mean(final_acc_by_seed[m])) / 10.0, 2)
        ps = privacy_score(METHODS[m])
        stab = stability_scores[m]
        comm_eff = round(min(10.0, 10.0 * ref_payload / payload_records[m]), 2)
        comp_speed = round(min(10.0, 10.0 * ref_client / client_time(m)), 2)
        radar_data["data"][m] = [acc_score, ps, stab, comm_eff, comp_speed]

    radar_json_path = os.path.join(cfg.output_dir, "radar_metrics.json")
    with open(radar_json_path, "w", encoding="utf-8") as f:
        json.dump(radar_data, f, indent=4)
    print(f"[SUCCESS] Saved empirical radar metrics to {radar_json_path}")

    fairness_rows = []
    for m in methods_to_run:
        mean_acc = float(np.mean(final_acc_by_seed[m]))
        ps = privacy_score(METHODS[m])
        pfi = round(ps * (mean_acc / 100.0), 3)
        fairness_rows.append(
            {
                "Algorithm": m,
                "Accuracy (%)": round(mean_acc, 2),
                "Privacy Score": ps,
                "Privacy Fairness Index": pfi,
                "Final Acc Std (s_j)": round(stds[m], 3),
                "Seed Stability Score": stability_scores[m],
            }
        )
    pd.DataFrame(fairness_rows).to_csv(os.path.join(cfg.output_dir, "Privacy_Fairness_Table.csv"), index=False)

    overhead_rows = []
    wall_df = pd.DataFrame(wall_records)
    for m in methods_to_run:
        avg_local_ms = np.mean([t["client_train"] + t["client_dpops"] for t in timing_records[m]]) * 1000.0
        avg_encmask_ms = np.mean([t["client_encode_mask"] for t in timing_records[m]]) * 1000.0
        avg_encrypt_ms = np.mean([t["client_encrypt"] for t in timing_records[m]]) * 1000.0
        avg_crypto_ms = avg_encmask_ms + avg_encrypt_ms
        avg_server_ms = np.mean([t["server_aggregate"] for t in timing_records[m]]) * 1000.0
        w_run = wall_df[wall_df.Algorithm == m]["Run_Wall_Clock_s"].values
        w_rnd = wall_df[wall_df.Algorithm == m]["Per_Round_Wall_Clock_s"].values
        overhead_rows.append(
            {
                "Algorithm": m,
                "Payload (Bytes)": int(payload_records[m]),
                "Payload (KiB)": round(payload_records[m] / 1024.0, 2),
                "Client Local Train+DP (ms)": round(float(avg_local_ms), 2),
                "Client Encode+Mask (ms)": round(float(avg_encmask_ms), 2),
                "Client Encrypt (ms)": round(float(avg_encrypt_ms), 2),
                "Client Crypto Overhead (ms)": round(float(avg_crypto_ms), 2),
                "Server Aggregation Overhead (ms)": round(float(avg_server_ms), 2),
                "Per-Round Wall-Clock (s, mean)": round(float(np.mean(w_rnd)), 3),
                "Per-Round Wall-Clock (s, std)": round(float(np.std(w_rnd, ddof=1)), 3),
                "Run Wall-Clock T=200 (s, mean)": round(float(np.mean(w_run)), 1),
                "Run Wall-Clock T=200 (s, std)": round(float(np.std(w_run, ddof=1)), 1),
            }
        )
    pd.DataFrame(overhead_rows).to_csv(os.path.join(cfg.output_dir, "Cryptographic_Overhead_Metrics.csv"), index=False)
    wall_df.round(4).to_csv(os.path.join(cfg.output_dir, "Timing_Raw_Per_Seed.csv"), index=False)
    pd.DataFrame(diag_records).round(5).to_csv(os.path.join(cfg.output_dir, "Run_Diagnostics_Per_Seed.csv"), index=False)

    with open(os.path.join(cfg.output_dir, "environment.txt"), "w", encoding="utf-8") as f:
        f.write(
            f"platform: {platform.platform()}\n"
            f"processor: {platform.processor()}\n"
            f"python: {sys.version.split()[0]}\n"
            f"torch: {torch.__version__} (torch.set_num_threads=1, CPU only)\n"
            f"numpy: {np.__version__}\n"
            f"pandas: {pd.__version__}\n"
            f"scipy: {scipy.__version__}\n"
            f"scikit-learn: {sklearn.__version__}\n"
            f"dp-accounting: {dp_accounting_version()}\n"
            f"pairwise_setup_N100_mean_s: {np.mean(setup_times_s):.3f}\n"
            f"pairwise_setup_N100_std_s: {np.std(setup_times_s, ddof=1):.3f}\n"
            f"pairwise_setup_N100_mismatches: {setup_mismatches}\n"
            f"timing boundary: time.perf_counter around each stage inside src/fl/trainer.py; "
            f"wall-clock = whole run_method call executing 10 clients sequentially on 1 CPU thread\n"
        )

    print("\n[PHASE 2B] Running Multi-Budget & Privacy-Matched Ablation on 6G-IoT (5 seeds)...")
    n_part = cfg.clients_per_round
    sqrt_n = math.sqrt(n_part)
    extra_runs_spec = [
        ("PQ-FL (sigma=2.0)", METHODS["PQ-FL (Proposed)"], 2.0),
        ("PQ-FL (sigma=sqrt(10))", METHODS["PQ-FL (Proposed)"], sqrt_n),
        ("PQ-FL (sigma=4.0)", METHODS["PQ-FL (Proposed)"], 4.0),
        ("FL + DistNoise (sigma=sqrt(10))", METHODS["FL + Distributed Noise"], sqrt_n),
    ]
    ablation_accs = {
        "PQ-FL (sigma=1.0)": list(final_acc_by_seed["PQ-FL (Proposed)"]),
        "FL + DistNoise (sigma=1.0)": list(final_acc_by_seed["FL + Distributed Noise"]),
        "DP-FedAvg (sigma_local=1.0)": list(final_acc_by_seed["DP-FedAvg"]),
    }
    ablation_f1s = {
        "PQ-FL (sigma=1.0)": list(final_metrics_by_seed["PQ-FL (Proposed)"]["f1"]),
        "FL + DistNoise (sigma=1.0)": list(final_metrics_by_seed["FL + Distributed Noise"]["f1"]),
        "DP-FedAvg (sigma_local=1.0)": list(final_metrics_by_seed["DP-FedAvg"]["f1"]),
    }
    for label, _, _ in extra_runs_spec:
        ablation_accs[label] = []
        ablation_f1s[label] = []

    for seed in cfg.seeds:
        clients, X_test, y_test = datasets_by_seed[seed]
        for label, spec, sig in extra_runs_spec:
            t0 = time.time()
            pair_sec = pairwise_secrets_by_seed[seed] if spec["secagg"] else None
            res = run_method(label, spec, seed, cfg, clients, X_test, y_test, sig, pair_sec, quiet=True)
            acc_fin = res["metrics"]["accuracy"][-1]
            f1_fin = res["metrics"]["f1"][-1]
            ablation_accs[label].append(acc_fin)
            ablation_f1s[label].append(f1_fin)
            print(f"  Seed {seed:4d} | {label:<32} | sigma={sig:.4f} | Final Acc={acc_fin:6.2f}% ({time.time()-t0:4.1f}s)")

    q_6g = cfg.sampling_rate
    T_6g = cfg.num_rounds
    delta_6g = cfg.delta

    def _budgets(sig_eff: float):
        ep_rdp, ord_rdp = epsilon_rdp_with_order(sig_eff, q_6g, T_6g, delta_6g)
        ep_pld = epsilon_pld(sig_eff, q_6g, T_6g, delta_6g)
        return round(ep_rdp, 2), round(ord_rdp, 2), round(ep_pld, 2)

    ablation_table_meta = [
        ("PQ-FL (sigma=1.0)", 1.0, 1.0, 1.0, True, True),
        ("PQ-FL (sigma=2.0)", 2.0, 2.0, 2.0, True, True),
        ("PQ-FL (sigma=sqrt(10))", sqrt_n, sqrt_n, sqrt_n, True, True),
        ("PQ-FL (sigma=4.0)", 4.0, 4.0, 4.0, True, True),
        ("FL + DistNoise (sigma=1.0)", 1.0, 1.0, 1.0 / sqrt_n, False, False),
        ("FL + DistNoise (sigma=sqrt(10))", sqrt_n, sqrt_n, 1.0, False, False),
        ("DP-FedAvg (sigma_local=1.0)", 1.0, sqrt_n, 1.0, False, False),
    ]
    ablation_rows = []
    for label, sig_nom, sig_agg, sig_srv, has_sa, has_pq in ablation_table_meta:
        agg_rdp, agg_ord, agg_pld = _budgets(sig_agg)
        srv_rdp, _, srv_pld = _budgets(sig_srv)
        sig_b = cfg.sigma_b_factor * sig_agg
        sig_g = gradient_sigma_for_adaptive_clipping(sig_agg, sig_b)
        accs = ablation_accs[label]
        f1s = ablation_f1s[label]
        ablation_rows.append(
            {
                "Configuration": label,
                "Nominal_Sigma": round(sig_nom, 4),
                "Grad_Sigma_Agg": round(sig_g, 4),
                "Quantile_Sigma_b": round(sig_b, 4),
                "Effective_Sigma_Agg": round(sig_agg, 4),
                "Effective_Sigma_ServerView": round(sig_srv, 4),
                "Eps_Agg_RDP": agg_rdp,
                "Optimal_RDP_Alpha": agg_ord,
                "Eps_Agg_PLD": agg_pld,
                "Eps_ServerView_RDP": srv_rdp,
                "Eps_ServerView_PLD": srv_pld,
                "SecAgg": has_sa,
                "PQ_Encryption": has_pq,
                "Accuracy_Mean": round(float(np.mean(accs)), 2),
                "Accuracy_SD": round(float(np.std(accs, ddof=1)), 2),
                "F1_Mean": round(float(np.mean(f1s)), 4),
                "F1_SD": round(float(np.std(f1s, ddof=1)), 4),
                "Per_Seed_Acc": str([round(x, 2) for x in accs]),
            }
        )
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_csv_path = os.path.join(cfg.output_dir, "Privacy_Matched_Ablation.csv")
    ablation_df.to_csv(ablation_csv_path, index=False)

    ablation_stats = SeedAlignedStatsAnalyzer(ablation_accs)
    ablation_report = ablation_stats.generate_report(
        [
            ("PQ-FL (sigma=1.0)", "DP-FedAvg (sigma_local=1.0)"),
            ("PQ-FL (sigma=sqrt(10))", "DP-FedAvg (sigma_local=1.0)"),
            ("PQ-FL (sigma=sqrt(10))", "FL + DistNoise (sigma=sqrt(10))"),
            ("PQ-FL (sigma=2.0)", "DP-FedAvg (sigma_local=1.0)"),
            ("PQ-FL (sigma=4.0)", "DP-FedAvg (sigma_local=1.0)"),
        ]
    )
    print(ablation_report)
    with open(os.path.join(cfg.output_dir, "privacy_ablation_report.txt"), "w", encoding="utf-8") as f:
        f.write(ablation_df.to_string(index=False) + "\n\n" + ablation_report + "\n")

    print("\n[PHASE 2C] External Validation on Public UCI Handwritten Digits Benchmark (5 seeds)...")
    cfg_digits = ExperimentConfig(
        num_clients=50,
        clients_per_round=10,
        num_rounds=cfg.num_rounds,
        local_epochs=5,
        lr=0.05,
        batch_size=16,
        dirichlet_alpha=0.5,
        fedprox_mu=0.01,
        seeds=list(cfg.seeds),
        model_dims=(64, 64, 10),
        clip_init=0.50,
        clip_min=0.05,
        clip_max=0.80,
    )
    digits_secrets_by_seed = {}
    for seed in cfg.seeds:
        sec_d, _, mism_d = establish_pairwise_secrets(rl, cfg_digits.num_clients, np.random.default_rng(seed))
        assert mism_d == 0
        digits_secrets_by_seed[seed] = sec_d

    digits_methods = [
        ("Vanilla FedAvg", METHODS["Vanilla FedAvg"], 1.0),
        ("FedProx", METHODS["FedProx"], 1.0),
        ("SCAFFOLD", METHODS["SCAFFOLD"], 1.0),
        ("DP-FedAvg", METHODS["DP-FedAvg"], 1.0),
        ("FL + Distributed Noise", METHODS["FL + Distributed Noise"], 1.0),
        ("PQ-FL (Proposed)", METHODS["PQ-FL (Proposed)"], 1.0),
        ("PQ-FL (sigma=sqrt(10))", METHODS["PQ-FL (Proposed)"], sqrt_n),
    ]
    digits_runs = {
        label: {k: np.zeros((len(cfg.seeds), cfg_digits.num_rounds), dtype=np.float64) for k in metrics_to_track}
        for label, _, _ in digits_methods
    }
    digits_final_acc = {label: [] for label, _, _ in digits_methods}
    digits_final_f1 = {label: [] for label, _, _ in digits_methods}
    digits_payload = {}
    digits_mismatches = 0

    for s_idx, seed in enumerate(cfg.seeds):
        clients_d, Xte_d, yte_d = load_federated_digits(
            seed=seed, num_clients=cfg_digits.num_clients, alpha=cfg_digits.dirichlet_alpha
        )
        print(f"  --- UCI Digits Seed {seed} ({s_idx + 1}/{len(cfg.seeds)}) ---")
        for label, spec, sig in digits_methods:
            t0 = time.time()
            pair_sec = digits_secrets_by_seed[seed] if spec["secagg"] else None
            res = run_method(label, spec, seed, cfg_digits, clients_d, Xte_d, yte_d, sig, pair_sec, quiet=True)
            for k in metrics_to_track:
                digits_runs[label][k][s_idx, :] = res["metrics"][k]
            acc_fin = res["metrics"]["accuracy"][-1]
            f1_fin = res["metrics"]["f1"][-1]
            digits_final_acc[label].append(acc_fin)
            digits_final_f1[label].append(f1_fin)
            digits_payload[label] = res["payload_bytes_per_client"]
            digits_mismatches += res["diagnostics"]["mismatched_coords"]
            print(f"    [{label:<26}] Final Acc: {acc_fin:6.2f}% | F1: {f1_fin:.4f} | ({time.time()-t0:4.1f}s)")

    q_dig = cfg_digits.sampling_rate
    T_dig = cfg_digits.num_rounds
    delta_dig = cfg_digits.delta

    def _digits_budgets(label, spec, sig):
        if spec["dp"] is None:
            return None, None, None, None
        sig_agg = sig * sqrt_n if spec["dp"] == "local" else sig
        sig_srv = sig if (spec["dp"] == "local" or spec["secagg"]) else (sig / sqrt_n)
        e_agg_rdp, _ = epsilon_rdp_with_order(sig_agg, q_dig, T_dig, delta_dig)
        e_agg_pld = epsilon_pld(sig_agg, q_dig, T_dig, delta_dig)
        e_srv_rdp, _ = epsilon_rdp_with_order(sig_srv, q_dig, T_dig, delta_dig)
        e_srv_pld = epsilon_pld(sig_srv, q_dig, T_dig, delta_dig)
        return round(e_agg_rdp, 2), round(e_agg_pld, 2), round(e_srv_rdp, 2), round(e_srv_pld, 2)

    digits_rows = []
    digits_summary = {}
    for label, spec, sig in digits_methods:
        digits_summary[label] = {"rounds": list(range(1, cfg_digits.num_rounds + 1))}
        for k in metrics_to_track:
            mat = digits_runs[label][k]
            digits_summary[label][k] = {
                "mean": np.mean(mat, axis=0).tolist(),
                "std": np.std(mat, axis=0, ddof=1).tolist(),
            }
        ea_rdp, ea_pld, es_rdp, es_pld = _digits_budgets(label, spec, sig)
        accs = digits_final_acc[label]
        f1s = digits_final_f1[label]
        digits_rows.append(
            {
                "Algorithm": label,
                "Nominal_Sigma": round(sig, 4) if spec["dp"] is not None else "None",
                "Eps_Agg_RDP": ea_rdp if ea_rdp is not None else "Inf",
                "Eps_Agg_PLD": ea_pld if ea_pld is not None else "Inf",
                "Eps_ServerView_RDP": es_rdp if es_rdp is not None else "Inf",
                "Eps_ServerView_PLD": es_pld if es_pld is not None else "Inf",
                "Payload_KiB": round(digits_payload[label] / 1024.0, 2),
                "Accuracy_Mean": round(float(np.mean(accs)), 2),
                "Accuracy_SD": round(float(np.std(accs, ddof=1)), 2),
                "F1_Mean": round(float(np.mean(f1s)), 4),
                "F1_SD": round(float(np.std(f1s, ddof=1)), 4),
                "AUC_Mean": round(float(np.mean(digits_runs[label]["auc"][:, -1])), 4),
                "MAE_Mean": round(float(np.mean(digits_runs[label]["mae"][:, -1])), 4),
                "Per_Seed_Acc": str([round(x, 2) for x in accs]),
            }
        )

    digits_df = pd.DataFrame(digits_rows)
    digits_df.to_csv(os.path.join(cfg.output_dir, "Public_Benchmark_UCI_Digits.csv"), index=False)
    with open(os.path.join(cfg.output_dir, "public_digits_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(digits_summary, f, indent=4)

    digits_stats = SeedAlignedStatsAnalyzer(digits_final_acc).generate_report(
        [
            ("PQ-FL (Proposed)", "DP-FedAvg"),
            ("PQ-FL (Proposed)", "Vanilla FedAvg"),
            ("PQ-FL (Proposed)", "FL + Distributed Noise"),
            ("PQ-FL (sigma=sqrt(10))", "DP-FedAvg"),
        ]
    )
    print(digits_stats)
    with open(os.path.join(cfg.output_dir, "public_digits_stats.txt"), "w", encoding="utf-8") as f:
        f.write(
            f"UCI Optical Recognition of Handwritten Digits (N=50, |S_t|=10, q=0.20, T=200, dims=(64,64,10), "
            f"d=4810, total_crypto_mismatches={digits_mismatches})\n\n"
            + digits_df.to_string(index=False)
            + "\n\n"
            + digits_stats
            + "\n"
        )

    run_femnist_benchmark(cfg.seeds, num_rounds=cfg.num_rounds, output_dir=cfg.output_dir)

    print("\n[PHASE 3] Formal Differential Privacy Accounting, Payload Audit & Decryption Verification...")
    run_privacy_accounting(cfg.output_dir)
    rep_payload = payload_audit.build_report()
    print(rep_payload)
    with open(os.path.join(cfg.output_dir, "payload_audit.txt"), "w", encoding="utf-8") as fh:
        fh.write(rep_payload + "\n")
    rep_decrypt = decryption_failure.build_report()
    print(rep_decrypt)
    with open(os.path.join(cfg.output_dir, "decryption_failure_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(rep_decrypt + "\n")

    print("\n[PHASE 4] Generating Publication-Quality Visualizations...")
    viz = Visualizer(cfg.output_dir)
    viz.plot_all_curves(summary_metrics, n_seeds=len(cfg.seeds))
    viz.plot_radar(radar_data)
    viz.plot_tradeoff(summary_metrics, radar_data)
    viz.plot_public_digits(digits_summary, n_seeds=len(cfg.seeds))

    print("\n[PHASE 5] Running Live Ring-LWE Scaling Analysis...")
    run_scaling_analysis(cfg)

    import subprocess
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "plot_manuscript_figures.py"), cfg.output_dir], check=True)

    total_min = (time.time() - t_pipeline_start) / 60.0
    print("\n" + "=" * 84)
    print(f" [COMPLETE] Entire Empirical Pipeline Executed Successfully in {total_min:.2f} min!")
    print("=" * 84)


if __name__ == "__main__":
    main()
