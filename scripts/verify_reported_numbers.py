import ast
import os
import sys

import torch
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "Supplementary_Data_Sheet", "Training_Raw_Metrics.csv")
RESULTS_DIR = os.path.join(ROOT, "results")

MANUSCRIPT_FINAL_ACC = {
    "Vanilla FedAvg": [95.25, 96.30, 96.15, 96.10, 95.60],
    "FedProx": [95.25, 96.30, 96.15, 96.10, 95.60],
    "SCAFFOLD": [95.25, 96.30, 96.15, 96.15, 95.75],
    "DP-FedAvg": [91.20, 91.15, 89.60, 92.30, 91.65],
    "FL + Distributed Noise": [95.05, 96.05, 95.95, 96.15, 96.00],
    "FL + Secure Aggregation": [95.25, 96.30, 96.15, 96.10, 95.60],
    "FL + PQ Encryption Only": [95.25, 96.30, 96.15, 96.10, 95.60],
    "PQ-FL (Proposed)": [95.10, 95.65, 95.65, 95.70, 95.40],
}
MANUSCRIPT_PAIRED_TESTS = {
    "Vanilla FedAvg": (-0.38, -4.09, 1.503e-2, -1.83),
    "FedProx": (-0.38, -4.09, 1.503e-2, -1.83),
    "SCAFFOLD": (-0.42, -5.06, 7.200e-3, -2.26),
    "FL + Distributed Noise": (-0.34, -3.12, 3.541e-2, -1.40),
    "DP-FedAvg": (4.32, 9.24, 7.630e-4, 4.13),
}
MANUSCRIPT_SUMMARY_TABLE = {
    "PQ-FL (Proposed)": (95.50, 0.95, 1.00, 0.06),
    "SCAFFOLD": (95.92, 0.96, 1.00, 0.05),
    "FedProx": (95.88, 0.96, 1.00, 0.05),
    "Vanilla FedAvg": (95.88, 0.96, 1.00, 0.05),
    "DP-FedAvg": (91.18, 0.91, 0.94, 0.12),
}
MANUSCRIPT_PAYLOAD_BYTES = {
    "Vanilla FedAvg": 38160,
    "FedProx": 38160,
    "SCAFFOLD": 76320,
    "DP-FedAvg": 38164,
    "FL + Distributed Noise": 38164,
    "FL + Secure Aggregation": 114480,
    "FL + PQ Encryption Only": 458752,
    "PQ-FL (Proposed)": 458752,
}

failures = []


def check(label, ok, detail=""):
    print(f"  [{'OK ' if ok else 'FAIL'}] {label} {detail}")
    if not ok:
        failures.append(label)


df = pd.read_csv(CSV)
seeds = sorted(df["Seed"].unique())
algos = list(dict.fromkeys(df["Algorithm"]))
last = int(df["Round"].max())
print(f"Loaded {len(df)} rows: {len(algos)} algorithms x {len(seeds)} seeds x {last} rounds  ({CSV})\n")

final = {a: df[(df.Algorithm == a) & (df.Round == last)].sort_values("Seed")["Accuracy"].values for a in algos}

print("1. Primary 6G-IoT final-round accuracy per seed (seed order %s)" % seeds)
for a in algos:
    v = final[a]
    print(f"   {a:<26} {np.round(v, 2).tolist()}  mean={v.mean():.2f}  sd={v.std(ddof=1):.3f}")
for a, ref in MANUSCRIPT_FINAL_ACC.items():
    check(f"per-seed accuracies of {a}", np.allclose(final[a], ref, atol=0.006))

print("\n2. Seed-aligned paired t-tests (scipy.stats.ttest_rel), PQ-FL vs baseline")
pq = final["PQ-FL (Proposed)"]
for b, (d_ref, t_ref, p_ref, dz_ref) in MANUSCRIPT_PAIRED_TESTS.items():
    diff = pq - final[b]
    t, p = stats.ttest_rel(pq, final[b])
    dz = diff.mean() / diff.std(ddof=1)
    print(f"   vs {b:<24} delta={diff.mean():+.2f} pp  t(4)={t:6.2f}  p={p:.3e}  dz={dz:+.2f}")
    check(
        f"Paired t-test row vs {b}",
        abs(diff.mean() - d_ref) < 0.006
        and abs(t - t_ref) < 0.01
        and abs(dz - dz_ref) < 0.01
        and abs(p - p_ref) / p_ref < 0.015,
    )

print("\n3. Final-round Macro-F1 / Macro-OvR AUC / MAE (mean over 5 seeds)")
for a, (acc, f1, auc, mae) in MANUSCRIPT_SUMMARY_TABLE.items():
    g = df[(df.Algorithm == a) & (df.Round == last)]
    got = (g.Accuracy.mean(), g.F1_Score.mean(), g.AUC.mean(), g.MAE.mean())
    print(f"   {a:<22} acc={got[0]:.2f} f1={got[1]:.4f} auc={got[2]:.4f} mae={got[3]:.4f}")
    check(
        f"Summary table row {a}",
        abs(got[0] - acc) < 0.006
        and round(got[1], 2) == f1
        and round(got[2], 2) == auc
        and round(got[3], 2) == mae,
    )
mae_red = 1.0 - (
    df[(df.Algorithm == "PQ-FL (Proposed)") & (df.Round == last)].MAE.mean()
    / df[(df.Algorithm == "DP-FedAvg") & (df.Round == last)].MAE.mean()
)
print(f"   MAE reduction PQ-FL vs DP-FedAvg (unrounded) = {100.0 * mae_red:.1f} %")
check("MAE reduction ~ 49.5 %", abs(100.0 * mae_red - 49.5) < 0.2)

print("\n4. Convergence milestone & noiseless cryptographic coincidence")
mean_curve = df[df.Algorithm == "PQ-FL (Proposed)"].groupby("Round")["Accuracy"].mean()
r85 = int(mean_curve[mean_curve >= 85.0].index.min())
r90 = int(mean_curve[mean_curve >= 90.0].index.min())
print(f"   PQ-FL mean accuracy first reaches 85 %% at round {r85}, 90 %% at round {r90}")
check("PQ-FL first reaches 85 % at round 9", r85 == 9)

base = df[df.Algorithm == "Vanilla FedAvg"].sort_values(["Seed", "Round"])["Accuracy"].values
for a in ["FL + Secure Aggregation", "FL + PQ Encryption Only"]:
    cur = df[df.Algorithm == a].sort_values(["Seed", "Round"])["Accuracy"].values
    max_diff = float(np.max(np.abs(cur - base)))
    check(f"Final-round coincidence {a} == Vanilla FedAvg", np.allclose(final[a], final["Vanilla FedAvg"], atol=1e-9),
          f"(max round |diff| = {max_diff:.2f} pp)")

print("\n5. Privacy Fairness Index & Seed Stability (Privacy_Fairness_Table.csv)")
pf_csv = os.path.join(RESULTS_DIR, "Privacy_Fairness_Table.csv")
if os.path.exists(pf_csv):
    pf = pd.read_csv(pf_csv).set_index("Algorithm")
    stds = {a: float(final[a].std(ddof=1)) for a in algos}
    s_max = max(stds.values())
    for a in algos:
        r_calc = round(1.0 + 9.0 * (1.0 - stds[a] / s_max), 2)
        pfi_calc = round((final[a].mean() / 100.0) * float(pf.loc[a, "Privacy Score"]), 3)
        check(
            f"PFI & Seed Stability of {a}",
            abs(float(pf.loc[a, "Seed Stability Score"]) - r_calc) <= 0.01
            and abs(float(pf.loc[a, "Privacy Fairness Index"]) - pfi_calc) <= 0.002,
            f"(s_j={stds[a]:.3f}%, R_j={r_calc:.2f}, PFI={pfi_calc:.3f})",
        )

print("\n6. Cryptographic payload & server aggregation speedup (Cryptographic_Overhead_Metrics.csv)")
ov_csv = os.path.join(RESULTS_DIR, "Cryptographic_Overhead_Metrics.csv")
if os.path.exists(ov_csv):
    ov = pd.read_csv(ov_csv).set_index("Algorithm")
    for a, b_ref in MANUSCRIPT_PAYLOAD_BYTES.items():
        check(f"Payload bytes of {a}", int(ov.loc[a, "Payload (Bytes)"]) == b_ref, f"({b_ref} B)")
    speedup = float(ov.loc["FL + PQ Encryption Only", "Server Aggregation Overhead (ms)"]) / float(
        ov.loc["PQ-FL (Proposed)", "Server Aggregation Overhead (ms)"]
    )
    print(f"   Homomorphic server speedup (PQ-Only / PQ-FL) = {speedup:.2f}x")
    check("Homomorphic server speedup > 8x", speedup > 8.0)

ab_csv = os.path.join(RESULTS_DIR, "Privacy_Matched_Ablation.csv")
if os.path.exists(ab_csv):
    print("\n7. Multi-budget & privacy-matched sweep (Privacy_Matched_Ablation.csv)")
    ab = pd.read_csv(ab_csv)
    for _, row in ab.iterrows():
        arr = np.array(ast.literal_eval(row["Per_Seed_Acc"]), dtype=float)
        check(
            f"Ablation mean/SD {row['Configuration']}",
            abs(arr.mean() - float(row["Accuracy_Mean"])) < 0.011
            and abs(arr.std(ddof=1) - float(row["Accuracy_SD"])) < 0.011,
            f"(Mean={arr.mean():.2f}% +/- {arr.std(ddof=1):.2f}%, eps_RDP={row['Eps_Agg_RDP']})",
        )

dig_csv = os.path.join(RESULTS_DIR, "Public_Benchmark_UCI_Digits.csv")
if os.path.exists(dig_csv):
    print("\n8. Public UCI Handwritten Digits benchmark (Public_Benchmark_UCI_Digits.csv)")
    dg = pd.read_csv(dig_csv)
    for _, row in dg.iterrows():
        arr = np.array(ast.literal_eval(row["Per_Seed_Acc"]), dtype=float)
        check(
            f"UCI Digits mean/SD {row['Algorithm']}",
            abs(arr.mean() - float(row["Accuracy_Mean"])) < 0.011
            and abs(arr.std(ddof=1) - float(row["Accuracy_SD"])) < 0.011,
            f"(Mean={arr.mean():.2f}% +/- {arr.std(ddof=1):.2f}%)",
        )

fem_csv = os.path.join(RESULTS_DIR, "Public_Benchmark_LEAF_FEMNIST.csv")
fem_raw = os.path.join(ROOT, "Supplementary_Data_Sheet", "FEMNIST_Raw_Metrics.csv")
if os.path.exists(fem_csv) and os.path.exists(fem_raw):
    print("\n9. Public LEAF FEMNIST benchmark (Public_Benchmark_LEAF_FEMNIST.csv vs FEMNIST_Raw_Metrics.csv)")
    fm = pd.read_csv(fem_csv)
    fr = pd.read_csv(fem_raw)
    f_last = int(fr["Round"].max())
    f_final = {
        a: fr[(fr.Algorithm == a) & (fr.Round == f_last)].sort_values("Seed")["Accuracy"].values
        for a in fm["Algorithm"]
    }
    for _, row in fm.iterrows():
        arr = np.array(ast.literal_eval(row["Per_Seed_Acc"]), dtype=float)
        raw = f_final[row["Algorithm"]]
        check(
            f"FEMNIST mean/SD/raw {row['Algorithm']}",
            np.allclose(arr, raw, atol=0.006)
            and abs(raw.mean() - float(row["Accuracy_Mean"])) < 0.011
            and abs(raw.std(ddof=1) - float(row["Accuracy_SD"])) < 0.011
            and int(row["Crypto_Mismatches"]) == 0,
            f"(Mean={raw.mean():.2f}% +/- {raw.std(ddof=1):.2f}%)",
        )
    for a, b in [("PQ-FL (Proposed)", "DP-FedAvg"), ("PQ-FL (sigma=sqrt(10))", "DP-FedAvg"),
                 ("PQ-FL (Proposed)", "FL + Distributed Noise")]:
        t, p = stats.ttest_rel(f_final[a], f_final[b])
        diff = f_final[a] - f_final[b]
        print(f"   {a} vs {b}: delta={diff.mean():+.2f} pp  t(4)={t:.2f}  p={p:.3e}  dz={diff.mean() / diff.std(ddof=1):+.2f}")

print("\nRESULT:", "ALL MANUSCRIPT VALUES REPRODUCED" if not failures else f"{len(failures)} MISMATCH(ES): {failures}")
sys.exit(1 if failures else 0)
