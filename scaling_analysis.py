import csv
import os
import time
import numpy as np

from src.crypto.rlwe import RLWE, RLWEParams


def benchmark_live_rlwe_exchange(
    n_ring: int = 256, q: int = 12289, eta: int = 2, t: int = 12, n_samples: int = 500, seed: int = 42
):
    rl = RLWE(RLWEParams(n_ring=n_ring, q=q, eta=eta, t=t))
    rng = np.random.default_rng(seed)

    for _ in range(5):
        rl.establish_pair_secret(rng)

    times_us = []
    mismatches = 0
    for _ in range(n_samples):
        t0 = time.perf_counter()
        sa, sb = rl.establish_pair_secret(rng)
        t1 = time.perf_counter()
        if sa != sb:
            mismatches += 1
        times_us.append((t1 - t0) * 1e6)

    arr = np.asarray(times_us, dtype=np.float64)
    return float(np.median(arr)), float(np.mean(arr)), float(np.std(arr, ddof=1)), mismatches


def measure_all_pairs(rl: RLWE, num_clients: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    mismatches = 0
    t0 = time.perf_counter()
    for k in range(num_clients):
        for v in range(k + 1, num_clients):
            sa, sb = rl.establish_pair_secret(rng)
            if sa != sb:
                mismatches += 1
    elapsed_s = time.perf_counter() - t0
    return elapsed_s, mismatches


def run_scaling_analysis(config=None):
    from src.config import ExperimentConfig
    if config is None:
        config = ExperimentConfig()

    output_dir = getattr(config, "output_dir", "results")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 78)
    print("  PQ-FL Pairwise Key Establishment Scalability Benchmark")
    print("=" * 78)
    print(f"Benchmarking live Ring-LWE key exchange (N_R={config.rlwe_n}, q={config.rlwe_q}, 500 trials)...")

    median_us, mean_us, std_us, mismatches = benchmark_live_rlwe_exchange(
        n_ring=config.rlwe_n, q=config.rlwe_q, eta=config.rlwe_eta, t=config.rlwe_t, n_samples=500, seed=42
    )
    med_1024_us, mean_1024_us, std_1024_us, mism_1024 = benchmark_live_rlwe_exchange(
        n_ring=1024, q=config.rlwe_q, eta=config.rlwe_eta, t=2, n_samples=500, seed=42
    )
    print(f"  Prototype (N_R={config.rlwe_n}, t={config.rlwe_t}): median = {median_us:.2f} us "
          f"(mean {mean_us:.2f} +/- {std_us:.2f} us), mismatches = {mismatches}/500")
    print(f"  Reference (N_R=1024, t=2) : median = {med_1024_us:.2f} us "
          f"(mean {mean_1024_us:.2f} +/- {std_1024_us:.2f} us), mismatches = {mism_1024}/500")

    rl = RLWE(RLWEParams(n_ring=config.rlwe_n, q=config.rlwe_q, eta=config.rlwe_eta, t=config.rlwe_t))
    ms_per_pair = median_us / 1000.0
    MEASURED_N = {100, 200, 500}
    N_VALUES = [100, 200, 500, 1000, 2000, 5000, 10000]
    rows = []
    for N in N_VALUES:
        pairs = N * (N - 1) // 2
        if N in MEASURED_N:
            total_s, mism_n = measure_all_pairs(rl, N, seed=42 + N)
            total_ms = total_s * 1000.0
            source = "Measured"
        else:
            total_ms = pairs * ms_per_pair
            total_s = total_ms / 1000.0
            mism_n = 0
            source = "Extrapolated"
        total_min = total_s / 60.0
        rows.append({
            "N": N,
            "Pairs": pairs,
            "Source": source,
            "Time_ms": round(total_ms, 1),
            "Time_s": round(total_s, 2),
            "Time_min": round(total_min, 2),
            "Mismatches": mism_n,
        })

    print("\n" + "=" * 88)
    print(f"{'N':>7}  {'C(N,2) Pairs':>13}  {'Source':>13}  {'Time (ms)':>13}  {'Time (s)':>11}  {'Time (min)':>11}")
    print("=" * 88)
    for r in rows:
        print(f"{r['N']:>7,}  {r['Pairs']:>13,}  {r['Source']:>13}  "
              f"{r['Time_ms']:>13,.1f}  {r['Time_s']:>11,.2f}  {r['Time_min']:>11.2f}")
    print("=" * 88)

    csv_path = os.path.join(output_dir, "scaling_analysis.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["N", "Pairs", "Source", "Time_ms", "Time_s", "Time_min", "Mismatches"]
        )
        writer.writeheader()
        writer.writerows(rows)

    rep_path = os.path.join(output_dir, "scaling_report.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(
            f"Per-pair Ring-LWE key exchange (500 trials):\n"
            f"  N_R={config.rlwe_n}, q={config.rlwe_q}, t={config.rlwe_t}: "
            f"median={median_us:.2f} us, mean={mean_us:.2f} us, std={std_us:.2f} us, mismatches={mismatches}\n"
            f"  N_R=1024, q={config.rlwe_q}, t=2: "
            f"median={med_1024_us:.2f} us, mean={mean_1024_us:.2f} us, std={std_1024_us:.2f} us, mismatches={mism_1024}\n"
        )
    print(f"\n[SUCCESS] Scalability CSV saved -> {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.linewidth": 1.2})
        C_LINE = "#1D4ED8"
        C_POINT = "#2563EB"
        C_PAIR = "#DC2626"
        C_THEORY = "#6B7280"
        C_GRID = "#E5E7EB"

        ns = [r["N"] for r in rows]
        times_s = [r["Time_s"] for r in rows]
        pairs_list = [r["Pairs"] for r in rows]
        meas_ns = [r["N"] for r in rows if r["Source"] == "Measured"]
        meas_ts = [r["Time_s"] for r in rows if r["Source"] == "Measured"]
        ext_ns = [r["N"] for r in rows if r["Source"] == "Extrapolated"]
        ext_ts = [r["Time_s"] for r in rows if r["Source"] == "Extrapolated"]

        n_theory = np.geomspace(ns[0], ns[-1], 200)
        t_theory = n_theory * (n_theory - 1) / 2 * ms_per_pair / 1000.0

        fig1, ax1 = plt.subplots(figsize=(7.2, 5.0), dpi=300)
        ax1.plot(n_theory, t_theory, "--", color=C_THEORY, lw=1.4, zorder=2,
                 label=rf"$\mathcal{{O}}(N^2)$ curve ({median_us:.1f} $\mu$s/pair)")
        ax1.plot(meas_ns, meas_ts, "o-", color=C_LINE, lw=2.2, ms=7.5, markerfacecolor=C_POINT,
                 markeredgecolor="white", mew=1.4, zorder=4, label=r"Measured end-to-end ($N \leq 500$)")
        ax1.plot([meas_ns[-1]] + ext_ns, [meas_ts[-1]] + ext_ts, "D:", color=C_LINE, lw=1.8, ms=6.5,
                 markerfacecolor="white", markeredgecolor=C_LINE, mew=1.8, zorder=3,
                 label=r"Extrapolated ($N \geq 1000$)")
        ax1.set_xlabel("Number of Clients ($N$)", fontsize=12)
        ax1.set_ylabel("Pairwise Key Establishment Time (seconds)", fontsize=12)
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xticks(ns)
        ax1.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax1.grid(True, which="major", color=C_GRID, lw=0.8)
        ax1.grid(True, which="minor", color="#F3F4F6", lw=0.5, ls=":")
        ax1.legend(fontsize=9.5, loc="upper left", framealpha=0.92)
        ax1.set_title(f"PQ-FL Ring-LWE Key Setup Scalability ($N_R={config.rlwe_n}, q={config.rlwe_q}$)",
                      fontsize=12, fontweight="bold")
        fig1.tight_layout()
        fig1.savefig(os.path.join(output_dir, "scaling_keytime.png"), dpi=300, bbox_inches="tight")
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(7.2, 5.0), dpi=300)
        ax2.plot(ns, pairs_list, "s-", color=C_PAIR, lw=2.2, ms=7, markerfacecolor="#EF4444",
                 markeredgecolor="white", mew=1.5, label=r"Pairwise exchanges $\binom{N}{2}$", zorder=3)
        ax2.set_xlabel("Number of Clients ($N$)", fontsize=12)
        ax2.set_ylabel(r"Number of Pairwise Key Exchanges $\binom{N}{2}$", fontsize=12)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xticks(ns)
        ax2.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax2.grid(True, which="major", color=C_GRID, lw=0.8)
        ax2.legend(fontsize=10, loc="upper left", framealpha=0.9)
        ax2.set_title(r"Pairwise Exchange Growth $\mathcal{O}(N^2)$ vs. Network Scale", fontsize=12, fontweight="bold")
        fig2.tight_layout()
        fig2.savefig(os.path.join(output_dir, "scaling_pairwise.png"), dpi=300, bbox_inches="tight")
        plt.close(fig2)

        fig3, ax3 = plt.subplots(figsize=(7.8, 5.0), dpi=300)
        ln_th, = ax3.plot(n_theory, t_theory, "--", color=C_THEORY, lw=1.3, zorder=1,
                          label=rf"$\mathcal{{O}}(N^2)$ ref. ({median_us:.1f} $\mu$s/pair)")
        ln_m, = ax3.plot(meas_ns, meas_ts, "o-", color=C_LINE, lw=2.2, ms=7.5, markerfacecolor=C_POINT,
                         markeredgecolor="white", mew=1.4, zorder=4, label=r"Measured setup time ($N \leq 500$)")
        ln_e, = ax3.plot([meas_ns[-1]] + ext_ns, [meas_ts[-1]] + ext_ts, "D:", color=C_LINE, lw=1.8, ms=6.5,
                         markerfacecolor="white", markeredgecolor=C_LINE, mew=1.8, zorder=3,
                         label=r"Extrapolated setup time ($N \geq 1000$)")
        ax3.set_xlabel("Number of Clients ($N$)", fontsize=12)
        ax3.set_ylabel("Pairwise Key Setup Time (s)", fontsize=12, color=C_LINE)
        ax3.tick_params(axis="y", labelcolor=C_LINE)
        ax3.set_xscale("log")
        ax3.set_yscale("log")
        ax3.set_xticks(ns)
        ax3.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax3.grid(True, color=C_GRID, lw=0.8)
        ax3b = ax3.twinx()
        ln_p, = ax3b.plot(ns, pairs_list, "s--", color=C_PAIR, lw=1.9, ms=6.5, markerfacecolor="#991B1B",
                          markeredgecolor="white", mew=1.4, label=r"$\binom{N}{2}$ pairs")
        ax3b.set_ylabel(r"Number of Pairwise Exchanges $\binom{N}{2}$", fontsize=12, color=C_PAIR)
        ax3b.tick_params(axis="y", labelcolor=C_PAIR)
        ax3b.set_yscale("log")
        lines = [ln_m, ln_e, ln_th, ln_p]
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, fontsize=9.2, loc="upper left", framealpha=0.92)
        ax3.set_title(
            f"PQ-FL Key Establishment Scaling Analysis\n"
            f"(Ring-LWE: $N_R={config.rlwe_n}$, $q={config.rlwe_q}$, Median {median_us:.2f} $\\mu$s/pair)",
            fontsize=11.5,
            fontweight="bold",
        )
        fig3.tight_layout()
        fig3.savefig(os.path.join(output_dir, "scaling_combined.png"), dpi=300, bbox_inches="tight")
        plt.close(fig3)
        print(f"[SUCCESS] Scaling plots saved to {output_dir}")

    except Exception as e:
        print(f"[WARNING] Matplotlib error: {e}")

    return rows, median_us, med_1024_us


if __name__ == "__main__":
    run_scaling_analysis()
