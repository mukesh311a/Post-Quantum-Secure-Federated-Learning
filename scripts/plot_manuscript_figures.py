import csv
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results")
OUT = sys.argv[2] if len(sys.argv) > 2 else SRC
os.makedirs(OUT, exist_ok=True)
DPI = 600
META = {"Author": "Rakesh Badoni", "Software": None}

sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

STYLES = {
    "PQ-FL (Proposed)": {"color": "#d62728", "linewidth": 2.5, "linestyle": "-"},
    "PQ-FL (sigma=sqrt(10))": {"color": "#17becf", "linewidth": 2.0, "linestyle": "-."},
    "Vanilla FedAvg": {"color": "#1f77b4", "linewidth": 1.6, "linestyle": "--"},
    "FedProx": {"color": "#2ca02c", "linewidth": 1.6, "linestyle": "-."},
    "SCAFFOLD": {"color": "#9467bd", "linewidth": 1.6, "linestyle": ":"},
    "DP-FedAvg": {"color": "#ff7f0e", "linewidth": 1.8, "linestyle": "-"},
    "FL + Distributed Noise": {"color": "#8c564b", "linewidth": 1.6, "linestyle": "--"},
    "FL + Secure Aggregation": {"color": "#e377c2", "linewidth": 1.6, "linestyle": "-."},
    "FL + PQ Encryption Only": {"color": "#7f7f7f", "linewidth": 1.6, "linestyle": ":"},
}

LEGEND_NAMES = {
    "PQ-FL (Proposed)": r"PQ-FL (Proposed, $\sigma=1.0$)",
    "PQ-FL (sigma=sqrt(10))": r"PQ-FL ($\sigma=\sqrt{10}$)",
}


def accuracy_curves(data_file, title, filename, legend_kw):
    with open(os.path.join(SRC, data_file), encoding="utf-8") as fh:
        data = json.load(fh)
    n = 5
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    plt.figure(figsize=(9.5, 6))
    for algo, d in data.items():
        r = d["rounds"]
        m = np.array(d["accuracy"]["mean"])
        h = t_crit * np.array(d["accuracy"]["std"]) / np.sqrt(n)
        s = STYLES[algo]
        plt.plot(r, m, label=LEGEND_NAMES.get(algo, algo), color=s["color"],
                 linestyle=s["linestyle"], linewidth=s["linewidth"])
        plt.fill_between(r, m - h, m + h, color=s["color"], alpha=0.15)
    plt.title(title, fontsize=14.5, fontweight="bold", pad=14)
    plt.xlabel("Communication Rounds", fontsize=13, fontweight="bold")
    plt.ylabel("Test Accuracy (%)", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=10.5, frameon=True, fancybox=True, **legend_kw)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, filename), dpi=DPI, bbox_inches="tight", metadata=META)
    plt.close("all")


def radar():
    with open(os.path.join(SRC, "radar_metrics.json"), encoding="utf-8") as fh:
        rd = json.load(fh)
    labels = rd["labels"]
    k = len(labels)
    ang = np.linspace(0, 2 * np.pi, k, endpoint=False).tolist()
    closed = ang + ang[:1]
    fig, ax = plt.subplots(figsize=(8.6, 8.6), subplot_kw=dict(polar=True))
    ax.set_xticks(ang)
    ax.set_xticklabels([])
    for a, lab in zip(ang, labels):
        c = np.cos(a)
        ha = "left" if c > 0.3 else ("right" if c < -0.3 else "center")
        s = np.sin(a)
        va = "bottom" if s > 0.3 else ("top" if s < -0.3 else "center")
        ax.text(a, 10.75, lab, ha=ha, va=va, fontsize=11, fontweight="bold")
    ax.set_rlabel_position(36)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], color="grey", fontsize=10)
    ax.set_ylim(0, 10)
    for algo, v in rd["data"].items():
        vals = v + v[:1]
        c = STYLES[algo]["color"]
        if algo == "PQ-FL (Proposed)":
            ax.plot(closed, vals, linewidth=2.5, linestyle="-", label=algo, color=c, zorder=5)
            ax.fill(closed, vals, color=c, alpha=0.25)
        else:
            ax.plot(closed, vals, linewidth=1.5, linestyle="--", label=algo, color=c, alpha=0.7)
    plt.title("Normalized Multi-Dimensional Profile (Measured & Ordinal Descriptors)",
              size=14.5, fontweight="bold", y=1.1)
    plt.legend(loc="upper right", bbox_to_anchor=(1.28, 1.1), fontsize=9.8)
    plt.savefig(os.path.join(OUT, "comparison_radar.png"), dpi=DPI, bbox_inches="tight", metadata=META)
    plt.close("all")


def median_per_pair_us():
    with open(os.path.join(SRC, "scaling_report.txt"), encoding="utf-8") as fh:
        text = fh.read()
    m = re.search(r"N_R=256,.*?median=([0-9.]+)\s*us", text)
    if m is None:
        raise RuntimeError("Median per-pair cost for N_R=256 not found in scaling_report.txt")
    return float(m.group(1))


def key_scaling():
    with open(os.path.join(SRC, "scaling_analysis.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    N = np.array([int(r["N"]) for r in rows])
    P = np.array([int(r["Pairs"]) for r in rows])
    T = np.array([float(r["Time_s"]) for r in rows])
    meas = np.array([r["Source"] == "Measured" for r in rows])
    per_pair_us = median_per_pair_us()

    c_line = "#1D4ED8"
    c_point = "#2563EB"
    c_pair = "#DC2626"
    c_pair_face = "#991B1B"
    c_theory = "#6B7280"

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.8, 4.9))
    comma = mticker.FuncFormatter(lambda v, _: f"{int(v):,}" if v >= 1 else f"{v:.1f}")

    def style_axes(ax):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(N)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.yaxis.set_major_formatter(comma)
        ax.tick_params(axis="x", labelrotation=35, labelsize=11)
        ax.tick_params(axis="y", labelsize=11)
        ax.set_xlim(80, 13000)
        ax.grid(True, which="major", linestyle="--", alpha=0.7)
        ax.grid(False, which="minor")
        ax.set_xlabel("Number of Clients $N$", fontsize=13, fontweight="bold")

    nt = np.geomspace(N[0], N[-1], 200)
    a1.plot(nt, nt * (nt - 1) / 2 * per_pair_us * 1e-6, "--", color=c_theory, lw=1.4, zorder=1,
            label=rf"$\mathcal{{O}}(N^2)$ reference (${per_pair_us:.2f}\ \mu$s/pair)")
    a1.plot(N[meas], T[meas], "o-", color=c_line, lw=2.4, ms=8, markerfacecolor=c_point,
            markeredgecolor="white", mew=1.3, zorder=4, label=r"Measured ($N \leq 500$)")
    ie = np.where(~meas)[0]
    a1.plot(np.concatenate([[N[meas][-1]], N[ie]]), np.concatenate([[T[meas][-1]], T[ie]]), "D:",
            color=c_line, lw=2.0, ms=7.5, markerfacecolor="white", markeredgecolor=c_line, mew=1.8,
            zorder=3, markevery=list(range(1, len(ie) + 1)), label=r"Extrapolated ($N \geq 1{,}000$)")
    style_axes(a1)
    a1.set_ylim(1, 1e5)
    a1.set_ylabel("Setup Time (s)", fontsize=13, fontweight="bold", color=c_line)
    a1.set_title("(a) Pairwise key setup time", fontsize=13, fontweight="bold", pad=10)
    a1.legend(fontsize=10, frameon=True, fancybox=True, loc="upper left")

    a2.plot(N, P, "s--", color=c_pair, lw=2.0, ms=7.5, markerfacecolor=c_pair_face,
            markeredgecolor="white", mew=1.3, label=r"$\binom{N}{2}$ client pairs")
    style_axes(a2)
    a2.set_ylim(1e3, 1e8)
    a2.set_ylabel(r"Number of Pairs $\binom{N}{2}$", fontsize=13, fontweight="bold", color=c_pair)
    a2.set_title("(b) Number of client pairs", fontsize=13, fontweight="bold", pad=10)
    a2.legend(fontsize=10, frameon=True, fancybox=True, loc="upper left")

    fig.suptitle(r"Pairwise Key Establishment Scaling (Ring-LWE, $N_R=256$, $q=12{,}289$)",
                 fontsize=14.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "scaling_combined.png"), dpi=DPI, bbox_inches="tight", metadata=META)
    plt.close(fig)


if __name__ == "__main__":
    radar()
    accuracy_curves("public_digits_metrics.json",
                    "Test Accuracy on UCI Handwritten Digits (Mean & 95% Student-t CI, 5 Seeds)",
                    "public_digits_accuracy.png", {"loc": "lower right"})
    accuracy_curves("femnist_metrics.json",
                    "Test Accuracy on LEAF FEMNIST (Mean & 95% Student-t CI, 5 Seeds)",
                    "femnist_accuracy.png", {"loc": "center right", "bbox_to_anchor": (1.0, 0.6)})
    key_scaling()
    print("Figures 6, 8, 9, and 10 written to " + OUT)
