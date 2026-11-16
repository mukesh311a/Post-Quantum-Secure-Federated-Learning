import os
from typing import Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as _st


class Visualizer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
        self.styles = {
            "PQ-FL (Proposed)":        {"color": "#d62728", "linewidth": 2.5, "linestyle": "-"},
            "PQ-FL (sigma=sqrt(10))":  {"color": "#17becf", "linewidth": 2.0, "linestyle": "-."},
            "Vanilla FedAvg":          {"color": "#1f77b4", "linewidth": 1.6, "linestyle": "--"},
            "FedProx":                 {"color": "#2ca02c", "linewidth": 1.6, "linestyle": "-."},
            "SCAFFOLD":                {"color": "#9467bd", "linewidth": 1.6, "linestyle": ":"},
            "DP-FedAvg":               {"color": "#ff7f0e", "linewidth": 1.8, "linestyle": "-"},
            "FL + Distributed Noise":  {"color": "#8c564b", "linewidth": 1.6, "linestyle": "--"},
            "FL + Secure Aggregation": {"color": "#e377c2", "linewidth": 1.6, "linestyle": "-."},
            "FL + PQ Encryption Only": {"color": "#7f7f7f", "linewidth": 1.6, "linestyle": ":"},
        }

    def _plot_metric_with_ci(
        self,
        metrics_data: Dict,
        metric_key: str,
        title: str,
        ylabel: str,
        filename: str,
        n_seeds: int = 5,
    ):
        t_crit = float(_st.t.ppf(0.975, df=n_seeds - 1))
        plt.figure(figsize=(9.5, 6), dpi=300)
        for algo, data in metrics_data.items():
            rounds = data["rounds"]
            mean_vals = np.array(data[metric_key]["mean"])
            half_width = t_crit * np.array(data[metric_key]["std"]) / np.sqrt(n_seeds)
            style = self.styles.get(algo, {"color": "gray", "linewidth": 1.5, "linestyle": "-"})
            c = style["color"]
            ls = style.get("linestyle", "-")
            lw = style.get("linewidth", 1.5)
            plt.plot(rounds, mean_vals, label=algo, color=c, linestyle=ls, linewidth=lw)
            plt.fill_between(rounds, mean_vals - half_width, mean_vals + half_width, color=c, alpha=0.15)

        plt.title(title, fontsize=14.5, fontweight="bold", pad=14)
        plt.xlabel("Communication Rounds", fontsize=13, fontweight="bold")
        plt.ylabel(ylabel, fontsize=13, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(fontsize=10.5, frameon=True, fancybox=True, loc="lower right" if metric_key != "mae" else "upper right")
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close("all")
        print(f"[SUCCESS] Plot saved: {save_path}")

    def plot_all_curves(self, metrics_summary: Dict, n_seeds: int = 5):
        self._plot_metric_with_ci(
            metrics_summary,
            "accuracy",
            "Test Accuracy over Communication Rounds (Mean & 95% Student-t CI, 5 Seeds)",
            "Test Accuracy (%)",
            "comparison_accuracy.png",
            n_seeds,
        )
        self._plot_metric_with_ci(
            metrics_summary,
            "f1",
            "Macro F1-Score over Communication Rounds (Mean & 95% Student-t CI)",
            "Macro F1-Score",
            "comparison_f1.png",
            n_seeds,
        )
        self._plot_metric_with_ci(
            metrics_summary,
            "auc",
            "Macro AUC-ROC over Communication Rounds (Mean & 95% Student-t CI)",
            "Macro AUC-ROC",
            "comparison_auc.png",
            n_seeds,
        )
        self._plot_metric_with_ci(
            metrics_summary,
            "mae",
            "Mean Absolute Error over Communication Rounds (Mean & 95% Student-t CI)",
            "Mean Absolute Error",
            "comparison_mae.png",
            n_seeds,
        )

    def plot_public_digits(self, digits_summary: Dict, n_seeds: int = 5):
        self._plot_metric_with_ci(
            digits_summary,
            "accuracy",
            "UCI Handwritten Digits Benchmark (Non-IID Dirichlet alpha=0.5, Mean & 95% t-CI)",
            "Test Accuracy (%)",
            "public_digits_accuracy.png",
            n_seeds,
        )

    def plot_femnist(self, femnist_summary: Dict, n_seeds: int = 5):
        self._plot_metric_with_ci(
            femnist_summary,
            "accuracy",
            "LEAF FEMNIST (62 Classes, Writer-Partitioned, N=100, Mean & 95% t-CI)",
            "Test Accuracy (%)",
            "femnist_accuracy.png",
            n_seeds,
        )

    def plot_radar(self, radar_data: Dict):
        labels = radar_data["labels"]
        num_vars = len(labels)
        angle_vals = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angle_vals += angle_vals[:1]

        fig, ax = plt.subplots(figsize=(8.6, 8.6), subplot_kw=dict(polar=True), dpi=300)
        plt.xticks(angle_vals[:-1], labels, size=11, fontweight="bold")
        ax.set_rlabel_position(0)
        plt.yticks([2, 4, 6, 8, 10], ["2", "4", "6", "8", "10"], color="grey", size=10)
        plt.ylim(0, 10)

        for algo, values in radar_data["data"].items():
            vals = values + values[:1]
            style = self.styles.get(algo, {"color": "gray"})
            c = style["color"]
            if algo == "PQ-FL (Proposed)":
                ax.plot(angle_vals, vals, linewidth=2.5, linestyle="-", label=algo, color=c, zorder=5)
                ax.fill(angle_vals, vals, color=c, alpha=0.25)
            else:
                ax.plot(angle_vals, vals, linewidth=1.5, linestyle="--", label=algo, color=c, alpha=0.7)

        plt.title("Normalized Multi-Dimensional Profile (Measured & Ordinal Descriptors)", size=14.5, fontweight="bold", y=1.1)
        plt.legend(loc="upper right", bbox_to_anchor=(1.28, 1.1), fontsize=9.8)
        save_path = os.path.join(self.output_dir, "comparison_radar.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close("all")
        print(f"[SUCCESS] Radar chart saved: {save_path}")

    def plot_tradeoff(self, metrics_summary: Dict, radar_data: Dict):
        plt.figure(figsize=(10.5, 6.5), dpi=300)
        x_priv, y_acc, sizes, colors, labels = [], [], [], [], []
        radar_map = radar_data["data"]

        for algo, data in metrics_summary.items():
            if algo not in radar_map:
                continue
            final_acc = data["accuracy"]["mean"][-1]
            priv_score = radar_map[algo][1]
            comp_speed = radar_map[algo][4]
            x_priv.append(priv_score)
            y_acc.append(final_acc)
            sizes.append(max(comp_speed, 1.0) * 110)
            colors.append(self.styles.get(algo, {"color": "black"})["color"])
            labels.append(algo)

        plt.scatter(x_priv, y_acc, s=sizes, c=colors, alpha=0.75, edgecolors="black", linewidth=1.5)
        plt.axvline(x=5.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        plt.axhline(y=92.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)

        plt.text(
            8.3,
            max(y_acc) + 2.4,
            "Multi-Channel Protection & High Utility Region",
            fontsize=9.8,
            fontweight="bold",
            color="darkgreen",
            ha="center",
            bbox=dict(facecolor="#e6f5e6", edgecolor="green", boxstyle="round,pad=0.4"),
        )

        offsets = {
            "SCAFFOLD": (26, 16),
            "Vanilla FedAvg": (58, 0),
            "FedProx": (26, -20),
            "FL + PQ Encryption Only": (0, -22),
            "FL + Secure Aggregation": (0, 15),
            "DP-FedAvg": (0, -22),
            "FL + Distributed Noise": (0, -22),
            "PQ-FL (Proposed)": (0, -22),
        }
        for i, txt in enumerate(labels):
            dx, dy = offsets.get(txt, (0, 12))
            plt.annotate(
                txt,
                (x_priv[i], y_acc[i]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="center",
                fontsize=8.8,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.78),
            )

        plt.title("Descriptive Protection-Coverage vs. Test Accuracy Profile", fontsize=14.5, fontweight="bold", pad=14)
        plt.xlabel("Author-Defined Protection-Channel Rubric (Ordinal, 1-10)", fontsize=12.5, fontweight="bold")
        plt.ylabel("Final Test Accuracy (%)", fontsize=12.5, fontweight="bold")
        plt.xlim(-0.3, 11.2)
        plt.ylim(min(y_acc) - 4.5, max(y_acc) + 4.0)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "comparison_tradeoff.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close("all")
        print(f"[SUCCESS] Tradeoff plot saved: {save_path}")
