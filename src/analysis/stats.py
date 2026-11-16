import numpy as np
from scipy import stats
from typing import Dict, List, Tuple


class SeedAlignedStatsAnalyzer:
    def __init__(self, final_accuracies_by_algo: Dict[str, List[float]]):
        self.data = final_accuracies_by_algo

    def paired_compare(self, algo_a: str, algo_b: str) -> Dict:
        acc_a = np.array(self.data[algo_a], dtype=np.float64)
        acc_b = np.array(self.data[algo_b], dtype=np.float64)
        if len(acc_a) != len(acc_b):
            raise ValueError(f"Length mismatch between {algo_a} ({len(acc_a)}) and {algo_b} ({len(acc_b)})")

        diff = acc_a - acc_b
        n = len(diff)
        df = n - 1
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1))

        if std_diff < 1e-12:
            t_stat = 0.0 if abs(mean_diff) < 1e-12 else float('inf') * np.sign(mean_diff)
            p_val = 1.0 if abs(mean_diff) < 1e-12 else 0.0
            d_z = 0.0
        else:
            t_stat = mean_diff / (std_diff / np.sqrt(n))
            p_val = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df))
            d_z = mean_diff / std_diff

        return {
            "algo_a": algo_a,
            "algo_b": algo_b,
            "mean_a": float(np.mean(acc_a)),
            "mean_b": float(np.mean(acc_b)),
            "delta_acc": mean_diff,
            "df": df,
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "cohens_dz": float(d_z),
            "significant": bool(p_val < 0.05)
        }

    def generate_report(self, comparisons: List[Tuple[str, str]]) -> str:
        lines = []
        lines.append("=" * 80)
        lines.append("   Statistical Significance Analysis (Seed-Aligned Paired t-tests)   ")
        lines.append("=" * 80)
        lines.append(f"{'Comparison':<36} | {'Delta Acc':<10} | {'t(df)':<10} | {'p-value':<12} | {'Cohen dz':<8}")
        lines.append("-" * 80)

        for a, b in comparisons:
            res = self.paired_compare(a, b)
            sig_marker = "***" if res["p_value"] < 0.001 else "**" if res["p_value"] < 0.01 else "*" if res["p_value"] < 0.05 else "ns"
            lines.append(
                f"{a + ' vs ' + b:<36} | "
                f"{res['delta_acc']:+6.2f} pp  | "
                f"t({res['df']})={res['t_statistic']:6.2f} | "
                f"{res['p_value']:10.3e} {sig_marker:<2} | "
                f"{res['cohens_dz']:7.2f}"
            )
        lines.append("=" * 80)
        lines.append("Note: * p < 0.05, ** p < 0.01, *** p < 0.001, ns = not statistically significant.")
        return "\n".join(lines)
