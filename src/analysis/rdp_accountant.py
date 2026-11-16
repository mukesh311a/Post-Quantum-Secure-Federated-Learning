import math
import os
from typing import Dict

from src.privacy.accounting import (
    dp_accounting_version,
    epsilon_pld,
    epsilon_rdp,
    epsilon_rdp_with_order,
    gradient_sigma_for_adaptive_clipping,
    verify_joint_composition_rdp,
)


class FormalPrivacyAccountant:
    def __init__(
        self,
        num_rounds: int = 200,
        sampling_rate: float = 0.1,
        delta: float = 1e-5,
        sigma_b_factor: float = 5.0,
    ):
        self.T = num_rounds
        self.q = sampling_rate
        self.delta = delta
        self.sigma_b_factor = sigma_b_factor

    def compute_pqfl(self, sigma: float = 1.0) -> Dict[str, float]:
        sigma_b = self.sigma_b_factor * sigma
        sigma_g = gradient_sigma_for_adaptive_clipping(sigma, sigma_b)
        eps_rdp, opt_order = epsilon_rdp_with_order(sigma, self.q, self.T, self.delta)
        eps_pld = epsilon_pld(sigma, self.q, self.T, self.delta)
        return {
            "sigma": sigma,
            "sigma_g": sigma_g,
            "sigma_b": sigma_b,
            "eps_rdp": eps_rdp,
            "opt_order": opt_order,
            "eps_pld": eps_pld,
        }

    def compute_dp_fedavg(self, sigma_client: float = 1.0, n_selected: int = 10) -> Dict[str, float]:
        sigma_eff = sigma_client * math.sqrt(n_selected)
        eps_rdp, opt_order = epsilon_rdp_with_order(sigma_eff, self.q, self.T, self.delta)
        eps_pld = epsilon_pld(sigma_eff, self.q, self.T, self.delta)
        eps_server_rdp = epsilon_rdp(sigma_client, self.q, self.T, self.delta)
        eps_server_pld = epsilon_pld(sigma_client, self.q, self.T, self.delta)
        return {
            "sigma_client": sigma_client,
            "sigma_eff": sigma_eff,
            "eps_rdp": eps_rdp,
            "opt_order": opt_order,
            "eps_pld": eps_pld,
            "eps_server_rdp": eps_server_rdp,
            "eps_server_pld": eps_server_pld,
        }

    def generate_report(self, output_dir: str = None) -> str:
        ver = dp_accounting_version()
        res_pqfl = self.compute_pqfl(sigma=1.0)
        res_dp = self.compute_dp_fedavg(sigma_client=1.0, n_selected=10)
        res_sig2 = self.compute_pqfl(sigma=2.0)
        res_matched = self.compute_pqfl(sigma=math.sqrt(10.0))
        res_sig4 = self.compute_pqfl(sigma=4.0)
        e_single, e_comp = verify_joint_composition_rdp(1.0, 5.0, self.q, self.T, self.delta)

        eps_unmasked_view_rdp = epsilon_rdp(1.0 / math.sqrt(10.0), self.q, self.T, self.delta)

        lines = [
            "=" * 88,
            f"  Differential Privacy Accounting Report (Google dp-accounting v{ver})",
            "=" * 88,
            "Protocol & Accounting Configuration:",
            f"  - Adjacency relation          : Add/remove one client's complete local dataset",
            f"  - Subsampling model           : Poisson subsampling (q = {self.q:.2f}, |S_t|=10, N=100)",
            f"  - Communication rounds (T)    : {self.T}",
            f"  - Target failure prob (delta) : {self.delta:.1e}",
            f"  - Adaptive clipping budget    : Joint (g_tilde, b_tilde) release per Andrew et al. (2021),",
            f"                                  sigma_b = {self.sigma_b_factor:.1f} * sigma, "
            f"sigma_g = (sigma^-2 - sigma_b^-2)^(-1/2)",
            f"  - Joint composition check     : single-event RDP = {e_single:.12f} vs "
            f"composed-event RDP = {e_comp:.12f} (|diff| = {abs(e_single - e_comp):.2e})",
            "-" * 88,
            "[1] PQ-FL (Distributed Noise + Private Adaptive Quantile Clipping + SecAgg):",
            f"    - Target joint sigma        : {res_pqfl['sigma']:.4f} "
            f"(sigma_g = {res_pqfl['sigma_g']:.6f}, sigma_b = {res_pqfl['sigma_b']:.2f})",
            f"    - RDP composed epsilon      : {res_pqfl['eps_rdp']:.3f} "
            f"(optimal order alpha* = {res_pqfl['opt_order']:.2f}, approx. {res_pqfl['eps_rdp']:.1f})",
            f"    - PLD composed epsilon      : {res_pqfl['eps_pld']:.3f} (approx. {res_pqfl['eps_pld']:.1f})",
            f"    - Note (FL + Dist. Noise without SecAgg): aggregate release has eps_RDP = {res_pqfl['eps_rdp']:.3f},",
            f"      but an honest-but-curious server observing raw shares (sigma_view = 1/sqrt(10) = 0.3162)",
            f"      faces weak server-view privacy loss eps_RDP(server) = {eps_unmasked_view_rdp:.2f}.\n",
            "[2] DP-FedAvg (Independent Per-Client Local Noise, sigma_local = 1.0):",
            f"    - Per-client local sigma    : {res_dp['sigma_client']:.4f} "
            f"-> server-view eps_RDP = {res_dp['eps_server_rdp']:.3f} (eps_PLD = {res_dp['eps_server_pld']:.3f})",
            f"    - Aggregate effective sigma : {res_dp['sigma_eff']:.4f} (sqrt(10) * sigma_local)",
            f"    - Aggregate RDP epsilon     : {res_dp['eps_rdp']:.3f} "
            f"(optimal order alpha* = {res_dp['opt_order']:.2f}, approx. {res_dp['eps_rdp']:.2f})",
            f"    - Aggregate PLD epsilon     : {res_dp['eps_pld']:.3f} (approx. {res_dp['eps_pld']:.2f})\n",
            "[3] Multi-Budget & Privacy-Matched Sensitivity Table (T=200, q=0.10, delta=1e-5):",
            "    | Configuration / Noise Multiplier         | sigma_g | sigma_b | RDP Eps | PLD Eps |",
            "    |------------------------------------------|---------|---------|---------|---------|",
            f"    | PQ-FL (sigma = 1.000, server-matched)    | {res_pqfl['sigma_g']:7.4f} | {res_pqfl['sigma_b']:7.2f} "
            f"| {res_pqfl['eps_rdp']:7.3f} | {res_pqfl['eps_pld']:7.3f} |",
            f"    | PQ-FL (sigma = 2.000, moderate budget)   | {res_sig2['sigma_g']:7.4f} | {res_sig2['sigma_b']:7.2f} "
            f"| {res_sig2['eps_rdp']:7.3f} | {res_sig2['eps_pld']:7.3f} |",
            f"    | PQ-FL (sigma = sqrt(10), release-matched)| {res_matched['sigma_g']:7.4f} | {res_matched['sigma_b']:7.2f} "
            f"| {res_matched['eps_rdp']:7.3f} | {res_matched['eps_pld']:7.3f} |",
            f"    | DP-FedAvg (sigma_local = 1.000)          | {res_pqfl['sigma_g']:7.4f} | {res_pqfl['sigma_b']:7.2f} "
            f"| {res_dp['eps_rdp']:7.3f} | {res_dp['eps_pld']:7.3f} |",
            f"    | PQ-FL (sigma = 4.000, tightest budget)   | {res_sig4['sigma_g']:7.4f} | {res_sig4['sigma_b']:7.2f} "
            f"| {res_sig4['eps_rdp']:7.3f} | {res_sig4['eps_pld']:7.3f} |",
            "=" * 88,
        ]

        report_str = "\n".join(lines)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            report_path = os.path.join(output_dir, "privacy_budget_report.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_str + "\n")
            print(f"[SUCCESS] Formal Privacy Budget Report saved to {report_path}")
        return report_str


def run_privacy_accounting(output_dir: str = "results") -> Dict[str, float]:
    accountant = FormalPrivacyAccountant(num_rounds=200, sampling_rate=0.1, delta=1e-5)
    report = accountant.generate_report(output_dir)
    print(report)
    pqfl_res = accountant.compute_pqfl(sigma=1.0)
    dp_res = accountant.compute_dp_fedavg(sigma_client=1.0)
    return {
        "epsilon_pqfl_rdp": pqfl_res["eps_rdp"],
        "epsilon_pqfl_pld": pqfl_res["eps_pld"],
        "epsilon_dpfedavg_rdp": dp_res["eps_rdp"],
        "epsilon_dpfedavg_pld": dp_res["eps_pld"],
    }


if __name__ == "__main__":
    run_privacy_accounting()
