import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import ExperimentConfig


def analyze_key_setup_scalability():
    print('=========================================')
    print('   Scalability of Key Setup (O(N^2))     ')
    print('=========================================')
    for N in [100, 1000, 10000]:
        print(f'Clients (N={N}): {N * (N - 1) // 2:,} pairwise key exchanges.')
    print('Per-pair latency is measured (scaling_analysis.py -> results/scaling_analysis.csv); larger N are extrapolations.')
    print('-----------------------------------------\n')


def analyze_communication_overhead():
    from src.analysis.payload_audit import build_report
    print('=========================================')
    print('   Communication payload (from serialised messages)')
    print('=========================================')
    print(build_report())
    print('-----------------------------------------\n')


def analyze_dropout_variance(n=10, sigma=1.0, C=1.0, k_min=8, tau=0):
    print('=========================================')
    print('   Dropout Analysis (noise variance vs. survivors)')
    print('=========================================')
    target = (sigma * C) ** 2
    print(f'n={n}, k_min={k_min}, target variance (sigma C)^2 = {target:.2f}; ratios are relative to the target')
    print(f'{"survivors m":>12} {"equal-share m/n":>16} {"survivor-aware m/k_min":>24} {"honest only (tau=2)":>22}')
    for m in range(n, 4, -1):
        print(f'{m:>12} {m / n:>16.2f} {m / k_min:>24.2f} {max(m - 2, 0) / k_min:>22.2f}')
    print(f'Survivor-aware sum noise >= target iff m >= k_min = {k_min}; the round must abort when m < k_min.')
    print(f'Utility cost: with no dropout the noise is n/k_min = {n / k_min:.2f}x the target variance.')
    print('A variance >= target is necessary but NOT sufficient for a DP guarantee: it additionally requires dropout to be')
    print('independent of the data/noise (oblivious), unmasking of orphan masks (not implemented), a data-independent')
    print('release rule (sum or division by a public constant), and RDP accounting for the realised sampling.')
    print('-----------------------------------------\n')


def analyze_decryption_failure():
    from src.analysis.decryption_failure import analytic_failure
    cfg = ExperimentConfig()
    print('=========================================')
    print('   Decryption Failure (Gaussian-approx.)  ')
    print('=========================================')
    digits = 9540 * cfg.frac_bits
    for n_ring in (cfg.rlwe_n, 1024):
        coeffs = -(-digits // n_ring) * n_ring
        a = analytic_failure(n_ring, cfg.rlwe_q, cfg.rlwe_eta, cfg.rlwe_t, cfg.clients_per_round, coeffs, cfg.num_rounds)
        print(f'N_R={n_ring}: noise sd={a["sigma"]:.1f}, margin Delta/2={a["margin"]:.0f} ({a["z"]:.2f} sd), '
              f'P_fail/coeff={a["p_coeff"]:.2e}, per round={a["p_round"]:.2e}, per run={a["p_run"]:.2e}')
    print('See src/analysis/decryption_failure.py for the measured residual noise and decoding-error counts.')
    print('-----------------------------------------\n')


def analyze_rdp_privacy_budget():
    print('=========================================')
    print('   RDP / PLD Privacy Accounting (dp-accounting)')
    print('=========================================')
    from src.analysis.rdp_accountant import FormalPrivacyAccountant
    cfg = ExperimentConfig()
    accountant = FormalPrivacyAccountant(num_rounds=cfg.num_rounds, sampling_rate=cfg.sampling_rate, delta=cfg.delta)
    res_pqfl = accountant.compute_pqfl(sigma=1.0)
    res_dp = accountant.compute_dp_fedavg(sigma_client=1.0, n_selected=cfg.clients_per_round)
    print(f"PQ-FL      (sigma_sum=1.0,   T={cfg.num_rounds}, q={cfg.sampling_rate}, delta={cfg.delta}): "
          f"RDP eps = {res_pqfl['eps_rdp']:.3f}, PLD eps = {res_pqfl['eps_pld']:.3f}")
    print(f"DP-FedAvg  (sigma_sum={res_dp['sigma_eff']:.3f}):                      "
          f"RDP eps = {res_dp['eps_rdp']:.3f}, PLD eps = {res_dp['eps_pld']:.3f}")
    print('Scope: the adaptive clipping threshold C_t is treated as a public constant (its privacy cost is not accounted).')
    print('-----------------------------------------\n')


if __name__ == '__main__':
    analyze_key_setup_scalability()
    analyze_communication_overhead()
    analyze_dropout_variance()
    analyze_decryption_failure()
    analyze_rdp_privacy_budget()
