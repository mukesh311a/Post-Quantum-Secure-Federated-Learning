import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import ExperimentConfig
from src.crypto import encoding
from src.crypto.rlwe import RLWE, RLWEParams
from src.data.public_digits import DIGITS_DIMS
from src.data.leaf_femnist import FEMNIST_DIMS
from src.models.mlp import DEFAULT_DIMS, num_parameters


def build_report(seed: int = 7) -> str:
    cfg = ExperimentConfig()
    d = num_parameters(DEFAULT_DIMS)
    d_pqfl = d + 1
    d_dig = num_parameters(DIGITS_DIMS)
    F, N, q, t = cfg.frac_bits, cfg.rlwe_n, cfg.rlwe_q, cfg.rlwe_t

    rng = np.random.default_rng(seed)
    rl = RLWE(RLWEParams(N, q, cfg.rlwe_eta, t))
    pk, _ = rl.keygen(rng)
    prep = rl.prepare_public_key(pk)

    update = rng.normal(0.0, 0.05, d)
    plain_bytes = update.astype(np.float32).tobytes()
    plain_dp_bytes = np.append(update, 0.0).astype(np.float32).tobytes()

    digits_nodp, _, _ = encoding.encode(update, 0.25, F, rng)
    secagg_bytes = digits_nodp.astype(np.uint8).tobytes()

    blocks_nodp = encoding.to_blocks(digits_nodp.astype(np.int64), N)
    u0, v0 = rl.encrypt_blocks(prep, blocks_nodp, rng)
    pq_only_wire = u0.astype(np.uint16).tobytes() + v0.astype(np.uint16).tobytes()

    digits_g, _, _ = encoding.encode(update, 0.6, F, rng)
    digits_b, _, _ = encoding.encode(np.array([0.1]), 3.0, F, rng)
    digits_pqfl = np.concatenate([digits_g, digits_b]).astype(np.int64)
    blocks_pqfl = encoding.to_blocks(digits_pqfl, N)
    u1, v1 = rl.encrypt_blocks(prep, blocks_pqfl, rng)
    pqfl_wire = u1.astype(np.uint16).tobytes() + v1.astype(np.uint16).tobytes()
    packed14 = -(-(2 * blocks_pqfl.shape[0] * N * 14) // 8)

    blocks_dig = -(-((d_dig + 1) * F) // N)
    dig_wire_nbytes = rl.ciphertext_nbytes(blocks_dig)

    d_fem = num_parameters(FEMNIST_DIMS)
    blocks_fem = -(-((d_fem + 1) * F) // N)
    fem_wire_nbytes = rl.ciphertext_nbytes(blocks_fem)
    assert d_fem == 16638 and blocks_fem == 780 and fem_wire_nbytes == 798720

    assert len(pq_only_wire) == rl.ciphertext_nbytes(blocks_nodp.shape[0]) == 458752
    assert len(pqfl_wire) == rl.ciphertext_nbytes(blocks_pqfl.shape[0]) == 458752

    L = [
        f"Payload Audit (6G-IoT d={d:,d} params, F={F} bit-planes, N_R={N}, q={q}, t={t})",
        "=" * 94,
        f"1. Vanilla FedAvg / FedProx (float32)       : {len(plain_bytes):>9,d} B = {len(plain_bytes) / 1024:8.2f} KiB  (d * 4 B)",
        f"2. SCAFFOLD (float32 w_k + control delta_c) : {2 * len(plain_bytes):>9,d} B = {2 * len(plain_bytes) / 1024:8.2f} KiB  (2 * d * 4 B)",
        f"3. DP-FedAvg / FL+DistNoise (float32 + b_k) : {len(plain_dp_bytes):>9,d} B = {len(plain_dp_bytes) / 1024:8.2f} KiB  ((d + 1) * 4 B)",
        f"4. FL + Secure Aggregation (uint8 in Z_t)   : {len(secagg_bytes):>9,d} B = {len(secagg_bytes) / 1024:8.2f} KiB  (d * F * 1 B)",
        f"5. FL + PQ Encryption Only (u,v uint16)     : {len(pq_only_wire):>9,d} B = {len(pq_only_wire) / 1024:8.2f} KiB  "
        f"({blocks_nodp.shape[0]} blocks, {blocks_nodp.size - d * F} zero-padded slots)",
        f"6. PQ-FL (Proposed, d+1 scalars encrypted)  : {len(pqfl_wire):>9,d} B = {len(pqfl_wire) / 1024:8.2f} KiB  "
        f"({blocks_pqfl.shape[0]} blocks, {blocks_pqfl.size - d_pqfl * F} zero-padded slots)",
        f"   - Hypothetical 14-bit bit-packed size    : {packed14:>9,d} B = {packed14 / 1024:8.2f} KiB",
        f"   - Expansion vs float32 plaintext         : {len(pqfl_wire) / len(plain_bytes):6.2f}x     "
        f"vs SecAgg masked digits: {len(pqfl_wire) / len(secagg_bytes):.2f}x",
        f"7. UCI Digits PQ-FL (d={d_dig:,d}+1 scalars)    : {dig_wire_nbytes:>9,d} B = {dig_wire_nbytes / 1024:8.2f} KiB  "
        f"({blocks_dig} blocks of {N} coeffs)",
        f"8. LEAF FEMNIST PQ-FL (d={d_fem:,d}+1 scalars)  : {fem_wire_nbytes:>9,d} B = {fem_wire_nbytes / 1024:8.2f} KiB  "
        f"({blocks_fem} blocks of {N} coeffs; float32 FedAvg = {d_fem * 4:,d} B = {d_fem * 4 / 1024:.2f} KiB)",
        "",
        "Exact arithmetic reconciliation with '10 blocks x 2 x 1024 x 14 bits = 35 KiB':",
        f"  (a) Bit-plane expansion: each coefficient in Z_t (t={t}) carries 1 bit-plane digit in {{0,1}} prior to",
        f"      masking so n=10 client digits sum in [0,10] < t without modular wraparound. Encoding {d:,d} parameters",
        f"      (+1 clipping indicator in PQ-FL) at F={F} bit-planes produces {d_pqfl * F:,d} coefficients, not {d:,d}.",
        f"  (b) Block count: at N_R={N}, ceil({d_pqfl * F:,d} / {N}) = {blocks_pqfl.shape[0]} blocks "
        f"(equivalent in total coefficients to 112 blocks at N_R=1024, i.e., 11.2x more than 10 blocks).",
        f"  (c) Coefficient alignment: residues mod q={q} (< 2^16) are serialised as 16-bit uint16 (2 B/coeff)",
        f"      rather than 14-bit packed streams: 448 blocks * 2 polys * 256 coeffs * 2 B = 458,752 B = 448.00 KiB",
        f"      (note: (112 / 10) * 35 KiB * (16 / 14) = 448.00 KiB).",
        "  (d) Scope: counts per-round client-to-server ciphertext upload; excludes one-time public-key broadcast,",
        "      one-time pairwise key establishment, and transport-layer (TCP/TLS) framing headers.",
    ]
    return "\n".join(L)


if __name__ == "__main__":
    rep = build_report()
    print(rep)
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "payload_audit.txt"), "w", encoding="utf-8") as fh:
        fh.write(rep + "\n")
