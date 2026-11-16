import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.crypto.ring import mul_small, schoolbook_negacyclic, centered_binomial
from src.crypto.rlwe import RLWE, RLWEParams
from src.crypto.secagg import client_mask
from src.crypto import encoding
from src.models.mlp import num_parameters
from src.privacy.accounting import verify_joint_composition_rdp


class TestRing(unittest.TestCase):
    def test_fft_multiplication_matches_schoolbook(self):
        rng = np.random.default_rng(0)
        q = 12289
        for n in (16, 64, 256):
            for _ in range(5):
                a = rng.integers(0, q, n)
                s = centered_binomial((n,), 2, rng)
                self.assertTrue(np.array_equal(mul_small(a, s, q), schoolbook_negacyclic(a, s % q, q)))


class TestRLWE(unittest.TestCase):
    def setUp(self):
        self.rl = RLWE(RLWEParams(n_ring=256))
        self.rng = np.random.default_rng(1)

    def test_encrypt_decrypt_roundtrip(self):
        pk, sk = self.rl.keygen(self.rng)
        prep = self.rl.prepare_public_key(pk)
        msg = self.rng.integers(0, 2, size=(8, 256))
        u, v = self.rl.encrypt_blocks(prep, msg, self.rng)
        self.assertTrue(np.array_equal(self.rl.decrypt_blocks(sk, u, v), msg))

    def test_additive_aggregation_of_ten_ciphertexts(self):
        pk, sk = self.rl.keygen(self.rng)
        prep = self.rl.prepare_public_key(pk)
        msgs = [self.rng.integers(0, 2, size=(4, 256)) for _ in range(10)]
        agg = None
        for m in msgs:
            c = self.rl.encrypt_blocks(prep, m, self.rng)
            agg = c if agg is None else self.rl.add(agg, c)
        dec = self.rl.decrypt_blocks(sk, *agg)
        self.assertTrue(np.array_equal(dec, sum(msgs)))

    def test_pair_secret_agreement(self):
        for _ in range(20):
            a, b = self.rl.establish_pair_secret(self.rng)
            self.assertEqual(a, b)


class TestSecAggAndEncoding(unittest.TestCase):
    def test_masks_cancel(self):
        rng = np.random.default_rng(2)
        S = [3, 7, 11, 20, 42]
        secrets = {(k, v): rng.bytes(32) for k in S for v in S if k < v}
        t, L = 12, 1000
        total = sum(client_mask(k, S, secrets, 5, L, t) for k in S) % t
        self.assertTrue(np.all(total == 0))

    def test_encode_decode_sum_is_exact_up_to_quantisation(self):
        rng = np.random.default_rng(3)
        d, F, B, n = 500, 12, 0.7, 10
        xs = [rng.normal(0, 0.1, d) for _ in range(n)]
        digit_sum = np.zeros(d * F, dtype=np.int64)
        ref_ints = np.zeros(d, dtype=np.int64)
        for x in xs:
            digits, ints, _ = encoding.encode(x, B, F, rng)
            digit_sum += digits
            ref_ints += ints
        rec, total_ints = encoding.decode_sum(digit_sum, n, B, F)
        self.assertTrue(np.array_equal(total_ints, ref_ints))
        step = 2 * B / ((1 << F) - 1)
        self.assertLess(np.max(np.abs(rec - sum(xs))), n * step)

    def test_full_pipeline_secagg_plus_rlwe_equals_plain_sum(self):
        rng = np.random.default_rng(4)
        rl = RLWE(RLWEParams(n_ring=256))
        pk, sk = rl.keygen(rng)
        prep = rl.prepare_public_key(pk)
        n, d, F, B, t = 10, 300, 12, 0.5, rl.p.t
        S = list(range(n))
        secrets = {(k, v): rng.bytes(32) for k in S for v in S if k < v}
        agg, ref = None, np.zeros(d, dtype=np.int64)
        for k in S:
            x = rng.normal(0, 0.05, d)
            digits, ints, _ = encoding.encode(x, B, F, rng)
            ref += ints
            y = (digits.astype(np.int64) + client_mask(k, S, secrets, 1, d * F, t)) % t
            c = rl.encrypt_blocks(prep, encoding.to_blocks(y, rl.p.n_ring), rng)
            agg = c if agg is None else rl.add(agg, c)
        dec = encoding.from_blocks(rl.decrypt_blocks(sk, *agg), d * F)
        _, total_ints = encoding.decode_sum(dec, n, B, F)
        self.assertTrue(np.array_equal(total_ints, ref))

    def test_full_9540_model_plus_indicator_pqfl_wire_and_recovery(self):
        rng = np.random.default_rng(2026)
        rl = RLWE(RLWEParams(n_ring=256, q=12289, eta=2, t=12))
        pk, sk = rl.keygen(rng)
        prep = rl.prepare_public_key(pk)
        n, d, F, B_g, B_b, t = 10, num_parameters(), 12, 0.8, 4.0, rl.p.t
        self.assertEqual(d, 9540)
        S = list(range(n))
        secrets = {(k, v): rng.bytes(32) for k in S for v in S if k < v}
        total_len = (d + 1) * F
        agg_ct = None
        ref_g_ints = np.zeros(d, dtype=np.int64)
        ref_b_ints = np.zeros(1, dtype=np.int64)
        for k in S:
            g_hat = rng.normal(0.0, 0.05, d)
            b_hat = np.array([rng.normal(0.1, 0.3)])
            dg, ig, _ = encoding.encode(g_hat, B_g, F, rng)
            db, ib, _ = encoding.encode(b_hat, B_b, F, rng)
            ref_g_ints += ig
            ref_b_ints += ib
            digits = np.concatenate([dg, db]).astype(np.int64)
            masked = (digits + client_mask(k, S, secrets, 1, total_len, t)) % t
            blocks = encoding.to_blocks(masked, rl.p.n_ring)
            self.assertEqual(blocks.shape, (448, 256))
            self.assertEqual(rl.ciphertext_nbytes(blocks.shape[0]), 458752)
            ct = rl.encrypt_blocks(prep, blocks, rng)
            agg_ct = ct if agg_ct is None else rl.add(agg_ct, ct)
        dec_digits = encoding.from_blocks(rl.decrypt_blocks(sk, *agg_ct), total_len)
        _, got_g_ints = encoding.decode_sum(dec_digits[:d * F], n, B_g, F)
        _, got_b_ints = encoding.decode_sum(dec_digits[d * F:], n, B_b, F)
        self.assertTrue(np.array_equal(got_g_ints, ref_g_ints))
        self.assertTrue(np.array_equal(got_b_ints, ref_b_ints))


class TestAdaptiveClippingAccounting(unittest.TestCase):
    def test_joint_gaussian_budget_split_matches_composed_rdp(self):
        for sigma in (1.0, 2.0, float(np.sqrt(10.0)), 4.0):
            e_single, e_joint = verify_joint_composition_rdp(sigma, 5.0 * sigma, q=0.1, rounds=200, delta=1e-5)
            self.assertAlmostEqual(e_single, e_joint, places=11)


if __name__ == "__main__":
    unittest.main()
