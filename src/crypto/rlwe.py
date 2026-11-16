from dataclasses import dataclass
import hashlib
import numpy as np

from .ring import centered_binomial, rfft_padded, negacyclic_mul_fft, mul_small


@dataclass(frozen=True)
class RLWEParams:
    n_ring: int = 256
    q: int = 12289
    eta: int = 2
    t: int = 12

    @property
    def delta(self):
        return self.q // self.t


class RLWE:
    def __init__(self, params: RLWEParams):
        self.p = params

    def keygen(self, rng):
        p = self.p
        a = rng.integers(0, p.q, size=p.n_ring, dtype=np.int64)
        s = centered_binomial((p.n_ring,), p.eta, rng)
        e = centered_binomial((p.n_ring,), p.eta, rng)
        b = (mul_small(a, s, p.q) + e) % p.q
        return (a, b), s

    def prepare_public_key(self, pk):
        p = self.p
        from .ring import _center
        a, b = pk
        return (rfft_padded(_center(a, p.q), p.n_ring), rfft_padded(_center(b, p.q), p.n_ring))

    def encrypt_blocks(self, prepared_pk, msg, rng):
        p = self.p
        fa, fb = prepared_pk
        B = msg.shape[0]
        noise = centered_binomial((3, B, p.n_ring), p.eta, rng)
        r, e1, e2 = noise[0], noise[1], noise[2]
        fr = rfft_padded(r, p.n_ring)
        u = (negacyclic_mul_fft(fa, fr, p.n_ring, p.q) + e1) % p.q
        v = (negacyclic_mul_fft(fb, fr, p.n_ring, p.q) + e2 + p.delta * msg.astype(np.int64)) % p.q
        return u, v

    def decrypt_blocks(self, s, u, v):
        p = self.p
        x = (v - mul_small(u, s, p.q)) % p.q
        return np.rint(x / p.delta).astype(np.int64) % p.t

    def add(self, c1, c2):
        q = self.p.q
        return ((c1[0] + c2[0]) % q, (c1[1] + c2[1]) % q)

    def ciphertext_nbytes(self, n_blocks):
        return 2 * n_blocks * self.p.n_ring * 2

    def establish_pair_secret(self, rng):
        p = self.p
        pk, sk = self.keygen(rng)
        prepared = self.prepare_public_key(pk)
        key_bits = rng.integers(0, 2, size=(1, p.n_ring), dtype=np.int64)
        u, v = self.encrypt_blocks(prepared, key_bits, rng)
        recovered = self.decrypt_blocks(sk, u, v)
        secret_b = hashlib.sha256(key_bits.astype(np.uint8).tobytes()).digest()
        secret_a = hashlib.sha256(recovered.astype(np.uint8).tobytes()).digest()
        return secret_a, secret_b
