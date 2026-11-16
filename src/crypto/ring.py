import numpy as np


def centered_binomial(shape, eta, rng):
    bits = rng.integers(0, 2, size=tuple(shape) + (2 * eta,), dtype=np.int8)
    return (bits[..., :eta].sum(-1, dtype=np.int16)
            - bits[..., eta:].sum(-1, dtype=np.int16)).astype(np.int64)


def _center(a, q):
    a = np.asarray(a, dtype=np.int64) % q
    return np.where(a > q // 2, a - q, a)


def rfft_padded(x, n_ring):
    return np.fft.rfft(np.asarray(x, dtype=np.float64), n=2 * n_ring, axis=-1)


def negacyclic_mul_fft(fa, fs, n_ring, q):
    c = np.fft.irfft(fa * fs, n=2 * n_ring, axis=-1)
    c = np.rint(c).astype(np.int64)
    return (c[..., :n_ring] - c[..., n_ring:]) % q


def mul_small(a, s, q):
    n_ring = a.shape[-1]
    return negacyclic_mul_fft(rfft_padded(_center(a, q), n_ring),
                              rfft_padded(s, n_ring), n_ring, q)


def schoolbook_negacyclic(a, b, q):
    n = len(a)
    out = [0] * n
    for i in range(n):
        ai = int(a[i])
        for j in range(n):
            k = i + j
            if k < n:
                out[k] += ai * int(b[j])
            else:
                out[k - n] -= ai * int(b[j])
    return np.array([x % q for x in out], dtype=np.int64)
