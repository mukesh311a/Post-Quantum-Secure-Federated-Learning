import numpy as np


def encode(x, bound, frac_bits, rng):
    levels = (1 << frac_bits) - 1
    clamped = np.clip(x, -bound, bound)
    n_clamped = int(np.sum(np.abs(x) > bound))
    scaled = (clamped + bound) / (2.0 * bound) * levels
    ints = np.floor(scaled + rng.random(scaled.shape)).astype(np.int64)
    ints = np.clip(ints, 0, levels)
    shifts = np.arange(frac_bits, dtype=np.int64)
    digits = ((ints[:, None] >> shifts[None, :]) & 1).astype(np.uint8)
    return digits.reshape(-1), ints, n_clamped


def decode_sum(digit_sums, n_clients, bound, frac_bits):
    levels = (1 << frac_bits) - 1
    ds = np.asarray(digit_sums, dtype=np.int64).reshape(-1, frac_bits)
    total_ints = (ds << np.arange(frac_bits, dtype=np.int64)[None, :]).sum(axis=1)
    return total_ints * (2.0 * bound / levels) - n_clients * bound, total_ints


def to_blocks(digits, n_ring):
    d = digits.shape[0]
    n_blocks = -(-d // n_ring)
    padded = np.zeros(n_blocks * n_ring, dtype=np.int64)
    padded[:d] = digits
    return padded.reshape(n_blocks, n_ring)


def from_blocks(blocks, length):
    return blocks.reshape(-1)[:length]
