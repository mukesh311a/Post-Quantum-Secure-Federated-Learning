import hashlib
import numpy as np


def prg_mask(secret: bytes, round_idx: int, length: int, modulus: int) -> np.ndarray:
    stream = hashlib.shake_256(secret + round_idx.to_bytes(4, "little")).digest(4 * length)
    return (np.frombuffer(stream, dtype=np.uint32).astype(np.int64) % modulus)


def client_mask(k: int, participants, secrets: dict, round_idx: int, length: int, modulus: int):
    m = np.zeros(length, dtype=np.int64)
    for v in participants:
        if v == k:
            continue
        if v > k:
            m += prg_mask(secrets[(k, v)], round_idx, length, modulus)
        else:
            m -= prg_mask(secrets[(v, k)], round_idx, length, modulus)
    return m % modulus
