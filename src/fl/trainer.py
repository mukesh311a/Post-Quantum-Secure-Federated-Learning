import math
import time
import zlib
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from src.models.mlp import DEFAULT_DIMS, init_flat, forward_flat, num_parameters
from src.crypto.rlwe import RLWE, RLWEParams
from src.crypto.secagg import client_mask
from src.crypto import encoding
from src.privacy.accounting import gradient_sigma_for_adaptive_clipping

torch.set_num_threads(1)


def evaluate(w: torch.Tensor, X: torch.Tensor, y: torch.Tensor, dims=DEFAULT_DIMS):
    with torch.no_grad():
        logits = forward_flat(w, X, dims=dims)
        prob = torch.softmax(logits, dim=1).numpy()
    if not np.all(np.isfinite(prob)):
        prob = np.full_like(prob, 1.0 / prob.shape[1])
    pred = prob.argmax(1)
    yn = y.numpy()
    return {
        "accuracy": float((pred == yn).mean() * 100.0),
        "f1": float(f1_score(yn, pred, average="macro")),
        "auc": float(roc_auc_score(yn, prob, multi_class="ovr", average="macro")),
        "mae": float(np.abs(pred - yn).mean()),
    }


def local_train(w_g, X, y, cfg, rb, opt, c_global=None, c_local=None, dims=DEFAULT_DIMS):
    w = w_g.clone()
    n = X.shape[0]
    steps = 0
    corr = (c_global - c_local) if opt == "scaffold" else None
    for _ in range(cfg.local_epochs):
        perm = rb.permutation(n)
        for i in range(0, n, cfg.batch_size):
            idx = torch.from_numpy(perm[i:i + cfg.batch_size])
            w.requires_grad_(True)
            loss = F.cross_entropy(forward_flat(w, X[idx], dims=dims), y[idx])
            g, = torch.autograd.grad(loss, w)
            w = w.detach()
            if opt == "fedprox":
                g = g + cfg.fedprox_mu * (w - w_g)
            elif opt == "scaffold":
                g = g + corr
            w = w - cfg.lr * g
            steps += 1
    return w.detach(), steps


def establish_pairwise_secrets(rlwe, num_clients, rng):
    secrets, mism = {}, 0
    t0 = time.perf_counter()
    for k in range(num_clients):
        for v in range(k + 1, num_clients):
            sa, sb = rlwe.establish_pair_secret(rng)
            mism += int(sa != sb)
            secrets[(k, v)] = sa
    return secrets, time.perf_counter() - t0, mism


def run_method(name, spec, seed, cfg, clients, X_test, y_test, sigma, pair_secrets=None, quiet=True):
    dims = getattr(cfg, "model_dims", DEFAULT_DIMS)
    d = num_parameters(dims)
    T, n_sel, N = cfg.num_rounds, cfg.clients_per_round, cfg.num_clients
    Xt = torch.from_numpy(X_test)
    yt = torch.from_numpy(y_test)
    cl = {k: (torch.from_numpy(X), torch.from_numpy(y)) for k, (X, y) in clients.items()}

    w = init_flat(seed, dims=dims)
    rng_sample = np.random.default_rng(seed)
    rng_noise = np.random.default_rng([seed, zlib.crc32(name.encode()), 1])
    rng_crypto = np.random.default_rng([seed, zlib.crc32(name.encode()), 2])

    use_crypto = spec["secagg"] or spec["pq"]
    has_dp = spec["dp"] is not None
    F_bits = cfg.frac_bits
    rl = RLWE(RLWEParams(cfg.rlwe_n, cfg.rlwe_q, cfg.rlwe_eta, cfg.rlwe_t)) if spec["pq"] else None
    if spec["pq"]:
        pk, sk = rl.keygen(rng_crypto)
        prep = rl.prepare_public_key(pk)
    t_mod = cfg.rlwe_t

    c_global = torch.zeros(d)
    c_local = {k: torch.zeros(d) for k in cl} if spec["opt"] == "scaffold" else None

    C_t = float(cfg.clip_init)
    p_target = float(cfg.clip_percentile) / 100.0
    if has_dp:
        sigma_b_base = cfg.sigma_b_factor * sigma
        sigma_g = gradient_sigma_for_adaptive_clipping(sigma, sigma_b_base)
    else:
        sigma_b_base = sigma_g = 0.0

    hist = {m: [] for m in ("accuracy", "f1", "auc", "mae")}
    diag = dict(
        mismatched_coords=0,
        clamped_coords=0,
        clip_values=[],
        clipped_fraction=[],
        update_norm_mean=[],
    )
    tm = {k: [] for k in ("client_train", "client_dpops", "client_encode_mask", "client_encrypt", "server_aggregate")}

    if spec["opt"] == "scaffold":
        payload_bytes = 2 * d * 4
    elif has_dp and not use_crypto:
        payload_bytes = (d + 1) * 4
    else:
        payload_bytes = d * 4

    for t in range(1, T + 1):
        S = sorted(rng_sample.choice(N, size=n_sel, replace=False).tolist())
        n = len(S)

        updates, norms = {}, {}
        tr_t = 0.0
        dc_sum = torch.zeros(d)
        for k in S:
            X, y = cl[k]
            rb = np.random.default_rng([seed, t, k])
            t0 = time.perf_counter()
            wk, steps = local_train(
                w, X, y, cfg, rb, spec["opt"], c_global, c_local[k] if c_local is not None else None, dims=dims
            )
            tr_t += time.perf_counter() - t0
            g = wk - w
            if spec["opt"] == "scaffold":
                ci_new = c_local[k] - c_global + (w - wk) / (steps * cfg.lr)
                dc_sum += ci_new - c_local[k]
                c_local[k] = ci_new
            updates[k] = g.numpy().astype(np.float64)
            norms[k] = float(np.linalg.norm(updates[k]))
        if spec["opt"] == "scaffold":
            c_global = c_global + dc_sum / N
        tm["client_train"].append(tr_t / n)
        diag["update_norm_mean"].append(float(np.mean(list(norms.values()))))

        t0 = time.perf_counter()
        b_noisy = {}
        if has_dp:
            std_g = (sigma_g * C_t / math.sqrt(n)) if spec["dp"] == "dist" else (sigma_g * C_t)
            std_b = (sigma_b_base * 0.5 / math.sqrt(n)) if spec["dp"] == "dist" else (sigma_b_base * 0.5)
            clipped = 0
            for k in S:
                scale = 1.0 if norms[k] == 0.0 else min(1.0, C_t / max(norms[k], 1e-8))
                clipped += int(scale < 1.0)
                updates[k] = updates[k] * scale + rng_noise.normal(0.0, std_g, d)
                b_tilde = (1.0 if norms[k] <= C_t else 0.0) - 0.5
                b_noisy[k] = float(b_tilde + rng_noise.normal(0.0, std_b))
            diag["clip_values"].append(C_t)
            diag["clipped_fraction"].append(clipped / n)
            bound_g = C_t + 5.0 * std_g
            bound_b = 0.5 + 5.0 * std_b
        else:
            bound_g = cfg.enc_bound_nodp
            bound_b = 1.0
        tm["client_dpops"].append((time.perf_counter() - t0) / n)

        if not use_crypto:
            t0 = time.perf_counter()
            agg = np.sum([updates[k] for k in S], axis=0)
            b_sum = float(sum(b_noisy[k] for k in S)) if has_dp else 0.0
            if has_dp:
                eff_sigma = sigma if spec["dp"] == "dist" else (sigma * math.sqrt(n))
                eta_c = cfg.clip_lr / max(1.0, eff_sigma)
                gamma_hat = b_sum / n + 0.5
                C_t = min(max(C_t * math.exp(-eta_c * (gamma_hat - p_target)), cfg.clip_min), cfg.clip_max)
            tm["server_aggregate"].append(time.perf_counter() - t0)
        else:
            enc_t = mask_t = 0.0
            n_scalars = (d + 1) if has_dp else d
            total_digits = n_scalars * F_bits
            digit_sum = np.zeros(total_digits, dtype=np.int64)
            ref_g_ints = np.zeros(d, dtype=np.int64)
            ref_b_int = 0
            agg_ct = None
            dec_each_t = 0.0

            for k in S:
                t0 = time.perf_counter()
                dg, ig, nclamp_g = encoding.encode(updates[k], bound_g, F_bits, rng_crypto)
                diag["clamped_coords"] += nclamp_g
                ref_g_ints += ig
                if has_dp:
                    db, ib, nclamp_b = encoding.encode(np.array([b_noisy[k]]), bound_b, F_bits, rng_crypto)
                    diag["clamped_coords"] += nclamp_b
                    ref_b_int += int(ib[0])
                    y = np.concatenate([dg, db]).astype(np.int64)
                else:
                    y = dg.astype(np.int64)

                if spec["secagg"]:
                    y = (y + client_mask(k, S, pair_secrets, t, total_digits, t_mod)) % t_mod
                mask_t += time.perf_counter() - t0

                if spec["pq"]:
                    t0 = time.perf_counter()
                    blocks = encoding.to_blocks(y, cfg.rlwe_n)
                    ct = rl.encrypt_blocks(prep, blocks, rng_crypto)
                    enc_t += time.perf_counter() - t0
                    payload_bytes = rl.ciphertext_nbytes(blocks.shape[0])
                    if spec["secagg"]:
                        agg_ct = ct if agg_ct is None else rl.add(agg_ct, ct)
                    else:
                        t1 = time.perf_counter()
                        dec = encoding.from_blocks(rl.decrypt_blocks(sk, *ct), total_digits)
                        dec_each_t += time.perf_counter() - t1
                        digit_sum += dec
                else:
                    payload_bytes = total_digits
                    digit_sum += y

            t0 = time.perf_counter()
            if spec["pq"] and spec["secagg"]:
                digit_sum = encoding.from_blocks(rl.decrypt_blocks(sk, *agg_ct), total_digits)
            elif spec["secagg"]:
                digit_sum = digit_sum % t_mod

            agg, total_g_ints = encoding.decode_sum(digit_sum[:d * F_bits], n, bound_g, F_bits)
            mism = int(np.sum(total_g_ints != ref_g_ints))
            if has_dp:
                b_rec, total_b_ints = encoding.decode_sum(digit_sum[d * F_bits:], n, bound_b, F_bits)
                mism += int(int(total_b_ints[0]) != ref_b_int)
                b_sum = float(b_rec[0])
                eff_sigma = sigma if spec["dp"] == "dist" else (sigma * math.sqrt(n))
                eta_c = cfg.clip_lr / max(1.0, eff_sigma)
                gamma_hat = b_sum / n + 0.5
                C_t = min(max(C_t * math.exp(-eta_c * (gamma_hat - p_target)), cfg.clip_min), cfg.clip_max)
            tm["server_aggregate"].append(time.perf_counter() - t0 + dec_each_t)
            diag["mismatched_coords"] += mism
            tm["client_encode_mask"].append(mask_t / n)
            tm["client_encrypt"].append(enc_t / n)

        w = w + torch.from_numpy((agg / n).astype(np.float32))
        m = evaluate(w, Xt, yt, dims=dims)
        for key in hist:
            hist[key].append(m[key])
        if not quiet and t % 20 == 0:
            print(f"    [{name}] seed={seed} round={t:3d} acc={m['accuracy']:.2f}")

    return {
        "metrics": hist,
        "timing_s": {k: (float(np.mean(v)) if v else 0.0) for k, v in tm.items()},
        "payload_bytes_per_client": int(payload_bytes),
        "diagnostics": {
            "mismatched_coords": diag["mismatched_coords"],
            "clamped_coords": diag["clamped_coords"],
            "coords_per_round": (d + 1) if (use_crypto and has_dp) else (d if use_crypto else 0),
            "clip_trajectory": [float(c) for c in diag["clip_values"]],
            "update_norm_trajectory": diag["update_norm_mean"],
            "mean_clip_threshold": float(np.mean(diag["clip_values"])) if diag["clip_values"] else None,
            "mean_clipped_fraction": float(np.mean(diag["clipped_fraction"])) if diag["clipped_fraction"] else None,
            "mean_update_norm": float(np.mean(diag["update_norm_mean"])),
        },
    }
