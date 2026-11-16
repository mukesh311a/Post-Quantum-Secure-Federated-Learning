import importlib.metadata
import math
from typing import Tuple

from dp_accounting import dp_event
from dp_accounting import pld, rdp


def dp_accounting_version() -> str:
    try:
        return importlib.metadata.version("dp-accounting")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def gradient_sigma_for_adaptive_clipping(sigma_total: float, sigma_b: float) -> float:
    if sigma_b <= sigma_total:
        raise ValueError(f"sigma_b ({sigma_b}) must exceed sigma_total ({sigma_total}) for budget splitting.")
    return 1.0 / math.sqrt((1.0 / (sigma_total ** 2)) - (1.0 / (sigma_b ** 2)))


def joint_effective_sigma(sigma_g: float, sigma_b: float) -> float:
    return 1.0 / math.sqrt((1.0 / (sigma_g ** 2)) + (1.0 / (sigma_b ** 2)))


def _event(sigma_sum: float, q: float, rounds: int):
    return dp_event.SelfComposedDpEvent(
        dp_event.PoissonSampledDpEvent(q, dp_event.GaussianDpEvent(sigma_sum)), rounds
    )


def epsilon_rdp(sigma_sum: float, q: float, rounds: int, delta: float) -> float:
    acc = rdp.RdpAccountant()
    acc.compose(_event(sigma_sum, q, rounds))
    return float(acc.get_epsilon(delta))


def epsilon_rdp_with_order(sigma_sum: float, q: float, rounds: int, delta: float) -> Tuple[float, float]:
    acc = rdp.RdpAccountant()
    acc.compose(_event(sigma_sum, q, rounds))
    eps, opt_order = acc.get_epsilon_and_optimal_order(delta)
    return float(eps), float(opt_order)


def epsilon_pld(sigma_sum: float, q: float, rounds: int, delta: float) -> float:
    acc = pld.PLDAccountant()
    acc.compose(_event(sigma_sum, q, rounds))
    return float(acc.get_epsilon(delta))


def verify_joint_composition_rdp(
    sigma_total: float, sigma_b: float, q: float, rounds: int, delta: float
) -> Tuple[float, float]:
    sigma_g = gradient_sigma_for_adaptive_clipping(sigma_total, sigma_b)
    ev_single = _event(sigma_total, q, rounds)
    ev_joint = dp_event.SelfComposedDpEvent(
        dp_event.PoissonSampledDpEvent(
            q,
            dp_event.ComposedDpEvent([dp_event.GaussianDpEvent(sigma_g), dp_event.GaussianDpEvent(sigma_b)]),
        ),
        rounds,
    )
    a1, a2 = rdp.RdpAccountant(), rdp.RdpAccountant()
    a1.compose(ev_single)
    a2.compose(ev_joint)
    return float(a1.get_epsilon(delta)), float(a2.get_epsilon(delta))


def calibrate_sigma(
    target_eps: float,
    q: float,
    rounds: int,
    delta: float,
    method: str = "rdp",
    lo: float = 0.3,
    hi: float = 50.0,
    tol: float = 1e-3,
) -> float:
    f = epsilon_rdp if method == "rdp" else epsilon_pld
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if f(mid, q, rounds, delta) > target_eps:
            lo = mid
        else:
            hi = mid
    return hi


def sigma_sum_for(method_dp: str, sigma: float, n_selected: int) -> float:
    return sigma * math.sqrt(n_selected) if method_dp == "local" else sigma
