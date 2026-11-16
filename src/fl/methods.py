METHODS = {
    "Vanilla FedAvg":          dict(opt="fedavg",   dp=None,    secagg=False, pq=False),
    "FedProx":                 dict(opt="fedprox",  dp=None,    secagg=False, pq=False),
    "SCAFFOLD":                dict(opt="scaffold", dp=None,    secagg=False, pq=False),
    "DP-FedAvg":               dict(opt="fedavg",   dp="local", secagg=False, pq=False),
    "FL + Distributed Noise":  dict(opt="fedavg",   dp="dist",  secagg=False, pq=False),
    "FL + Secure Aggregation": dict(opt="fedavg",   dp=None,    secagg=True,  pq=False),
    "FL + PQ Encryption Only": dict(opt="fedavg",   dp=None,    secagg=False, pq=True),
    "PQ-FL (Proposed)":        dict(opt="fedavg",   dp="dist",  secagg=True,  pq=True),
}


def privacy_score(spec):
    dp, sa, pq = spec["dp"] is not None, spec["secagg"], spec["pq"]
    if dp and sa and pq:
        return 9.8
    if dp:
        return 7.0
    if sa:
        return 5.0
    if pq:
        return 3.0
    return 1.0
