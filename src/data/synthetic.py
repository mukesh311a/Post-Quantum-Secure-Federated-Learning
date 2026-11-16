import os
import numpy as np
import pandas as pd

CLASSES = ["SmartSensor", "Wearable", "AutonomousVehicle", "IndustrialController"]

FEATURE_RANGES = {
    "Signal_Strength_dBm": (-120.0, -50.0),
    "Latency_ms": (5.0, 200.0),
    "Heart_Rate_bpm": (40.0, 180.0),
    "Location_X": (0.0, 1.0),
    "Location_Y": (0.0, 1.0),
    "Battery_Level": (10.0, 100.0),
    "Data_Volume_MB": (0.1, 500.0),
}
FEATURES = list(FEATURE_RANGES)

CLASS_PARAMS = {
    "SmartSensor":          {"Signal_Strength_dBm": (-88, 10), "Latency_ms": (110, 40), "Heart_Rate_bpm": (72, 20),
                             "Location_X": (0.50, 0.30), "Location_Y": (0.50, 0.30), "Battery_Level": (80, 12),
                             "Data_Volume_MB": (1.0, 0.6)},
    "Wearable":             {"Signal_Strength_dBm": (-72, 10), "Latency_ms": (60, 25), "Heart_Rate_bpm": (90, 14),
                             "Location_X": (0.50, 0.25), "Location_Y": (0.40, 0.25), "Battery_Level": (55, 18),
                             "Data_Volume_MB": (2.3, 0.6)},
    "AutonomousVehicle":    {"Signal_Strength_dBm": (-62, 8), "Latency_ms": (22, 10), "Heart_Rate_bpm": (70, 18),
                             "Location_X": (0.25, 0.15), "Location_Y": (0.65, 0.15), "Battery_Level": (85, 10),
                             "Data_Volume_MB": (4.6, 0.6)},
    "IndustrialController": {"Signal_Strength_dBm": (-80, 10), "Latency_ms": (40, 15), "Heart_Rate_bpm": (74, 20),
                             "Location_X": (0.80, 0.10), "Location_Y": (0.20, 0.10), "Battery_Level": (95, 5),
                             "Data_Volume_MB": (3.6, 0.6)},
}


def generate_records(num_records: int, rng: np.random.Generator) -> pd.DataFrame:
    labels = rng.integers(0, len(CLASSES), size=num_records)
    cols = {}
    for f in FEATURES:
        lo, hi = FEATURE_RANGES[f]
        x = np.empty(num_records)
        for c, name in enumerate(CLASSES):
            idx = labels == c
            mu, sd = CLASS_PARAMS[name][f]
            if f == "Data_Volume_MB":
                x[idx] = np.exp(rng.normal(mu, sd, idx.sum()))
            else:
                x[idx] = rng.normal(mu, sd, idx.sum())
        cols[f] = np.clip(x, lo, hi)
    df = pd.DataFrame(cols)
    df.insert(0, "Record_ID", [f"REC_{i:05d}" for i in range(num_records)])
    df["Device_Type"] = [CLASSES[i] for i in labels]
    df["Label"] = labels
    return df


def dirichlet_partition(labels: np.ndarray, num_clients: int, alpha: float, rng: np.random.Generator,
                        min_size: int = 8, max_tries: int = 200):
    n_classes = int(labels.max()) + 1
    for _ in range(max_tries):
        parts = [[] for _ in range(num_clients)]
        for c in range(n_classes):
            idx = np.where(labels == c)[0]
            rng.shuffle(idx)
            p = rng.dirichlet(alpha * np.ones(num_clients))
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for client, chunk in enumerate(np.split(idx, cuts)):
                parts[client].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [np.array(p) for p in parts]
    raise RuntimeError("could not satisfy min_size in Dirichlet partition")


def build_dataset(path: str, seed: int, num_records: int = 10000, num_clients: int = 100,
                  alpha: float = 0.5, test_fraction: float = 0.2):
    rng = np.random.default_rng(seed)
    df = generate_records(num_records, rng)
    split = np.array(["train"] * num_records, dtype=object)
    for c in range(len(CLASSES)):
        idx = np.where(df["Label"].values == c)[0]
        test_idx = rng.choice(idx, size=int(round(test_fraction * len(idx))), replace=False)
        split[test_idx] = "test"
    df["Split"] = split
    train_idx = np.where(split == "train")[0]
    parts = dirichlet_partition(df["Label"].values[train_idx], num_clients, alpha, rng)
    client_id = np.full(num_records, -1, dtype=int)
    for k, p in enumerate(parts):
        client_id[train_idx[p]] = k
    df["Client_ID"] = client_id
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.round(4).to_csv(path, index=False)
    return df


def load_federated(path: str):
    df = pd.read_csv(path)
    tr, te = df[df["Split"] == "train"], df[df["Split"] == "test"]
    mu, sd = tr[FEATURES].values.mean(0), tr[FEATURES].values.std(0) + 1e-8
    f = lambda d: ((d[FEATURES].values - mu) / sd).astype(np.float32)
    clients = {}
    for k, g in tr.groupby("Client_ID"):
        clients[int(k)] = (f(g), g["Label"].values.astype(np.int64))
    return clients, f(te), te["Label"].values.astype(np.int64)


def write_description(path: str):
    lines = ["# Synthetic 6G-IoT federated dataset", "",
             "Label rule: Device_Type (4 classes, uniform prior) is drawn first; the 7 features are drawn from",
             "class-conditional Gaussians (log-normal for Data_Volume_MB, where mu and sigma specify ln(MB))",
             "and clipped to the physical ranges below.", "",
             "| Feature | Range |", "|---|---|"]
    lines += [f"| {k} | [{lo}, {hi}] |" for k, (lo, hi) in FEATURE_RANGES.items()]
    lines += ["", "Class-conditional (mean, std):", ""]
    for c, p in CLASS_PARAMS.items():
        lines.append(f"- {c}: " + ", ".join(f"{k}=({m},{s})" for k, (m, s) in p.items()))
    lines += ["",
              "Columns: Record_ID, 7 features, Device_Type, Label (0-3), Split (train/test), Client_ID (-1 for test records).",
              "Client partition: per-class Dirichlet(alpha=0.5) over 100 clients (min_size=8); 20 % stratified test split (8,000 train / 2,000 test).",
              "Preprocessing: zero-mean unit-variance standardisation fitted exclusively on the 8,000 training records (zero test-set leakage)."]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
