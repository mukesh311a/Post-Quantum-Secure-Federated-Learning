from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ExperimentConfig:
    num_clients: int = 100
    clients_per_round: int = 10
    num_rounds: int = 200
    local_epochs: int = 5
    lr: float = 0.01
    batch_size: int = 32
    dirichlet_alpha: float = 0.5
    fedprox_mu: float = 0.01
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])
    num_records: int = 10000
    model_dims: Tuple[int, ...] = (7, 128, 64, 4)

    delta: float = 1e-5
    target_epsilon: float = 11.1
    clip_init: float = 1.0
    clip_percentile: float = 70.0
    clip_lr: float = 0.30
    clip_min: float = 0.05
    clip_max: float = 1.0
    sigma_b_factor: float = 5.0

    frac_bits: int = 12
    enc_bound_nodp: float = 0.25
    rlwe_n: int = 256
    rlwe_q: int = 12289
    rlwe_eta: int = 2
    rlwe_t: int = 12

    output_dir: str = "results"
    dataset_dir: str = "dataset"

    @property
    def sampling_rate(self) -> float:
        return self.clients_per_round / self.num_clients
