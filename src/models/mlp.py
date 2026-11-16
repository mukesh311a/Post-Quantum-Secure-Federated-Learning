from typing import Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn

IN_DIM, H1, H2, N_CLASSES = 7, 128, 64, 4
DEFAULT_DIMS: Tuple[int, ...] = (IN_DIM, H1, H2, N_CLASSES)


class PQFL_MLP(nn.Module):
    def __init__(self, dims: Sequence[int] = DEFAULT_DIMS):
        super().__init__()
        self.dims = tuple(int(x) for x in dims)
        self.layers = nn.ModuleList(
            [nn.Linear(self.dims[i], self.dims[i + 1]) for i in range(len(self.dims) - 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


def _shapes_and_sizes(dims: Sequence[int]):
    shapes = []
    for i in range(len(dims) - 1):
        fi, fo = int(dims[i]), int(dims[i + 1])
        shapes.append((fo, fi))
        shapes.append((fo,))
    sizes = [int(np.prod(s)) for s in shapes]
    return shapes, sizes


_DEFAULT_SHAPES, _DEFAULT_SIZES = _shapes_and_sizes(DEFAULT_DIMS)


def num_parameters(dims: Sequence[int] = DEFAULT_DIMS) -> int:
    dims = tuple(int(x) for x in dims)
    return sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))


def init_flat(seed: int, dims: Sequence[int] = DEFAULT_DIMS) -> torch.Tensor:
    torch.manual_seed(seed)
    m = PQFL_MLP(dims=dims)
    return torch.cat([p.detach().reshape(-1) for p in m.parameters()]).clone()


def forward_flat(w: torch.Tensor, x: torch.Tensor, dims: Sequence[int] = DEFAULT_DIMS) -> torch.Tensor:
    dims_t = tuple(int(d) for d in dims)
    if dims_t == DEFAULT_DIMS:
        shapes, sizes = _DEFAULT_SHAPES, _DEFAULT_SIZES
    else:
        shapes, sizes = _shapes_and_sizes(dims_t)
    parts, idx = [], 0
    for shp, n in zip(shapes, sizes):
        parts.append(w[idx:idx + n].view(*shp))
        idx += n
    h = x
    n_layers = len(parts) // 2
    for li in range(n_layers):
        W, b = parts[2 * li], parts[2 * li + 1]
        h = h @ W.T + b
        if li < n_layers - 1:
            h = torch.relu(h)
    return h


if __name__ == "__main__":
    print("6G-IoT MLP parameters:", num_parameters(DEFAULT_DIMS))
    print("UCI Digits MLP parameters:", num_parameters((64, 64, 10)))
