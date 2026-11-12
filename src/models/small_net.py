"""Optional tiny MLP swept across hidden widths: does a TRAINED network show double descent too?

One hidden ReLU layer, BOTH layers trained with full-batch Adam on the MSE, no weight decay and no
early stopping (both would act as regularizers and shrink the peak). Requires the ``torch`` extra;
torch is imported lazily so the rest of the library works without it.

Unlike the linear models, a trained network's peak sits NEAR, not exactly at, n_params == n_train:
its effective capacity depends on optimisation, not just on the parameter count.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.arrays import FloatArray
from src.datasets.synthetic import Task
from src.evaluation.capacity_sweep import COLUMNS


def torch_available() -> bool:
    """True when the optional torch extra is installed."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def mlp_n_params(dim: int, width: int) -> int:
    """Weights and biases of a dim -> width -> 1 network."""
    return (dim + 1) * width + width + 1


def train_mlp(
    x_train: FloatArray,
    y_train: FloatArray,
    x_test: FloatArray,
    width: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    threads: int = 2,
) -> tuple[FloatArray, FloatArray, float]:
    """Train one network; return (train predictions, test predictions, parameter L2 norm)."""
    import torch
    from torch import nn

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    xt = torch.as_tensor(np.asarray(x_train, dtype=np.float32).reshape(len(x_train), -1))
    yt = torch.as_tensor(np.asarray(y_train, dtype=np.float32)).reshape(-1, 1)
    xs = torch.as_tensor(np.asarray(x_test, dtype=np.float32).reshape(len(x_test), -1))
    net = nn.Sequential(nn.Linear(xt.shape[1], width), nn.ReLU(), nn.Linear(width, 1))
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(net(xt), yt)
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        train_pred = net(xt).squeeze(1).double().numpy()
        test_pred = net(xs).squeeze(1).double().numpy()
        sq = torch.zeros((), dtype=torch.float64)
        for param in net.parameters():
            sq = sq + (param.double() ** 2).sum()
        norm = float(torch.sqrt(sq))
    return train_pred, test_pred, norm


def mlp_width_sweep(
    task: Task,
    n_train: int,
    widths: Sequence[int],
    noise: float,
    epochs: int,
    learning_rate: float,
    n_seeds: int,
    seed: int,
    threads: int = 2,
) -> pd.DataFrame:
    """Train/test MSE of trained MLPs across hidden widths. Same columns as capacity sweeps."""
    x_test = task.test_x()
    f_test = task.f(x_test)
    records: list[dict[str, float | int | str]] = []
    for s, rng in enumerate(np.random.default_rng(seed).spawn(n_seeds)):
        x = task.sample_x(n_train, rng)
        y = task.f(x) + noise * rng.standard_normal(n_train)
        torch_seed = int(rng.integers(0, 2**31 - 1))
        for w in widths:
            train_pred, test_pred, norm = train_mlp(
                x, y, x_test, int(w), epochs, learning_rate, torch_seed, threads
            )
            records.append(
                {
                    "model": "mlp",
                    "capacity": int(w),
                    "n_params": mlp_n_params(task.dim, int(w)),
                    "n_train": n_train,
                    "noise": noise,
                    "seed": s,
                    "train_mse": float(np.mean((train_pred - y) ** 2)),
                    "test_mse": float(np.mean((test_pred - f_test) ** 2)),
                    "coef_norm": norm,
                }
            )
    return pd.DataFrame.from_records(records, columns=COLUMNS)
