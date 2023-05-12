import argparse
import os
import random
from typing import Tuple

import numpy as np
import pyjuice as juice
import torch
import torch.backends.cudnn  # this is not exported
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda")


def seed_all() -> None:
    """Seed all random generators to guarantee reproducible results (may limit performance)."""
    seed = 0xC699345C  # CRC32 of 'april-tools'
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


class _ArgsNamespace(argparse.Namespace):  # pylint: disable=too-few-public-methods
    batch_size: int
    num_latents: int


def process_args() -> _ArgsNamespace:
    """Process command line arguments.

    Returns:
        ArgsNamespace: Parsed args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=512, help="batch_size")
    parser.add_argument("--num_latents", type=int, default=32, help="num_latents")
    return parser.parse_args(namespace=_ArgsNamespace())


@torch.no_grad()
def evaluate(pc: juice.ProbCircuit, data_loader: DataLoader[Tuple[Tensor, ...]]) -> float:
    """Evaluate circuit on given data.

    Args:
        pc (juice.ProbCircuit): The PC to evaluate.
        data_loader (DataLoader[Tuple[Tensor, ...]]): The evaluation data.

    Returns:
        float: The average LL.
    """
    ll_total = 0.0
    batch: Tuple[Tensor]
    for batch in data_loader:
        x = batch[0].to(device)
        ll: Tensor = pc(x)
        ll_total += ll.mean().item()
    return ll_total / len(data_loader)


def batch_em_epoch(
    pc: juice.ProbCircuit,
    optimizer: juice.optim.CircuitOptimizer,
    data_loader: DataLoader[Tuple[Tensor, ...]],
) -> float:
    """Perform EM optimization on mini batches for one epoch.

    Args:
        pc (juice.ProbCircuit): The PC to optimize.
        optimizer (juice.optim.CircuitOptimizer): The optimizer for circuit.
        data_loader (DataLoader[Tuple[Tensor, ...]]): The training data.

    Returns:
        float: The average LL.
    """
    ll_total = 0.0
    batch: Tuple[Tensor]
    for batch in data_loader:
        optimizer.zero_grad()
        x = batch[0].to(device)
        ll: Tensor = pc(x)
        ll = ll.mean()
        ll.backward()
        optimizer.step()
        ll_total += ll.item()
    return ll_total / len(data_loader)


@torch.no_grad()
def full_em_epoch(
    pc: juice.ProbCircuit,
    data_loader: DataLoader[Tuple[Tensor, ...]],
) -> float:
    """Perform EM optimization on full dataset for one epoch.

    Args:
        pc (juice.ProbCircuit): The PC to optimize.
        data_loader (DataLoader[Tuple[Tensor, ...]]): The training data.

    Returns:
        float: The average LL.
    """
    ll_total = 0.0
    batch: Tuple[Tensor]
    for batch in data_loader:
        x = batch[0].to(device)
        ll: Tensor = pc(x)
        pc.backward(x, flows_memory=1.0)
        ll_total += ll.mean().item()
    pc.mini_batch_em(step_size=1.0, pseudocount=0.1)
    return ll_total / len(data_loader)


def main() -> None:
    """TODO."""
    args = process_args()

    seed_all()

    num_features = 28 * 28
    data_size = 60000
    rand_data = torch.randint(256, (data_size, num_features), dtype=torch.uint8)
    data_loader = DataLoader(
        dataset=TensorDataset(rand_data),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    num_bins = 32
    sigma = 0.5 / 32
    chunk_size = 32
    pc = juice.structures.HCLT(
        rand_data.to(device),
        num_bins=num_bins,
        sigma=sigma,
        chunk_size=chunk_size,
        num_latents=args.num_latents,
    )
    pc.to(device)

    optimizer = juice.optim.CircuitOptimizer(pc)  # just keep it default

    batch_em_epoch(pc, optimizer, data_loader)
    full_em_epoch(pc, data_loader)
    evaluate(pc, data_loader)


if __name__ == "__main__":
    main()
