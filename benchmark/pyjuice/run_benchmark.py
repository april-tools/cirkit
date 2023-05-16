import argparse
import enum
import functools
import logging
import os
import random
from typing import Callable, Tuple, TypeVar

import numpy as np
import torch
import torch.backends.cudnn  # this is not exported
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

# cat  prod ---sum---
# 0011_1111_1111_1111
os.environ["JUICE_COMPILE_FLAG"] = str(0b0011_1111_1111_1111)
print(os.environ["JUICE_COMPILE_FLAG"])
import pyjuice as juice  # pylint: disable=wrong-import-position

# disable only the following warnings (there shouldn't be other warnings by torch._inductor.utils)
# torch._inductor.utils: [WARNING] skipping cudagraphs due to input mutation
logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)

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


class _Modes(str, enum.Enum):
    SANITY = "sanity"
    BATCH_EM = "batch_em"
    FULL_EM = "full_em"
    EVAL = "eval"


class _ArgsNamespace(argparse.Namespace):  # pylint: disable=too-few-public-methods
    mode: _Modes
    first_pass_only: bool
    batch_size: int
    num_latents: int


def process_args() -> _ArgsNamespace:
    """Process command line arguments.

    Returns:
        ArgsNamespace: Parsed args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=_Modes,
        default="sanity",
        choices=["sanity", "batch_em", "full_em", "eval"],
        help="mode",
    )
    parser.add_argument("--first_pass_only", action="store_true", help="first_pass_only")
    parser.add_argument("--batch_size", type=int, default=512, help="batch_size")
    parser.add_argument("--num_latents", type=int, default=32, help="num_latents")
    return parser.parse_args(namespace=_ArgsNamespace())


T = TypeVar("T")


def benchmarker(fn: Callable[[], T]) -> Tuple[T, Tuple[float, float]]:
    """Benchmark a given function and record time and GPU memory cost.

    Args:
        fn (Callable[[], T]): The function to benchmark.

    Returns:
        Tuple[T, Tuple[float, float]]: The original return value, \
            followed by time in milliseconds and peak memory in megabytes.
    """
    torch.cuda.synchronize()  # finish all prev ops and reset mem counter
    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record(torch.cuda.current_stream())

    ret = fn()

    end_event.record(torch.cuda.current_stream())
    torch.cuda.synchronize()  # wait for event finish
    elapsed_time: float = start_event.elapsed_time(end_event)  # ms
    peak_memory = torch.cuda.max_memory_allocated() / 2**20  # MB
    return ret, (elapsed_time, peak_memory)


@torch.no_grad()
def evaluate(pc: juice.ProbCircuit, data_loader: DataLoader[Tuple[Tensor, ...]]) -> float:
    """Evaluate circuit on given data.

    Args:
        pc (juice.ProbCircuit): The PC to evaluate.
        data_loader (DataLoader[Tuple[Tensor, ...]]): The evaluation data.

    Returns:
        float: The average LL.
    """

    def _iter(x: Tensor) -> Tensor:
        return pc(x)  # type: ignore[no-any-return,misc]

    ll_total = 0.0
    batch: Tuple[Tensor]
    for batch in data_loader:
        x = batch[0].to(device)
        ll, (t, m) = benchmarker(functools.partial(_iter, x))
        print(t, m)
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

    def _iter(x: Tensor) -> Tensor:
        optimizer.zero_grad()
        ll: Tensor = pc(x)
        ll = ll.mean()
        ll.backward()
        optimizer.step()
        return ll

    ll_total = 0.0
    batch: Tuple[Tensor]
    for batch in data_loader:
        x = batch[0].to(device)
        ll, (t, m) = benchmarker(functools.partial(_iter, x))
        print(t, m)
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

    def _iter(x: Tensor) -> Tensor:
        ll: Tensor = pc(x)
        pc.backward(x, flows_memory=1.0)
        return ll

    ll_total = 0.0
    batch: Tuple[Tensor]
    for batch in data_loader:
        x = batch[0].to(device)
        ll, (t, m) = benchmarker(functools.partial(_iter, x))
        print(t, m)
        ll_total += ll.mean().item()

    em_step = functools.partial(pc.mini_batch_em, step_size=1.0, pseudocount=0.1)
    _, (t, m) = benchmarker(em_step)  # type: ignore[misc]
    # TODO: this is mypy bug  # pylint: disable=fixme
    print(t, m)
    return ll_total / len(data_loader)


def main() -> None:
    """Execute the main procedure."""
    args = process_args()
    print(args)
    assert (
        not args.mode == _Modes.SANITY or args.batch_size == 512 and args.num_latents == 32
    ), "Must use default hyper-params for sanity check."

    seed_all()

    num_features = 28 * 28
    data_size = (
        10240
        if args.mode == _Modes.SANITY
        else args.batch_size
        if args.first_pass_only
        else 100 * args.batch_size
    )
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

    if args.mode in (_Modes.BATCH_EM, _Modes.SANITY):
        optimizer = juice.optim.CircuitOptimizer(pc)  # just keep it default
        ll_batch = batch_em_epoch(pc, optimizer, data_loader)
        print("batch_em LL:", ll_batch)
    if args.mode in (_Modes.FULL_EM, _Modes.SANITY):
        ll_full = full_em_epoch(pc, data_loader)
        print("full_em LL:", ll_full)
    if args.mode in (_Modes.EVAL, _Modes.SANITY):
        ll_eval = evaluate(pc, data_loader)
        print("eval LL:", ll_eval)
        if args.mode == _Modes.SANITY:
            # -4285.7583... is an exact float32 number, 2^-10 is 2eps
            assert abs(ll_eval - -4285.75830078125) < 2**-10, "eval LL out of allowed error range"


if __name__ == "__main__":
    main()
