import argparse
import enum
import functools
import logging
import os
import random
from dataclasses import dataclass
from typing import Callable, Tuple, TypeVar

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from .load_rg import load_region_graph

# cat  prod ---sum---
# 0011_1111_1111_1111
if "JUICE_COMPILE_FLAG" not in os.environ:
    os.environ["JUICE_COMPILE_FLAG"] = str(0b0011_1111_1111_1111)
print(os.environ["JUICE_COMPILE_FLAG"])
import pyjuice as juice  # pylint: disable=wrong-import-order,wrong-import-position
from pyjuice import ProbCircuit  # pylint: disable=wrong-import-order,wrong-import-position

# disable only the following warnings (there shouldn't be other warnings by torch._inductor.utils)
# torch._inductor.utils: [WARNING] skipping cudagraphs due to input mutation
logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)

device = torch.device("cuda")


def seed_all(seed: int) -> None:
    """Seed all random generators and enforce deterministic algorithms to \
        guarantee reproducible results (may limit performance).

    Args:
        seed (int): The seed shared by all RNGs.
    """
    seed %= 2**32  # some only accept 32bit seed
    assert os.environ.get("PYTHONHASHSEED", "") == str(
        seed
    ), "Must set PYTHONHASHSEED to the same seed before starting python."
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


class _Modes(str, enum.Enum):
    """Execution modes."""

    SANITY = "sanity"
    BATCH_EM = "batch_em"
    FULL_EM = "full_em"
    EVAL = "eval"


@dataclass
class _ArgsNamespace(argparse.Namespace):
    mode: _Modes = _Modes.SANITY
    seed: int = 0xC699345C  # default is CRC32 of 'april-tools' = 3331929180
    num_batches: int = 20
    batch_size: int = 512
    region_graph: str = "from_data"
    num_latents: int = 32
    first_pass_only: bool = False


def process_args() -> _ArgsNamespace:
    """Process command line arguments.

    Returns:
        ArgsNamespace: Parsed args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=_Modes, choices=_Modes.__members__.values(), help="mode")
    parser.add_argument("--seed", type=int, help="seed")
    parser.add_argument("--num_batches", type=int, help="num_batches")
    parser.add_argument("--batch_size", type=int, help="batch_size")
    parser.add_argument("--region_graph", type=str, help="region_graph filename")
    parser.add_argument("--num_latents", type=int, help="num_latents")
    parser.add_argument("--first_pass_only", action="store_true", help="first_pass_only")
    return parser.parse_args(namespace=_ArgsNamespace())


T = TypeVar("T")


def benchmarker(fn: Callable[[], T]) -> Tuple[T, Tuple[float, float]]:
    """Benchmark a given function and record time and GPU memory cost.

    Args:
        fn (Callable[[], T]): The function to benchmark.

    Returns:
        Tuple[T, Tuple[float, float]]: The original return value, followed by \
            time in milliseconds and peak memory in megabytes (1024 scale).
    """
    torch.cuda.synchronize()  # finish all prev ops and reset mem counter
    torch.cuda.reset_peak_memory_stats()
    # TODO: repeated with benchmark/utils/gpu_benchmark.py
    start_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
    end_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
    start_event.record(torch.cuda.current_stream())  # type: ignore[no-untyped-call]

    ret = fn()

    end_event.record(torch.cuda.current_stream())  # type: ignore[no-untyped-call]
    torch.cuda.synchronize()  # wait for event finish
    elapsed_time: float = start_event.elapsed_time(end_event)  # type: ignore[no-untyped-call]  # ms
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
        print("t/m:", t, m)
        ll_total += ll.mean().item()
        del x, ll
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
        ll.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        return ll.detach()

    ll_total = 0.0
    batch: Tuple[Tensor]
    for batch in data_loader:
        x = batch[0].to(device)
        ll, (t, m) = benchmarker(functools.partial(_iter, x))
        print("t/m:", t, m)
        ll_total += ll.item()
        del x, ll
    return ll_total / len(data_loader)


@torch.no_grad()
def full_em_epoch(pc: juice.ProbCircuit, data_loader: DataLoader[Tuple[Tensor, ...]]) -> float:
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
        print("t/m:", t, m)
        ll_total += ll.mean().item()
        del x, ll

    # TODO: this is mypy bug
    _, (t, m) = benchmarker(  # type: ignore[misc]
        functools.partial(pc.mini_batch_em, step_size=1.0, pseudocount=0.1)
    )
    print("t/m:", t, m)
    return ll_total / len(data_loader)


def main() -> None:
    """Execute the main procedure."""
    args = process_args()
    print(args)
    assert not args.mode == _Modes.SANITY or (
        args.seed == _ArgsNamespace.seed
        and args.num_batches == _ArgsNamespace.num_batches
        and args.batch_size == _ArgsNamespace.batch_size
        and args.region_graph == _ArgsNamespace.region_graph
        and args.num_latents == _ArgsNamespace.num_latents
        and args.first_pass_only == _ArgsNamespace.first_pass_only
    ), "Must use default hyper-params for sanity check."

    seed_all(args.seed)

    num_features = 28 * 28
    data_size = args.batch_size if args.first_pass_only else args.num_batches * args.batch_size
    rand_data = torch.randint(256, (data_size, num_features), dtype=torch.uint8)
    data_loader = DataLoader(
        dataset=TensorDataset(rand_data),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    if args.region_graph == _ArgsNamespace.region_graph:
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
    else:
        rg = load_region_graph(args.region_graph, args.num_latents)
        pc = ProbCircuit(rg)
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
