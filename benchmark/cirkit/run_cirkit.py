import argparse
import enum
import functools
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor, optim
from torch.utils.data import DataLoader, TensorDataset

from benchmark.utils import benchmarker
from cirkit.layers.input.exp_family import CategoricalLayer
from cirkit.layers.sum_product.cp import UncollapsedCPLayer
from cirkit.models import TensorizedPC
from cirkit.region_graph import RegionGraph
from cirkit.utils import RandomCtx, set_determinism

device = torch.device("cuda")


class _Modes(str, enum.Enum):  # TODO: StrEnum introduced in 3.11
    """Execution modes."""

    TRAIN = "train"
    EVAL = "eval"


@dataclass
class _ArgsNamespace(argparse.Namespace):
    mode: _Modes = _Modes.TRAIN
    seed: int = 42
    num_batches: int = 20
    batch_size: int = 128
    region_graph: str = ""
    num_latents: int = 32  # TODO: rename this
    first_pass_only: bool = False


def process_args() -> _ArgsNamespace:
    """Process command line arguments.

    Returns:
        ArgsNamespace: Parsed args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=_Modes, choices=_Modes.__members__.values(), help="mode")
    parser.add_argument("--seed", type=int, help="seed, 0 for disable")
    parser.add_argument("--num_batches", type=int, help="num_batches")
    parser.add_argument("--batch_size", type=int, help="batch_size")
    parser.add_argument("--region_graph", type=str, help="region_graph filename")
    parser.add_argument("--num_latents", type=int, help="num_latents")
    parser.add_argument("--first_pass_only", action="store_true", help="first_pass_only")
    return parser.parse_args(namespace=_ArgsNamespace())


@torch.no_grad()
def evaluate(
    pc: TensorizedPC, data_loader: DataLoader[Tuple[Tensor, ...]]
) -> Tuple[Tuple[List[float], List[float]], float]:
    """Evaluate circuit on given data.

    Args:
        pc (TensorizedPC): The PC to evaluate.
        data_loader (DataLoader[Tuple[Tensor, ...]]): The evaluation data.

    Returns:
        Tuple[Tuple[List[float], List[float]], float]:
         A tuple consisting of time and memory measurements, and the average LL.
    """

    def _iter(x: Tensor) -> Tensor:
        return pc(x)

    ll_total = 0.0
    ts, ms = [], []
    batch: Tuple[Tensor]
    for batch in data_loader:
        x = batch[0].to(device)
        ll, (t, m) = benchmarker(functools.partial(_iter, x))
        ts.append(t)
        ms.append(m)
        ll_total += ll.mean().item()
        del x, ll
    return (ts, ms), ll_total / len(data_loader)


def train(
    pc: TensorizedPC, optimizer: optim.Optimizer, data_loader: DataLoader[Tuple[Tensor, ...]]
) -> Tuple[Tuple[List[float], List[float]], float]:
    """Train circuit on given data.

    Args:
        pc (TensorizedPC): The PC to optimize.
        optimizer (optim.Optimizer): The optimizer for circuit.
        data_loader (DataLoader[Tuple[Tensor, ...]]): The training data.

    Returns:
        Tuple[Tuple[List[float], List[float]], float]:
         A tuple consisting of time and memory measurements, and the average LL.
    """

    def _iter(x: Tensor) -> Tensor:
        optimizer.zero_grad()
        ll = pc(x)
        ll = ll.mean()
        (-ll).backward()  # type: ignore[no-untyped-call]  # we optimize NLL
        optimizer.step()
        return ll.detach()

    ll_total = 0.0
    ts, ms = [], []
    batch: Tuple[Tensor]
    for batch in data_loader:
        x = batch[0].to(device)
        ll, (t, m) = benchmarker(functools.partial(_iter, x))
        ts.append(t)
        ms.append(m)
        ll_total += ll.item()
        del x, ll  # TODO: is everything released properly
    return (ts, ms), ll_total / len(data_loader)


def main() -> None:
    """Execute the main procedure."""
    args = process_args()
    assert args.region_graph, "Must provide a RG filename"
    print(args)

    if args.seed:
        # TODO: find a way to set w/o with
        RandomCtx(args.seed).__enter__()  # pylint: disable=unnecessary-dunder-call
        set_determinism(check_hash_seed=True)

    num_vars = 28 * 28
    data_size = args.batch_size if args.first_pass_only else args.num_batches * args.batch_size
    rand_data = torch.randint(256, (data_size, num_vars), dtype=torch.uint8)
    data_loader = DataLoader(
        dataset=TensorDataset(rand_data),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    pc = TensorizedPC.from_region_graph(
        RegionGraph.load(args.region_graph),
        layer_cls=UncollapsedCPLayer,
        efamily_cls=CategoricalLayer,
        layer_kwargs={"rank": 1},  # type: ignore[misc]
        efamily_kwargs={"num_categories": 256},  # type: ignore[misc]
        num_inner_units=args.num_latents,
        num_input_units=args.num_latents,
    )
    pc.to(device)
    print(pc)
    print(f"Number of parameters: {sum(p.numel() for p in pc.parameters())}")

    if args.mode == _Modes.TRAIN:
        optimizer = optim.Adam(pc.parameters())  # just keep everything default
        (ts, ms), ll_train = train(pc, optimizer, data_loader)
        print("Train LL:", ll_train)
    elif args.mode == _Modes.EVAL:
        (ts, ms), ll_eval = evaluate(pc, data_loader)
        print("Evaluation LL:", ll_eval)
    else:
        assert False, "Something is wrong here"
    if not args.first_pass_only and args.num_batches > 1:
        # Skip warmup step
        ts, ms = ts[1:], ms[1:]
    mu_t, sigma_t = np.mean(ts).item(), np.std(ts).item()  # type: ignore[misc]
    mu_m, sigma_m = np.mean(ms).item(), np.std(ms).item()  # type: ignore[misc]
    print(f"Time (ms): {mu_t:.3f}+-{sigma_t:.3f}")
    print(f"Memory (MiB): {mu_m:.3f}+-{sigma_m:.3f}")


if __name__ == "__main__":
    main()
