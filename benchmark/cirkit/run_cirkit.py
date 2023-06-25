import argparse
import enum
import functools
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.backends.cudnn  # TODO: this is not exported
from torch import Tensor, optim
from torch.utils.data import DataLoader, TensorDataset

from cirkit.layers.einsum.cp import CPLayer  # TODO: rework interfaces for import
from cirkit.layers.exp_family import CategoricalLayer
from cirkit.models import TensorizedPC
from cirkit.region_graph import RegionGraph
from cirkit.utils import RandomCtx, set_determinism

from ..utils import benchmarker

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
def evaluate(pc: TensorizedPC, data_loader: DataLoader[Tuple[Tensor, ...]]) -> float:
    """Evaluate circuit on given data.

    Args:
        pc (TensorizedPC): The PC to evaluate.
        data_loader (DataLoader[Tuple[Tensor, ...]]): The evaluation data.

    Returns:
        float: The average LL.
    """

    def _iter(x: Tensor) -> Tensor:
        return pc(x)

    ll_total = 0.0
    batch: Tuple[Tensor]
    for batch in data_loader:
        x = batch[0].to(device)
        ll, (t, m) = benchmarker(functools.partial(_iter, x))
        print("t/m:", t, m)
        ll_total += ll.mean().item()
        del x, ll
    return ll_total / len(data_loader)


def train(
    pc: TensorizedPC, optimizer: optim.Optimizer, data_loader: DataLoader[Tuple[Tensor, ...]]
) -> float:
    """Train circuit on given data.

    Args:
        pc (TensorizedPC): The PC to optimize.
        optimizer (optim.Optimizer): The optimizer for circuit.
        data_loader (DataLoader[Tuple[Tensor, ...]]): The training data.

    Returns:
        float: The average LL.
    """

    def _iter(x: Tensor) -> Tensor:
        optimizer.zero_grad()
        ll = pc(x)
        ll = ll.mean()
        (-ll).backward()  # we optimize NLL
        optimizer.step()
        return ll.detach()

    ll_total = 0.0
    batch: Tuple[Tensor]
    for batch in data_loader:
        x = batch[0].to(device)
        ll, (t, m) = benchmarker(functools.partial(_iter, x))
        print("t/m:", t, m)
        ll_total += ll.item()
        del x, ll  # TODO: is everything released properly
    return ll_total / len(data_loader)


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

    pc = TensorizedPC(
        RegionGraph.load(args.region_graph),
        num_vars=num_vars,
        layer_cls=CPLayer,  # type: ignore[misc]
        efamily_cls=CategoricalLayer,
        layer_kwargs={"rank": 1, "prod_exp": True},  # type: ignore[misc]
        efamily_kwargs={"num_categories": 256},  # type: ignore[misc]
        num_inner_units=args.num_latents,
        num_input_units=args.num_latents,
    )
    pc.to(device)

    if args.mode == _Modes.TRAIN:
        optimizer = optim.Adam(pc.parameters())  # just keep everything default
        ll_train = train(pc, optimizer, data_loader)
        print("train LL:", ll_train)
    if args.mode == _Modes.EVAL:
        ll_eval = evaluate(pc, data_loader)
        print("eval LL:", ll_eval)


if __name__ == "__main__":
    main()
