import argparse
import logging
import time
import warnings
from typing import Optional, Tuple

import numpy as np
import pyjuice as juice
import torch
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)
logging.getLogger("torch._inductor.compile_fx").setLevel(logging.ERROR)


def process_args() -> argparse.Namespace:
    """Process command line arguments.

    Returns:
        argparse.Namespace: Parsed args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=512, help="batch_size")
    parser.add_argument("--cuda", type=int, default=0, help="cuda idx")
    parser.add_argument("--num_latents", type=int, default=32, help="num_latents")
    parser.add_argument("--mode", type=str, default="train", help="options: 'train', 'load'")
    # parser.add_argument("--dataset", type=str, default="mnist", help="mnist, fashion")
    parser.add_argument(
        "--input_circuit",
        type=str,
        default=None,
        help="load circuit from file instead of learning structure",
    )
    parser.add_argument("--output_dir", type=str, default="examples", help="output directory")
    args = parser.parse_args()  # pylint: disable=redefined-outer-name
    return args


def evaluate(
    pc: juice.ProbCircuit,
    loader: DataLoader[Tuple[torch.Tensor, ...]],
    alphas: Optional[torch.Tensor] = None,
) -> float:
    """Evaluate the circuit.

    Args:
        pc (juice.ProbCircuit): The PC to evaluate.
        loader (DataLoader[Tuple[torch.Tensor, ...]]): The evaluation data.
        alphas (Optional[torch.Tensor], optional): Alpha on image. Defaults to None.

    Returns:
        float: The average LL on data.
    """
    lls_total = 0.0
    for batch in loader:
        x = batch[0].to(pc.device)
        lls = pc(x, alphas=alphas)
        lls_total += lls.mean().detach().cpu().numpy().item()

    lls_total /= len(loader)
    return lls_total


def evaluate_miss(
    pc: juice.ProbCircuit,
    loader: DataLoader[Tuple[torch.Tensor, ...]],
    alphas: Optional[torch.Tensor] = None,
) -> float:
    """Evaluate with missing data.

    Args:
        pc (juice.ProbCircuit): The PC to evaluate.
        loader (DataLoader[Tuple[torch.Tensor, ...]]): The evaluation data.
        alphas (Optional[torch.Tensor], optional): Alpha on image. Defaults to None.

    Returns:
        float: The average LL on data.
    """
    lls_total = 0.0
    for batch in loader:
        x = batch[0].to(pc.device)
        mask = batch[1].to(pc.device)
        lls = pc(x, missing_mask=mask, alphas=alphas)
        lls_total += lls.mean().detach().cpu().numpy().item()

    lls_total /= len(loader)
    return lls_total


def mini_batch_em_epoch(  # pylint: disable=too-many-arguments,too-many-locals
    num_epochs: int,
    pc: juice.ProbCircuit,
    optimizer: juice.optim.CircuitOptimizer,
    scheduler: juice.optim.CircuitScheduler,
    train_loader: DataLoader[Tuple[torch.Tensor, ...]],
    test_loader: DataLoader[Tuple[torch.Tensor, ...]],
    device: torch.device,
) -> None:
    """Perform EM on mini batches.

    Args:
        num_epochs (int): Number of epochs to do.
        pc (juice.ProbCircuit): The PC to optimize.
        optimizer (juice.optim.CircuitOptimizer): The optimizer.
        scheduler (juice.optim.CircuitScheduler): The lr scheduler.
        train_loader (DataLoader[Tuple[torch.Tensor, ...]]): Training data.
        test_loader (DataLoader[Tuple[torch.Tensor, ...]]): Testing data.
        device (torch.device): The device to put data on.
    """
    for epoch in range(num_epochs):
        t_start = time.time()
        train_ll = 0.0
        for batch in train_loader:
            x = batch[0].to(device)

            optimizer.zero_grad()

            lls = pc(x)
            lls.mean().backward()

            train_ll += lls.mean().detach().cpu().numpy().item()

            optimizer.step()
            scheduler.step()

        train_ll /= len(train_loader)

        t_train = time.time()
        test_ll = evaluate(pc, loader=test_loader)
        t_eval = time.time()

        print(
            f"[Epoch {epoch}/{num_epochs}][train LL: {train_ll:.2f}; test LL: {test_ll:.2f}]....."
            f"[train forward+backward+step {t_train-t_start:.2f};"
            f" test forward {t_eval-t_train:.2f}] "
        )


def full_batch_em_epoch(
    pc: juice.ProbCircuit,
    train_loader: DataLoader[Tuple[torch.Tensor, ...]],
    test_loader: DataLoader[Tuple[torch.Tensor, ...]],
    device: torch.device,
) -> None:
    """Perform EM on whole dataset.

    Args:
        pc (juice.ProbCircuit): The PC to optimize.
        train_loader (DataLoader[Tuple[torch.Tensor, ...]]): Training data.
        test_loader (DataLoader[Tuple[torch.Tensor, ...]]): Testing data.
        device (torch.device): The device to put data on.
    """
    with torch.no_grad():
        t_start = time.time()
        train_ll = 0.0
        for batch in train_loader:
            x = batch[0].to(device)

            lls = pc(x)
            pc.backward(x, flows_memory=1.0)

            train_ll += lls.mean().detach().cpu().numpy().item()

        pc.mini_batch_em(step_size=1.0, pseudocount=0.1)

        train_ll /= len(train_loader)

        t_train = time.time()
        test_ll = evaluate(pc, loader=test_loader)
        t_eval = time.time()
        print(
            f"[train LL: {train_ll:.2f}; test LL: {test_ll:.2f}]....."
            f"[train forward+backward+step {t_train-t_start:.2f};"
            f" test forward {t_eval-t_train:.2f}] "
        )


def load_circuit(
    filename: str, verbose: bool = False, device: Optional[torch.device] = None
) -> juice.ProbCircuit:
    """Load circuit from file.

    Args:
        filename (str): The file name of saved circuit.
        verbose (bool, optional): Whether to print extra info. Defaults to False.
        device (Optional[torch.device], optional): The device to load the circuit on.
            Defaults to None.

    Returns:
        juice.ProbCircuit: Loaded circuit.
    """
    t_start = time.time()
    if verbose:
        print(f"Loading circuit....{filename}.....", end="")
    pc = juice.model.ProbCircuit.load(filename)
    if device is not None:
        print(f"...into device {device}...", end="")
        pc.to(device)
    t_loaded = time.time()
    if verbose:
        print(f"Took {t_loaded-t_start:.2f} (s)")
        print("pc params size", pc.params.size())
        print("pc num nodes ", pc.num_nodes)

    return pc


def save_circuit(pc: juice.ProbCircuit, filename: str, verbose: bool = False) -> None:
    """Save circuit to file.

    Args:
        pc (juice.ProbCircuit): The circuit to save.
        filename (str): The file name to save circuit.
        verbose (bool, optional): Whether to print extra info. Defaults to False.
    """
    if verbose:
        print(f"Saving pc into {filename}.....", end="")
    t_start = time.time()
    torch.save(pc, filename)
    t_saved = time.time()
    if verbose:
        print(f"took {t_saved - t_start:.2f} (s)")


# pylint: disable-next=redefined-outer-name,too-many-locals,too-many-statements
def main(args: argparse.Namespace) -> None:
    """Invoke different procedures based on command line args.

    Args:
        args (argparse.Namespace): The arguments passed in.
    """
    torch.cuda.set_device(args.cuda)
    device = torch.device(f"cuda:{args.cuda}")
    filename = f"{args.output_dir}/dummydataset_{args.num_latents}.torch"

    train_data = torch.randint(256, (60000, 28 * 28), dtype=torch.uint8)
    test_data = torch.randint(256, (10000, 28 * 28), dtype=torch.uint8)

    num_features = train_data.size(1)

    train_loader = DataLoader(
        dataset=TensorDataset(train_data), batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        dataset=TensorDataset(test_data), batch_size=args.batch_size, shuffle=False, drop_last=True
    )

    if args.mode == "train":
        print("===========================Train===============================")
        if args.input_circuit is None:
            pc = juice.structures.HCLT(
                train_data.float().to(device),
                num_bins=32,
                sigma=0.5 / 32,
                num_latents=args.num_latents,
                chunk_size=32,
            )
            pc.to(device)
        else:
            pc = load_circuit(args.input_circuit, verbose=True, device=device)

        optimizer = juice.optim.CircuitOptimizer(pc, lr=0.1, pseudocount=0.1)
        scheduler = juice.optim.CircuitScheduler(
            optimizer,
            method="multi_linear",
            lrs=[0.9, 0.1, 0.05],
            milestone_steps=[0, len(train_loader) * 100, len(train_loader) * 350],
        )

        mini_batch_em_epoch(350, pc, optimizer, scheduler, train_loader, test_loader, device)
        full_batch_em_epoch(pc, train_loader, test_loader, device)
        save_circuit(pc, filename, verbose=True)

    elif args.mode == "load":
        print("===========================LOAD===============================")
        pc = load_circuit(filename, verbose=True, device=device)

        t_start = time.time()
        test_ll = evaluate(pc, loader=test_loader)  # force compilation

        t_compile = time.time()
        train_ll = evaluate(pc, loader=train_loader)
        t_evaltrain = time.time()
        test_ll = evaluate(pc, loader=test_loader)
        t_evaltest = time.time()

        train_bpd = -train_ll / (num_features * np.log(2))
        test_bpd = -test_ll / (num_features * np.log(2))

        print(
            f"Compilation+test took {t_compile-t_start:.2f} (s); "
            f"train_ll {t_evaltrain-t_compile:.2f} (s); test_ll {t_evaltest-t_evaltrain:.2f} (s)"
        )
        print(f"train_ll: {train_ll:.2f}, test_ll: {test_ll:.2f}")
        print(f"train_bpd: {train_bpd:.2f}, test_bpd: {test_bpd:.2f}")

    elif args.mode == "miss":
        print("===========================MISS===============================")
        print(f"Loading {filename} into {device}.......")
        pc = load_circuit(filename, verbose=True, device=device)

        # test_miss_mask = torch.zeros(test_data.size(), dtype=torch.bool)
        # test_miss_mask[1:5000, 0:392] = 1 # for first half of images make first half missing
        # test_miss_mask[5000:, 392:] = 1   # for second half of images make second half missing
        test_miss_mask = torch.rand(test_data.size()) < 0.5

        test_loader_miss = DataLoader(
            dataset=TensorDataset(test_data, test_miss_mask),
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
        )
        t_start = time.time()
        test_ll = evaluate(pc, loader=test_loader)
        test_ll_miss = evaluate_miss(pc, loader=test_loader_miss)

        t_compile = time.time()
        train_ll = evaluate(pc, loader=train_loader)
        t_evaltrain = time.time()
        test_ll = evaluate(pc, loader=test_loader)
        t_evaltest = time.time()
        test_ll_miss = evaluate_miss(pc, loader=test_loader_miss)
        t_evalmiss = time.time()

        train_bpd = -train_ll / (num_features * np.log(2))
        test_bpd = -test_ll / (num_features * np.log(2))
        test_miss_bpd = -test_ll_miss / (num_features * np.log(2))

        print(
            f"train_ll: {train_ll:.2f}, train_bpd: {train_bpd:.2f}; "
            f"time = {t_evaltrain-t_compile:.2f} (s)"
        )
        print(
            f"test_ll: {test_ll:.2f}, test_bpd: {test_bpd:.2f}; "
            f"time = {t_evaltest-t_evaltrain:.2f} (s)"
        )
        print(
            f"test_miss_ll: {test_ll_miss:.2f}, test_miss_bpd: {test_miss_bpd:.2f}; "
            f"time = {t_evalmiss-t_evaltest:.2f} (s)"
        )
    elif args.mode == "alphas":
        print("===========================ALPHAS===============================")
        pc = load_circuit(filename, verbose=True, device=device)

        alphas = 0.99 * torch.ones((args.batch_size, 28 * 28), device=device)
        test_ll = evaluate(pc, loader=test_loader)
        train_ll = evaluate(pc, loader=train_loader)

        t_compile = time.time()
        train_ll_alpha = evaluate(pc, loader=train_loader, alphas=alphas)
        t_evaltrain = time.time()
        test_ll_alpha = evaluate(pc, loader=test_loader, alphas=alphas)
        t_evaltest = time.time()

        print(f"train_ll: {train_ll:.2f}, test_ll: {test_ll:.2f}")
        print(f"train_ll_alpha: {train_ll_alpha:.2f}, test_ll_alpha: {test_ll_alpha:.2f}")
        print(f"train {t_evaltrain-t_compile:.2f} (s); test {t_evaltest-t_evaltrain:.2f} (s)")

    print(f"Memory allocated: {torch.cuda.memory_allocated(device) / 1024 / 1024 / 1024:.1f}GB")
    print(f"Memory reserved: {torch.cuda.memory_reserved(device) / 1024 / 1024 / 1024:.1f}GB")
    print(
        f"Max memory reserved: {torch.cuda.max_memory_reserved(device) / 1024 / 1024 / 1024:.1f}GB"
    )


if __name__ == "__main__":
    args = process_args()
    print(args)
    main(args)