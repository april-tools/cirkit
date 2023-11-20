from typing import Callable, Tuple, TypeVar

import torch

T = TypeVar("T")


def timer(fn: Callable[[], T]) -> Tuple[T, float]:
    """Time a given function for GPU time cost.

    Args:
        fn (Callable[[], T]): The function to time.

    Returns:
        Tuple[T, float]: The original return value, and time in ms.
    """
    # TODO: torch typing issue
    start_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
    end_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
    start_event.record(torch.cuda.current_stream())  # type: ignore[no-untyped-call]

    ret = fn()

    end_event.record(torch.cuda.current_stream())  # type: ignore[no-untyped-call]
    torch.cuda.synchronize()  # wait for event to be recorded
    # TODO: Event.elapsed_time is not typed
    elapsed_time: float = start_event.elapsed_time(end_event)  # type: ignore[no-untyped-call]
    return ret, elapsed_time


def benchmarker(fn: Callable[[], T]) -> Tuple[T, Tuple[float, float]]:
    """Benchmark a given function for GPU time and (peak) memory cost.

    Args:
        fn (Callable[[], T]): The function to benchmark.

    Returns:
        Tuple[T, Tuple[float, float]]: The original return value, and time in ms and memory in MiB.
    """
    torch.cuda.synchronize()  # finish all previous work before resetting mem stats
    torch.cuda.reset_peak_memory_stats()

    ret, elapsed_time = timer(fn)

    peak_memory = torch.cuda.max_memory_allocated() / 2**20
    return ret, (elapsed_time, peak_memory)
