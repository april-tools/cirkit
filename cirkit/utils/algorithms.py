from collections import deque, defaultdict
from typing import Set, Any, Callable, List, Iterator, Dict, Optional, Union, Tuple


def bfs(
        roots: Set[Any],
        incomings_fn: Callable[[Any], Union[List[Any], Tuple[Any]]]
) -> Iterator[Any]:
    seen, to_visit = set(roots), deque(roots)
    while to_visit:
        node = to_visit.popleft()
        yield node
        for ch in incomings_fn(node):
            if ch not in seen:
                seen.add(ch)
                to_visit.append(ch)


def topological_ordering(
        roots: Set[Any],
        incomings_fn: Callable[[Any], Union[List[Any], Tuple[Any]]]
) -> Optional[List[Any]]:
    num_incomings: Dict[Any, int] = defaultdict(int)
    outgoings: Dict[Any, List[Any]] = defaultdict(list)
    for n in bfs(roots, incomings_fn):
        incomings = incomings_fn(n)
        num_incomings[n] = len(incomings)
        for ch in incomings:
            outgoings[ch].append(n)
    in_nodes = filter(lambda x: x[1] == 0, num_incomings.items())

    ordering = []
    to_visit = deque(in_nodes)
    while to_visit:
        child = to_visit.popleft()
        ordering.append(child)
        for n in outgoings[child]:
            num_incomings[n] -= 1
            if num_incomings[n] == 0:
                to_visit.append(n)

    if sum(num_incomings.values()) != 0:
        return None  # There is at least one cycle, no topological ordering is possible
    return ordering
