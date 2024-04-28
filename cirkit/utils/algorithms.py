from collections import defaultdict, deque
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

NodeType = TypeVar("NodeType")


def bfs(
    roots: Set[NodeType], incomings_fn: Callable[[NodeType], Sequence[NodeType]]
) -> Iterator[NodeType]:
    seen, to_visit = set(roots), deque(roots)
    while to_visit:
        node = to_visit.popleft()
        yield node
        for ch in incomings_fn(node):
            if ch not in seen:
                seen.add(ch)
                to_visit.append(ch)


def topological_ordering(
    roots: Set[NodeType], incomings_fn: Callable[[NodeType], Sequence[NodeType]]
) -> Optional[List[NodeType]]:
    num_incomings: Dict[NodeType, int] = defaultdict(int)
    outgoings: Dict[NodeType, List[NodeType]] = defaultdict(list)
    for n in bfs(roots, incomings_fn):
        incomings = incomings_fn(n)
        num_incomings[n] = len(incomings)
        for ch in incomings:
            outgoings[ch].append(n)
    in_nodes = map(lambda x: x[0], filter(lambda x: x[1] == 0, num_incomings.items()))

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
