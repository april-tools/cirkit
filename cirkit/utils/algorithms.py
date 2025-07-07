from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Generic, TypeVar

# pylint:
NodeT = TypeVar("NodeT")


def graph_nodes_outgoings(
    nodes: Iterable[NodeT], incomings_fn: Callable[[NodeT], Sequence[NodeT]]
) -> dict[NodeT, Sequence[NodeT]]:
    outgoings: dict[NodeT, list[NodeT]] = {}
    for n in nodes:
        incomings = incomings_fn(n)
        for ch in incomings:
            if ch in outgoings:
                outgoings[ch].append(n)
            else:
                outgoings[ch] = [n]
    return outgoings


def bfs(
    roots: Iterable[NodeT], incomings_fn: Callable[[NodeT], Sequence[NodeT]]
) -> Iterator[NodeT]:
    if isinstance(roots, Iterator):
        roots = list(roots)
    seen, to_visit = set(roots), deque(roots)
    while to_visit:
        n = to_visit.popleft()
        yield n
        for ch in incomings_fn(n):
            if ch not in seen:
                seen.add(ch)
                to_visit.append(ch)


def subgraph(
    roots: Iterable[NodeT], incomings_fn: Callable[[NodeT], Sequence[NodeT]]
) -> tuple[Sequence[NodeT], dict[NodeT, Sequence[NodeT]]]:
    nodes = list(bfs(roots, incomings_fn))
    incomings: dict[NodeT, Sequence[NodeT]] = {}
    for n in nodes:
        incomings[n] = incomings_fn(n)
    return nodes, incomings


def topological_ordering(
    nodes: Iterable[NodeT],
    incomings_fn: Callable[[NodeT], Sequence[NodeT]],
    outcomings_fn: Callable[[NodeT], Sequence[NodeT]] | None = None,
) -> Iterator[NodeT]:
    if outcomings_fn is None:
        if isinstance(nodes, Iterator):
            nodes = list(nodes)
        outgoings = graph_nodes_outgoings(nodes, incomings_fn)
        outcomings_fn = lambda n: outgoings.get(n, [])
    num_incomings: dict[NodeT, int] = {n: len(incomings_fn(n)) for n in nodes}
    inputs = map(lambda x: x[0], filter(lambda x: x[1] == 0, num_incomings.items()))
    to_visit = deque(inputs)
    while to_visit:
        child = to_visit.popleft()
        yield child
        for n in outcomings_fn(child):
            num_incomings[n] -= 1
            if num_incomings[n] == 0:
                to_visit.append(n)
    if sum(num_incomings.values()):
        raise ValueError("The graph has at least one cycle. No topological ordering exists.")


def layerwise_topological_ordering(
    nodes: Iterable[NodeT],
    incomings_fn: Callable[[NodeT], Sequence[NodeT]],
    outcomings_fn: Callable[[NodeT], Sequence[NodeT]] | None = None,
) -> Iterator[list[NodeT]]:
    if outcomings_fn is None:
        if isinstance(nodes, Iterator):
            nodes = list(nodes)
        outgoings = graph_nodes_outgoings(nodes, incomings_fn)
        outcomings_fn = lambda n: outgoings.get(n, [])
    num_incomings: dict[NodeT, int] = {n: len(incomings_fn(n)) for n in nodes}
    inputs = list(map(lambda x: x[0], filter(lambda x: x[1] == 0, num_incomings.items())))
    yield inputs
    prev_ordering = inputs
    while True:
        ls = []
        for n in prev_ordering:
            for i in outcomings_fn(n):
                num_incomings[i] -= 1
                if num_incomings[i] == 0:
                    ls.append(i)
        if not ls:
            break
        yield ls
        prev_ordering = ls
    if sum(num_incomings.values()) != 0:
        raise ValueError("The graph has at least one cycle. No topological ordering exists.")


def topologically_process_nodes(
    ordering: Iterable[NodeT],
    outputs: Iterable[NodeT],
    process_fn: Callable[[NodeT], NodeT],
    *,
    incomings_fn: Callable[[NodeT], Sequence[NodeT]],
) -> tuple[Sequence[NodeT], dict[NodeT, Sequence[NodeT]], Sequence[NodeT]]:
    nodes_map = {}
    in_nodes = {}
    for n in ordering:
        new_n = process_fn(n)
        nodes_map[n] = new_n
        in_nodes[new_n] = [nodes_map[ni] for ni in incomings_fn(n)]
    nodes = list(nodes_map.values())
    outputs = [nodes_map[n] for n in outputs]
    return nodes, in_nodes, outputs


class Graph(Generic[NodeT]):
    def __init__(
        self,
        nodes: Sequence[NodeT],
        in_nodes: Mapping[NodeT, Sequence[NodeT]],
    ):
        self._nodes = nodes
        self._in_nodes = in_nodes
        self._out_nodes = graph_nodes_outgoings(nodes, self.node_inputs)

    def node_inputs(self, n: NodeT) -> Sequence[NodeT]:
        return self._in_nodes.get(n, [])

    def node_outputs(self, n: NodeT) -> Sequence[NodeT]:
        return self._out_nodes.get(n, [])

    @property
    def nodes(self) -> Sequence[NodeT]:
        return self._nodes

    @property
    def nodes_inputs(self) -> Mapping[NodeT, Sequence[NodeT]]:
        return self._in_nodes

    @property
    def nodes_outputs(self) -> Mapping[NodeT, Sequence[NodeT]]:
        return self._out_nodes

    @property
    def inputs(self) -> Iterator[NodeT]:
        return (n for n in self._nodes if not self.node_inputs(n))


class DiAcyclicGraph(Graph[NodeT]):
    def __init__(
        self,
        nodes: Sequence[NodeT],
        in_nodes: Mapping[NodeT, Sequence[NodeT]],
        outputs: Sequence[NodeT],
    ):
        super().__init__(nodes, in_nodes)
        self._outputs = outputs

    @property
    def outputs(self) -> Sequence[NodeT]:
        return self._outputs

    def topological_ordering(self) -> Iterator[NodeT]:
        return topological_ordering(self._nodes, self.node_inputs, self.node_outputs)

    def layerwise_topological_ordering(self) -> Iterator[list[NodeT]]:
        return layerwise_topological_ordering(self._nodes, self.node_inputs, self.node_outputs)

    def subgraph(self, *roots: NodeT) -> "DiAcyclicGraph[NodeT]":
        nodes, in_nodes = subgraph(roots, self.node_inputs)
        return DiAcyclicGraph[NodeT](nodes, in_nodes, outputs=roots)


class RootedDiAcyclicGraph(DiAcyclicGraph[NodeT]):
    def __init__(
        self,
        nodes: Sequence[NodeT],
        in_nodes: Mapping[NodeT, Sequence[NodeT]],
        outputs: Sequence[NodeT],
    ):
        if len(outputs) != 1:
            raise ValueError("The graph should have exactly one output node.")
        super().__init__(nodes, in_nodes, outputs)

    @property
    def output(self) -> NodeT:
        (output,) = self._outputs
        return output


BiMapLeftT = TypeVar("BiMapLeftT")
BiMapRightT = TypeVar("BiMapRightT")


class BiMap(Generic[BiMapLeftT, BiMapRightT]):
    def __init__(self):
        self._lhs_map: dict[BiMapLeftT, BiMapRightT] = {}
        self._rhs_map: dict[BiMapRightT, BiMapLeftT] = {}

    def has_left(self, lhs: BiMapLeftT) -> bool:
        return lhs in self._lhs_map

    def has_right(self, rhs: BiMapRightT) -> bool:
        return rhs in self._rhs_map

    def get_left(self, lhs: BiMapLeftT) -> BiMapRightT:
        return self._lhs_map[lhs]

    def get_right(self, rhs: BiMapRightT) -> BiMapLeftT:
        return self._rhs_map[rhs]

    def add(self, lhs: BiMapLeftT, rhs: BiMapRightT):
        assert not self.has_left(lhs)
        assert not self.has_right(rhs)
        self._lhs_map[lhs] = rhs
        self._rhs_map[rhs] = lhs
