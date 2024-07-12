from collections import defaultdict, deque
from functools import cached_property
from typing import Callable, Dict, Generic, Iterable, Iterator, List, Optional, Sequence, TypeVar

NodeType = TypeVar("NodeType")


def graph_outgoings(
    nodes: Iterable[NodeType], incomings_fn: Callable[[NodeType], Sequence[NodeType]]
) -> Dict[NodeType, List[NodeType]]:
    outgoings: Dict[NodeType, List[NodeType]] = defaultdict(list)
    for n in nodes:
        incomings = incomings_fn(n)
        for ch in incomings:
            outgoings[ch].append(n)
    return outgoings


def bfs(
    roots: Iterable[NodeType], incomings_fn: Callable[[NodeType], Sequence[NodeType]]
) -> Iterator[NodeType]:
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


def topological_ordering(
    nodes: Iterable[NodeType],
    incomings_fn: Callable[[NodeType], Sequence[NodeType]],
    outcomings_fn: Optional[Callable[[NodeType], Sequence[NodeType]]] = None,
) -> Iterator[NodeType]:
    if outcomings_fn is None:
        if isinstance(nodes, Iterator):
            nodes = list(nodes)
        outgoings = graph_outgoings(nodes, incomings_fn)
        outcomings_fn = lambda n: outgoings.get(n, [])
    num_incomings: Dict[NodeType, int] = {n: len(incomings_fn(n)) for n in nodes}
    inputs = map(lambda x: x[0], filter(lambda x: x[1] == 0, num_incomings.items()))
    to_visit = deque(inputs)
    while to_visit:
        child = to_visit.popleft()
        yield child
        for n in outcomings_fn(child):
            num_incomings[n] -= 1
            if num_incomings[n] == 0:
                to_visit.append(n)
    if sum(num_incomings.values()) != 0:
        raise ValueError("The graph has at least one cycle. No topological ordering exists.")


def layerwise_topological_ordering(
    nodes: Iterable[NodeType],
    incomings_fn: Callable[[NodeType], Sequence[NodeType]],
    outcomings_fn: Optional[Callable[[NodeType], Sequence[NodeType]]] = None,
) -> Iterator[List[NodeType]]:
    if outcomings_fn is None:
        if isinstance(nodes, Iterator):
            nodes = list(nodes)
        outgoings = graph_outgoings(nodes, incomings_fn)
        outcomings_fn = lambda n: outgoings.get(n, [])
    num_incomings: Dict[NodeType, int] = {n: len(incomings_fn(n)) for n in nodes}
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


class Graph(Generic[NodeType]):
    def __init__(
        self,
        nodes: List[NodeType],
        in_nodes: Dict[NodeType, List[NodeType]],
        out_nodes: Dict[NodeType, List[NodeType]],
    ):
        self._nodes = nodes
        self._in_nodes = in_nodes
        self._out_nodes = out_nodes

    def node_inputs(self, n: NodeType) -> List[NodeType]:
        return self._in_nodes.get(n, [])

    def node_outputs(self, n: NodeType) -> List[NodeType]:
        return self._out_nodes.get(n, [])

    @property
    def nodes(self) -> List[NodeType]:
        return self._nodes

    @property
    def nodes_inputs(self) -> Dict[NodeType, List[NodeType]]:
        return self._in_nodes

    @property
    def nodes_outputs(self) -> Dict[NodeType, List[NodeType]]:
        return self._out_nodes

    @property
    def inputs(self) -> Iterator[NodeType]:
        return (n for n in self._nodes if not self.node_inputs(n))

    @property
    def outputs(self) -> Iterator[NodeType]:
        return (n for n in self._nodes if not self.node_outputs(n))


class DiAcyclicGraph(Graph[NodeType]):
    def __init__(
        self,
        nodes: List[NodeType],
        in_nodes: Dict[NodeType, List[NodeType]],
        out_nodes: Dict[NodeType, List[NodeType]],
        *,
        topologically_ordered: bool = False,
    ):
        super().__init__(nodes, in_nodes, out_nodes)
        self._topologically_ordered = topologically_ordered

    @property
    def is_topologically_ordered(self) -> bool:
        return self._topologically_ordered

    def topological_ordering(
        self, roots: Optional[Iterable[NodeType]] = None
    ) -> Iterator[NodeType]:
        if self.is_topologically_ordered and roots is None:
            return iter(self.nodes)
        nodes = self._nodes if roots is None else bfs(roots, self.node_inputs)
        return topological_ordering(self._nodes, self.node_inputs, self.node_outputs)

    def layerwise_topological_ordering(self) -> Iterator[List[NodeType]]:
        return layerwise_topological_ordering(self._nodes, self.node_inputs, self.node_outputs)


class RootedDiAcyclicGraph(DiAcyclicGraph[NodeType]):
    @cached_property
    def output(self) -> NodeType:
        outputs = list(self.outputs)
        if len(outputs) != 1:
            raise ValueError("The graph should have exactly one output node.")
        (output,) = outputs
        return output


class BiRootedDiAcyclicGraph(RootedDiAcyclicGraph[NodeType]):
    @cached_property
    def input(self) -> NodeType:
        inputs = list(self.inputs)
        if len(inputs) != 1:
            raise ValueError("The graph should have exactly one input node.")
        (input,) = inputs
        return input


BimapLeftType = TypeVar("BimapLeftType")
BimapRightType = TypeVar("BimapRightType")


class BiMap(Generic[BimapLeftType, BimapRightType]):
    def __init__(self):
        self._lhs_map: Dict[BimapLeftType, BimapRightType] = {}
        self._rhs_map: Dict[BimapRightType, BimapLeftType] = {}

    def has_left(self, lhs: BimapLeftType) -> bool:
        return lhs in self._lhs_map

    def has_right(self, rhs: BimapRightType) -> bool:
        return rhs in self._rhs_map

    def get_left(self, lhs: BimapLeftType) -> BimapRightType:
        return self._lhs_map[lhs]

    def get_right(self, rhs: BimapRightType) -> BimapLeftType:
        return self._rhs_map[rhs]

    def add(self, lhs: BimapLeftType, rhs: BimapRightType):
        assert not self.has_left(lhs)
        assert not self.has_right(rhs)
        self._lhs_map[lhs] = rhs
        self._rhs_map[rhs] = lhs
