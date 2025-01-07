import itertools
import re
from collections import defaultdict, deque
from itertools import chain

from cirkit.templates.logic.graph import (
    BottomNode,
    ConjunctionNode,
    DisjunctionNode,
    LiteralNode,
    LogicalCircuit,
    LogicalCircuitNode,
    NegatedLiteralNode,
    TopNode,
)


def sliding_window(iterable, n):
    """Collect data into overlapping fixed-length chunks or blocks.
    taken from https://docs.python.org/3/library/itertools.html
    """
    # sliding_window('ABCDEFG', 4) → ABCD BCDE CDEF DEFG
    iterator = iter(iterable)
    window = deque(itertools.islice(iterator, n - 1), maxlen=n)
    for x in iterator:
        window.append(x)
        yield tuple(window)


class SDD(LogicalCircuit):
    @staticmethod
    def load(filename: str) -> "SDD":
        """Load the SDD from a file.
        The file will be opened with mode="r" and encoding="utf-8".

        Syntax of each line in the file:
            sdd count-of-sdd-nodes
            F id-of-false-sdd-node
            T id-of-true-sdd-node
            L id-of-literal-sdd-node id-of-vtree literal
            D id-of-decomposition-sdd-node id-of-vtree number-of-elements {id-of-prime id-of-sub}*

        The ids of sdd nodes start at 0. Nodes appear bottom-up, children before parents.

        Args:
            filename (str): The file name for loading.

        Returns:
            LogicalCircuit: The loaded logic graph.
        """
        tag_re = re.compile(r"^(c|sdd|F|T|L|D)")
        line_re = re.compile(r"(-?\d+)")

        nodes_map: dict[int, LogicalCircuitNode] = {}
        literal_map: dict[tuple[int, bool], LogicalCircuitNode] = {}
        in_nodes: dict[LogicalCircuitNode, list[LogicalCircuitNode]] = defaultdict(list)

        with open(filename, encoding="utf-8") as f:
            for line in f.readlines():
                tag = tag_re.findall(line)[0]
                args = map(int, line_re.findall(line))

                match tag:
                    case "L":
                        # literal numbering starts from 1
                        n_id, _, l = args

                        if l > 0:
                            node = LiteralNode(abs(l) - 1)
                            nodes_map[n_id] = node
                            literal_map[(abs(l), True)] = node
                        else:
                            node = NegatedLiteralNode(abs(l) - 1)
                            nodes_map[n_id] = node
                            literal_map[(abs(l), False)] = node
                    case "F":
                        (n_id,) = args
                        nodes_map[n_id] = BottomNode()
                    case "T":
                        (n_id,) = args
                        nodes_map[n_id] = TopNode()
                    case "D":
                        n_id, _, _, *ds = args
                        decomposition_node = DisjunctionNode()
                        nodes_map[n_id] = decomposition_node

                        for prime, sub in zip(*([iter(ds)] * 2), strict=True):
                            conjunct = ConjunctionNode()
                            in_nodes[conjunct] = [nodes_map[prime], nodes_map[sub]]
                            in_nodes[decomposition_node].append(conjunct)

        nodes = list(set(chain(*in_nodes.values())).union(in_nodes.keys()))
        graph = SDD(nodes, in_nodes, [nodes_map[0]])

        return graph
