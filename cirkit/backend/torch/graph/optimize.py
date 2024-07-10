from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Callable, Dict, Generic, Iterable, Iterator, List, Optional, Tuple, Type

from joblib import Parallel, delayed

from cirkit.backend.torch.graph.modules import TorchDiAcyclicGraph, TorchModule


class OptMatchStrategy(IntEnum):
    LARGEST_MATCH = auto()


@dataclass(frozen=True)
class GraphOptPatternDefn(Generic[TorchModule]):
    entries: Dict[int, Type[TorchModule]]
    output: bool = False


GraphOptPattern = Type[GraphOptPatternDefn[TorchModule]]


@dataclass(frozen=True)
class GraphOptMatch(Generic[TorchModule]):
    pattern: GraphOptPattern[TorchModule]
    entries: Dict[int, TorchModule]

    def __hash__(self):
        return hash(id(self))


def match_optimization_patterns(
    graph: TorchDiAcyclicGraph[TorchModule],
    patterns: Iterable[GraphOptPattern[TorchModule]],
    *,
    num_jobs: int = 1,
    strategy: OptMatchStrategy = OptMatchStrategy.LARGEST_MATCH,
) -> Tuple[List[GraphOptMatch[TorchModule]], Dict[TorchModule, GraphOptMatch[TorchModule]]]:
    # A map from modules to the list of found matches they belong to
    module_matches: Dict[TorchModule, List[GraphOptMatch[TorchModule]]] = defaultdict(list)

    # For each given pattern, match it on the graph
    for pattern in patterns:
        # Get an iterator of matches, for a given pattern
        for match in _match_pattern_graph(graph, pattern, num_jobs=num_jobs):
            # For each module found in a match, update the map from modules to found matches
            for matched_module in match.entries.values():
                module_matches[matched_module].append(match)

    # Prioritize the matched patterns
    prioritized_module_matches = _prioritize_optimization_strategy(
        graph, module_matches, strategy=strategy, in_place=True
    )

    # Extract all the matches that are still active
    prioritized_matches = list(set(prioritized_module_matches.values()))

    return prioritized_matches, prioritized_module_matches


def _prioritize_optimization_strategy(
    graph: TorchDiAcyclicGraph[TorchModule],
    module_matches: Dict[TorchModule, List[GraphOptMatch[TorchModule]]],
    *,
    strategy: OptMatchStrategy = OptMatchStrategy.LARGEST_MATCH,
    in_place: bool = True,
) -> Dict[TorchModule, GraphOptMatch[TorchModule]]:
    if not in_place:
        module_matches = module_matches.copy()
    prioritized_module_matches: Dict[TorchModule, GraphOptMatch[TorchModule]] = {}

    # Follow the topological ordering of the computational graph and prune
    # pattern matches, according to the given prioritization strategy
    for module in graph.topological_ordering():
        matches = module_matches[module]
        if not matches:
            continue
        if len(matches) == 1:
            prioritized_module_matches[module] = matches[0]

        # Sort the matches based on the given strategy
        sorted_matches = _sort_matches_priority(matches, strategy=strategy)

        # Prune the 'excess' pattern matches
        for match in sorted_matches[1:]:
            for mid, m in match.entries:
                module_matches[m].remove(match)
        prioritized_module_matches[module] = sorted_matches[0]

    return prioritized_module_matches


def _sort_matches_priority(
    matches: List[GraphOptMatch[TorchModule]],
    *,
    strategy: OptMatchStrategy,
) -> List[GraphOptMatch[TorchModule]]:
    if strategy == OptMatchStrategy.LARGEST_MATCH:
        return sorted(matches, key=lambda m: len(m.entries), reverse=True)
    assert False


def _match_pattern_graph(
    graph: TorchDiAcyclicGraph[TorchModule],
    pattern: GraphOptPattern[TorchModule],
    *,
    num_jobs: int = 1,
) -> Iterator[GraphOptMatch[TorchModule]]:
    # Tries to match a pattern by rooting it in all the modules of the computational graph
    # This can be parallelized through joblib
    modules = list(graph.outputs) if pattern.output else graph.nodes
    optional_matches = Parallel(n_jobs=num_jobs, backend="threading")(
        delayed(_match_pattern_rooted)(m, pattern, incomings_fn=graph.node_inputs) for m in modules
    )
    return filter(lambda match: match is not None, optional_matches)


def _match_pattern_rooted(
    module: TorchModule,
    pattern: GraphOptPattern[TorchModule],
    *,
    incomings_fn: Callable[[TorchModule], List[TorchModule]],
) -> Optional[GraphOptMatch[TorchModule]]:
    pattern_entries = pattern.entries
    match_entries: Dict[int, TorchModule] = {}
    num_entries = len(pattern_entries)

    # Start matching the pattern from the root
    # TODO: generalize to match DAGs or binary trees
    for mid in range(num_entries):
        if not isinstance(module, pattern_entries[mid]):
            return None
        in_modules = incomings_fn(module)
        if len(in_modules) > 1 and mid != num_entries - 1:
            return None
        match_entries[mid] = module
        if mid != num_entries - 1:
            (module,) = in_modules

    return GraphOptMatch(pattern, match_entries)
