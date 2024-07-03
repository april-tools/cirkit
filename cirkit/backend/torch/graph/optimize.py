from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Generic, Iterator, List, Optional, Type

from joblib import Parallel, delayed

from cirkit.backend.torch.graph.modules import TorchDiAcyclicGraph
from cirkit.backend.torch.graph.nodes import TorchModule


@dataclass(frozen=True)
class GraphOptEntry(Generic[TorchModule]):
    cls: Type[TorchModule]


@dataclass(frozen=True)
class GraphOptPatternDefn(Generic[TorchModule]):
    entries: Dict[int, GraphOptEntry[TorchModule]]
    output: bool = False


GraphOptPattern = Type[GraphOptPatternDefn[TorchModule]]


@dataclass(frozen=True)
class GraphOptMatch(Generic[TorchModule]):
    pattern: GraphOptPattern[TorchModule]
    entries: Dict[int, TorchModule]


def match_optimization_patterns(
    graph: TorchDiAcyclicGraph[TorchModule],
    patterns: List[GraphOptPattern[TorchModule]],
    *,
    num_jobs: int = 1,
) -> Dict[TorchModule, List[GraphOptMatch[TorchModule]]]:
    # A map from modules to the list of found matches they belong to
    matches: Dict[TorchModule, List[GraphOptMatch[TorchModule]]] = defaultdict(list)

    # For each given pattern, match it on the graph
    for pattern in patterns:
        # Get an iterator of matches, for a given pattern
        for match in _match_pattern_graph(graph, pattern, num_jobs=num_jobs):
            # For each module found in a match, update the map from modules to found matches
            for matched_module in match.entries.values():
                matches[matched_module].append(match)

    return matches


def prioritize_optimization_strategy(
    graph: TorchDiAcyclicGraph[TorchModule],
    matches: Dict[TorchModule, List[GraphOptMatch[TorchModule]]],
    *,
    strategy: str = "largest-match",
) -> Dict[TorchModule, GraphOptMatch[TorchModule]]:
    matches = matches.copy()
    prioritized_matches: Dict[TorchModule, GraphOptMatch[TorchModule]] = {}

    # Follow the topological ordering of the computational graph and prune
    # pattern matches, according to the given prioritization strategy
    for module in graph.topological_ordering():
        module_matches = matches[module]
        if len(module_matches) <= 1:
            continue

        # Sort the matches based on the given strategy
        sorted_module_matches = _sort_matches_priority(module_matches, strategy=strategy)

        # Prune the 'excess' pattern matches
        for match in sorted_module_matches[1:]:
            for mid, m in match.entries:
                matches[m].remove(match)
        prioritized_matches[module] = sorted_module_matches[0]

    return prioritized_matches


def _sort_matches_priority(
    matches: List[GraphOptMatch[TorchModule]], *, strategy: str = "largest"
) -> List[GraphOptMatch[TorchModule]]:
    if strategy not in ["largest"]:
        raise ValueError(f"Unknown prioritization strategy named '{strategy}'")
    if strategy == "largest-match":
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
        mcls = pattern_entries[mid].cls
        if not issubclass(module, mcls):
            return None
        in_modules = incomings_fn(module)
        if len(in_modules) > 1 or (not in_modules and mid < num_entries - 1):
            return None
        match_entries[mid] = module
        if mid != num_entries - 1:
            (module,) = in_modules

    return GraphOptMatch(pattern, match_entries)
