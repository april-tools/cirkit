from collections import defaultdict
from enum import IntEnum, auto
from typing import (
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
)

from joblib import Parallel, delayed

from cirkit.backend.torch.graph.modules import TorchModule


class OptMatchStrategy(IntEnum):
    LARGEST_MATCH = auto()


class GraphOptPatternDefn(Generic[TorchModule]):
    @classmethod
    def is_output(cls) -> bool:
        return False

    @classmethod
    def entries(cls) -> List[Type[TorchModule]]:
        ...


GraphOptPattern = Type[GraphOptPatternDefn[TorchModule]]


class GraphOptMatch(Generic[TorchModule]):
    def __init__(self, pattern: GraphOptPattern[TorchModule], entries: List[TorchModule]):
        self._pattern = pattern
        self._entries = entries

    @property
    def pattern(self) -> GraphOptPattern[TorchModule]:
        return self._pattern

    @property
    def entries(self) -> List[TorchModule]:
        return self._entries


class PatternMatcherFunc(Protocol):
    def __call__(
        self,
        module: TorchModule,
        pattern: GraphOptPattern[TorchModule],
        *,
        incomings_fn: Callable[[TorchModule], List[TorchModule]],
    ) -> Optional[GraphOptMatch[TorchModule]]:
        ...


class MatchOptimizerFunc(Protocol):
    def __call__(
        self,
        match: GraphOptMatch[TorchModule],
    ) -> Tuple[TorchModule, ...]:
        ...


def optimize_graph(
    ordering: Iterable[TorchModule],
    outputs: Iterable[TorchModule],
    patterns: Iterable[GraphOptPattern],
    *,
    incomings_fn: Callable[[TorchModule], List[TorchModule]],
    pattern_matcher_fn: PatternMatcherFunc,
    match_optimizer_fn: MatchOptimizerFunc,
    strategy: OptMatchStrategy = OptMatchStrategy.LARGEST_MATCH,
) -> Optional[
    Tuple[
        List[TorchModule],
        Dict[TorchModule, List[TorchModule]],
        Dict[TorchModule, List[TorchModule]],
    ]
]:
    # TODO: generalize this as to cover patterns with multiply entry or exit points? (much more difficult)

    ordering = list(ordering) if isinstance(ordering, Iterator) else ordering
    outputs = list(outputs) if isinstance(outputs, Iterator) else outputs

    # Match optimization patterns
    # matches: list of all matched and grounded optimization rules
    # module_matches: a map from modules to the matches they belong to, if any
    matches, module_matches = match_optimization_patterns(
        ordering,
        outputs,
        patterns,
        incomings_fn=incomings_fn,
        pattern_matcher_fn=pattern_matcher_fn,
        strategy=strategy,
    )

    # Check if no matches have been found. If so, then just return None
    if not matches:
        return None

    # Run the matched optimization rules and collect the optimized modules
    match_opt_modules: Dict[GraphOptMatch, Tuple[TorchModule, ...]] = {}
    for match in matches:
        match_opt_modules[match] = match_optimizer_fn(match)

    # The list of optimized layer and the inputs/outputs of each optimized module
    modules: List[TorchModule] = []
    in_modules: Dict[TorchModule, List[TorchModule]] = {}
    out_modules: Dict[TorchModule, List[TorchModule]] = defaultdict(list)

    # A map from matches to their entry point unoptimized modules
    match_entry_points: Dict[GraphOptMatch, TorchModule] = {}

    # A map from matches to their exit point unoptimized modules
    match_exit_points: Dict[GraphOptMatch, TorchModule] = {}

    # Build the optimize graph by following the topological ordering
    for module in ordering:
        match = module_matches.get(module, None)

        # Check if the layer does not belong to any matched pattern
        # If so, then just add it to the optimize layer as is
        if match is None:
            modules.append(module)
            module_ins = [
                match_exit_points[module_matches[mi]] if mi in module_matches else mi
                for mi in incomings_fn(module)
            ]
            in_modules[module] = module_ins
            for mi in module_ins:
                out_modules[mi].append(module)
            continue

        # Check if the module is the root within the matched pattern
        # If so, then add the corresponding sub-computational-graph optimization to the
        # optimized graph, and build the connections
        if module == match.entries[0]:
            opt_modules = match_opt_modules[match]
            modules.extend(opt_modules)
            for i, om in enumerate(opt_modules):
                if i == 0:
                    om_module_ins = [
                        match_exit_points[module_matches[mi]] if mi in module_matches else mi
                        for mi in incomings_fn(match_entry_points[match])
                    ]
                else:
                    om_module_ins = [opt_modules[i - 1]]
                in_modules[om] = om_module_ins
                for mi in om_module_ins:
                    out_modules[mi].append(om)
            # Set the root model of the match as the exit point of the matched pattern
            match_exit_points[match] = opt_modules[-1]
            continue

        # If the module belongs to a matched pattern (there can only be a single one by construction),
        # but it is not the root in that pattern,
        # then register it as the entry point of the matched sub-computational-graph, if not other entry
        # point has been registered before.
        if match not in match_entry_points:
            match_entry_points[match] = module

    return modules, in_modules, out_modules


def match_optimization_patterns(
    ordering: Iterable[TorchModule],
    outputs: Iterable[TorchModule],
    patterns: Iterable[GraphOptPattern[TorchModule]],
    *,
    incomings_fn: Callable[[TorchModule], List[TorchModule]],
    pattern_matcher_fn: PatternMatcherFunc,
    strategy: OptMatchStrategy = OptMatchStrategy.LARGEST_MATCH,
    num_jobs: int = 1,
) -> Tuple[List[GraphOptMatch[TorchModule]], Dict[TorchModule, GraphOptMatch[TorchModule]]]:
    # A map from modules to the list of found matches they belong to
    module_matches: Dict[TorchModule, List[GraphOptMatch[TorchModule]]] = defaultdict(list)

    # For each given pattern, match it on the graph
    for pattern in patterns:
        # Get an iterator of matches, for a given pattern
        for match in _match_pattern_graph(
            ordering,
            outputs,
            pattern,
            incomings_fn=incomings_fn,
            pattern_matcher_fn=pattern_matcher_fn,
            num_jobs=num_jobs,
        ):
            # For each module found in a match, update the map from modules to found matches
            for matched_module in match.entries:
                module_matches[matched_module].append(match)

    # Prioritize the matched patterns
    prioritized_module_matches = _prioritize_optimization_strategy(
        ordering, module_matches, strategy=strategy, in_place=True
    )

    # Extract all the matches that are still active
    prioritized_matches = list(set(prioritized_module_matches.values()))

    return prioritized_matches, prioritized_module_matches


def _prioritize_optimization_strategy(
    ordering: Iterable[TorchModule],
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
    for module in ordering:
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
    nodes: Iterable[TorchModule],
    outputs: Iterable[TorchModule],
    pattern: GraphOptPattern[TorchModule],
    *,
    incomings_fn: Callable[[TorchModule], List[TorchModule]],
    pattern_matcher_fn: PatternMatcherFunc,
    num_jobs: int = 1,
) -> Iterator[GraphOptMatch[TorchModule]]:
    # Tries to match a pattern by rooting it in all the modules of the computational graph
    # This can be parallelized through joblib
    modules = outputs if pattern.is_output() else nodes
    optional_matches = Parallel(n_jobs=num_jobs, backend="threading")(
        delayed(pattern_matcher_fn)(m, pattern, incomings_fn=incomings_fn) for m in modules
    )
    return filter(lambda match: match is not None, optional_matches)
