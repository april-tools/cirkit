from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from enum import IntEnum, auto
from functools import cached_property
from typing import Any, Generic, Protocol

from cirkit.backend.torch.graph.modules import TorchModuleT


class OptMatchStrategy(IntEnum):
    """Strategy used to sort the matches and determine the one to keep"""

    LARGEST_MATCH = auto()


class GraphOptPatternDefn(Generic[TorchModuleT]):
    """Class defining a pattern in a graph."""

    @classmethod
    def is_output(cls) -> bool:
        """Define if the pattern should be searched in the graph's outputs only.

        Returns:
            bool: if True, search only in the outputs.
        """
        return False

    @classmethod
    def entries(cls) -> Sequence[type[TorchModuleT]]:
        """Returns an ordered sequence of module type that need to be matched in the
        order of the sequence when going through the graph in the reverse topological order.

        For example: the entry `[LayerType3, LayerType2]` will match the graph:

        `LayerType1 -> LayerType2 -> LayerType3 -> LayerType4`

        The match will be the graph rooted at `LayerType3`.

        Raises:
            NotImplementedError: This method need to be implemented for any pattern

        Returns:
            Sequence[type[TorchModuleT]]: the sequence in revese topological order.
        """
        raise NotImplementedError

    @classmethod
    def config_patterns(cls) -> Sequence[Mapping[str, Any]]:
        """Returns a list of dictionaries that match a config name to a config value.

        The “config” of a layer / parameter node is simply the dictionary returned by `Layer.config`.

        The dictionary at position x in the list define the config for the x-th element of the `entries` list.

        For example, the sum layer with config:

        ```python
        {
        "num_input_units":2,
        "num_output_units":1,
        "arity":1
        }
        ```

        Will match the pattern:

        ```python
        class ExamplePattern(LayerOptPatternDefn):
            def config_patterns():
                return [{"arity":1}]
            def entries():
                return [TorchSumLayer]
        ```


        Returns:
            Sequence[Mapping[str, Any]]: List of config name -> config value mappings
        """
        return ()

    @classmethod
    def sub_patterns(cls) -> Sequence[Mapping[str, type["GraphOptPatternDefn"]]]:
        """Returns a list of dictionaries that map layer's parameter names to a `ParameterOptPattern`.

        The dictionary at position x in the list define the config for the x-th element of the `entries` list.

        For example, you can match the weight parameter of a sum layer to be of a certain `ParameterType`:

        ```python
        class LayerPatternOne(LayerOptPatternDefn):
            @classmethodParameterType
            def entries(cls) -> Sequence[type[TorchLayer]]:
                return [TorchSumLayer]

            @classmethod
            def sub_patterns(cls) -> Sequence[dict[str, ParameterOptPattern]]:
                return [{"weight": ParameterPatternOne}]

        class ParameterPatternOne(ParameterOptPatternDefn):
            @classmethod
            def entries(cls) -> list[type[TorchParameterNode]]:
                return [ParameterType]
        ```

        `LayerPatternOne` will match the following layer:

        ```python
        TorchSumLayer(1,1,1,weight=ParameterType(...)
        ```

        Returns:
            Sequence[Mapping[str, type[GraphOptPatternDefn]]]: List of dictionaries that
             map layer's parameter names to a `ParameterOptPattern`
        """
        return ()


GraphOptPattern = type[GraphOptPatternDefn[TorchModuleT]]


class GraphOptMatch(Generic[TorchModuleT]):
    """Class storing data related to a single match:
    - pattern (GraphOptPattern[TorchModuleT]): the pattern of the match.
    - entries (Sequence[TorchModuleT]): Modules, from the graph being searched,
        matching the `entries` types of the pattern.
    - sub_entries (Sequence[Mapping[str, Sequence[GraphOptMatch]]]):
        Modules corresponding to the `sub_pattern` method of the pattern.
    """

    def __init__(
        self,
        pattern: GraphOptPattern[TorchModuleT],
        entries: Sequence[TorchModuleT],
        sub_entries: Sequence[Mapping[str, Sequence["GraphOptMatch"]]] | None = None,
    ):
        self._pattern = pattern
        self._entries = entries
        self._sub_entries = sub_entries if sub_entries is not None else ()

    @property
    def pattern(self) -> GraphOptPattern[TorchModuleT]:
        return self._pattern

    @property
    def entries(self) -> Sequence[TorchModuleT]:
        return self._entries

    @property
    def sub_entries(self) -> Sequence[Mapping[str, Sequence["GraphOptMatch"]]]:
        return self._sub_entries

    @cached_property
    def size(self) -> int:
        """Count the number of entries and sub_entries in the match

        Returns:
            int: Number of entries and sub_entries
        """
        size = len(self._entries)
        for sub_entry in self.sub_entries:
            for matches in sub_entry.values():
                size += sum(match.size for match in matches)
        return size


class PatternMatcherFunc(Protocol[TorchModuleT]):
    """Defines the signature of a valid pattern matching function.

    Pattern matching function are functions which attempt to match
    a pattern in a graph starting at a given module.

    They return either a match object or None if the match fails.
    """

    def __call__(
        self,
        module: TorchModuleT,
        pattern: GraphOptPattern[TorchModuleT],
        /,
        *,
        incomings_fn: Callable[[TorchModuleT], Sequence[TorchModuleT]],
        outcomings_fn: Callable[[TorchModuleT], Sequence[TorchModuleT]],
    ) -> GraphOptMatch[TorchModuleT] | None: ...


class MatchOptimizerFunc(Protocol[TorchModuleT]):
    """Defines the signature of a valide match optimizer.

    Match optimizer are function which take a match object and returns
    the tuple of module that should be put in its place to optimize
    the graph.
    """

    def __call__(
        self,
        match: GraphOptMatch[TorchModuleT],
    ) -> tuple[TorchModuleT, ...]: ...


def optimize_graph(
    ordering: Iterable[TorchModuleT],
    outputs: Iterable[TorchModuleT],
    patterns: Iterable[GraphOptPattern[TorchModuleT]],
    *,
    incomings_fn: Callable[[TorchModuleT], Sequence[TorchModuleT]],
    outcomings_fn: Callable[[TorchModuleT], Sequence[TorchModuleT]],
    pattern_matcher_fn: PatternMatcherFunc[TorchModuleT],
    match_optimizer_fn: MatchOptimizerFunc[TorchModuleT],
    strategy: OptMatchStrategy = OptMatchStrategy.LARGEST_MATCH,
) -> (
    tuple[
        list[TorchModuleT],
        dict[TorchModuleT, list[TorchModuleT]],
        list[TorchModuleT],
    ]
    | None
):
    """Search and Apply the optimization patterns on the graph.

    Args:
        ordering (Iterable[TorchModuleT]): Torch modules in the graph to optimize.
        outputs (Iterable[TorchModuleT]): Torch modules acting as the graph's outputs.
        patterns (Iterable[GraphOptPattern[TorchModuleT]]): Iterable of patterns to search for.
        incomings_fn (Callable[[TorchModuleT], Sequence[TorchModuleT]]):
            Function that returns the inputs of the given module.
        outcomings_fn (Callable[[TorchModuleT], Sequence[TorchModuleT]]):
            Function that returns the outputs of the given module.
        pattern_matcher_fn (PatternMatcherFunc[TorchModuleT]):
            Function that tries to match a pattern using the given node as the root.
        match_optimizer_fn (MatchOptimizerFunc[TorchModuleT]):
            Function that takes a match as parameter and return the tuple of module to replace it.
        strategy (OptMatchStrategy, optional): Optimization strategy to deciding
            which match should be applied. Defaults to OptMatchStrategy.LARGEST_MATCH.

    Returns:
        tuple[list[TorchModuleT],dict[TorchModuleT, list[TorchModuleT]],list[TorchModuleT]] | None:
            - The list of all modules in the optimized graph.
            - The adjacency dictionary of the graph.
            - The list of modules from the graph that are outputs.
            If there are no optimization applied, the function simply returns None.
    """
    # TODO: generalize this as to cover patterns with multiply entry or exit points?

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
        outcomings_fn=outcomings_fn,
        pattern_matcher_fn=pattern_matcher_fn,
        strategy=strategy,
    )

    # Check if no matches have been found. If so, then just return None
    if not matches:
        return None

    # Run the matched optimization rules and collect the optimized modules
    match_opt_modules: dict[GraphOptMatch[TorchModuleT], tuple[TorchModuleT, ...]] = {}
    for match in matches:
        match_opt_modules[match] = match_optimizer_fn(match)

    # The list of optimized layer and the inputs of each optimized module
    modules: list[TorchModuleT] = []
    in_modules: dict[TorchModuleT, list[TorchModuleT]] = {}

    # A map from matches to their entry point unoptimized modules
    match_entry_points: dict[GraphOptMatch[TorchModuleT], TorchModuleT] = {}

    # A map from matches to their exit point unoptimized modules
    match_exit_points: dict[GraphOptMatch[TorchModuleT], TorchModuleT] = {}

    # Build the optimize graph by following the topological ordering
    for module in ordering:
        optional_match = module_matches.get(module, None)

        # Check if the layer does not belong to any matched pattern
        # If so, then just add it to the optimize layer as is
        if optional_match is None:
            modules.append(module)
            in_modules[module] = [
                match_exit_points[module_matches[mi]] if mi in module_matches else mi
                for mi in incomings_fn(module)
            ]
            continue
        match = optional_match

        # If the module belongs to a matched pattern (there can only be a single one
        # by construction), but it is not the root in that pattern,
        # then register it as the entry point of the matched sub-computational-graph,
        # if not other entry point has been registered before.
        if match not in match_entry_points:
            match_entry_points[match] = module

        # Check if the module is the root within the matched pattern
        # If so, then add the corresponding sub-computational-graph optimization to the
        # optimized graph, and build the connections
        if module == match.entries[0]:
            opt_modules = match_opt_modules[match]
            modules.extend(opt_modules)
            for i, om in enumerate(opt_modules):
                if not i:
                    in_modules[om] = [
                        match_exit_points[module_matches[mi]]
                        if mi in module_matches
                        else mi
                        for mi in incomings_fn(match_entry_points[match])
                    ]
                else:
                    in_modules[om] = [opt_modules[i - 1]]
            # Set the root model of the match as the exit point of the matched pattern
            match_exit_points[match] = opt_modules[-1]
            continue

    # Retrieve the sequence of output modules of the computational graph
    opt_outputs = [
        match_exit_points[module_matches[m]] if m in module_matches else m
        for m in outputs
    ]

    return modules, in_modules, opt_outputs


def match_optimization_patterns(
    ordering: Iterable[TorchModuleT],
    outputs: Iterable[TorchModuleT],
    patterns: Iterable[GraphOptPattern[TorchModuleT]],
    *,
    incomings_fn: Callable[[TorchModuleT], Sequence[TorchModuleT]],
    outcomings_fn: Callable[[TorchModuleT], Sequence[TorchModuleT]],
    pattern_matcher_fn: PatternMatcherFunc[TorchModuleT],
    strategy: OptMatchStrategy = OptMatchStrategy.LARGEST_MATCH,
) -> tuple[
    list[GraphOptMatch[TorchModuleT]], dict[TorchModuleT, GraphOptMatch[TorchModuleT]]
]:
    """Find and filter sections of a graph matching patterns.
    The function works as follows:
    1. Use `pattern_matcher_function` to retrieve all matches
        for all patterns.
    2. Filter the matches according to the `strategy`.

    Args:
        ordering (Iterable[TorchModuleT]): Torch modules in the graph to optimize.
        outputs (Iterable[TorchModuleT]): Torch modules acting as the graph's outputs.
        patterns (Iterable[GraphOptPattern[TorchModuleT]]): Iterable of patterns to search for.
        incomings_fn (Callable[[TorchModuleT], Sequence[TorchModuleT]]):
            Function that returns the inputs of the given module.
        outcomings_fn (Callable[[TorchModuleT], Sequence[TorchModuleT]]):
            Function that returns the outputs of the given module.
        pattern_matcher_fn (PatternMatcherFunc[TorchModuleT]):
            Function that tries to match a pattern using the given node as the root.
        strategy (OptMatchStrategy, optional): Optimization strategy to deciding
            which match should be applied. Defaults to OptMatchStrategy.LARGEST_MATCH.

    Returns:
        tuple[list[GraphOptMatch[TorchModuleT]], dict[TorchModuleT, GraphOptMatch[TorchModuleT]]]:
            - List of all the matches after priority-based  filtering
            - Mapping from the module that matches the root of the pattern to the corresponding match.
    """
    ordering = list(ordering) if isinstance(ordering, Iterator) else ordering
    outputs = list(outputs) if isinstance(outputs, Iterator) else outputs

    # A map from modules to the list of found matches they belong to
    module_matches: dict[TorchModuleT, list[GraphOptMatch[TorchModuleT]]] = defaultdict(
        list
    )

    # For each given pattern, match it on the graph
    for pattern in patterns:
        # Get an iterator of matches, for a given pattern
        modules = outputs if pattern.is_output() else ordering
        for match in _match_pattern_graph(
            modules,
            pattern,
            incomings_fn=incomings_fn,
            outcomings_fn=outcomings_fn,
            pattern_matcher_fn=pattern_matcher_fn,
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
    ordering: Iterable[TorchModuleT],
    module_matches: dict[TorchModuleT, list[GraphOptMatch[TorchModuleT]]],
    *,
    strategy: OptMatchStrategy = OptMatchStrategy.LARGEST_MATCH,
    in_place: bool = True,
) -> dict[TorchModuleT, GraphOptMatch[TorchModuleT]]:
    """Sort matches according to an optimization strategy and select the highest priority.

    Args:
        ordering (Iterable[TorchModuleT]): List of compiled module in the graph.
        module_matches (dict[TorchModuleT, list[GraphOptMatch[TorchModuleT]]]): mapping between modules
            and list of matches.
        strategy (OptMatchStrategy, optional): The strategy to use when sorting the matches.
             Defaults to OptMatchStrategy.LARGEST_MATCH.
        in_place (bool, optional): if True, the function will directly modify `module_matches`.
             Defaults to True.

    Returns:
        dict[TorchModuleT, GraphOptMatch[TorchModuleT]]: mapping between modules and exactly one match.
    """

    if not in_place:
        module_matches = module_matches.copy()
    prioritized_module_matches: dict[TorchModuleT, GraphOptMatch[TorchModuleT]] = {}

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
            for m in match.entries:
                module_matches[m].remove(match)
        prioritized_module_matches[module] = sorted_matches[0]

    return prioritized_module_matches


def _sort_matches_priority(
    matches: Sequence[GraphOptMatch[TorchModuleT]],
    *,
    strategy: OptMatchStrategy,
) -> list[GraphOptMatch[TorchModuleT]]:
    """Return a sorted list of matches according to the strategy.

    Args:
        matches (Sequence[GraphOptMatch[TorchModuleT]]): List of matches to sort.
        strategy (OptMatchStrategy): Strategy to use for the priority.
            Only one strategy available: OptMatchStrategy.LARGEST_MATCH:

    Returns:
        list[GraphOptMatch[TorchModuleT]]: Sorted list of matches
    """
    if strategy == OptMatchStrategy.LARGEST_MATCH:
        return sorted(matches, key=lambda m: m.size, reverse=True)
    assert False


def _match_pattern_graph(
    modules: Iterable[TorchModuleT],
    pattern: GraphOptPattern[TorchModuleT],
    *,
    incomings_fn: Callable[[TorchModuleT], Sequence[TorchModuleT]],
    outcomings_fn: Callable[[TorchModuleT], Sequence[TorchModuleT]],
    pattern_matcher_fn: PatternMatcherFunc[TorchModuleT],
) -> Iterator[GraphOptMatch[TorchModuleT]]:
    """Return all matches found for one pattern on a graph.

    Args:
        modules (Iterable[TorchModuleT]): Modules in the graph to search.
        pattern (GraphOptPattern[TorchModuleT]): The pattern to search.
        incomings_fn (Callable[[TorchModuleT], Sequence[TorchModuleT]]):
            Function that returns the inputs of the given module.
        outcomings_fn (Callable[[TorchModuleT], Sequence[TorchModuleT]]):
            Function that returns the outputs of the given module.
        pattern_matcher_fn (PatternMatcherFunc[TorchModuleT]):
            Function that tries to match a pattern using the given node as the root.

    Returns:
        Iterator[GraphOptMatch[TorchModuleT]]: All matches found.
    """
    # Tries to match a pattern by rooting it in all the modules of the computational graph
    optional_matches = map(
        lambda m: pattern_matcher_fn(
            m, pattern, incomings_fn=incomings_fn, outcomings_fn=outcomings_fn
        ),
        modules,
    )
    return (match for match in optional_matches if match is not None)
