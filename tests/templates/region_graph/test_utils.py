import tempfile
from collections import defaultdict

from cirkit.templates.region_graph import RegionGraph


def check_equivalent_region_graphs(rg1: RegionGraph, rg2: RegionGraph) -> None:
    # TODO: this method only works when there is exactly one instance of region node for each scope
    assert (
        rg1.scope == rg2.scope
    ), f"Region graphs have different scopes: '{rg1.scope}' and '{rg2.scope}'"
    assert len(rg1.nodes) == len(
        rg2.nodes
    ), f"Region graphs have not the same number of nodes: '{len(rg1.nodes)}' and '{len(rg2.nodes)}'"

    rg1_scope_factorizations = defaultdict(set)
    for ptn in rg1.partition_nodes:
        rgns = rg1.partition_inputs(ptn)
        rg1_scope_factorizations[frozenset(ptn.scope)].add(
            tuple(frozenset(rgn.scope) for rgn in rgns)
        )
    rg2_scope_factorizations = defaultdict(set)
    for ptn in rg2.partition_nodes:
        rgns = rg2.partition_inputs(ptn)
        rg2_scope_factorizations[frozenset(ptn.scope)].add(
            tuple(frozenset(rgn.scope) for rgn in rgns)
        )
    print(rg1_scope_factorizations)
    print(rg2_scope_factorizations)
    assert (
        rg1_scope_factorizations == rg2_scope_factorizations
    ), "Region graphs have different scope factorizations"


def check_region_graph_save_load(rg: RegionGraph) -> None:
    with tempfile.NamedTemporaryFile("r+") as f:
        rg.dump(f.name)
        f.seek(0)
        loaded_rg = RegionGraph.load(f.name)
        check_equivalent_region_graphs(rg, loaded_rg)
