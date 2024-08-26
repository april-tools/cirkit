import tempfile

from cirkit.templates.region_graph import PartitionNode, RegionGraph, RegionNode


def check_equivalent_region_graphs(rg1: RegionGraph, rg2: RegionGraph) -> None:
    rg1_nodes = sorted(rg1.nodes)
    rg2_nodes = sorted(rg2.nodes)
    assert len(rg1_nodes) == len(
        rg2_nodes
    ), f"Region graphs have not the same number of nodes: {len(rg1_nodes)} and {len(rg2_nodes)}"
    for n, m in zip(rg1_nodes, rg2_nodes):
        assert (
            n.__class__ == m.__class__
        ), "Region graphs have nodes with different types at the same locations"
        assert n.scope == m.scope, (
            f"Region graphs have nodes with different scopes at the same locations:"
            f" {n.scope} and {m.scope}"
        )


def check_region_graph_save_load(rg: RegionGraph) -> None:
    with tempfile.NamedTemporaryFile("r+") as f:
        rg.dump(f.name)
        f.seek(0)
        loaded_rg = RegionGraph.load(f.name)
        check_equivalent_region_graphs(rg, loaded_rg)
