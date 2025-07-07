from collections.abc import Callable
from os import PathLike
from pathlib import Path

import graphviz

from cirkit.templates.region_graph.graph import RegionGraph, RegionNode, PartitionNode


def plot_region_graph(
    region_graph: RegionGraph,
    out_path: str | PathLike[str] | None = None,
    orientation: str = "vertical",
    region_node_shape: str = "box",
    partition_node_shape: str = "point",
    label_font: str = "times italic bold",
    label_size: str = "21pt",
    label_color: str = "white",
    region_label: str | Callable[[RegionNode], str] | None = None,
    region_color: str | Callable[[RegionNode], str] = "#607d8b",
    partition_label: str | Callable[[PartitionNode], str] | None = None,
    partition_color: str | Callable[[PartitionNode], str] = "#ffbd2a",
) -> graphviz.Digraph:
    """Plot the region graph using graphviz.

    Args:
        region_graph (RegionGraph): The region graph to plot.
        out_path (str | PathLike[str] | None, optional): The output path where the plot is saved.
            If it is None, the plot is not saved to a file. Defaults to None.
            The Output file format is deduced from the path. Possible formats are:
            {'jp2', 'plain-ext', 'sgi', 'x11', 'pic', 'jpeg', 'imap', 'psd', 'pct',
             'json', 'jpe', 'tif', 'tga', 'gif', 'tk', 'xlib', 'vmlz', 'json0', 'vrml',
             'gd', 'xdot', 'plain', 'cmap', 'canon', 'cgimage', 'fig', 'svg', 'dot_json',
             'bmp', 'png', 'cmapx', 'pdf', 'webp', 'ico', 'xdot_json', 'gtk', 'svgz',
             'xdot1.4', 'cmapx_np', 'dot', 'tiff', 'ps2', 'gd2', 'gv', 'ps', 'jpg',
             'imap_np', 'wbmp', 'vml', 'eps', 'xdot1.2', 'pov', 'pict', 'ismap', 'exr'}.
             See https://graphviz.org/docs/outputs/ for more.
        orientation (str, optional): Orientation of the graph. "vertical" puts the root
            node at the top, "horizontal" at left. Defaults to "vertical".
        node_shape (str, optional): Default shape for a node in the graph. Defaults to "box".
            See https://graphviz.org/doc/info/shapes.html for the supported shapes.
        label_font (str, optional): Font used to render labels. Defaults to "times italic bold".
            See https://graphviz.org/faq/font/ for the available fonts.
        label_size (str, optional): Size of the font for labels in points. Defaults to "21pt".
        label_color (str, optional): Color for the labels in the nodes. Defaults to "white".
            See https://graphviz.org/docs/attr-types/color/ for supported color.
        region_label (str | Callable[[RegionNode], str] | None, optional): Either a string or a function.
            If a function is provided, then it must take as input a region node and returns a string
            that will be used as label. If None, it defaults to the string representation of the scope of the
            region node.
        region_color (str | Callable[[RegionNode], str], optional): Either a string or a function.
            If a function is provided, then it must take as input a region node and returns a string
            that will be used as color for the region node. Defaults to "#607d8b".
        partition_label (str | Callable[[PartitionNode], str] | None, optional): Either a string or a
            function. If a function is provided, then it must take as input a partition node and returns a
            string that will be used as label. If None, it defaults to an empty string.
        partition_color (str | Callable[[PartitionNode], str], optional): Either a string or a function.
            If a function is provided, then it must take as input a partition node and returns a string
            that will be used as color for the partition node. Defaults to "#ffbd2a".

    Raises:
        ValueError: The format is not among the supported ones.
        ValueError: The direction is not among the supported ones.

    Returns:
        graphviz.Digraph: The graphviz object representing the region graph.
    """

    if out_path is None:
        fmt: str = "svg"
    else:
        fmt: str = Path(out_path).suffix.replace(".", "")
        if fmt not in graphviz.FORMATS:
            raise ValueError(f"Supported formats are {graphviz.FORMATS}.")

    if orientation not in ["vertical", "horizontal"]:
        raise ValueError("Supported graph directions are only 'vertical' and 'horizontal'.")

    def _default_region_label(rgn: RegionNode) -> str:
        return str(set(rgn.scope))

    def _default_partition_label(rgn: PartitionNode) -> str:
        return ""

    if region_label is None:
        region_label = _default_region_label
    if partition_label is None:
        partition_label = _default_partition_label

    dot: graphviz.Digraph = graphviz.Digraph(
        format=fmt,
        node_attr={
            "style": "filled",
            "fontcolor": label_color,
            "fontsize": label_size,
            "fontname": label_font,
        },
        engine="dot",
    )
    dot.graph_attr["rankdir"] = "BT" if orientation == "vertical" else "LR"

    for node in region_graph.nodes:
        match node:
            case RegionNode():
                dot.node(
                    str(id(node)),
                    region_label if isinstance(region_label, str) else region_label(node),
                    color=region_color if isinstance(region_color, str) else region_color(node),
                    shape=region_node_shape,
                )
            case PartitionNode():
                dot.node(
                    str(id(node)),
                    partition_label if isinstance(partition_label, str) else partition_label(node),
                    color=(
                        partition_color
                        if isinstance(partition_color, str)
                        else partition_color(node)
                    ),
                    shape=partition_node_shape,
                    width="0.2",
                )
        for node_in in region_graph.node_inputs(node):
            dot.edge(str(id(node_in)), str(id(node)))
    if out_path is not None:
        out_path: Path = Path(out_path).with_suffix("")

        if fmt == "dot":
            with open(out_path, "w", encoding="utf8") as f:
                f.write(dot.source)
        else:
            dot.format = fmt
            dot.render(out_path, cleanup=True)

    return dot
