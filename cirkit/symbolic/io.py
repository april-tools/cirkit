from collections.abc import Callable
from os import PathLike
from pathlib import Path

import graphviz

from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import HadamardLayer, InputLayer, KroneckerLayer, ProductLayer, SumLayer


def plot_circuit(
    circuit: Circuit,
    out_path: str | PathLike[str] | None = None,
    orientation: str = "vertical",
    node_shape: str = "box",
    label_font: str = "times italic bold",
    label_size: str = "21pt",
    label_color: str = "white",
    sum_label: str | Callable[[SumLayer], str] = "+",
    sum_color: str | Callable[[SumLayer], str] = "#607d8b",
    product_label: str | Callable[[ProductLayer], str] | None = None,
    product_color: str | Callable[[ProductLayer], str] = "#24a5af",
    input_label: str | Callable[[InputLayer], str] = None,
    input_color: str | Callable[[InputLayer], str] = "#ffbd2a",
) -> graphviz.Digraph:
    """Plot the current symbolic circuit using graphviz.
    A graphviz object is returned, which can be visualized in jupyter notebooks.
    If format is not provided, SVG is used for optimal rendering in notebooks.

    Args:
        circuit: The symbolic circuit to plot.
        out_path: The output path where the plot is save
            If it is None, the plot is not saved to a file. Defaults to None.
            The Output file format is deduce from the path. Possible formats are:
            {'jp2', 'plain-ext', 'sgi', 'x11', 'pic', 'jpeg', 'imap', 'psd', 'pct',
             'json', 'jpe', 'tif', 'tga', 'gif', 'tk', 'xlib', 'vmlz', 'json0', 'vrml',
             'gd', 'xdot', 'plain', 'cmap', 'canon', 'cgimage', 'fig', 'svg', 'dot_json',
             'bmp', 'png', 'cmapx', 'pdf', 'webp', 'ico', 'xdot_json', 'gtk', 'svgz',
             'xdot1.4', 'cmapx_np', 'dot', 'tiff', 'ps2', 'gd2', 'gv', 'ps', 'jpg',
             'imap_np', 'wbmp', 'vml', 'eps', 'xdot1.2', 'pov', 'pict', 'ismap', 'exr'}.
             See https://graphviz.org/docs/outputs/ for more.
        orientation: Orientation of the graph. "vertical" puts the root
            node at the top, "horizontal" at left. Defaults to "vertical".
        node_shape: Default shape for a node in the graph. Defaults to "box".
            See https://graphviz.org/doc/info/shapes.html for the supported shapes.
        label_font: Font used to render labels. Defaults to "times italic bold".
            See https://graphviz.org/faq/font/ for the available fonts.
        label_size: Size of the font for labels in points. Defaults to 21pt.
        label_color: Color for the labels in the nodes. Defaults to "white".
            See https://graphviz.org/docs/attr-types/color/ for supported color.
        sum_label: Either a string or a function.
            If a function is provided, then it must take as input a sum layer and returns a string
            that will be used as label. Defaults to "+".
        sum_color: Either a string or a function.
            If a function is provided, then it must take as input a sum layer and returns a string
            that will be used as color for the sum node. Defaults to "#607d8b".
        product_label: Either a string or a function.
            If a function is provided, then it must take as input a product layer and returns a
            string that will be used as label. If None, it defaults to "⊙" for Hadamard layers and
            "⊗" for Kronecker layers.
        product_color: Either a string or a function.
            If a function is provided, then it must take as input a product layer and returns a
            string that will be used as color for the product node. Defaults to "#24a5af".
        input_label: Either a string or a function.
            If a function is provided, then it must take as input an input layer and returns a
            string that will be used as label. If None, it defaults to using the scope of the layer.
        input_color: Either a string or a function.
            If a function is provided, then it must take as input an input layer and returns a
            string that will be used as color for the input layer node. Defaults to "#ffbd2a".

    Raises:
        ValueError: The format is not among the supported ones.
        ValueError: The direction is not among the supported ones.

    Returns:
        graphviz.Digraph: _description_
    """
    if out_path is None:
        fmt: str = "svg"
    else:
        fmt: str = Path(out_path).suffix.replace(".", "")
        if fmt not in graphviz.FORMATS:
            raise ValueError(f"Supported formats are {graphviz.FORMATS}.")

    if orientation not in ["vertical", "horizontal"]:
        raise ValueError("Supported graph directions are only 'vertical' and 'horizontal'.")

    def _default_product_label(sl: ProductLayer) -> str:
        match sl:
            case HadamardLayer():
                return "⊙"
            case KroneckerLayer():
                return "⊗"
            case _:
                raise NotImplementedError(
                    f"No default label for product layer of type {sl.__class__}"
                )

    def _default_input_label(sl: InputLayer) -> str:
        return " ".join(map(str, sl.scope))

    if product_label is None:
        product_label = _default_product_label
    if input_label is None:
        input_label = _default_input_label

    dot: graphviz.Digraph = graphviz.Digraph(
        format=fmt,
        node_attr={
            "shape": node_shape,
            "style": "filled",
            "fontcolor": label_color,
            "fontsize": label_size,
            "fontname": label_font,
        },
        engine="dot",
    )
    dot.graph_attr["rankdir"] = "BT" if orientation == "vertical" else "LR"

    for sl in circuit.layers:
        match sl:
            case ProductLayer():
                dot.node(
                    str(id(sl)),
                    product_label if isinstance(product_label, str) else product_label(sl),
                    color=product_color if isinstance(product_color, str) else product_color(sl),
                )
            case SumLayer():
                dot.node(
                    str(id(sl)),
                    sum_label if isinstance(sum_label, str) else sum_label(sl),
                    color=sum_color if isinstance(sum_color, str) else sum_color(sl),
                )
            case InputLayer():
                dot.node(
                    str(id(sl)),
                    input_label if isinstance(input_label, str) else input_label(sl),
                    color=input_color if isinstance(input_color, str) else input_color(sl),
                )

        for sli in circuit.layer_inputs(sl):
            dot.edge(str(id(sli)), str(id(sl)))

    if out_path is not None:
        out_path: Path = Path(out_path).with_suffix("")

        if fmt == "dot":
            with open(out_path, "w", encoding="utf8") as f:
                f.write(dot.source)
        else:
            dot.format = fmt
            dot.render(out_path, cleanup=True)

    return dot
