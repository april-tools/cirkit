from typing import List, Dict

from cirkit.symbolic.layers import Layer


class CircuitBlock:
    def __init__(
        self,
        layers: List[Layer],
        in_layers: Dict[Layer, List[Layer]],
        out_layers: Dict[Layer, List[Layer]]
    ):
        self._layers = layers
        self._in_layers = in_layers
        self._out_layers = out_layers
        ins = [l for l in layers if l not in in_layers or not in_layers[l]]
        outs = [l for l in layers if l not in out_layers or not out_layers[l]]
        assert len(ins) == 1 and len(outs) == 1
        self._input = ins[0]
        self._output = outs[0]

    @property
    def input(self) -> Layer:
        return self._input

    @property
    def output(self) -> Layer:
        return self._output

    @property
    def layers(self) -> List[Layer]:
        return self._layers

    @property
    def layer_inputs(self):
        return self._in_layers

    @property
    def layer_outputs(self):
        return self._out_layers

    @staticmethod
    def from_layer(sl: Layer) -> 'CircuitBlock':
        return CircuitBlock([sl], {}, {})

    @staticmethod
    def from_layer_composition(*sl: Layer) -> 'CircuitBlock':
        layers = list(sl)
        in_layers = {}
        out_layers = {}
        assert len(layers) > 1, "Expected a composition of at least 2 layers"
        for i, l in enumerate(layers):
            if i - 1 >= 0:
                in_layers[l] = [layers[i - 1]]
            if i + 1 < len(layers):
                out_layers[l] = [layers[i + 1]]
        return CircuitBlock(layers, in_layers, out_layers)
