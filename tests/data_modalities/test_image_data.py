import itertools

import pytest
import torch

from cirkit.pipeline import compile
from cirkit.symbolic.layers import (
    BinomialLayer,
    CategoricalLayer,
    EmbeddingLayer,
    GaussianLayer,
)
from cirkit.templates import utils
from cirkit.templates.data_modalities import image_data

REGION_GRAPHS = [
    "quad-tree-2",
    "quad-tree-4",
    "quad-graph",
    "random-binary-tree",
    "poon-domingos",
    "chow-liu-tree",
    "linear-tree",
]

INPUT_LAYERS = ["categorical", "binomial", "embedding", "gaussian"]

EXPECTED_LAYER_TYPE = {
    "categorical": CategoricalLayer,
    "binomial": BinomialLayer,
    "embedding": EmbeddingLayer,
    "gaussian": GaussianLayer,
}


def _make_image_data(input_layer: str, num_samples: int, num_variables: int) -> torch.Tensor:
    if input_layer == "gaussian":
        return torch.randn(num_samples, num_variables)
    return torch.randint(0, 256, (num_samples, num_variables)).long()


def _input_params(input_layer: str) -> dict | None:
    """Ensure the Embedding layer encodes a proper distribution."""
    if input_layer == "embedding":
        return {"weight": utils.Parameterization(activation="softmax", initialization="normal")}
    return None


@pytest.mark.parametrize(
    "region_graph,input_layer",
    itertools.product(REGION_GRAPHS, INPUT_LAYERS),
)
def test_image_data_modality(region_graph: str, input_layer: str):
    image_shape = (1, 3, 3)
    num_variables = image_shape[0] * image_shape[1] * image_shape[2]
    num_samples = 64
    data = _make_image_data(input_layer, num_samples, num_variables)

    symbolic_circuit = image_data(
        image_shape,
        region_graph=region_graph,
        input_layer=input_layer,
        num_input_units=2,
        sum_product_layer="cp",
        num_sum_units=2,
        input_params=_input_params(input_layer),
        data=data,
        sum_weight_param=utils.Parameterization(activation="softmax", initialization="normal"),
        use_mixing_weights=True,
    )

    assert len(symbolic_circuit.scope) == num_variables

    expected_type = EXPECTED_LAYER_TYPE[input_layer]
    for circuit_input_layer in symbolic_circuit.input_layers:
        assert len(circuit_input_layer.scope) == 1
        assert isinstance(
            circuit_input_layer, expected_type
        ), f"Expected {expected_type.__name__}, got {type(circuit_input_layer).__name__}"

    # Check if the finite log-likelihoods has the expected shape
    circuit = compile(symbolic_circuit)
    ll = circuit(data)
    assert ll.shape == (num_samples, 1, 1)
    assert torch.isfinite(ll).all()


@pytest.mark.parametrize(
    "input_layer,num_bins,mi_chunk_size",
    itertools.product(["categorical", "binomial", "embedding", "gaussian"], [None, 8], [None, 16]),
)
def test_image_data_chow_liu_tree_options(
    input_layer: str, num_bins: int | None, mi_chunk_size: int | None
):

    image_shape = (1, 3, 3)
    num_variables = image_shape[0] * image_shape[1] * image_shape[2]
    num_samples = 128
    data = _make_image_data(input_layer, num_samples, num_variables)

    symbolic_circuit = image_data(
        image_shape,
        region_graph="chow-liu-tree",
        input_layer=input_layer,
        num_input_units=2,
        sum_product_layer="cp",
        num_sum_units=2,
        input_params=_input_params(input_layer),
        data=data,
        num_bins=num_bins,
        mi_chunk_size=mi_chunk_size,
        sum_weight_param=utils.Parameterization(activation="softmax", initialization="normal"),
    )

    # Check if the finite log-likelihoods has the expected shape
    assert len(symbolic_circuit.scope) == num_variables
    circuit = compile(symbolic_circuit)
    ll = circuit(data)
    assert ll.shape == (num_samples, 1, 1)
    assert torch.isfinite(ll).all()
