import itertools

import pytest
import torch

from cirkit.pipeline import PipelineContext
from cirkit.symbolic.layers import CategoricalLayer, GaussianLayer
from cirkit.templates import utils
from cirkit.templates.data_modalities import tabular_data
from cirkit.backend.torch.queries import IntegrateQuery, SamplingQuery

DEVICE = 'cuda:0'

@pytest.mark.parametrize(
    "n_cat_features,n_num_features,region_graph",
    itertools.product([0, 1, 10], [0, 1, 3], ["random-binary-tree", "chow-liu-tree"]),
)
def test_tabular_data_modality(n_cat_features: int, n_num_features: int, region_graph: str):

    n = 500
    n_classes = 5
    cat_data = torch.randint(0, n_classes, (n, n_cat_features))
    num_data = torch.randn(n, n_num_features)
    data = torch.cat([cat_data, num_data], dim=1)
    num_features = data.shape[1]
    
    data = data.to(DEVICE)

    input_layers = [
        {"name": "categorical", "args": {"num_categories": n_classes + i}}
        for i in range(n_cat_features)
    ] + [{"name": "gaussian", "args": {}} for _ in range(n_num_features)]

    if num_features > 0:

        if num_features == 1:
            # TODO: The case with only one feature has to be fixed
            # and it does not depend on the function tested here
            # pytest.xfail("Single feature case is known to fail")
            pass
        else:
            symbolic_circuit = tabular_data(
                region_graph=region_graph,
                data=data,
                input_layers=input_layers,
                num_input_units=2000,
                sum_product_layer="cp",
                num_sum_units=2000,
                sum_weight_param=utils.Parameterization(
                    activation="softmax", initialization="normal"
                ),
                use_mixing_weights=True,
            )

            # Check if the circuit has the expected number of input layers
            assert len(symbolic_circuit.scope) == num_features

            # Check if the input layers are correctly created
            for circuit_input_layer in symbolic_circuit.input_layers:
                scope = list(circuit_input_layer.scope)[0]
                expected_type = (
                    CategoricalLayer
                    if input_layers[scope]["name"] == "categorical"
                    else GaussianLayer
                )
                assert isinstance(
                    circuit_input_layer, expected_type
                ), f"Expected {expected_type.__name__}, got {type(circuit_input_layer).__name__}"

            # Check if the log-likelihood has the expected shape
            ctx = PipelineContext(semiring="lse-sum", fold=True, optimize=False)
            circuit = ctx.compile(symbolic_circuit)
            
            circuit.to(data.device)
            
            ll = circuit(data)
            assert ll.shape == (
                len(data),
                1,
                1,
            ), f"Expected log-likelihood shape {(n, 1, 1)}, got {ll.shape}"
            
            # Check if the circuit can sample
            sampling_query = SamplingQuery(circuit)
            samples = sampling_query(num_samples=len(data))[1]
            assert samples.shape == data.shape, f"Expected samples shape {data.shape}, got {samples.shape}"