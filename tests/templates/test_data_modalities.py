import itertools

import pytest

import torch

from cirkit.templates import data_modalities, utils
from cirkit.pipeline import PipelineContext

@pytest.mark.parametrize(
    "image_shape,region_graph,input_layer,sum_product_layer,num_units,activation,initialization,fold,optimize", 
    itertools.product(
        [(1, 4, 4), (3, 4, 5)],
        ['quad-tree-2', 'quad-tree-4', 'quad-graph', 'random-binary-tree', 'poon-domingos'],
        ['categorical', 'binomial', 'embedding', 'gaussian'],
        ['cp', 'cp-t', 'tucker'],
        [1, 3],
        ["none", "softmax", "sigmoid", "positive-clamp"],
        ["uniform", "normal", "dirichlet"],
        [False, True],
        [False, True],
    )
)
def test_image_data(image_shape, region_graph, input_layer, sum_product_layer, 
                    num_units, activation, initialization, fold, optimize):
    sc = data_modalities.image_data(
        image_shape,
        region_graph=region_graph,
        input_layer=input_layer,
        num_input_units=num_units,
        sum_product_layer=sum_product_layer,
        num_sum_units=num_units,
        sum_weight_param=utils.Parameterization(
            activation=activation,
            initialization=initialization
        )
    )

    assert sc.is_smooth
    assert sc.is_decomposable
    
    ctx = PipelineContext(backend="torch", fold=fold, optimize=optimize)

    # test on dummy input
    tc = ctx.compile(sc)
    tc(torch.randint(1, 255, size=(2, sc.num_variables)))