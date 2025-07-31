import itertools
import tempfile
from functools import partial

import pytest
from tests.floats import allclose

import torch

from cirkit.pipeline import PipelineContext
import cirkit.symbolic.functional as SF
from cirkit.templates.logic import SDD

# A & (~ B | C)
SDD_s = "L 1 0 1\nL 3 2 2\nL 4 4 3\nL 5 2 -2\nT 6\nD 2 3 2 3 4 5 6\nL 7 0 -1\nF 8\nD 0 1 2 1 2 7 8"

@pytest.mark.parametrize(
    "fold,optimize,input_layer",
    itertools.product([False, True], [False, True], ["categorical", "embedding"]),
)
def test_compile_sdd(fold, optimize, input_layer):
    ctx = PipelineContext(optimize=optimize, fold=fold)
    
    sdd_c = SDD.from_string(SDD_s)
    # construct circuit without enforcing smoothness
    s_c = sdd_c.build_circuit(enforce_smoothness=False, input_layer=input_layer)
    t_c = ctx.compile(s_c)

    assert t_c.properties.decomposable
    assert not t_c.properties.smooth

    sdd_c = SDD.from_string(SDD_s)
    # construct circuit and enforce smoothness
    s_c = sdd_c.build_circuit(input_layer=input_layer)
    t_c = ctx.compile(s_c)

    assert t_c.properties.decomposable
    assert t_c.properties.smooth
    assert t_c.properties.structured_decomposable
    
    # model checking
    worlds = torch.tensor(
        list(itertools.product([0, 1], repeat=s_c.num_variables)), dtype=torch.long
    )
    gt = worlds[:, 0] & ((1 - worlds[:, 1]) | worlds[:, 2])
    assert allclose(t_c(worlds).flatten(), gt)

    # model counting by integrating all variables
    mc_s_c = SF.integrate(s_c)
    mc_t_c = ctx.compile(mc_s_c)
    assert mc_t_c().item() == gt.sum()

@pytest.mark.parametrize(
    "fold,optimize,input_layer,num_units",
    itertools.product([False, True], [False, True], ["categorical", "embedding"], [2, 10, 100]),
)
def test_overparameterized_sdd(fold, optimize, input_layer, num_units):
    ctx = PipelineContext(optimize=optimize, fold=fold)
    
    sdd_c = SDD.from_string(SDD_s)
    # construct circuit without enforcing smoothness
    s_c = sdd_c.build_circuit(enforce_smoothness=False, input_layer=input_layer, num_units=num_units)
    t_c = ctx.compile(s_c)

    assert t_c.properties.decomposable
    assert not t_c.properties.smooth

    sdd_c = SDD.from_string(SDD_s)
    # construct circuit and enforce smoothness
    s_c = sdd_c.build_circuit(input_layer=input_layer, num_units=num_units)
    t_c = ctx.compile(s_c)

    assert t_c.properties.decomposable
    assert t_c.properties.smooth
    assert t_c.properties.structured_decomposable
    
    pf_c = ctx.compile(SF.integrate(s_c))
    pf = pf_c()

    # model checking
    worlds = torch.tensor(
        list(itertools.product([0, 1], repeat=s_c.num_variables)), dtype=torch.long
    )
    gt = worlds[:, 0] & ((1 - worlds[:, 1]) | worlds[:, 2])
    assert allclose(t_c(worlds).flatten().nonzero(), gt.flatten().nonzero())

@pytest.mark.parametrize(
    "fold,optimize,input_layer",
    itertools.product([False, True], [False, True], ["categorical", "embedding"]),
)
def test_model_checking_sdd(fold, optimize, input_layer):
    ctx = PipelineContext(optimize=optimize, fold=fold)
    
    sdd_c = SDD.from_string(SDD_s)
    s_c = sdd_c.build_circuit(input_layer=input_layer)
    t_c = ctx.compile(s_c)

    # model checking
    worlds = torch.tensor(
        list(itertools.product([0, 1], repeat=s_c.num_variables)), dtype=torch.long
    )
    gt = worlds[:, 0] & ((1 - worlds[:, 1]) | worlds[:, 2])
    assert allclose(t_c(worlds).flatten(), gt)

    # model counting by integrating all variables
    mc_s_c = SF.integrate(s_c)
    mc_t_c = ctx.compile(mc_s_c)
    assert mc_t_c().item() == gt.sum()

@pytest.mark.parametrize(
    "fold,optimize,input_layer",
    itertools.product([False, True], [False, True], ["categorical", "embedding"]),
)
def test_model_counting_sdd(fold, optimize, input_layer):
    ctx = PipelineContext(optimize=optimize, fold=fold)
    
    sdd_c = SDD.from_string(SDD_s)
    s_c = sdd_c.build_circuit(input_layer=input_layer)
    t_c = ctx.compile(s_c)

    # model checking
    worlds = torch.tensor(
        list(itertools.product([0, 1], repeat=s_c.num_variables)), dtype=torch.long
    )
    gt = worlds[:, 0] & ((1 - worlds[:, 1]) | worlds[:, 2])
    
    # model counting by integrating all variables
    mc_s_c = SF.integrate(s_c)
    mc_t_c = ctx.compile(mc_s_c)
    assert mc_t_c().item() == gt.sum()

@pytest.mark.parametrize(
    "fold,optimize,input_layer",
    itertools.product([False, True], [False, True], ["categorical", "embedding"]),
)
def test_conditional_sdd_circuit(fold, optimize, input_layer):
    sdd_c = SDD.from_string(SDD_s)
    s_c = sdd_c.build_circuit(
        input_layer=input_layer,
        sum_weight_activation="softmax"
    )
    cond_s_c, gf_specs = SF.condition_circuit(s_c, gate_functions={ 
        "sum": list(s_c.sum_layers)
    })
    
    ctx = PipelineContext(optimize=optimize, fold=fold)
    def f(shape, x): return x.view(-1, *shape)
    for k, shape in gf_specs.items():
        ctx.add_gate_function(k, partial(f, shape))
    
    t_c = ctx.compile(cond_s_c)

    # model checking
    worlds = torch.tensor(
        list(itertools.product([0, 1], repeat=s_c.num_variables)), dtype=torch.long
    )
    gt = worlds[:, 0] & ((1 - worlds[:, 1]) | worlds[:, 2])   
    params = { k: {"x" : torch.ones(1, *shape)} for k, shape in gf_specs.items()}
    assert (t_c(worlds, gate_function_kwargs=params).flatten().nonzero() == gt.nonzero()).all()