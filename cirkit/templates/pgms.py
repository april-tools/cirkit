from collections.abc import Mapping, Sequence
from typing import Any

from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import HadamardLayer, Layer, SumLayer
from cirkit.templates.utils import (
    Parameterization,
    name_to_input_layer_factory,
    named_parameterizations_to_factories,
    parameterization_to_factory,
)
from cirkit.utils.scope import Scope


def fully_factorized(
    num_variables: int,
    input_layer: str = "categorical",
    input_params: Mapping[str, Parameterization] | None = None,
    input_layer_kwargs: Mapping[str, Any] | list[Mapping[str, Any]] | None = None,
) -> Circuit:
    """Construct a circuit encoding a fully-factorized model.

    Args:
        num_variables: The number of variables.
        input_layer: The input layer to use for the factors. It can be 'categorical', 'binomial' or
            'gaussian'. Defaults to 'categorical'.
        input_params: A dictionary mapping each name of a parameter of the input layer to
            its parameterization. If it is None, then the default parameterization of the chosen
            input layer will be chosen.
        input_layer_kwargs: Additional optional arguments to pass when constructing input layers.
            If it is a dictionary, then it is interpreted as kwargs passed to all input layers
            constructors. If it is a list of dictionaries, then it must contain as many
            dictionaries as the number of variables, and each dictionary is interpreted as kwargs
            passed to each input layer constructor (in the order given by range(num_variables)).
    """
    if num_variables <= 0:
        raise ValueError("The number of variables should be a positive integer")
    if input_layer not in ["categorical", "binomial", "gaussian"]:
        raise ValueError(f"Unknown input layer called {input_layer}")
    input_layer_kwargs_ls: list[Mapping[str, Any]]
    if input_layer_kwargs is None:
        input_layer_kwargs_ls = [{}] * num_variables
    elif isinstance(input_layer_kwargs, Mapping):
        input_layer_kwargs_ls = [input_layer_kwargs] * num_variables
    elif isinstance(input_layer_kwargs, list):
        if len(input_layer_kwargs) != num_variables:
            raise ValueError(
                f"The list of input layer kwargs should have length num_variables={num_variables}"
            )
        if not all(isinstance(kwargs, Mapping) for kwargs in input_layer_kwargs):
            raise ValueError("The list of input layer kwargs should be a list of dictionaries")
        input_layer_kwargs_ls = input_layer_kwargs

    # Construct the input layers
    input_param_kwargs: Mapping[str, Any]
    if input_params is None:
        input_param_kwargs = {}
    else:
        input_param_kwargs = named_parameterizations_to_factories(input_params)
    input_factories = [
        name_to_input_layer_factory(input_layer, **kwargs, **input_param_kwargs)
        for kwargs in input_layer_kwargs_ls
    ]
    input_layers = [f(Scope([i]), 1) for i, f in enumerate(input_factories)]

    # Catch the case there is only one variable
    if len(input_layers) == 1:
        return Circuit(input_layers, in_layers={}, outputs=[input_layers[0]])

    # Construct a product layer
    prod_sl = HadamardLayer(1, arity=len(input_layers))

    return Circuit(input_layers + [prod_sl], in_layers={prod_sl: input_layers}, outputs=[prod_sl])


def hmm(
    ordering: Sequence[int],
    input_layer: str = "categorical",
    num_latent_states: int = 1,
    input_params: Mapping[str, Parameterization] | None = None,
    input_layer_kwargs: Mapping[str, Any] | list[Mapping[str, Any]] | None = None,
    weight_param: Parameterization | None = None,
) -> Circuit:
    """Construct a symbolic circuit mimicking a hidden markov model (HMM) of a given variable
    ordering. Product Layers are of type [HadamardLayer][cirkit.symbolic.layers.HadamardLayer],
    and sum layers are of type [SumLayer][cirkit.symbolic.layers.SumLayer]. Note that the HMM
    constructed is inhomogeneous, i.e., the emission probability tables and the transition
    probability tables are not shared between time steps.

    Args:
        ordering: The input order of variables of the HMM.
        input_layer: The input layer to use for the factors. It can be 'categorical', 'binomial' or
            'gaussian'. Defaults to 'categorical'.
        num_latent_states: The number of states the latent variables can assume or, equivalently,
            the number of sum units per sum layer.
        input_params: A dictionary mapping each name of a parameter of the input layer to
            its parameterization. If it is None, then the default parameterization of the chosen
            input layer will be chosen.
        input_layer_kwargs: Additional optional arguments to pass when constructing input layers.
            If it is a dictionary, then it is interpreted as kwargs passed to all input layers
            constructors. If it is a list of dictionaries, then it must contain as many
            dictionaries as the number of variables, and each dictionary is interpreted as kwargs
            passed to each input layer constructor (in the order given by range(num_variables)).
        weight_param: The parameterization to use for the weight coefficients.
            If None, then it defaults to using a softmax parameterization for the transition
            probability tables.

    Returns:
        Circuit: A symbolic circuit encoding an HMM.

    Raises:
        ValueError: order must consists of consistent numbers, starting from 0.
    """
    if not ordering:
        raise ValueError("The ordering should be non-empty")
    num_variables = len(ordering)
    if set(ordering) != set(range(num_variables)):
        raise ValueError("The 'ordering' of variables is not valid")
    if input_layer not in ["categorical", "binomial", "gaussian"]:
        raise ValueError(f"Unknown input layer called {input_layer}")
    input_layer_kwargs_ls: list[Mapping[str, Any]]
    if input_layer_kwargs is None:
        input_layer_kwargs_ls = [{}] * num_variables
    elif isinstance(input_layer_kwargs, Mapping):
        input_layer_kwargs_ls = [input_layer_kwargs] * num_variables
    elif isinstance(input_layer_kwargs, list):
        if len(input_layer_kwargs) != num_variables:
            raise ValueError(
                f"The list of input layer kwargs should have length num_variables={num_variables}"
            )
        if not all(isinstance(kwargs, Mapping) for kwargs in input_layer_kwargs):
            raise ValueError("The list of input layer kwargs should be a list of dictionaries")
        input_layer_kwargs_ls = input_layer_kwargs

    # Get the input layer factories
    input_param_kwargs: Mapping[str, Any]
    if input_params is None:
        input_param_kwargs = {}
    else:
        input_param_kwargs = named_parameterizations_to_factories(input_params)
    input_factories = [
        name_to_input_layer_factory(input_layer, **kwargs, **input_param_kwargs)
        for kwargs in input_layer_kwargs_ls
    ]

    layers: list[Layer] = []
    in_layers: dict[Layer, list[Layer]] = {}
    input_sl = input_factories[-1](Scope([ordering[-1]]), num_latent_states)
    layers.append(input_sl)

    # Set the sum weight factory
    if weight_param is None:
        weight_param = Parameterization(activation="softmax", initialization="normal")
    weight_factory = parameterization_to_factory(weight_param)

    num_units_out = 1 if num_variables == 1 else num_latent_states
    sum_sl = SumLayer(num_latent_states, num_units_out, weight_factory=weight_factory)
    layers.append(sum_sl)
    in_layers[sum_sl] = [input_sl]

    # Loop over the number of variables
    for i in reversed(range(num_variables - 1)):
        last_dense = layers[-1]

        input_sl = input_factories[i](Scope([ordering[i]]), num_latent_states)
        layers.append(input_sl)
        prod_sl = HadamardLayer(num_latent_states, 2)
        layers.append(prod_sl)
        in_layers[prod_sl] = [last_dense, input_sl]

        num_units_out = 1 if i == 0 else num_latent_states
        sum_sl = SumLayer(
            num_latent_states,
            num_units_out,
            weight_factory=weight_factory,
        )
        layers.append(sum_sl)
        in_layers[sum_sl] = [prod_sl]

    return Circuit(layers, in_layers, [layers[-1]])
