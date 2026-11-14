"""Initialize categorical circuits from observations on a fixed circuit structure.

The instance-partitioning approach is inspired by Vergari, Di Mauro and Esposito,
*Simplifying, Regularizing and Strengthening Sum-Product Network Structure Learning*,
ECML PKDD 2015, pp. 343-358, https://doi.org/10.1007/978-3-319-23525-7_21.
This module does not implement their structure learner, tree leaves or bagging.
Euclidean k-means, pooling at shared nodes and perturbations of replicated units are
initialization choices made here for already constructed tensorized circuits.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast, get_args

import numpy as np
import torch
from fast_pytorch_kmeans import KMeans
from torch import Tensor

from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.initializers import ConstantTensorInitializer
from cirkit.symbolic.layers import (
    CategoricalLayer,
    HadamardLayer,
    KroneckerLayer,
    Layer,
    MultichannelCategoricalLayer,
    SumLayer,
)
from cirkit.symbolic.parameters import (
    MixingWeightParameter,
    Parameter,
    TensorParameter,
    mixing_weight_factory,
)
from cirkit.templates.utils import (
    POSITIVE_PARAMETER_MIN,
    InputSharing,
    Parameterization,
    name_to_parameter_activation,
)

EstimatedActivation: TypeAlias = Literal["none", "softmax", "softplus", "sigmoid", "positive-clamp"]


@dataclass(frozen=True)
class DataDrivenInitialization:
    """Settings for initialization from categorical observations.

    Args:
        alpha: Non-negative pseudocount added to every category or mixture branch.
            Empty populations always receive uniform probabilities.
        noise_scale: Non-negative standard deviation of symmetry-breaking noise.
            Zero preserves the estimated probabilities, up to numerical clipping.
        seed: Non-negative seed for a local NumPy generator, or None for fresh entropy.
            Initialization does not reset NumPy or Torch global random states. Results
            are reproducible on the same device and software configuration.

    Raises:
        ValueError: If alpha or noise_scale is negative/non-finite, or seed is invalid.
    """

    alpha: float = 0.5
    noise_scale: float = 0.2
    seed: int | None = 42

    def __post_init__(self) -> None:
        for name, value in (("alpha", self.alpha), ("noise_scale", self.noise_scale)):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"Expected {name} to be finite and non-negative, found {value}")
        if self.seed is not None and (not isinstance(self.seed, int) or self.seed < 0):
            raise ValueError("Expected seed to be a non-negative integer or None")


@torch.no_grad()
def initialize_parameters(
    circuit: Circuit,
    data: Tensor,
    *,
    initialization: DataDrivenInitialization,
    sum_weight_param: Parameterization,
    input_sharing: InputSharing = "none",
) -> Circuit:
    """Replace the parameters of a newly built categorical circuit in place.

    Sum layers partition samples across their child *layers*, using Euclidean k-means.
    Dense layers with one child distribute mass uniformly across its units. Product
    layers pass their samples to every child. A shared node pools the union of incoming
    row indices before processing it, counting each observation once. Input distributions
    and branch weights use smoothed counts; replicated units receive independent noise.

    With full input sharing, marginal counts are pooled over all observations and all
    distinct variables at each channel position. All leaves must have matching types,
    units and category counts. Inputs always use softmax probabilities.
    Structured mixing weights retain their MixingWeightParameter operator.

    Args:
        circuit: A smooth, decomposable symbolic circuit with categorical input layers,
            sums and Hadamard/Kronecker products. Call before compilation or creating
            derived circuits: existing parameter references are replaced.
        data: Non-empty (num_samples, num_variables) tensor. Column indices must match
            variable IDs. Values must be finite integers in each variable's category
            range; integer-valued floating tensors are accepted. Computation uses its device.
        initialization: Smoothing, perturbation and local random-state settings.
        sum_weight_param: Parameterization for estimated sum weights. Its initialization
            is superseded by the estimates. Real dtype and the activations none, softmax,
            softplus, sigmoid and positive-clamp are supported; activation kwargs are not.
        input_sharing: Either 'none' for local marginals or 'full' for pooled marginals
            and a single shared input parameter tensor.

    Returns:
        The same circuit, with new learnable symbolic parameters. Softmax inputs and
        sum weights guarantee local normalization after perturbation.

    Raises:
        ValueError: If data, sharing, circuit structure or parameterization is unsupported.
    """
    if input_sharing not in get_args(InputSharing):
        raise ValueError(f"Unknown input sharing mode {input_sharing!r}")
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("Expected a non-empty data matrix of shape (num_samples, num_variables)")
    if data.is_complex() or not bool(torch.isfinite(data).all()):
        raise ValueError("Categorical observations must be finite real integers")
    if data.is_floating_point() and not bool((data == data.round()).all()):
        raise ValueError("Categorical observations must be integer-valued")
    if set(circuit.scope) != set(range(data.shape[1])):
        raise ValueError("Data columns must match the circuit's variable IDs")
    if not circuit.is_smooth or not circuit.is_decomposable:
        raise ValueError("Initialization requires a smooth, decomposable circuit")
    if sum_weight_param.dtype != "real" or sum_weight_param.activation_kwargs:
        raise ValueError("Estimated weights require real dtype and no activation kwargs")
    if sum_weight_param.activation not in get_args(EstimatedActivation):
        raise ValueError(f"Unsupported estimated-weight activation {sum_weight_param.activation!r}")

    signatures = set()
    for layer in circuit.layers:
        if isinstance(layer, (CategoricalLayer, MultichannelCategoricalLayer)):
            # Promote uint8 pixels before comparing with 256, which otherwise wraps to zero.
            values = data[:, sorted(layer.scope)].long()
            if bool(((values < 0) | (values >= layer.num_categories)).any()):
                raise ValueError(
                    f"Observations in scope {layer.scope} are outside the category range"
                )
            signatures.add(
                (type(layer), layer.num_output_units, layer.num_categories, len(layer.scope))
            )
        elif not isinstance(layer, (SumLayer, HadamardLayer, KroneckerLayer)):
            raise ValueError("Data-driven initialization supports only categorical input circuits")
    if input_sharing == "full" and len(signatures) != 1:
        raise ValueError("Full sharing requires identical input shapes and category counts")

    rng = np.random.default_rng(initialization.seed)
    all_rows = torch.arange(data.shape[0], device=data.device)
    pending: dict[Layer, list[Tensor]] = defaultdict(list)
    for output in circuit.outputs:
        pending[output].append(all_rows)
    shared_parameter = (
        _shared_input_parameter(circuit, data, initialization, rng)
        if input_sharing == "full"
        else None
    )

    ordering = list(circuit.topological_ordering())
    shared_owner = next(layer for layer in ordering if not circuit.layer_inputs(layer))
    for layer in reversed(ordering):
        populations = pending.pop(layer)
        rows = populations[0] if len(populations) == 1 else torch.unique(torch.cat(populations))
        children = circuit.layer_inputs(layer)
        if isinstance(layer, (CategoricalLayer, MultichannelCategoricalLayer)):
            if shared_parameter is None:
                marginals = np.stack(
                    [
                        _estimate_marginal(
                            data[rows, feature], layer.num_categories, initialization
                        )
                        for feature in sorted(layer.scope)
                    ]
                )
                probs = marginals[0] if isinstance(layer, CategoricalLayer) else marginals
                parameter = _make_input_parameter(
                    probs, layer.num_output_units, initialization, rng
                )
            else:
                parameter = shared_parameter if layer is shared_owner else shared_parameter.ref()
            layer.probs = parameter
            layer.logits = None
        elif isinstance(layer, SumLayer):
            features = torch.tensor(sorted(circuit.layer_scope(layer)), device=data.device)
            clusters = _cluster_instances(data, rows, features, len(children), rng)
            layer.weight = _make_sum_parameter(
                layer, clusters, sum_weight_param, initialization, rng
            )
            for child, cluster in zip(children, clusters):
                pending[child].append(cluster)
        else:
            for child in children:
                pending[child].append(rows)
    return circuit


def _cluster_instances(
    data: Tensor, rows: Tensor, features: Tensor, num_clusters: int, rng: np.random.Generator
) -> list[Tensor]:
    """Partition rows, padding with empty clusters when fewer samples are available."""
    if num_clusters == 1:
        return [rows]
    if rows.numel() == 0:
        return [rows] * num_clusters
    values = data.index_select(0, rows).index_select(1, features).float()
    effective_clusters = min(num_clusters, len(rows))
    indices = torch.as_tensor(
        rng.choice(len(rows), size=effective_clusters, replace=False), device=data.device
    )
    centroids = values.index_select(0, indices).clone()
    labels = KMeans(n_clusters=effective_clusters, mode="euclidean").fit_predict(
        values, centroids=centroids
    )
    return [rows[labels == index] for index in range(num_clusters)]


def _normalize_counts(counts: np.ndarray, config: DataDrivenInitialization) -> np.ndarray:
    counts = counts + config.alpha
    total = counts.sum()
    return counts / total if total > 0 else np.full(counts.shape, 1.0 / counts.size)


def _estimate_marginal(
    values: Tensor, num_categories: int, config: DataDrivenInitialization
) -> np.ndarray:
    counts = torch.bincount(values.reshape(-1).long(), minlength=num_categories)
    return _normalize_counts(counts.cpu().numpy().astype(np.float64), config)


def _shared_input_parameter(
    circuit: Circuit,
    data: Tensor,
    config: DataDrivenInitialization,
    rng: np.random.Generator,
) -> Parameter:
    inputs = list(circuit.input_layers)
    first = inputs[0]
    assert isinstance(first, (CategoricalLayer, MultichannelCategoricalLayer))
    scopes = [sorted(layer.scope) for layer in inputs]
    marginals = np.stack(
        [
            _estimate_marginal(data[:, sorted(set(features))], first.num_categories, config)
            for features in zip(*scopes)
        ]
    )
    probs = marginals[0] if isinstance(first, CategoricalLayer) else marginals
    return _make_input_parameter(probs, first.num_output_units, config, rng)


def _make_input_parameter(
    probs: np.ndarray,
    num_units: int,
    config: DataDrivenInitialization,
    rng: np.random.Generator,
) -> Parameter:
    values = np.broadcast_to(probs, (num_units, *probs.shape)).copy()
    return _parameter_from_probabilities(values, "softmax", config, rng)


def _make_sum_parameter(
    layer: SumLayer,
    clusters: list[Tensor],
    param: Parameterization,
    config: DataDrivenInitialization,
    rng: np.random.Generator,
) -> Parameter:
    weights = _normalize_counts(np.array([len(rows) for rows in clusters], dtype=float), config)
    weights = np.broadcast_to(weights, (layer.num_output_units, len(clusters))).copy()
    activation = cast(EstimatedActivation, param.activation)
    if isinstance(layer.weight.output, MixingWeightParameter):
        estimated = _parameter_from_probabilities(weights, activation, config, rng)
        return mixing_weight_factory(layer.weight.shape, param_factory=lambda _: estimated)
    weights = np.repeat(weights / layer.num_input_units, layer.num_input_units, axis=1)
    return _parameter_from_probabilities(weights, activation, config, rng)


def _parameter_from_probabilities(
    probs: np.ndarray,
    activation: EstimatedActivation,
    config: DataDrivenInitialization,
    rng: np.random.Generator,
) -> Parameter:
    """Invert a supported activation and perturb its unconstrained parameters."""
    if activation in {"none", "positive-clamp"}:
        theta = probs * np.exp(rng.normal(0.0, config.noise_scale, size=probs.shape))
        theta = np.maximum(theta, POSITIVE_PARAMETER_MIN)
    else:
        positive = np.clip(probs, 1e-12, 1 - 1e-12 if activation == "sigmoid" else None)
        if activation == "softplus":
            theta = np.log(np.expm1(positive))
        elif activation == "sigmoid":
            theta = np.log(positive) - np.log1p(-positive)
        else:
            theta = np.log(positive)
        theta += rng.normal(0.0, config.noise_scale, size=probs.shape)
    tensor = TensorParameter(*probs.shape, initializer=ConstantTensorInitializer(theta))
    factory = name_to_parameter_activation(activation)
    if factory is None:
        return Parameter.from_input(tensor)
    return Parameter.from_unary(factory(probs.shape), tensor)
