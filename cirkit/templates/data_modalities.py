"""Construct symbolic circuits for images and tables, optionally initialized from data."""

import functools
from typing import Any, Literal, Mapping, TypeAlias, get_args

import numpy as np
from torch import Tensor

from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.parameters import ParameterFactory, mixing_weight_factory
from cirkit.templates.initialization import DataDrivenInitialization, initialize_parameters
from cirkit.templates.region_graph import (
    ChowLiuTree,
    LinearTree,
    PoonDomingos,
    QuadGraph,
    QuadTree,
    RandomBinaryTree,
    RegionGraph,
)
from cirkit.templates.utils import (
    InputLayerFactory,
    InputLayerName,
    InputSharing,
    Parameterization,
    SumProductLayerName,
    categorical_input_layer_factory,
    name_to_input_layer_factory,
    parameterization_to_factory,
)
from cirkit.utils.scope import Scope

ImageRegionGraph: TypeAlias = Literal[
    "quad-tree-2",
    "quad-tree-4",
    "quad-graph",
    "random-binary-tree",
    "poon-domingos",
    "chow-liu-tree",
    "linear-tree",
]
TabularRegionGraph: TypeAlias = Literal["random-binary-tree", "chow-liu-tree"]


def _flatten_image_data(data: Tensor, image_shape: tuple[int, int, int]) -> Tensor:
    """Flattens image samples into the tabular form required for region-graph learning.

    Accepts data already in tabular form (num_samples, C * H * W), as a full image tensor
    (num_samples, C, H, W) matching image_shape, or as a single-channel image without the
    channel axis (num_samples, H, W) when C == 1.

    Args:
        data (Tensor): The image data to flatten, whose leading axis indexes the samples.
            Excluding that axis, its shape must be one of (C * H * W,), (C, H, W) matching
            image_shape, or (H, W) for single-channel images (i.e. when C == 1).
        image_shape (tuple[int, int, int]): The image shape, given as (C, H, W).

    Returns:
        The data reshaped into tabular form, i.e. a matrix of shape (num_samples, C * H * W).

    Raises:
        ValueError: If the shape of data is not compatible with image_shape.
    """
    num_variables = image_shape[0] * image_shape[1] * image_shape[2]
    valid_feature_shapes = [(num_variables,), tuple(image_shape)]
    if image_shape[0] == 1:
        valid_feature_shapes.append(tuple(image_shape[1:]))
    if tuple(data.shape[1:]) not in valid_feature_shapes:
        raise ValueError(
            f"Expected image data of shape (num_samples, {image_shape[0]}, {image_shape[1]},"
            f" {image_shape[2]}) or (num_samples, {num_variables}),"
            f" but found a tensor of shape {tuple(data.shape)}"
        )
    return data.reshape(data.shape[0], num_variables)


def _build_image_region_graph(
    image_shape: tuple[int, int, int],
    region_graph: ImageRegionGraph,
    *,
    input_layer: InputLayerName,
    data: Tensor | None,
    num_bins: int | None,
    mi_chunk_size: int | None,
    validate_initialization: bool,
) -> RegionGraph:
    """Build the requested image region graph and validate structure-learning data."""
    num_variables = int(np.prod(image_shape))
    match region_graph:
        case "quad-tree-2":
            return QuadTree(image_shape, num_patch_splits=2)
        case "quad-tree-4":
            return QuadTree(image_shape, num_patch_splits=4)
        case "quad-graph":
            return QuadGraph(image_shape)
        case "random-binary-tree":
            return RandomBinaryTree(num_variables)
        case "poon-domingos":
            delta = int(max(np.ceil(image_shape[1] / 8), np.ceil(image_shape[2] / 8)))
            return PoonDomingos(image_shape, delta=delta)
        case "linear-tree":
            return LinearTree(num_variables)
        case "chow-liu-tree":
            if data is None:
                raise ValueError("A data tensor is required when region_graph is 'chow-liu-tree'")
            flattened_data = _flatten_image_data(data, image_shape)
            if validate_initialization:
                if flattened_data.shape[0] == 0:
                    raise ValueError("Expected non-empty categorical observations")
                if flattened_data.is_complex() or not bool(flattened_data.isfinite().all()):
                    raise ValueError("Categorical observations must be finite real integers")
                if flattened_data.is_floating_point() and not bool(
                    (flattened_data == flattened_data.round()).all()
                ):
                    raise ValueError("Categorical observations must be integer-valued")
                values = flattened_data.long()
                if bool(((values < 0) | (values >= 256)).any()):
                    raise ValueError("Categorical image observations must be in [0, 256)")
            if input_layer == "gaussian":
                if not flattened_data.is_floating_point():
                    raise ValueError(
                        "A floating-point data tensor is required when input_layer is "
                        f"'gaussian', but found a tensor of dtype {flattened_data.dtype}"
                    )
                result = ChowLiuTree(
                    data=flattened_data,
                    input_type="gaussian",
                    chunk_size=mi_chunk_size,
                    as_region_graph=True,
                )
            else:
                clt_data = flattened_data.long()
                num_categories = 256
                if num_bins is not None:
                    if not 1 <= num_bins <= 256:
                        raise ValueError(
                            f"Expected num_bins to be in [1, 256], but found {num_bins}"
                        )
                    clt_data = (clt_data * num_bins).div(256, rounding_mode="floor")
                    clt_data = clt_data.clamp_max(num_bins - 1)
                    num_categories = num_bins
                result = ChowLiuTree(
                    data=clt_data,
                    input_type="categorical",
                    num_categories=num_categories,
                    chunk_size=mi_chunk_size,
                    as_region_graph=True,
                )
            if not isinstance(result, RegionGraph):
                raise ValueError(f"Expected a RegionGraph, but got {type(result).__name__}")
            return result
    raise AssertionError(f"Unhandled image region graph {region_graph!r}")


def _build_tabular_region_graph(
    region_graph: TabularRegionGraph,
    *,
    num_features: int | None,
    data: Tensor | None,
    input_layers: dict | list[dict],
    validate_initialization: bool,
) -> RegionGraph:
    """Build a fixed or learned tabular region graph."""
    if region_graph == "random-binary-tree":
        if num_features is None:
            if data is None:
                raise ValueError(f"You must pass `num_features=` if you ask for {region_graph}.")
            num_features = data.shape[1]
        return RandomBinaryTree(num_features)

    if data is None:
        raise ValueError("You must pass `data=` if you ask for `chow-liu-tree`.")
    if validate_initialization:
        if data.is_complex() or not bool(data.isfinite().all()):
            raise ValueError("Categorical observations must be finite real integers")
        if data.is_floating_point() and not bool((data == data.round()).all()):
            raise ValueError("Categorical observations must be integer-valued")
        feature_specs = (
            [input_layers] * data.shape[1] if isinstance(input_layers, dict) else input_layers
        )
        if len(feature_specs) != data.shape[1]:
            raise ValueError("Number of input layers must match the number of features")
        values = data.long()
        for feature, spec in enumerate(feature_specs):
            num_categories = spec["args"]["num_categories"]
            column = values[:, feature]
            if bool(((column < 0) | (column >= num_categories)).any()):
                raise ValueError(
                    f"Observations in feature {feature} must be in [0, {num_categories})"
                )

    result = ChowLiuTree(
        data=data,
        input_type=(
            input_layers["name"]
            if isinstance(input_layers, dict)
            else [spec["name"] for spec in input_layers]
        ),
        num_categories=(
            input_layers["args"]["num_categories"]
            if isinstance(input_layers, dict) and input_layers["name"] == "categorical"
            else None
        ),
        as_region_graph=True,
    )
    if not isinstance(result, RegionGraph):
        raise ValueError(f"Expected a RegionGraph, but got {type(result).__name__}.")
    return result


def image_data(  # pylint: disable=too-many-arguments
    image_shape: tuple[int, int, int],
    region_graph: ImageRegionGraph = "quad-graph",
    *,
    input_layer: InputLayerName,
    num_input_units: int,
    sum_product_layer: SumProductLayerName,
    num_sum_units: int,
    num_classes: int = 1,
    input_params: dict[str, Parameterization] | None = None,
    sum_weight_param: Parameterization | None = None,
    use_mixing_weights: bool = True,
    data: Tensor | None = None,
    num_bins: int | None = None,
    mi_chunk_size: int | None = None,
    initialization: DataDrivenInitialization | None = None,
    input_sharing: InputSharing = "none",
) -> Circuit:
    """Constructs a symbolic circuit whose structure is tailored for image data sets.

    Args:
        image_shape: The image shape (C, H, W), where C is the number of channels, H is the height
            of the images, and W is their width.
        region_graph: The name of the region graph to use. It can be one of the following:
            'quad-tree-2' (the Quad-Tree with two splits per region node),
            'quad-tree-4' (the Quad-Tree with four splits per region node),
            'quad-graph'  (the Quad-Graph region graph),
            'random-binary-tree' (the random binary tree on flattened image pixels),
            'poon-domingos' (the Poon-Domingos architecture),
            'chow-liu-tree' (the Chow-Liu tree / HCLT, learned from data; requires the 'data'
            argument),
            'linear-tree' (the linear chain over the flattened image
            pixels, i.e. the HMM region graph).
        input_layer: The name of the input layer. It can be one of the following:
            'categorical' (encoding a Categorical distribution over pixel channel values),
            'binomial' (encoding a Binomial distribution over pixel channel values),
            'embedding' (encoding an Embedding vector over pixel channel values),
            'gaussian' (encoding a Gaussian distribution over pixel channel values).
        num_input_units: The number of input units per input layer.
        sum_product_layer: The name of the sum-product inner layer. It can be one of the following:
            'cp' (the canonical decomposition layer, consisting of dense layers followed by a
            hadamard product layer), 'cp-t' (the transposed canonical decomposition layer,
            consisting of a hadamard product layer followed by a single dense layer), 'tucker'
            (the Tucker decomposition layer, consisting of a kronecker product layer followed by
            a single dense layer).
        num_classes: The number of output classes (default=1).
        num_sum_units: The number of sum units in each sum layer, i.e., either dense or mixing
            layer.
        input_params: A dictionary mapping each name of a parameter of the input layer to
            its parameterization. If it is None, then the default parameterization of the chosen
            input layer will be chosen.
        sum_weight_param: The parameterization to use for sum layers parameters. If it None,
            then a softmax parameterization of the sum weights will be used.
        use_mixing_weights: Whether to parameterize sum layers having arity > 1 in a way such
            that they compute a linear combinations of the input vectors, instead of computing
            a matrix-vector product where the vector is the concatenation of input vectors.
            Sum layers having this semantics are also sometimes referred to as "mixing" layers.
            Defaults to True.
        data: Data for learning the 'chow-liu-tree' structure or initializing parameters.
            Required for either operation and otherwise ignored. It can be of shape
            (num_samples, C, H, W) matching image_shape, flattened (num_samples, C * H * W),
            or (num_samples, H, W) for single-channel images (C == 1). When input_layer is
            'gaussian' it must be floating-point; otherwise it is treated as discrete
            pixel values.
        num_bins: Used by the categorical, binomial, and embedding 'chow-liu-tree' region graph.
            If given, the 256 pixel levels are binned into num_bins bins before learning the tree.
            Must be in [1, 256]. If None, the raw pixel values are used.
        mi_chunk_size: Only used by the 'chow-liu-tree' region graph. The number of samples
            per chunk passed to the chunked mutual-information estimators. If None, all samples
            are processed at once.
        initialization: Optional settings for categorical initialization from data on the
            selected structure. None preserves the usual random initialization. Requires
            input_layer='categorical', data with integer values in [0, 255], and no input_params.
            Uses data.device; num_bins affects only structure learning, not initialization.
        input_sharing: 'none' for independent input parameters or 'full' for shared categorical
            parameters. With multiple channels, full sharing uses a product of per-channel
            categorical distributions at each pixel and requires a QuadGraph or QuadTree.
            Parameters are shared across pixels, while channels have distinct distributions.
            With initialization enabled, shared marginals pool all samples and pixels.

    See:
        Gala, de Campos, Vergari and Quaeghebeur, *Scaling Continuous Latent Variable
        Models as Probabilistic Integral Circuits*, NeurIPS 2024, Appendix D.2:
        https://arxiv.org/abs/2406.06494. The shared-input representation follows their
        PC setup; empirical initialization is an independent option.

    Returns:
        Circuit: A symbolic circuit.

    Raises:
        ValueError:
            - If one of the arguments is not one of the specified allowed ones.
            - If region_graph is 'chow-liu-tree' but data is None, num_bins is outside [1, 256], or
                input_layer is 'gaussian' with non-floating-point data.
    """
    if (
        not isinstance(image_shape, tuple)
        or len(image_shape) != 3
        or any(d <= 0 for d in image_shape)
    ):
        raise ValueError(
            "Expected the image shape to be a tuple of three positive integers,"
            f" but found {image_shape}"
        )

    if region_graph not in get_args(ImageRegionGraph):
        raise ValueError(f"Unknown region graph called {region_graph}")
    if input_layer not in get_args(InputLayerName):
        raise ValueError(f"Unknown input layer called {input_layer}")
    feature_options_enabled = initialization is not None or input_sharing != "none"
    if feature_options_enabled and sum_product_layer not in get_args(SumProductLayerName):
        raise ValueError(f"Unknown sum-product layer called {sum_product_layer}")
    if input_sharing not in get_args(InputSharing):
        raise ValueError(f"Unknown input sharing mode {input_sharing!r}")
    if input_sharing == "full" and input_layer != "categorical":
        raise ValueError("Full input sharing requires categorical inputs")

    multichannel = input_sharing == "full" and image_shape[0] > 1
    if multichannel and region_graph not in ("quad-graph", "quad-tree-2", "quad-tree-4"):
        raise ValueError("Multichannel input sharing requires a QuadGraph or QuadTree")
    estimation_data = None
    if initialization is not None:
        if input_layer != "categorical" or input_params is not None:
            raise ValueError(
                "Data-driven initialization requires categorical inputs without input_params"
            )
        if data is None:
            raise ValueError("Data-driven initialization requires a data tensor")
        estimation_data = _flatten_image_data(data, image_shape)

    rg = _build_image_region_graph(
        image_shape,
        region_graph,
        input_layer=input_layer,
        data=data,
        num_bins=num_bins,
        mi_chunk_size=mi_chunk_size,
        validate_initialization=initialization is not None,
    )

    # Get the input layer factory
    input_kwargs: dict[str, Any]
    match input_layer:
        case "categorical":
            input_kwargs = {"num_categories": 256}
        case "binomial":
            input_kwargs = {"total_count": 255}
        case "embedding":
            input_kwargs = {"num_states": 256}
        case "gaussian":
            input_kwargs = {}
        case _:
            assert False
    if input_params is not None:
        input_kwargs.update(
            (name + "_factory", parameterization_to_factory(param))
            for name, param in input_params.items()
        )
    input_factory = (
        categorical_input_layer_factory(
            num_channels=image_shape[0] if multichannel else 1,
            sharing="full" if input_sharing == "full" else "none",
            **input_kwargs,
        )
        if input_layer == "categorical"
        else name_to_input_layer_factory(input_layer, **input_kwargs)
    )

    # Set the sum weight factory
    if sum_weight_param is None:
        sum_weight_param = Parameterization(activation="softmax", initialization="normal")
    sum_weight_factory = parameterization_to_factory(sum_weight_param)

    # Set the nary sum weight factory
    nary_sum_weight_factory: ParameterFactory
    if use_mixing_weights:
        nary_sum_weight_factory = functools.partial(
            mixing_weight_factory, param_factory=sum_weight_factory
        )
    else:
        nary_sum_weight_factory = sum_weight_factory

    # Build and return the symbolic circuit
    circuit = rg.build_circuit(
        input_factory=input_factory,
        sum_product=sum_product_layer,
        sum_weight_factory=sum_weight_factory,
        nary_sum_weight_factory=nary_sum_weight_factory,
        num_input_units=num_input_units,
        num_sum_units=num_sum_units,
        num_classes=num_classes,
        factorize_multivariate=not multichannel,
    )
    if initialization is not None:
        assert estimation_data is not None
        initialize_parameters(
            circuit,
            estimation_data,
            initialization=initialization,
            sum_weight_param=sum_weight_param,
            input_sharing=input_sharing,
        )
    return circuit


def tabular_data(
    region_graph: TabularRegionGraph = "random-binary-tree",
    *,
    num_features: int | None = None,
    data: Tensor | None = None,
    input_layers: dict | list[dict],
    num_input_units: int,
    sum_product_layer: SumProductLayerName,
    num_sum_units: int,
    num_classes: int = 1,
    sum_weight_param: Parameterization | None = None,
    use_mixing_weights: bool = True,
    initialization: DataDrivenInitialization | None = None,
    input_sharing: InputSharing = "none",
) -> Circuit:
    """
    Constructs a symbolic circuit whose structure is tailored for tabular data sets,
    supporting either a fixed random-binary-tree or a learned Chow–Liu tree.

    Args:
        region_graph:
            Which region graph to use.
            - `"random-binary-tree"`: build a random binary tree over the feature indices.
            - `"chow-liu-tree"`: learn a Chow–Liu tree from data.
        num_features:
            Number of features (columns) in the dataset.
            **Required** if `region_graph="random-binary-tree"`.
        data:
            A Torch tensor of shape `(n_samples, n_features)`.
            **Required** if `region_graph="chow-liu-tree"`, since the tree structure is
                learned from these samples.
        input_layers:
            Which per-feature distribution to use.
            The provided dictionaries should be of the following form:
            {
                'name': <name: str>,
                'args': <dictionary of arguments: dict>
            }
            for example: {'name': 'categorical', 'args': {'num_categories': 27}}
                or {'name': 'gaussian', 'args': {}}
            If a dict is provided, the same input layer is used for all features.
                If a list of dictionaries is provided,
            each feature will have its own input layer (input_layers[i] corresponds
                to feature i of the data).
        num_input_units:
            Number of parallel input units (e.g. mixtures/components) per feature.
        sum_product_layer:
            Which inner sum/product decomposition to use. E.g. `"cp"`, `"cp-t"`, or `"tucker"`.
        num_sum_units:
            Number of sum (or mixing) units in each sum layer.
        num_classes:
            Number of output classes (or root-layer mixtures). Often 1 for pure density estimation.
        sum_weight_param:
            If provided, a `Parameterization` object specifying activation & initialization
            for sum-layer weights.  Defaults internally to a softmax + Normal init.
        use_mixing_weights:
            Whether to use “mixing” sum layers (i.e. learn a linear combination of child outputs)
            for nodes of arity >1.  If False, falls back to a matrix-vector product.
        initialization:
            Optional initialization from categorical observations on the selected structure.
            Requires non-empty data and categorical input specifications containing only
            num_categories. Each feature may have a different category count. Values must
            be finite integers within the corresponding category range. None preserves
            the usual initialization, including support for mixed input distributions.
        input_sharing:
            'none' for independent parameters or 'full' for a single shared categorical
            parameter tensor. Full sharing requires identical input specifications and
            assumes the features have the same domain. When initializing from data, it
            pools observations across all features, unlike per-channel pooling for images.

    Returns:
        Circuit
            A fully-specified sum-product circuit over the given region graph with the chosen
            input distributions and inner decomposition layer.

    Raises:
        ValueError:
            - If one of the names of the input layers is not known, or the related arguments.
            - If the number of input layers (the length of the list) does not match the number
                of features (`num_features` or inferred from `data`).
            - If `region_graph="random-binary-tree"` but `num_features` is `None` and `data`
                is None.
            - If `region_graph="chow-liu-tree"` but `data` is `None`.
            - If `region_graph` is not one of the supported strings.
    """

    if region_graph not in get_args(TabularRegionGraph):
        raise ValueError(f"Unknown region graph called {region_graph}")
    feature_options_enabled = initialization is not None or input_sharing != "none"
    if feature_options_enabled and sum_product_layer not in get_args(SumProductLayerName):
        raise ValueError(f"Unknown sum-product layer called {sum_product_layer}")
    if input_sharing not in get_args(InputSharing):
        raise ValueError(f"Unknown input sharing mode {input_sharing!r}")
    specs = [input_layers] if isinstance(input_layers, dict) else input_layers
    if feature_options_enabled and (
        not specs or any("name" not in spec or "args" not in spec for spec in specs)
    ):
        raise ValueError("Expected input specifications with name and args")
    if initialization is not None:
        if data is None:
            raise ValueError("Data-driven initialization requires a data tensor")
        if any(
            spec["name"] != "categorical" or set(spec["args"]) != {"num_categories"}
            for spec in specs
        ):
            raise ValueError(
                "Data-driven initialization requires categorical inputs with num_categories only"
            )
        if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
            raise ValueError(
                "Expected a non-empty data matrix of shape (num_samples, num_features)"
            )
        if num_features is not None and data.shape[1] != num_features:
            raise ValueError("num_features must match the number of data columns")
    if input_sharing == "full":
        if specs[0]["name"] != "categorical" or any(spec != specs[0] for spec in specs):
            raise ValueError("Full sharing requires identical categorical input specifications")

    rg = _build_tabular_region_graph(
        region_graph,
        num_features=num_features,
        data=data,
        input_layers=input_layers,
        validate_initialization=initialization is not None,
    )

    if sum_weight_param is None:
        sum_weight_param = Parameterization(activation="softmax", initialization="normal")
    sum_weight_factory = parameterization_to_factory(sum_weight_param)

    nary_sum_weight_factory: ParameterFactory
    if use_mixing_weights:
        nary_sum_weight_factory = functools.partial(
            mixing_weight_factory, param_factory=sum_weight_factory
        )
    else:
        nary_sum_weight_factory = sum_weight_factory

    input_factories: InputLayerFactory | Mapping[Scope, InputLayerFactory]
    if input_sharing == "full":
        if isinstance(input_layers, list) and len(input_layers) != len(rg.scope):
            raise ValueError("Number of input layers must match the number of features")
        input_factories = categorical_input_layer_factory(sharing="full", **specs[0]["args"])
    elif isinstance(input_layers, dict):
        input_factories = name_to_input_layer_factory(input_layers["name"], **input_layers["args"])
    else:
        if len(input_layers) != len(rg.scope):
            raise ValueError(
                f"Number of provided input layers ({len(input_layers)}) does \
                not match the number of features ({len(rg.scope)})."
            )
        input_factories = {
            Scope([i]): name_to_input_layer_factory(input_layer["name"], **input_layer["args"])
            for i, input_layer in enumerate(input_layers)
        }

    circuit = rg.build_circuit(
        input_factory=input_factories,
        sum_product=sum_product_layer,
        sum_weight_factory=sum_weight_factory,
        nary_sum_weight_factory=nary_sum_weight_factory,
        num_input_units=num_input_units,
        num_sum_units=num_sum_units,
        num_classes=num_classes,
        factorize_multivariate=True,
    )
    if initialization is not None:
        assert data is not None
        initialize_parameters(
            circuit,
            data,
            initialization=initialization,
            sum_weight_param=sum_weight_param,
            input_sharing=input_sharing,
        )
    return circuit
