import functools
from abc import ABC
from collections.abc import Iterable
from typing import Any, Mapping

import torch
from torch import Tensor

from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.backend.torch.layers import TorchInnerLayer, TorchInputLayer, TorchLayer
from cirkit.utils.scope import Scope


class Query(ABC):
    """An object used to run queries of circuits compiled using the torch backend."""

    def __init__(self) -> None:
        ...


class IntegrateQuery(Query):
    """The integration query object allows marginalising out variables.

    Computes output in two forward passes:
        a) The normal circuit forward pass for input x
        b) The integration forward pass where all variables are marginalised

    A mask over random variables is computed based on the scopes passed as
    input. This determines whether the integrated or normal circuit result
    is returned for each variable.
    """

    def __init__(self, circuit: TorchCircuit) -> None:
        """Initialize an integration query object.

        Args:
            circuit: The circuit to integrate over.

        Raises:
            ValueError: If the circuit to integrate is not smooth or not decomposable.
        """
        if not circuit.properties.smooth or not circuit.properties.decomposable:
            raise ValueError(
                f"The circuit to integrate must be smooth and decomposable, "
                f"but found {circuit.properties}"
            )
        super().__init__()
        self._circuit = circuit

    def __call__(
        self,
        x: Tensor,
        *,
        integrate_vars: Tensor | Scope | Iterable[Scope],
        gate_function_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> Tensor:
        """Solve an integration query, given an input batch and the variables to integrate.

        Args:
            x: An input batch of shape $(B, D)$, where $B$ is the batch size,
                and $D$ is the number of variables.
            integrate_vars: The variables to integrate. It must be a subset of the variables on
                which the circuit given in the constructor is defined on.
                The format can be one of the following three:
                    1. Tensor of shape (B, D) where B is the batch size and D is the number of
                        variables in the scope of the circuit. Its dtype should be torch.bool
                        and have True in the positions of random variables that should be
                        marginalised out and False elsewhere.
                    2. Scope, in this case the same integration mask is applied for all entries
                        of the batch
                    3. List of Scopes, where the length of the list must be either 1 or B. If
                        the list has length 1, behaves as above.
        Returns:
            The result of the integration query, given as a tensor of shape $(B, O, K)$,
                where $B$ is the batch size, $O$ is the number of output vectors of the circuit, and
                $K$ is the number of units in each output vector.
        """
        if isinstance(integrate_vars, Tensor):
            # Check type of tensor is boolean
            if integrate_vars.dtype != torch.bool:
                raise ValueError(
                    f"Expected dtype of tensor to be torch.bool, got {integrate_vars.dtype}"
                )
            # If single dimensional tensor, assume batch size = 1
            if len(integrate_vars.shape) == 1:
                integrate_vars = torch.unsqueeze(integrate_vars, 0)
            # If the scope is correct, proceed, otherwise error
            num_vars = max(self._circuit.scope) + 1
            if integrate_vars.shape[1] == num_vars:
                integrate_vars_mask = integrate_vars
            else:
                raise ValueError(
                    f"Circuit scope has {num_vars} variables but integrate_vars "
                    f"was defined over {integrate_vars.shape[1]} != {num_vars} variables"
                )
        else:
            # Convert list of scopes to a boolean mask of dimension (B, N) where
            # N is the number of variables in the circuit's scope.
            integrate_vars_mask = IntegrateQuery.scopes_to_mask(self._circuit, integrate_vars)
            integrate_vars_mask = integrate_vars_mask.to(x.device)

        # Check batch sizes of input x and mask are compatible
        if integrate_vars_mask.shape[0] not in (1, x.shape[0]):
            raise ValueError(
                "The number of scopes to integrate over must "
                "either match the batch size of x, or be 1 if you "
                "want to broadcast. Found #inputs = "
                f"{x.shape[0]} != {integrate_vars_mask.shape[0]} = len(integrate_vars)"
            )

        gate_function_kwargs = {} if gate_function_kwargs is None else gate_function_kwargs
        for gate_function_id, gate_function_eval in self._circuit._gate_function_evals.items():
            kwargs = gate_function_kwargs.get(gate_function_id, {})
            gate_function_eval.memoize(**kwargs)

        output = self._circuit.evaluate(
            x,
            module_fn=functools.partial(
                IntegrateQuery._layer_fn, integrate_vars_mask=integrate_vars_mask
            ),
        )  # (O, B, K)
        return output.transpose(0, 1)  # (B, O, K)

    @staticmethod
    def _layer_fn(layer: TorchLayer, x: Tensor, *, integrate_vars_mask: Tensor) -> Tensor:
        # Evaluate a layer: if it is not an input layer, then evaluate it in the usual
        # feed-forward way. Otherwise, use the variables to integrate to solve the marginal
        # queries on the input layers.
        output = layer(x)  # (F, B, Ko)
        if not isinstance(layer, TorchInputLayer):
            return output
        if layer.num_variables > 1:
            raise NotImplementedError("Integration of multivariate input layers is not supported")
        # integrate_vars_mask is a boolean tensor of dim (B, N)
        # where N is the number of variables in the scope of the whole circuit.
        #
        # layer.scope_idx contains a subset of the variable_idxs of the scope
        # but may be a reshaped tensor; the shape and order of the variables may be different.
        #
        # as such, we need to use the idxs in layer.scope_idx to lookup the values from
        # the integrate_vars_mask - this will return the correct shape and values.
        #
        # if integrate_vars_mask was a vector, we could do integrate_vars_mask[layer.scope_idx]
        # the vmap below applies the above across the B dimension

        # integration_mask has dimension (B, F, Ko)
        integration_mask = torch.vmap(lambda x: x[layer.scope_idx])(integrate_vars_mask)
        # permute to match integration_output: integration_mask has dimension (F, B, Ko)
        integration_mask = integration_mask.permute([1, 0, 2])

        if not torch.any(integration_mask).item():
            return output

        integration_output = layer.integrate()
        # Use the integration mask to select which output should be the result of
        # an integration operation, and which should not be
        # This is done in parallel for all folds, and regardless of whether the
        # circuit is folded or unfolded
        return torch.where(integration_mask, integration_output, output)

    @staticmethod
    def scopes_to_mask(circuit: TorchCircuit, batch_integrate_vars: Scope | list[Scope]):
        """Accepts a batch of scopes and returns a boolean mask as a tensor with
        True in positions of specified scope indices and False otherwise.
        """
        # If we passed a single scope, assume B = 1
        if isinstance(batch_integrate_vars, Scope):
            batch_integrate_vars = [batch_integrate_vars]

        batch_size = len(tuple(batch_integrate_vars))
        # There are cases where the circuit.scope may change,
        # e.g. we may marginalise out X_1 and the length of the scope may be smaller
        # but the actual scope will not have been shifted.
        num_rvs = max(circuit.scope) + 1
        num_idxs = sum(len(s) for s in batch_integrate_vars)

        # TODO: Maybe consider using a sparse tensor
        mask = torch.zeros((batch_size, num_rvs), dtype=torch.bool)

        # Catch case of only empty scopes where the following command will fail
        if num_idxs == 0:
            return mask

        batch_idxs, rv_idxs = zip(
            *((i, idx) for i, idxs in enumerate(batch_integrate_vars) for idx in idxs if idxs)
        )

        # Check that we have not asked to marginalise variables that are not defined
        invalid_idxs = Scope(rv_idxs) - circuit.scope
        if invalid_idxs:
            raise ValueError(
                "The variables to marginalize must be a subset of "
                "the circuit scope. Invalid variables "
                f"not in scope: {list(invalid_idxs)} "
            )

        mask[batch_idxs, rv_idxs] = True

        return mask


class SamplingQuery(Query):
    """The sampling query object."""

    def __init__(self, circuit: TorchCircuit) -> None:
        """Initialize a sampling query object. Currently, only sampling from the joint distribution
            is supported, i.e., sampling won't work in the case of circuits obtained by
            marginalization, or by observing evidence. Conditional sampling is currently not
            implemented.

        Args:
            circuit: The circuit to sample from.

        Raises:
            ValueError: If the circuit to sample from is not normalised.
        """
        if not circuit.properties.smooth or not circuit.properties.decomposable:
            raise ValueError(
                f"The circuit to sample from must be smooth and decomposable, "
                f"but found {circuit.properties}"
            )
        # TODO: add a check to verify the circuit is monotonic and normalized?
        super().__init__()
        self._circuit = circuit

    def __call__(self, num_samples: int = 1) -> tuple[Tensor, list[Tensor]]:
        """Sample a number of data points.

        Args:
            num_samples: The number of samples to return.

        Return:
            A pair (samples, mixture_samples), consisting of (i) an assignment to the observed
            variables the circuit is defined on, and (ii) the samples of the finitely-discrete
            latent variables associated to the sum units. The samples (i) are returned as a
            tensor of shape (num_samples, num_variables).

        Raises:
            ValueError: if the number of samples is not a positive number.
        """
        if num_samples <= 0:
            raise ValueError("The number of samples must be a positive number")

        mixture_samples: list[Tensor] = []
        # samples: (O, K, num_samples, D)
        samples = self._circuit.evaluate(
            module_fn=functools.partial(
                self._layer_fn,
                num_samples=num_samples,
                mixture_samples=mixture_samples,
            ),
        )
        # samples: (num_samples, O, K, D)
        samples = samples.permute(2, 0, 1, 3)
        # TODO: fix for the case of multi-output circuits, i.e., O != 1 or K != 1
        samples = samples[:, 0, 0]  # (num_samples, D)
        return samples, mixture_samples

    def _layer_fn(
        self, layer: TorchLayer, *inputs: Tensor, num_samples: int, mixture_samples: list[Tensor]
    ) -> Tensor:
        # Sample from an input layer
        if not inputs:
            assert isinstance(layer, TorchInputLayer)
            samples = layer.sample(num_samples)
            samples = self._pad_samples(samples, layer.scope_idx)
            mixture_samples.append(samples)
            return samples

        # Sample through an inner layer
        assert isinstance(layer, TorchInnerLayer)
        samples, mix_samples = layer.sample(*inputs)
        if mix_samples is not None:
            mixture_samples.append(mix_samples)
        return samples

    def _pad_samples(self, samples: Tensor, scope_idx: Tensor) -> Tensor:
        """Pads univariate samples to the size of the scope of the circuit (output dimension)
        according to scope for compatibility in downstream inner nodes.
        """
        if scope_idx.shape[1] != 1:
            raise NotImplementedError("Padding is only implemented for univariate samples")

        # padded_samples: (F, K, num_samples, D)
        padded_samples = torch.zeros(
            (*samples.shape, len(self._circuit.scope)), device=samples.device, dtype=samples.dtype
        )
        fold_idx = torch.arange(samples.shape[0], device=samples.device)
        padded_samples[fold_idx, :, :, scope_idx.squeeze(dim=1)] = samples
        return padded_samples
