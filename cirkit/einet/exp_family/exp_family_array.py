from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Sequence, Union

import torch
from torch import Tensor, nn

# TODO: find a good way to doc tensor shape


# TODO: why this inherits Module separately? but not Layer?
class ExponentialFamilyArray(nn.Module, ABC):  # pylint: disable=too-many-instance-attributes
    """ExponentialFamilyArray computes log-densities of exponential families in \
        parallel. ExponentialFamilyArray is \
        abstract and needs to be derived, in order to implement a concrete exponential family.

    The main use of ExponentialFamilyArray is to compute the densities for \
        FactorizedLeafLayer, which computes products \
        of densities over single RVs. All densities over single RVs are computed in \
        parallel via ExponentialFamilyArray.

    Note that when we talk about single RVs, these can in fact be multi-dimensional. \
        A natural use-case is RGB image \
        data: it is natural to consider pixels as single RVs, which are, however, \
        3-dimensional vectors each.

    Although ExponentialFamilyArray is not derived from class Layer, it implements \
        a similar interface. It is intended \
        that ExponentialFamilyArray is a helper class for FactorizedLeafLayer, which \
        just forwards calls to the Layer \
        interface.

    Best to think of ExponentialFamilyArray as an array of log-densities, \
        of shape array_shape, parallel for each RV.
    When evaluated, it returns a tensor of shape (batch_size, num_var, *array_shape) \
        -- for each sample in the batch and \
        each RV, it evaluates an array of array_shape densities, each with their \
        own parameters. Here, num_var is the number \
        of random variables, i.e. the size of the set (boldface) X in the paper.

    The boolean use_em indicates if we want to use the on-board EM algorithm \
        (alternatives would be SGD, Adam,...).

    After the ExponentialFamilyArray has been generated, we need to initialize \
        it. There are several options for \
        initialization (see also method initialize(...) below):
        'default': use the default initializer (to be written in derived classes).
        Tensor: provide a custom initialization.

    In order to implement a concrete exponential family, we need to derive this class and implement

        sufficient_statistics(self, x)
        log_normalizer(self, theta)
        log_h(self, x)

        expectation_to_natural(self, phi)
        default_initializer(self)
        project_params(self, params)
        reparam_function(self)
        _sample(self, *args, **kwargs)

    Please see docstrings of these functions below, for further details.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self, num_var: int, num_dims: int, array_shape: Sequence[int], num_stats: int, use_em: bool
    ):
        """Init class.

        :param num_var: number of random variables (int)
        :param num_dims: dimensionality of random variables (int)
        :param array_shape: shape of log-probability tensor, (tuple of ints)
                            log-probability tensor will be of shape \
                                (batch_size, num_var,) + array_shape
        :param num_stats: number of sufficient statistics of exponential family (int)
        :param use_em: use internal EM algorithm? (bool)
        """
        super().__init__()  # TODO: multi-inherit init?

        self.num_var = num_var
        self.num_dims = num_dims
        self.array_shape = array_shape
        self.num_stats = num_stats
        self.params_shape = (num_var, *array_shape, num_stats)

        # TODO: is this a good init? (originally None)
        self.params: nn.Parameter = nn.Parameter()
        self.ll: Tensor = torch.Tensor()
        self.suff_stats: Tensor = torch.Tensor()

        self.marginalization_idx: Optional[Tensor] = None  # TODO: should this be Tensor?
        self.marginalization_mask: Optional[Tensor] = None

        self._use_em = use_em
        # TODO: types of all `None`s?
        # TODO: why allow None but not directly init?
        self._p_acc: Optional[Tensor] = None
        self._stats_acc: Optional[Tensor] = None
        self._online_em_frequency: Optional[int] = None
        self._online_em_stepsize: Optional[float] = None
        self._online_em_counter: int = 0

        # if em is switched off, we re-parametrize the expectation parameters
        # self.reparam holds the function object for this task
        self.reparam = None
        if not self._use_em:
            self.reparam = self.reparam_function

    @abstractmethod
    def sufficient_statistics(self, x: Tensor) -> Tensor:
        """Get sufficient statistics for the implemented exponential \
            family (called T(x) in the paper).

        :param x: observed data (Tensor).
                  If self.num_dims == 1, this can be either of shape \
                    (batch_size, self.num_var, 1) or
                  (batch_size, self.num_var).
                  If self.num_dims > 1, this must be of shape \
                    (batch_size, self.num_var, self.num_dims).
        :return: sufficient statistics of the implemented exponential family (Tensor).
                 Must be of shape (batch_size, self.num_var, self.num_stats)
        """

    @abstractmethod
    def log_normalizer(self, theta: Tensor) -> Tensor:
        """Log-normalizer of the implemented exponential family (called A(theta) in the paper).

        :param theta: natural parameters (Tensor). Must be of shape \
            (self.num_var, *self.array_shape, self.num_stats).
        :return: log-normalizer (Tensor). Must be of shape (self.num_var, *self.array_shape).
        """

    @abstractmethod
    def log_h(self, x: Tensor) -> Tensor:
        """Get the log of the base measure (called h(x) in the paper).

        :param x: observed data (Tensor).
                  If self.num_dims == 1, this can be either of shape \
                    (batch_size, self.num_var, 1) or
                  (batch_size, self.num_var).
                  If self.num_dims > 1, this must be of shape \
                    (batch_size, self.num_var, self.num_dims).
        :return: log(h) of the implemented exponential family (Tensor).
                 Can either be a scalar or must be of shape (batch_size, self.num_var)
        """

    @abstractmethod
    def expectation_to_natural(self, phi: Tensor) -> Tensor:
        """Conversion from expectations parameters phi to natural parameters \
            theta, for the implemented exponential family.

        :param phi: expectation parameters (Tensor). Must be of shape \
            (self.num_var, *self.array_shape, self.num_stats).
        :return: natural parameters theta (Tensor). Same shape as phi.
        """

    @abstractmethod
    def default_initializer(self) -> Tensor:
        """Get default initializer for params.

        :return: initial parameters for the implemented exponential family (Tensor).
                 Must be of shape (self.num_var, *self.array_shape, self.num_stats)
        """

    @abstractmethod
    def project_params(self, params: Tensor) -> Tensor:
        """Project onto parameters' constraint set.

        Exponential families are usually defined on a constrained domain, e.g. \
            the second parameter of a Gaussian needs \
            to be non-negative. The EM algorithm takes the parameters sometimes \
            out of their domain. This function projects \
            them back onto their domain.

        :param params: the current parameters, same shape as self.params.
        :return: projected parameters, same shape as self.params.
        """

    @abstractmethod
    def reparam_function(self, params: Tensor) -> Tensor:
        """Re-parameterize parameters, in order that they stay in their constrained domain.

        When we are not using the EM, we need to transform unconstrained \
            (real-valued) parameters to the constrained set \
            of the expectation parameter. This function should return such a \
            function (i.e. the return value should not be \
            a projection, but a function which does the projection).

        :param params: I don't know
        :return: function object f which takes as input unconstrained parameters (Tensor) \
            and returns re-parametrized parameters.
        """

    @abstractmethod
    def _sample(  # type: ignore[misc]
        self, num_samples: int, params: Tensor, **kwargs: Any
    ) -> Tensor:
        """Is helper function for sampling the exponential family.

        :param num_samples: number of samples to be produced
        :param params: expectation parameters (phi) of the exponential family, of shape
                       (self.num_var, *self.array_shape, self.num_stats)
        :param kwargs: keyword arguments
               Depending on the implementation, kwargs can also contain further arguments.
        :return: i.i.d. samples of the exponential family (Tensor).
                 Should be of shape (num_samples, self.num_var, self.num_dims, *self.array_shape)
        """

    @abstractmethod
    def _argmax(self, params: Tensor, **kwargs: Any) -> Tensor:  # type: ignore[misc]
        """Is helper function for getting the argmax of the exponential family.

        :param params: expectation parameters (phi) of the exponential family, of shape
                       (self.num_var, *self.array_shape, self.num_stats)
        :param kwargs: keyword arguments
               Depending on the implementation, kwargs can also contain further arguments.
        :return: argmax of the exponential family (Tensor).
                 Should be of shape (self.num_var, self.num_dims, *self.array_shape)
        """

    # TODO: don't use str, use None instead
    def initialize(self, initializer: Union[Literal["default"], Tensor] = "default") -> None:
        """Initialize the parameters for this ExponentialFamilyArray.

        :param initializer: denotes the initialization method.
               If 'default' (str): use the default initialization, and store the parameters locally.
               If Tensor: provide custom initial parameters.
        """
        if initializer == "default":
            # default initializer; when em is switched off, we reparametrize
            # and use Gaussian noise as init values.
            if self._use_em:
                self.params = nn.Parameter(self.default_initializer())
            else:
                self.params = nn.Parameter(torch.randn(self.params_shape))
        else:
            # provided initializer
            assert isinstance(initializer, Tensor)  # type: ignore[misc]  # TODO: dummy for str
            assert initializer.shape == self.params_shape, "Incorrect parameter shape."
            self.params = nn.Parameter(initializer)

        assert not torch.isnan(self.params).any()

    def __call__(self, x: Tensor) -> Tensor:
        """Invoke the forward.

        Args:
            x (Tensor): The input.

        Returns:
            Tensor: The output.
        """
        return super().__call__(x)  # type: ignore[no-any-return,misc]

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the exponential family, in log-domain.

        For a single log-density we would compute
            log_h(X) + <params, T(X)> + A(params)
        Here, we do this in parallel and compute an array of log-densities of \
            shape array_shape, for each sample in the \
            batch and each RV.

        :param x: input data (Tensor).
                  If self.num_dims == 1, this can be either of shape \
                    (batch_size, self.num_var, 1) or
                  (batch_size, self.num_var).
                  If self.num_dims > 1, this must be of shape \
                    (batch_size, self.num_var, self.num_dims).
        :return: log-densities of implemented exponential family (Tensor).
                 Will be of shape (batch_size, self.num_var, *self.array_shape)
        """
        if self.reparam is None:
            with torch.no_grad():
                theta = self.expectation_to_natural(self.params)
        else:
            # TODO: no_grad?
            phi = self.reparam(self.params)

            # assert not torch.isnan(self.params).any()
            # assert not torch.isnan(phi).any()

            theta = self.expectation_to_natural(phi)

            # assert not torch.isnan(theta).any()
            # assert not torch.isinf(theta).any()

        # suff_stats: (batch_size, self.num_var, self.num_stats)
        self.suff_stats = self.sufficient_statistics(x)

        # assert not torch.isnan(self.suff_stats).any()
        # assert not torch.isinf(self.suff_stats).any()

        # log_normalizer: (self.num_var, *self.array_shape)
        log_normalizer = self.log_normalizer(theta)

        # log_h: scalar, or (batch_size, self.num_var)
        log_h = self.log_h(x)
        if len(log_h.shape) > 0:
            # reshape for broadcasting
            # TODO: this line is definitely not written in a good way
            log_h = log_h.reshape(log_h.shape[:2] + (1,) * len(self.array_shape))

        # compute the exponential family tensor
        # (batch_size, self.num_var, *self.array_shape)

        # antonio_mari -> edit: (theta.unsqueeze(0) * self.suff_stats).sum(-1) is inefficient
        # example: for MNIST with PD structure, batch_size=100 and num_sums=128,
        # categorical distr. it's computed
        # a tensor (100, 784, 128, 1, 256) -> over 10GB
        # given by the tensor broadcasting (1, 784, 128, 1, 256) * (100, 784, 1, 1, 256).
        # I try with an einsum operation (x, o, d, s), (b, x, s) -> b, x, o, d.
        # That should have the same result

        crucial_quantity_einsum = torch.einsum("xods,bxs->bxod", theta, self.suff_stats)

        # assert not torch.isnan(crucial_quantity_einsum).any()

        # reshape for broadcasting
        # shape = self.suff_stats.shape
        # shape = shape[0:2] + (1,) * len(self.array_shape) + (shape[2],)
        # self.suff_stats = self.suff_stats.reshape(shape)
        # crucial_quantity_orig = (theta.unsqueeze(0) * self.suff_stats).sum(-1)
        # assert torch.all(torch.eq(crucial_quantity_einsum, crucial_quantity_orig))
        # TODO: check also for other cases, for now I checked and it's correct

        # TODO: does ll have grad now?
        self.ll = log_h + crucial_quantity_einsum - log_normalizer

        if self._use_em:
            # EM needs the gradient with respect to self.ll
            self.ll.requires_grad_()

        # Marginalization in PCs works by simply setting leaves corresponding to
        # marginalized variables to 1 (0 in
        # (log-domain). We achieve this by a simple multiplicative 0-1 mask, generated here.
        # TODO: the marginalization mask doesn't need to be computed every time;
        # only when marginalization_idx changes.
        if self.marginalization_idx is not None:
            with torch.no_grad():
                # TODO: is this better? torch.ones(self.num_var).to(self.ll)
                self.marginalization_mask = torch.ones(
                    self.num_var, dtype=self.ll.dtype, device=self.ll.device
                )
                self.marginalization_mask[self.marginalization_idx] = 0.0
                # TODO: find another way to reshape
                shape = (1, self.num_var) + (1,) * len(self.array_shape)
                self.marginalization_mask = self.marginalization_mask.reshape(shape)
            output = self.ll * self.marginalization_mask
        else:
            self.marginalization_mask = None
            output = self.ll

        # assert not torch.isnan(output).any()
        # assert not torch.isinf(output).any()
        return output

    def sample(self, num_samples: int = 1, **kwargs: Any) -> Tensor:  # type: ignore[misc]
        """Sample the dist.

        Args:
            num_samples (int, optional): Number of samples. Defaults to 1.
            kwargs (Any, optional): Any kwargs.

        Returns:
            Tensor: The sample.
        """
        if self.reparam is None:
            params: Tensor = self.params
        else:
            # TODO: maybe the function should be no_grad?
            with torch.no_grad():
                params = self.reparam(self.params)
        return self._sample(num_samples, params, **kwargs)  # type: ignore[misc]

    def argmax(self, **kwargs: Any) -> Tensor:  # type: ignore[misc]
        """Get the argmax.

        Args:
            kwargs (Any, optional): Any kwargs.

        Returns:
            Tensor: The argmax.
        """
        if self.reparam is None:
            params: Tensor = self.params
        else:
            with torch.no_grad():
                params = self.reparam(self.params)
        return self._argmax(params, **kwargs)  # type: ignore[misc]

    # TODO: types
    def em_set_hyperparams(
        self, online_em_frequency: int, online_em_stepsize: float, purge: bool = True
    ) -> None:
        """Set new setting for online EM.

        Args:
            online_em_frequency (int): I don't know.
            online_em_stepsize (float): I don't know.
            purge (bool, optional): Whether to purge. Defaults to True.
        """
        if purge:
            self.em_purge()
            self._online_em_counter = 0
        self._online_em_frequency = online_em_frequency
        self._online_em_stepsize = online_em_stepsize

    def em_purge(self) -> None:
        """Discard em statistics."""
        if self.ll.grad is not None:
            self.ll.grad.zero_()
        self._p_acc = None
        self._stats_acc = None

    def em_process_batch(self) -> None:
        """Accumulate EM statistics of current batch.

        This should typically be called via EinsumNetwork.em_process_batch().
        """
        assert self._use_em, "em_process_batch called while _use_em==False."
        # TODO: do we allow it's None? if so, we need to guard it every time using it.
        # if self.params is None:
        #     return

        with torch.no_grad():
            # TODO: is grad None?
            assert self.ll.grad is not None
            p = self.ll.grad
            # weighted_stats = (p.unsqueeze(-1) * self.suff_stats).sum(0)
            # antonio_mari --> correction
            weighted_stats = torch.einsum("bxod,bxs->xods", p, self.suff_stats)
            p = p.sum(dim=0)

            if self._p_acc is None:
                self._p_acc = torch.zeros_like(p)
            self._p_acc += p

            if self._stats_acc is None:
                self._stats_acc = torch.zeros_like(weighted_stats)
            self._stats_acc += weighted_stats

            self.ll.grad.zero_()

            if self._online_em_frequency is not None:
                self._online_em_counter += 1
                if self._online_em_counter == self._online_em_frequency:
                    self.em_update(True)
                    self._online_em_counter = 0

    def em_update(self, _triggered: bool = False) -> None:
        """Do an EM update.
        
        If the setting is online EM (online_em_stepsize is not None), \
            then this function does nothing, \
            since updates are triggered automatically. (Thus, leave the private \
            parameter _triggered alone)

        TODO: pylint is not happy with this _triggered: for internal use, don't set
        """
        assert self._use_em, "em_update called while _use_em==False."
        if self._online_em_stepsize is not None and not _triggered:
            return

        with torch.no_grad():
            # TODO: can we guarantee this?
            assert self._p_acc is not None and self._stats_acc is not None
            if self._online_em_stepsize is None:
                self.params[:] = self._stats_acc / (self._p_acc.unsqueeze(-1) + 1e-12)
            else:
                s = self._online_em_stepsize
                self.params[:] = (1 - s) * self.params + s * (
                    self._stats_acc / (self._p_acc.unsqueeze(-1) + 1e-12)
                )
            self.params[:] = self.project_params(self.params)

        self._p_acc = None
        self._stats_acc = None

    # TODO: why we need this for public attr?
    def set_marginalization_idx(self, idx: Tensor) -> None:
        """Set indices of marginalized variables.

        Args:
            idx (Tensor): The idx to set.
        """
        self.marginalization_idx = idx

    def get_marginalization_idx(self) -> Optional[Tensor]:
        """Get indices of marginalized variables.

        Returns:
            Optional[Tensor]: The idx got.
        """
        return self.marginalization_idx
