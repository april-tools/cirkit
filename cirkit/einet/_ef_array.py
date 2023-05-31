import torch
from cirkit.einet._utils import one_hot


class ExponentialFamilyArray(torch.nn.Module):
    """
    ExponentialFamilyArray computes log-densities of exponential families in parallel. ExponentialFamilyArray is
    abstract and needs to be derived, in order to implement a concrete exponential family.

    The main use of ExponentialFamilyArray is to compute the densities for FactorizedLeafLayer, which computes products
    of densities over single RVs. All densities over single RVs are computed in parallel via ExponentialFamilyArray.

    Note that when we talk about single RVs, these can in fact be multi-dimensional. A natural use-case is RGB image
    data: it is natural to consider pixels as single RVs, which are, however, 3-dimensional vectors each.

    Although ExponentialFamilyArray is not derived from class Layer, it implements a similar interface. It is intended
    that ExponentialFamilyArray is a helper class for FactorizedLeafLayer, which just forwards calls to the Layer
    interface.

    Best to think of ExponentialFamilyArray as an array of log-densities, of shape array_shape, parallel for each RV.
    When evaluated, it returns a tensor of shape (batch_size, num_var, *array_shape) -- for each sample in the batch and
    each RV, it evaluates an array of array_shape densities, each with their own parameters. Here, num_var is the number
    of random variables, i.e. the size of the set (boldface) X in the paper.

    The boolean use_em indicates if we want to use the on-board EM algorithm (alternatives would be SGD, Adam,...).

    After the ExponentialFamilyArray has been generated, we need to initialize it. There are several options for
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

    def __init__(self, num_var, num_dims, array_shape, num_stats, use_em):
        """
        :param num_var: number of random variables (int)
        :param num_dims: dimensionality of random variables (int)
        :param array_shape: shape of log-probability tensor, (tuple of ints)
                            log-probability tensor will be of shape (batch_size, num_var,) + array_shape
        :param num_stats: number of sufficient statistics of exponential family (int)
        :param use_em: use internal EM algorithm? (bool)
        """
        super(ExponentialFamilyArray, self).__init__()

        self.num_var = num_var
        self.num_dims = num_dims
        self.array_shape = array_shape
        self.num_stats = num_stats
        self.params_shape = (num_var, *array_shape, num_stats)

        self.params = None
        self.ll = None
        self.suff_stats = None

        self.marginalization_idx = None
        self.marginalization_mask = None

        self._use_em = use_em
        self._p_acc = None
        self._stats_acc = None
        self._online_em_frequency = None
        self._online_em_stepsize = None
        self._online_em_counter = 0

        # if em is switched off, we re-parametrize the expectation parameters
        # self.reparam holds the function object for this task
        self.reparam = None
        if not self._use_em:
            self.reparam = self.reparam_function

    # --------------------------------------------------------------------------------
    # The following functions need to be implemented to specify an exponential family.

    def sufficient_statistics(self, x):
        """
        The sufficient statistics function for the implemented exponential family (called T(x) in the paper).

        :param x: observed data (Tensor).
                  If self.num_dims == 1, this can be either of shape (batch_size, self.num_var, 1) or
                  (batch_size, self.num_var).
                  If self.num_dims > 1, this must be of shape (batch_size, self.num_var, self.num_dims).
        :return: sufficient statistics of the implemented exponential family (Tensor).
                 Must be of shape (batch_size, self.num_var, self.num_stats)
        """
        raise NotImplementedError

    def log_normalizer(self, theta):
        """
        Log-normalizer of the implemented exponential family (called A(theta) in the paper).

        :param theta: natural parameters (Tensor). Must be of shape (self.num_var, *self.array_shape, self.num_stats).
        :return: log-normalizer (Tensor). Must be of shape (self.num_var, *self.array_shape).
        """
        raise NotImplementedError

    def log_h(self, x):
        """
        The log of the base measure (called h(x) in the paper).

        :param x: observed data (Tensor).
                  If self.num_dims == 1, this can be either of shape (batch_size, self.num_var, 1) or
                  (batch_size, self.num_var).
                  If self.num_dims > 1, this must be of shape (batch_size, self.num_var, self.num_dims).
        :return: log(h) of the implemented exponential family (Tensor).
                 Can either be a scalar or must be of shape (batch_size, self.num_var)
        """
        raise NotImplementedError

    def expectation_to_natural(self, phi):
        """
        Conversion from expectations parameters phi to natural parameters theta, for the implemented exponential
        family.

        :param phi: expectation parameters (Tensor). Must be of shape (self.num_var, *self.array_shape, self.num_stats).
        :return: natural parameters theta (Tensor). Same shape as phi.
        """
        raise NotImplementedError

    def default_initializer(self):
        """
        Default initializer for params.

        :return: initial parameters for the implemented exponential family (Tensor).
                 Must be of shape (self.num_var, *self.array_shape, self.num_stats)
        """
        raise NotImplementedError

    def project_params(self, params):
        """
        Project onto parameters' constraint set.

        Exponential families are usually defined on a constrained domain, e.g. the second parameter of a Gaussian needs
        to be non-negative. The EM algorithm takes the parameters sometimes out of their domain. This function projects
        them back onto their domain.

        :param params: the current parameters, same shape as self.params.
        :return: projected parameters, same shape as self.params.
        """
        raise NotImplementedError

    def reparam_function(self, params):
        """
        Re-parameterize parameters, in order that they stay in their constrained domain.

        When we are not using the EM, we need to transform unconstrained (real-valued) parameters to the constrained set
        of the expectation parameter. This function should return such a function (i.e. the return value should not be
        a projection, but a function which does the projection).

        :return: function object f which takes as input unconstrained parameters (Tensor) and returns re-parametrized
                 parameters.
        """
        raise NotImplementedError

    def _sample(self, num_samples, params, **kwargs):
        """
        Helper function for sampling the exponential family.

        :param num_samples: number of samples to be produced
        :param params: expectation parameters (phi) of the exponential family, of shape
                       (self.num_var, *self.array_shape, self.num_stats)
        :param kwargs: keyword arguments
               Depending on the implementation, kwargs can also contain further arguments.
        :return: i.i.d. samples of the exponential family (Tensor).
                 Should be of shape (num_samples, self.num_var, self.num_dims, *self.array_shape)
        """
        raise NotImplementedError

    def _argmax(self, params, **kwargs):
        """
        Helper function for getting the argmax of the exponential family.

        :param params: expectation parameters (phi) of the exponential family, of shape
                       (self.num_var, *self.array_shape, self.num_stats)
        :param kwargs: keyword arguments
               Depending on the implementation, kwargs can also contain further arguments.
        :return: argmax of the exponential family (Tensor).
                 Should be of shape (self.num_var, self.num_dims, *self.array_shape)
        """
        raise NotImplementedError

    # --------------------------------------------------------------------------------

    def initialize(self, initializer='default'):
        """
        Initialize the parameters for this ExponentialFamilyArray.

        :param initializer: denotes the initialization method.
               If 'default' (str): use the default initialization, and store the parameters locally.
               If Tensor: provide custom initial parameters.
        :return: None
        """
        if type(initializer) == str and initializer == 'default':
            # default initializer; when em is switched off, we reparametrize and use Gaussian noise as init values.
            if self._use_em:
                self.params = torch.nn.Parameter(self.default_initializer())
            else:
                self.params = torch.nn.Parameter(torch.randn(self.params_shape))
        elif type(initializer) == torch.Tensor:
            # provided initializer
            if initializer.shape != self.params_shape:
                raise AssertionError("Incorrect parameter shape.")
            self.params = torch.nn.Parameter(initializer)
        else:
            raise AssertionError("Unknown initializer.")

        assert not torch.isnan(self.params).any()

    def forward(self, x):
        """
        Evaluates the exponential family, in log-domain. For a single log-density we would compute
            log_h(X) + <params, T(X)> + A(params)
        Here, we do this in parallel and compute an array of log-densities of shape array_shape, for each sample in the
        batch and each RV.

        :param x: input data (Tensor).
                  If self.num_dims == 1, this can be either of shape (batch_size, self.num_var, 1) or
                  (batch_size, self.num_var).
                  If self.num_dims > 1, this must be of shape (batch_size, self.num_var, self.num_dims).
        :return: log-densities of implemented exponential family (Tensor).
                 Will be of shape (batch_size, self.num_var, *self.array_shape)
        """

        if self._use_em:
            with torch.no_grad():
                theta = self.expectation_to_natural(self.params)
        else:
            phi = self.reparam(self.params)

            #assert not torch.isnan(self.params).any()
            #assert not torch.isnan(phi).any()

            theta = self.expectation_to_natural(phi)

            #assert not torch.isnan(theta).any()
            #assert not torch.isinf(theta).any()

        # suff_stats: (batch_size, self.num_var, self.num_stats)
        self.suff_stats = self.sufficient_statistics(x)

        #assert not torch.isnan(self.suff_stats).any()
        #assert not torch.isinf(self.suff_stats).any()

        # log_normalizer: (self.num_var, *self.array_shape)
        log_normalizer = self.log_normalizer(theta)

        # log_h: scalar, or (batch_size, self.num_var)
        log_h = self.log_h(x)
        if len(log_h.shape) > 0:
            # reshape for broadcasting
            log_h = log_h.reshape(log_h.shape[0:2] + (1,) * len(self.array_shape))

        # compute the exponential family tensor
        # (batch_size, self.num_var, *self.array_shape)

        # antonio_mari -> edit: (theta.unsqueeze(0) * self.suff_stats).sum(-1) is inefficient
        # example: for MNIST with PD structure, batch_size=100 and num_sums=128, categorical distr. it's computed
        # a tensor (100, 784, 128, 1, 256) -> over 10GB
        # given by the tensor broadcasting (1, 784, 128, 1, 256) * (100, 784, 1, 1, 256).
        # I try with an einsum operation (x, o, d, s), (b, x, s) -> b, x, o, d. That should have the same result

        crucial_quantity_einsum = torch.einsum("xods,bxs->bxod", theta, self.suff_stats)

        #assert not torch.isnan(crucial_quantity_einsum).any()

        # reshape for broadcasting
        # shape = self.suff_stats.shape
        # shape = shape[0:2] + (1,) * len(self.array_shape) + (shape[2],)
        # self.suff_stats = self.suff_stats.reshape(shape)
        # crucial_quantity_orig = (theta.unsqueeze(0) * self.suff_stats).sum(-1)
        # assert torch.all(torch.eq(crucial_quantity_einsum, crucial_quantity_orig))
        # TODO: check also for other cases, for now I checked and it's correct

        self.ll = log_h + crucial_quantity_einsum - log_normalizer

        if self._use_em:
            # EM needs the gradient with respect to self.ll
            self.ll.requires_grad_()

        # Marginalization in PCs works by simply setting leaves corresponding to marginalized variables to 1 (0 in
        # (log-domain). We achieve this by a simple multiplicative 0-1 mask, generated here.
        # TODO: the marginalization mask doesn't need to be computed every time; only when marginalization_idx changes.
        if self.marginalization_idx is not None:
            with torch.no_grad():
                self.marginalization_mask = torch.ones(self.num_var, dtype=self.ll.dtype, device=self.ll.device)
                self.marginalization_mask.data[self.marginalization_idx] = 0.0
                shape = (1, self.num_var) + (1,) * len(self.array_shape)
                self.marginalization_mask = self.marginalization_mask.reshape(shape)
                self.marginalization_mask.requires_grad_(False)
        else:
            self.marginalization_mask = None

        if self.marginalization_mask is not None:
            output = self.ll * self.marginalization_mask
        else:
            output = self.ll

        #assert not torch.isnan(output).any()
        #assert not torch.isinf(output).any()
        return output

    def sample(self, num_samples=1, **kwargs):
        if self._use_em:
            params = self.params
        else:
            with torch.no_grad():
                params = self.reparam(self.params)
        return self._sample(num_samples, params, **kwargs)

    def argmax(self, **kwargs):
        if self._use_em:
            params = self.params
        else:
            with torch.no_grad():
                params = self.reparam(self.params)
        return self._argmax(params, **kwargs)

    def em_set_hyperparams(self, online_em_frequency, online_em_stepsize, purge=True):
        """Set new setting for online EM."""
        if purge:
            self.em_purge()
            self._online_em_counter = 0
        self._online_em_frequency = online_em_frequency
        self._online_em_stepsize = online_em_stepsize

    def em_purge(self):
        """ Discard em statistics."""
        if self.ll is not None and self.ll.grad is not None:
            self.ll.grad.zero_()
        self._p_acc = None
        self._stats_acc = None

    def em_process_batch(self):
        """
        Accumulate EM statistics of current batch. This should typically be called via EinsumNetwork.em_process_batch().
        """
        if not self._use_em:
            raise AssertionError("em_process_batch called while _use_em==False.")
        if self.params is None:
            return

        with torch.no_grad():
            p = self.ll.grad
            # weighted_stats = (p.unsqueeze(-1) * self.suff_stats).sum(0)
            # antonio_mari --> correction
            weighted_stats = torch.einsum("bxod,bxs->xods", p, self.suff_stats)
            p = p.sum(0)

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

    def em_update(self, _triggered=False):
        """
        Do an EM update. If the setting is online EM (online_em_stepsize is not None), then this function does nothing,
        since updates are triggered automatically. (Thus, leave the private parameter _triggered alone)

        :param _triggered: for internal use, don't set
        :return: None
        """
        if not self._use_em:
            raise AssertionError("em_update called while _use_em==False.")
        if self._online_em_stepsize is not None and not _triggered:
            return

        with torch.no_grad():
            if self._online_em_stepsize is None:
                self.params.data = self._stats_acc / (self._p_acc.unsqueeze(-1) + 1e-12)
            else:
                s = self._online_em_stepsize
                self.params.data = (1. - s) * self.params + s * (self._stats_acc / (self._p_acc.unsqueeze(-1) + 1e-12))
            self.params.data = self.project_params(self.params.data)

        self._p_acc = None
        self._stats_acc = None

    def set_marginalization_idx(self, idx):
        """Set indices of marginalized variables."""
        self.marginalization_idx = idx

    def get_marginalization_idx(self):
        """Get indices of marginalized variables."""
        return self.marginalization_idx


def shift_last_axis_to(x, i):
    """This takes the last axis of tensor x and inserts it at position i"""
    num_axes = len(x.shape)
    return x.permute(tuple(range(i)) + (num_axes - 1,) + tuple(range(i, num_axes - 1)))


class NormalArray(ExponentialFamilyArray):
    """Implementation of Normal distribution."""

    def __init__(self, num_var, num_dims, array_shape, min_var=0.0001, max_var=10., use_em=True):
        super(NormalArray, self).__init__(num_var, num_dims, array_shape, 2 * num_dims, use_em=use_em)
        self.log_2pi = torch.tensor(1.8378770664093453)
        self.min_var = min_var
        self.max_var = max_var

    def default_initializer(self):
        phi = torch.empty(self.num_var, *self.array_shape, 2*self.num_dims)
        with torch.no_grad():
            phi[..., 0:self.num_dims] = torch.randn(self.num_var, *self.array_shape, self.num_dims)
            phi[..., self.num_dims:] = 1. + phi[..., 0:self.num_dims]**2
        return phi

    def project_params(self, phi):
        phi_project = phi.clone()
        mu2 = phi_project[..., 0:self.num_dims] ** 2
        phi_project[..., self.num_dims:] -= mu2
        phi_project[..., self.num_dims:] = torch.clamp(phi_project[..., self.num_dims:], self.min_var, self.max_var)
        phi_project[..., self.num_dims:] += mu2
        return phi_project

    def reparam_function(self, params_in):
        mu = params_in[..., 0:self.num_dims].clone()
        var = self.min_var + torch.sigmoid(params_in[..., self.num_dims:]) * (self.max_var - self.min_var)
        return torch.cat((mu, var + mu**2), -1)


    def sufficient_statistics(self, x):
        if len(x.shape) == 2:
            stats = torch.stack((x, x ** 2), -1)
        elif len(x.shape) == 3:
            stats = torch.cat((x, x**2), -1)
        else:
            raise AssertionError("Input must be 2 or 3 dimensional tensor.")
        return stats

    def expectation_to_natural(self, phi):
        var = phi[..., self.num_dims:] - phi[..., 0:self.num_dims] ** 2
        theta1 = phi[..., 0:self.num_dims] / var
        theta2 = - 1. / (2. * var)
        return torch.cat((theta1, theta2), -1)

    def log_normalizer(self, theta):
        log_normalizer = -theta[..., 0:self.num_dims] ** 2 / (4 * theta[..., self.num_dims:]) - 0.5 * torch.log(-2. * theta[..., self.num_dims:])
        log_normalizer = torch.sum(log_normalizer, -1)
        return log_normalizer

    def log_h(self, x):
        return -0.5 * self.log_2pi * self.num_dims

    def _sample(self, num_samples, params, std_correction=1.0):
        with torch.no_grad():
            mu = params[..., 0:self.num_dims]
            var = params[..., self.num_dims:] - mu**2
            std = torch.sqrt(var)
            shape = (num_samples,) + mu.shape
            samples = mu.unsqueeze(0) + std_correction * std.unsqueeze(0) * torch.randn(shape, dtype=mu.dtype, device=mu.device)
            return shift_last_axis_to(samples, 2)

    def _argmax(self, params, **kwargs):
        with torch.no_grad():
            mu = params[..., 0:self.num_dims]
            return shift_last_axis_to(mu, 1)


class BinomialArray(ExponentialFamilyArray):
    """Implementation of Binomial distribution."""

    def __init__(self, num_var, num_dims, array_shape, N, use_em=True):
        super(BinomialArray, self).__init__(num_var, num_dims, array_shape, num_dims, use_em=use_em)
        self.N = torch.tensor(float(N))

    def default_initializer(self):
        phi = (0.01 + 0.98 * torch.rand(self.num_var, *self.array_shape, self.num_dims)) * self.N
        return phi

    def project_params(self, phi):
        return torch.clamp(phi, 0.0, self.N)

    def reparam_function(self, params):
        return torch.sigmoid(params * 0.1) * float(self.N)

    def sufficient_statistics(self, x):
        if len(x.shape) == 2:
            stats = x.unsqueeze(-1)
        elif len(x.shape) == 3:
            stats = x
        else:
            raise AssertionError("Input must be 2 or 3 dimensional tensor.")
        return stats

    def expectation_to_natural(self, phi):
        theta = torch.clamp(phi / self.N, 1e-6, 1. - 1e-6)
        theta = torch.log(theta) - torch.log(1. - theta)
        return theta

    def log_normalizer(self, theta):
        return torch.sum(self.N * torch.nn.functional.softplus(theta), -1)

    def log_h(self, x):
        if self.N == 1:
            return torch.zeros([], device=x.device)
        else:
            log_h = torch.lgamma(self.N + 1.) - torch.lgamma(x + 1.) - torch.lgamma(self.N + 1. - x)
            if len(x.shape) == 3:
                log_h = log_h.sum(-1)
            return log_h

    def _sample(self, num_samples, params, dtype=torch.float32, memory_efficient_binomial_sampling=True):
        with torch.no_grad():
            params = params / self.N
            if memory_efficient_binomial_sampling:
                samples = torch.zeros((num_samples,) + params.shape, dtype=dtype, device=params.device)
                for n in range(int(self.N)):
                    rand = torch.rand((num_samples,) + params.shape, device=params.device)
                    samples += (rand < params).type(dtype)
            else:
                rand = torch.rand((num_samples,) + params.shape + (int(self.N),), device=params.device)
                samples = torch.sum(rand < params.unsqueeze(-1), -1).type(dtype)
            return shift_last_axis_to(samples, 2)

    def _argmax(self, params, dtype=torch.float32):
        with torch.no_grad():
            params = params / self.N
            mode = torch.clamp(torch.floor((self.N + 1.) * params), 0.0, self.N).type(dtype)
            return shift_last_axis_to(mode, 1)


class CategoricalArray(ExponentialFamilyArray):
    """Implementation of Categorical distribution."""

    def __init__(self, num_var, num_dims, array_shape, K, use_em=True):
        super(CategoricalArray, self).__init__(num_var, num_dims, array_shape, num_dims * K, use_em=use_em)
        self.K = K

    def default_initializer(self):
        phi = (0.01 + 0.98 * torch.rand(self.num_var, *self.array_shape, self.num_dims * self.K))
        return phi

    def project_params(self, phi):
        """Note that this is not actually l2-projection. For simplicity, we simply renormalize."""
        phi = phi.reshape(self.num_var, *self.array_shape, self.num_dims, self.K)
        phi = torch.clamp(phi, min=1e-12)
        phi = phi / torch.sum(phi, -1, keepdim=True)
        return phi.reshape(self.num_var, *self.array_shape, self.num_dims * self.K)

    def reparam_function(self, params):
        return torch.nn.functional.softmax(params, -1)

    def sufficient_statistics(self, x):
        if len(x.shape) == 2:
            stats = one_hot(x.long(), self.K)
        elif len(x.shape) == 3:
            stats = one_hot(x.long(), self.K).reshape(-1, self.num_dims * self.K)
        else:
            raise AssertionError("Input must be 2 or 3 dimensional tensor.")
        return stats

    def expectation_to_natural(self, phi):
        theta = torch.clamp(phi, 1e-12, 1.)
        theta = theta.reshape(self.num_var, *self.array_shape, self.num_dims, self.K)
        theta /= theta.sum(-1, keepdim=True)
        theta = theta.reshape(self.num_var, *self.array_shape, self.num_dims * self.K)
        theta = torch.log(theta)
        return theta

    def log_normalizer(self, theta):
        return 0.0

    def log_h(self, x):
        return torch.zeros([], device=x.device)

    def _sample(self, num_samples, params, dtype=torch.float32):
        with torch.no_grad():
            dist = params.reshape(self.num_var, *self.array_shape, self.num_dims, self.K)
            cum_sum = torch.cumsum(dist[..., 0:-1], -1)
            rand = torch.rand((num_samples,) + cum_sum.shape[0:-1] + (1,), device=cum_sum.device)
            samples = torch.sum(rand > cum_sum, -1).type(dtype)
            return shift_last_axis_to(samples, 2)

    def _argmax(self, params, dtype=torch.float32):
        with torch.no_grad():
            dist = params.reshape(self.num_var, *self.array_shape, self.num_dims, self.K)
            mode = torch.argmax(dist, -1).type(dtype)
            return shift_last_axis_to(mode, 1)
