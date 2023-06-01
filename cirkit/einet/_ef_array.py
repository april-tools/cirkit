import torch
from cirkit.einet._utils import one_hot


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
