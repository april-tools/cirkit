import torch
from cirkit.einet._utils import one_hot


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
