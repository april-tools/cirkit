import torch
from cirkit.einet._utils import one_hot


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
