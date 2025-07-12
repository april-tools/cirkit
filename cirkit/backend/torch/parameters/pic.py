import numpy as np
import torch
from torch import nn

from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.backend.torch.layers import (
    TorchCategoricalLayer,
    TorchCPTLayer,
    TorchGaussianLayer,
    TorchHadamardLayer,
    TorchKroneckerLayer,
    TorchSumLayer,
    TorchTuckerLayer,
)
from cirkit.backend.torch.parameters.nodes import TorchParameterOp, TorchTensorParameter


def zw_quadrature(
    integration_method: str,
    nip: int,
    a: float | None = -1,
    b: float | None = 1,
    return_log_weight: bool | None = False,
    dtype: torch.dtype | None = torch.float32,
):
    if integration_method == "leggauss":
        z_quad, w_quad = np.polynomial.legendre.leggauss(nip)
        z_quad = (b - a) * (z_quad + 1) / 2 + a
        w_quad = w_quad * (b - a) / 2
    elif integration_method == "midpoint":
        z_quad = np.linspace(a, b, num=nip + 1)
        z_quad = (z_quad[:-1] + z_quad[1:]) / 2
        w_quad = np.full_like(z_quad, (b - a) / nip)
    elif integration_method == "trapezoidal":
        z_quad = np.linspace(a, b, num=nip)
        w_quad = np.full((nip,), (b - a) / (nip - 1))
        w_quad[0] = w_quad[-1] = 0.5 * (b - a) / (nip - 1)
    elif integration_method == "simpson":
        assert nip % 2 == 1, "Number of integration points must be odd"
        z_quad = np.linspace(a, b, num=nip)
        w_quad = np.concatenate(
            [np.ones(1), np.tile(np.array([4, 2]), nip // 2 - 1), np.array([4, 1])]
        )
        w_quad = ((b - a) / (nip - 1)) / 3 * w_quad
    elif integration_method == "hermgauss":
        # https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature
        z_quad, w_quad = np.polynomial.hermite.hermgauss(nip)
    else:
        raise NotImplementedError("Integration method not implemented.")
    z_quad = torch.tensor(z_quad, dtype=dtype)
    w_quad = torch.tensor(w_quad, dtype=dtype)
    w_quad = w_quad.log() if return_log_weight else w_quad
    return z_quad, w_quad


class FourierLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma: float | None = 1.0,
        learnable: bool | None = False,
    ):
        super().__init__()
        assert out_features % 2 == 0, "Number of output features must be even."
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = sigma

        coeff = torch.normal(0.0, sigma, (in_features, out_features // 2))
        if learnable:
            self.coeff = nn.Parameter(coeff)
        else:
            self.register_buffer("coeff", coeff)

    def forward(self, z: torch.Tensor):
        z_proj = 2 * torch.pi * z @ self.coeff
        return torch.cat([z_proj.cos(), z_proj.sin()], dim=-1).transpose(-2, -1)

    def extra_repr(self) -> str:
        return f"{self.in_features}, {self.out_features}, sigma={self.sigma}"


class PICInputNet(nn.Module):
    def __init__(
        self,
        num_variables: int,
        num_param: int,
        net_dim: int | None = 64,
        bias: bool | None = False,
        sharing: str | None = "none",
        ff_dim: int | None = None,
        ff_sigma: float | None = 1.0,
        learn_ff: bool | None = False,
        z_quad: torch.Tensor | None = None,
        tensor_parameter: TorchTensorParameter | None = None,
        reparam: TorchParameterOp | None = None,
    ):
        super().__init__()
        assert sharing in {"none", "f", "c"}
        self.num_variables = num_variables
        self.num_param = num_param
        self.sharing = sharing
        self.tensor_parameter = tensor_parameter
        self.reparam = reparam
        if z_quad is not None:
            self.register_buffer("z_quad", z_quad)

        ff_dim = net_dim if ff_dim is None else ff_dim
        inner_conv_groups = 1 if sharing in {"f", "c"} else num_variables
        last_conv_groups = 1 if sharing == "f" else num_variables
        self.net = nn.Sequential(
            FourierLayer(1, ff_dim, sigma=ff_sigma, learnable=learn_ff),
            nn.Conv1d(
                ff_dim * inner_conv_groups,
                net_dim * inner_conv_groups,
                1,
                groups=inner_conv_groups,
                bias=bias,
            ),
            nn.Tanh(),
            nn.Conv1d(
                net_dim * last_conv_groups,
                num_param * last_conv_groups,
                1,
                groups=last_conv_groups,
                bias=bias,
            ),
        )

        # initialize all heads to be equal when using composite sharing
        if sharing == "c":
            self.net[-1].weight.data = (
                self.net[-1].weight.data[:num_param].repeat(num_variables, 1, 1)
            )
            if self.net[-1].bias is not None:
                self.net[-1].bias.data = self.net[-1].bias.data[:num_param].repeat(num_variables)

        if tensor_parameter is not None and z_quad is not None:
            with torch.no_grad():
                _ = self()  # initialize tensor_parameter as result of self.forward()

    def forward(self, z_quad: torch.Tensor | None = None, n_chunks: int | None = 1):
        z_quad = self.z_quad if z_quad is None else z_quad
        assert z_quad.ndim == 1
        self.net[1].groups = 1
        self.net[-1].groups = 1 if self.sharing in {"f", "c"} else self.num_variables
        param = torch.cat(
            [self.net(chunk.unsqueeze(1)) for chunk in z_quad.chunk(n_chunks, dim=0)], dim=1
        )
        if self.sharing == "f":
            param = param.unsqueeze(0).expand(self.num_variables, -1, -1)
        param = param.view(self.num_variables, self.num_param, len(z_quad)).transpose(1, 2)
        if self.tensor_parameter is not None:
            param = param.view_as(self.tensor_parameter._ptensor)
            self.tensor_parameter._ptensor = param
        if self.reparam is not None:
            param = self.reparam(param)
        return param

    def __repr__(self):
        return "\n".join(
            [line for line in super().__repr__().split("\n") if "tensor_parameter" not in line]
        )


class PICInnerNet(nn.Module):
    def __init__(
        self,
        num_dim: int,
        num_funcs: int,
        perm_dim: tuple[int] | None = None,
        norm_dim: tuple[int] | None = None,
        net_dim: int | None = 64,
        bias: bool | None = False,
        sharing: str | None = "none",
        ff_dim: int | None = None,
        ff_sigma: float | None = 1.0,
        learn_ff: bool | None = False,
        z_quad: torch.Tensor | None = None,
        w_quad: torch.Tensor | None = None,
        tensor_parameter: TorchTensorParameter | None = None,
    ):
        super().__init__()
        assert sharing in {"none", "f", "c"}
        self.num_dim = num_dim
        self.num_funcs = num_funcs
        self.sharing = sharing
        self.perm_dim = (0,) + (tuple(range(1, num_dim + 1)) if perm_dim is None else perm_dim)
        assert self.perm_dim[0] == 0 and set(self.perm_dim) == set(range(num_dim + 1))
        self.norm_dim = (tuple(range(1, num_dim + 1))) if norm_dim is None else norm_dim
        assert 0 not in self.norm_dim and set(self.norm_dim).issubset(self.perm_dim)
        self.eps = np.sqrt(torch.finfo(torch.get_default_dtype()).tiny)
        self.tensor_parameter = tensor_parameter

        assert (z_quad is None) == (w_quad is None), "must both be given or both be None"
        if z_quad is not None:
            self.register_buffer("z_quad", z_quad)
            self.register_buffer("w_quad", w_quad)

        ff_dim = net_dim if ff_dim is None else ff_dim
        inner_conv_groups = 1 if sharing in {"c", "f"} else num_funcs
        last_conv_groups = 1 if sharing == "f" else num_funcs
        self.net = nn.Sequential(
            FourierLayer(num_dim, ff_dim, sigma=ff_sigma, learnable=learn_ff),
            nn.Conv1d(
                inner_conv_groups * ff_dim,
                inner_conv_groups * net_dim,
                1,
                groups=inner_conv_groups,
                bias=bias,
            ),
            nn.Tanh(),
            nn.Conv1d(
                inner_conv_groups * net_dim,
                inner_conv_groups * net_dim,
                1,
                groups=inner_conv_groups,
                bias=bias,
            ),
            nn.Tanh(),
            nn.Conv1d(
                last_conv_groups * net_dim, last_conv_groups, 1, groups=last_conv_groups, bias=bias
            ),
            nn.Softplus(beta=1.0),
        )

        # initialize all heads to be equal when using composite sharing
        if sharing == "c":
            self.net[-2].weight.data = self.net[-2].weight.data[:1].repeat(num_funcs, 1, 1)
            if self.net[-2].bias is not None:
                self.net[-2].bias.data = self.net[-2].bias.data[:1].repeat(num_funcs)

        if tensor_parameter is not None and z_quad is not None:
            with torch.no_grad():
                _ = self()  # initialize tensor_parameter as result of self.forward()

    def forward(
        self,
        z_quad: torch.Tensor | None = None,
        w_quad: torch.Tensor | None = None,
        n_chunks: int | None = 1,
    ):
        z_quad = self.z_quad if z_quad is None else z_quad
        w_quad = self.w_quad if w_quad is None else w_quad
        assert z_quad.ndim == w_quad.ndim == 1 and len(z_quad) == len(w_quad)
        nip = z_quad.numel()  # number of integration points
        self.net[1].groups = 1
        self.net[-2].groups = 1 if self.sharing in {"c", "f"} else self.num_funcs
        z_meshgrid = (
            torch.stack(torch.meshgrid([z_quad] * self.num_dim, indexing="ij")).flatten(1).t()
        )
        logits = (
            torch.cat([self.net(chunk) for chunk in z_meshgrid.chunk(n_chunks, dim=0)], dim=1)
            + self.eps
        )
        # the expand actually does something when self.sharing is 'f'
        logits = (
            logits.expand(self.num_funcs, -1).view(-1, *[nip] * self.num_dim).permute(self.perm_dim)
        )
        w_shape = [nip if i in self.norm_dim else 1 for i in range(self.num_dim + 1)]
        w_meshgrid = (
            torch.stack(torch.meshgrid([w_quad] * len(self.norm_dim), indexing="ij"))
            .prod(0)
            .view(w_shape)
        )
        param = (logits / (logits * w_meshgrid).sum(self.norm_dim, True)) * w_meshgrid
        if self.tensor_parameter is not None:
            param = param.view_as(self.tensor_parameter._ptensor)
            self.tensor_parameter._ptensor = param
        return param

    def __repr__(self):
        return "\n".join(
            [line for line in super().__repr__().split("\n") if "tensor_parameter" not in line]
        )


@torch.no_grad()
def pc2qpc(
    pc: TorchCircuit,
    integration_method: str,
    net_dim: int | None = 128,
    bias: bool | None = True,
    input_sharing: str | None = "f",
    inner_sharing: str | None = "c",
    ff_dim: int | None = None,
    ff_sigma: float | None = 1.0,
    learn_ff: bool | None = False,
    freeze_mixing_layers: bool = True,
):
    def param_to_buffer(model: torch.nn.Module):
        """Turns all parameters of a module into buffers."""
        modules = model.modules()
        module = next(modules)
        named_parameters = list(module.named_parameters(recurse=False))
        for name, param in named_parameters:
            delattr(module, name)  # Unregister parameter
            module.register_buffer(name, param.data)
        for module in modules:
            param_to_buffer(module)

    qpc = pc  # copy.deepcopy(pc)
    param_to_buffer(qpc)

    for node in qpc.nodes:
        if isinstance(node, TorchCategoricalLayer):
            z_quad = zw_quadrature(
                integration_method=integration_method, nip=node.num_output_units
            )[0]
            if node.logits is None:
                tensor_parameter = node.probs.nodes[0]
                reparam = node.probs.nodes[1] if len(node.probs.nodes) == 2 else None
            else:
                tensor_parameter = node.logits.nodes[0]
                reparam = node.logits.nodes[1] if len(node.logits.nodes) == 2 else None
            input_net = PICInputNet(
                num_variables=node.num_variables * node.num_folds,
                num_param=node.num_categories,
                net_dim=net_dim,
                bias=bias,
                sharing=input_sharing,
                ff_dim=ff_dim,
                ff_sigma=ff_sigma,
                learn_ff=learn_ff,
                z_quad=z_quad,
                tensor_parameter=tensor_parameter,
                reparam=reparam,
            )
            if node.logits is None:
                node.probs = input_net
            else:
                node.logits = input_net
        elif isinstance(node, TorchGaussianLayer):
            assert len(node.mean.nodes) <= 2 and len(node.stddev.nodes) <= 2
            z_quad = zw_quadrature(
                integration_method=integration_method, nip=node.num_output_units
            )[0]
            node.mean = PICInputNet(
                num_variables=node.num_variables * node.num_folds,
                num_param=1,
                net_dim=net_dim,
                bias=bias,
                sharing=input_sharing,
                ff_dim=ff_dim,
                ff_sigma=ff_sigma,
                learn_ff=learn_ff,
                z_quad=z_quad,
                tensor_parameter=node.mean.nodes[0],
                reparam=None if len(node.mean.nodes) == 1 else node.mean.nodes[0],
            )
            node.stddev = PICInputNet(
                num_variables=node.num_variables * node.num_folds,
                num_param=1,
                net_dim=net_dim,
                bias=bias,
                sharing=input_sharing,
                ff_dim=ff_dim,
                ff_sigma=ff_sigma,
                learn_ff=learn_ff,
                z_quad=z_quad,
                tensor_parameter=node.stddev.nodes[0],
                reparam=None if len(node.stddev.nodes) == 1 else node.stddev.nodes[0],
            )
        elif isinstance(node, (TorchSumLayer, TorchTuckerLayer, TorchCPTLayer)):
            if isinstance(node, TorchSumLayer) and node.arity > 1:
                weight_nodes = list(node.weight.topological_ordering())
                assert len(weight_nodes) <= 2, (
                    "Sum layers with arity greater than one must have parameters "
                    "with at most 2 computational nodes"
                )
                tensor_parameter = node.weight.nodes[0]
                if freeze_mixing_layers:
                    tensor_parameter._ptensor.fill_(1 / node.arity)
                    tensor_parameter._ptensor.requires_grad = False
                    continue
            else:
                assert len(node.weight.nodes) == 1, (
                    "You are probably using a reparameterization. "
                    "Do not do that, QPCs are already normalized!"
                )
                tensor_parameter = node.weight.nodes[0]
            weight_shape = list(tensor_parameter._ptensor.shape)
            squeezed_weight_shape = [weight_shape[0]] + [
                dim_size for dim_size in weight_shape[1:] if dim_size != 1
            ]
            assert (
                sum(
                    (
                        dim_size % min(squeezed_weight_shape[1:])
                        for dim_size in squeezed_weight_shape[1:]
                    )
                )
                == 0
            ), f"Cannot model a sum layer with shape {weight_shape}!"
            is_tucker = isinstance(node, TorchTuckerLayer)
            nip = int(max(squeezed_weight_shape[1:]) ** (0.5 if is_tucker else 1))
            num_dim = sum(
                (int(np.emath.logn(nip, dim_size)) for dim_size in squeezed_weight_shape[1:])
            )
            z_quad, w_quad = zw_quadrature(integration_method=integration_method, nip=nip)
            node.weight = PICInnerNet(
                num_dim=num_dim,
                num_funcs=node.num_folds,
                perm_dim=tuple(range(1, num_dim + 1)),
                norm_dim=tuple(range(1, num_dim + 1))[-(2 if is_tucker else 1) :],
                net_dim=net_dim,
                bias=bias,
                sharing=inner_sharing,
                ff_dim=ff_dim,
                ff_sigma=ff_sigma,
                learn_ff=learn_ff,
                z_quad=z_quad,
                w_quad=w_quad,
                tensor_parameter=tensor_parameter,
            )
        elif isinstance(node, (TorchHadamardLayer, TorchKroneckerLayer)):
            pass
        else:
            raise NotImplementedError(f"Layer %s is not yet handled! {type(node)}")
