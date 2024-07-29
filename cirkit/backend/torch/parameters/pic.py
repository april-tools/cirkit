from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.backend.torch.layers import *
from cirkit.backend.torch.parameters.nodes import TorchTensorParameter


def zw_quadrature(
    integration_method: str,
    nip: int,
    a: Optional[float] = -1,
    b: Optional[float] = 1,
    return_log_weight: Optional[bool] = False,
    dtype: Optional[torch.dtype] = torch.float32,
    device: Optional[torch.device] = "cpu",
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
    z_quad = torch.tensor(z_quad, dtype=dtype).to(device)
    w_quad = torch.tensor(w_quad, dtype=dtype).to(device)
    w_quad = w_quad.log() if return_log_weight else w_quad
    return z_quad, w_quad


class FourierLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma: Optional[float] = 1.0,
        learnable: Optional[bool] = False,
    ):
        super(FourierLayer, self).__init__()
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
        return "{}, {}, sigma={}".format(self.in_features, self.out_features, self.sigma)


class PICInputNet(nn.Module):
    def __init__(
        self,
        num_vars: int,
        num_param: int,
        num_channels: Optional[bool] = 1,
        net_dim: Optional[int] = 64,
        bias: Optional[bool] = False,
        sharing: Optional[str] = "none",
        ff_dim: Optional[int] = None,
        ff_sigma: Optional[float] = 1.0,
        learn_ff: Optional[bool] = False,
        z_quad: Optional[torch.Tensor] = None,
        tensor_parameter: Optional[TorchTensorParameter] = None,
    ):
        super().__init__()
        assert sharing in ["none", "f", "c"]
        self.num_vars = num_vars
        self.num_param = num_param
        self.num_channels = num_channels
        self.sharing = sharing
        self.tensor_parameter = tensor_parameter
        if z_quad is not None:
            self.register_buffer("z_quad", z_quad)

        ff_dim = net_dim if ff_dim is None else ff_dim
        inner_conv_groups = num_channels * (1 if sharing in ["f", "c"] else num_vars)
        last_conv_groups = num_channels * (1 if sharing == "f" else num_vars)
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
                self.net[-1].weight.data[: num_param * num_channels].repeat(num_vars, 1, 1)
            )
            if self.net[-1].bias is not None:
                self.net[-1].bias.data = (
                    self.net[-1].bias.data[: num_param * num_channels].repeat(num_vars)
                )

        if tensor_parameter is not None and z_quad is not None:
            with torch.no_grad():
                _ = self()  # initialize tensor_parameter as result of self.forward()

    def forward(self, z_quad: Optional[torch.Tensor] = None, n_chunks: Optional[int] = 1):
        z_quad = self.z_quad if z_quad is None else z_quad
        assert z_quad.ndim == 1
        self.net[1].groups = 1
        self.net[-1].groups = self.num_channels * (
            1 if self.sharing in ["f", "c"] else self.num_vars
        )
        param = torch.cat(
            [self.net(chunk.unsqueeze(1)) for chunk in z_quad.chunk(n_chunks, dim=0)], dim=1
        )
        if self.sharing == "f":
            param = param.unsqueeze(0).expand(self.num_vars, -1, -1)
        param = param.view(
            self.num_vars, self.num_param * self.num_channels, len(z_quad)
        ).transpose(1, 2)
        if self.tensor_parameter is not None:
            param = param.view_as(self.tensor_parameter._ptensor)
            self.tensor_parameter._ptensor = param
        return param


class PICInnerNet(nn.Module):
    def __init__(
        self,
        num_dim: int,
        num_funcs: int,
        perm_dim: Optional[Tuple[int]] = None,
        norm_dim: Optional[Tuple[int]] = None,
        net_dim: Optional[int] = 64,
        bias: Optional[bool] = False,
        sharing: Optional[str] = "none",
        ff_dim: Optional[int] = None,
        ff_sigma: Optional[float] = 1.0,
        learn_ff: Optional[bool] = False,
        z_quad: Optional[torch.Tensor] = None,
        w_quad: Optional[torch.Tensor] = None,
        tensor_parameter: Optional[TorchTensorParameter] = None,
    ):
        super().__init__()
        assert sharing in ["none", "f", "c"]
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
        inner_conv_groups = 1 if sharing in ["c", "f"] else num_funcs
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
        z_quad: Optional[torch.Tensor] = None,
        w_quad: Optional[torch.Tensor] = None,
        n_chunks: Optional[int] = 1,
    ):
        z_quad = self.z_quad if z_quad is None else z_quad
        w_quad = self.w_quad if w_quad is None else w_quad
        assert z_quad.ndim == w_quad.ndim == 1 and len(z_quad) == len(w_quad)
        nip = z_quad.numel()  # number of integration points
        self.net[1].groups = 1
        self.net[-2].groups = 1 if self.sharing in ["c", "f"] else self.num_funcs
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


@torch.no_grad()
def pc2qpc(
    pc: TorchCircuit,
    integration_method: str,
    net_dim: Optional[int] = 128,
    bias: Optional[bool] = True,
    input_sharing: Optional[str] = "f",
    inner_sharing: Optional[str] = "c",
    ff_dim: Optional[int] = None,
    ff_sigma: Optional[float] = 1.0,
    learn_ff: Optional[bool] = False,
):
    def param_to_buffer(model: torch.nn.Module):
        """Turns all parameters of a module into buffers."""
        modules = model.modules()
        module = next(modules)
        for name, param in module.named_parameters(recurse=False):
            delattr(module, name)  # Unregister parameter
            module.register_buffer(name, param.data)
        for module in modules:
            param_to_buffer(module)

    qpc = pc  # copy.deepcopy(pc)
    param_to_buffer(qpc)

    for node in qpc._nodes:
        if isinstance(node, TorchCategoricalLayer):
            probs_shape = list(node.probs._nodes[0]._ptensor.shape)
            z_quad = zw_quadrature(integration_method=integration_method, nip=probs_shape[2])[0]
            node.probs._nodes[0] = PICInputNet(
                num_vars=probs_shape[0],
                num_param=probs_shape[-1],
                num_channels=probs_shape[-2],
                net_dim=net_dim,
                bias=bias,
                sharing=input_sharing,
                ff_dim=ff_dim,
                ff_sigma=ff_sigma,
                learn_ff=learn_ff,
                z_quad=z_quad,
                tensor_parameter=node.probs._nodes[0],
            )
        elif isinstance(node, TorchGaussianLayer):
            gauss_shape = list(
                node.mean._nodes[0]._ptensor.shape
            )  # or node.stddev._nodes[0]._ptensor.shape
            if len(gauss_shape) == 4:
                assert gauss_shape[1] == 1, f"Invalid Gaussian layer shape {gauss_shape}"
            z_quad = zw_quadrature(integration_method=integration_method, nip=gauss_shape[-2])[0]
            node.mean._nodes[0] = PICInputNet(
                num_vars=gauss_shape[0],
                num_param=1,
                num_channels=gauss_shape[-1],
                net_dim=net_dim,
                bias=bias,
                sharing=input_sharing,
                ff_dim=ff_dim,
                ff_sigma=ff_sigma,
                learn_ff=learn_ff,
                z_quad=z_quad,
                tensor_parameter=node.mean._nodes[0],
            )
            node.stddev._nodes[0] = PICInputNet(
                num_vars=gauss_shape[0],
                num_param=1,
                num_channels=gauss_shape[-1],
                net_dim=net_dim,
                bias=bias,
                sharing=input_sharing,
                ff_dim=ff_dim,
                ff_sigma=ff_sigma,
                learn_ff=learn_ff,
                z_quad=z_quad,
                tensor_parameter=node.stddev._nodes[0],
            )
        elif isinstance(node, (TorchDenseLayer, TorchTuckerLayer, TorchCPLayer)):
            assert (
                len(node.weight._nodes) == 1
            ), "You are probably using a reparameterization. Do not do that, QPCs are already normalized!"
            weight_shape = list(node.weight._nodes[0]._ptensor.shape)
            squeezed_weight_shape = [weight_shape[0]] + [
                dim_size for dim_size in weight_shape[1:] if dim_size != 1
            ]
            assert (
                sum(
                    [
                        dim_size % min(squeezed_weight_shape[1:])
                        for dim_size in squeezed_weight_shape[1:]
                    ]
                )
                == 0
            ), f"Cannot model a dense layer with shape {weight_shape}!"
            is_tucker = isinstance(node, TorchTuckerLayer)
            nip = int(max(squeezed_weight_shape[1:]) ** (0.5 if is_tucker else 1))
            num_dim = sum(
                [int(np.emath.logn(nip, dim_size)) for dim_size in squeezed_weight_shape[1:]]
            )
            z_quad, w_quad = zw_quadrature(integration_method=integration_method, nip=nip)
            node.weight._nodes = nn.ModuleList(
                [
                    PICInnerNet(
                        num_dim=num_dim,
                        num_funcs=weight_shape[0],
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
                        tensor_parameter=node.weight._nodes[0],
                    )
                ]
            )
        elif isinstance(node, TorchMixingLayer):
            assert (
                len(node.weight._nodes) == 1
            ), "You are probably using a reparameterization. Do not do that, QPCs are already normalized!"
            node.weight._nodes = node.weight._nodes[:1]  # ignore possible reparameterizations
            node.weight._nodes[0]._ptensor.fill_(1 / node.weight._nodes[0]._ptensor.size(-1))
        elif isinstance(node, (TorchHadamardLayer, TorchKroneckerLayer)):
            pass
        else:
            raise NotImplementedError("Layer %s is not yet handled!" % str(type(node)))
