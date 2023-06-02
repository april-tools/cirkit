import functools
import warnings
from itertools import count
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from cirkit.einet._Layer import Layer
from tensorly import set_backend
from torch.nn.functional import softmax
from cirkit.einet._utils import sample_matrix_categorical

set_backend('pytorch')


class CPSharedEinsumLayer(GenericEinsumLayer): # TODO edit this because of numerical stability
    def __init__(self, graph, products, layers, k: int, prod_exp=False, r=1):
        self.r = r
        super(CPSharedEinsumLayer, self).__init__(graph, products, layers, prod_exp, k=k)

    def build_params(self) -> (Dict[str, Union[None, torch.nn.Parameter]], Dict[str, Tuple[int, ...]]):
        """
        Create params dict for the layer (the parameters are uninitialized)
        :return: dict {params_name, params}
        """
        params_dict = {"cp_a": None, "cp_b": None, "cp_c": None, "cp_d": None}
        shapes_dict = {"cp_a": (self.num_input_dist, self.r),
                       "cp_b": (self.num_input_dist, self.r),
                       "cp_c": (self.num_sums, self.r),
                       "cp_d": (len(self.products), self.r)}

        return params_dict, shapes_dict

    @property
    def clamp_value(self) -> float:
        par_tensor = list(self.params_dict.items())[0][1]
        smallest_normal = torch.finfo(par_tensor.dtype).smallest_normal

        if self.prod_exp:
            return torch.pow(torch.Tensor([smallest_normal]), 1/4).item()
        else:
            return torch.sqrt(torch.Tensor([smallest_normal])).item()

    def central_einsum(self, left_prob: torch.Tensor, right_prob: torch.Tensor) -> torch.Tensor:
        left_max = torch.max(self.left_child_log_prob, 1, keepdim=True)[0]
        left_prob = torch.exp(self.left_child_log_prob - left_max)
        right_max = torch.max(self.right_child_log_prob, 1, keepdim=True)[0]
        right_prob = torch.exp(self.right_child_log_prob - right_max)

        pa = self.params_dict["cp_a"]
        pb = self.params_dict["cp_b"]
        pc = self.params_dict["cp_c"]
        pd = self.params_dict["cp_d"]

        left_hidden = torch.einsum('bip,ir->brp', left_prob, pa)
        right_hidden = torch.einsum('bjp,jr->brp', right_prob, pb)

        if self.prod_exp:
            hidden = left_hidden * right_hidden
            rescaled_hidden = torch.einsum('brp,pr->brp', hidden, pd)
            prob = torch.einsum('brp,or->bop', rescaled_hidden, pc)
            log_prob = torch.log(prob) + left_max + right_max
        else:
            log_left_hidden = torch.log(left_hidden) + left_max
            log_right_hidden = torch.log(right_hidden) + right_max
            log_hidden = log_left_hidden + log_right_hidden

            rescaled_log_hidden = log_hidden + torch.t(pd)
            hidden_max = torch.max(rescaled_log_hidden, 1, keepdim=True)[0]
            rescaled_hidden = torch.exp(rescaled_log_hidden - hidden_max)
            prob = torch.einsum('brp,or->bop', rescaled_hidden, pc)
            log_prob = torch.log(prob) + hidden_max

        return log_prob


class HCPTEinsumLayer(GenericEinsumLayer):
    def __init__(self, graph, products, layers, k: int, prod_exp=False):
        super(HCPTEinsumLayer, self).__init__(graph, products, layers, prod_exp, k=k)

    def build_params(self) -> (Dict[str, Union[None, torch.nn.Parameter]], Dict[str, Tuple[int, ...]]):
        """
        Create params dict for the layer (the parameters are uninitialized)
        :return: dict {params_name, params}
        """
        params_dict = {"cp_a": None, "cp_b": None}
        shapes_dict = {"cp_a": (self.num_input_dist, self.num_sums, len(self.products)),
                       "cp_b": (self.num_input_dist, self.num_sums, len(self.products))}

        return params_dict, shapes_dict

    @property
    def clamp_value(self) -> float:
        par_tensor = list(self.params_dict.items())[0][1]
        smallest_normal = torch.finfo(par_tensor.dtype).smallest_normal

        if self.prod_exp:
            return torch.sqrt(torch.Tensor([smallest_normal])).item()
        else:
            return torch.sqrt(torch.Tensor([smallest_normal])).item()# smallest_normal

    def central_einsum(self, left_prob: torch.Tensor, right_prob: torch.Tensor) -> torch.Tensor:
        left_max = torch.max(self.left_child_log_prob, 1, keepdim=True)[0]
        left_prob = torch.exp(self.left_child_log_prob - left_max)
        right_max = torch.max(self.right_child_log_prob, 1, keepdim=True)[0]
        right_prob = torch.exp(self.right_child_log_prob - right_max)

        pa = self.params_dict["cp_a"]
        pb = self.params_dict["cp_b"]

        left_hidden = torch.einsum('bip,irp->brp', left_prob, pa)
        right_hidden = torch.einsum('bjp,jrp->brp', right_prob, pb)

        if self.prod_exp:
            hidden = left_hidden * right_hidden
            log_prob = torch.log(hidden) + left_max + right_max
        else:
            log_left_hidden = torch.log(left_hidden) + left_max
            log_right_hidden = torch.log(right_hidden) + right_max
            log_prob = log_left_hidden + log_right_hidden

        return log_prob


class HCPTSharedEinsumLayer(GenericEinsumLayer):
    def __init__(self, graph, products, layers, k: int, prod_exp=False):
        super(HCPTSharedEinsumLayer, self).__init__(graph, products, layers, prod_exp, k=k)

    def build_params(self) -> (Dict[str, Union[None, torch.nn.Parameter]], Dict[str, Tuple[int, ...]]):
        """
        Create params dict for the layer (the parameters are uninitialized)
        :return: dict {params_name, params}
        """
        params_dict = {"cp_a": None, "cp_b": None, "cp_d": None}
        shapes_dict = {"cp_a": (self.num_input_dist, self.num_sums),
                       "cp_b": (self.num_input_dist, self.num_sums),
                       "cp_d": (len(self.products), self.num_sums)}

        return params_dict, shapes_dict

    @property
    def clamp_value(self) -> float:
        par_tensor = list(self.params_dict.items())[0][1]
        smallest_normal = torch.finfo(par_tensor.dtype).smallest_normal

        if self.prod_exp:
            return torch.sqrt(torch.Tensor([smallest_normal])).item()
        else:
            return torch.sqrt(torch.Tensor([smallest_normal])).item()

    def central_einsum(self, left_prob: torch.Tensor, right_prob: torch.Tensor) -> torch.Tensor:
        left_max = torch.max(self.left_child_log_prob, 1, keepdim=True)[0]
        left_prob = torch.exp(self.left_child_log_prob - left_max)
        right_max = torch.max(self.right_child_log_prob, 1, keepdim=True)[0]
        right_prob = torch.exp(self.right_child_log_prob - right_max)

        pa = self.params_dict["cp_a"]
        pb = self.params_dict["cp_b"]
        pd = self.params_dict["cp_d"]

        left_hidden = torch.einsum('bip,ir->brp', left_prob, pa)
        right_hidden = torch.einsum('bjp,jr->brp', right_prob, pb)

        if self.prod_exp:
            hidden = left_hidden * right_hidden
            log_prob = torch.log(hidden) + left_max + right_max
        else:
            log_left_hidden = torch.log(left_hidden) + left_max
            log_right_hidden = torch.log(right_hidden) + right_max
            log_prob = log_left_hidden + log_right_hidden

        log_prob = log_prob + torch.t(pd)

        return log_prob


class HCPTLoLoEinsumLayer(GenericEinsumLayer):
    def __init__(self, graph, products, layers, k: int, prod_exp=False, r=1):
        self.r = r
        super(HCPTLoLoEinsumLayer, self).__init__(graph, products, layers, prod_exp, k=k)

    def build_params(self) -> (Dict[str, Union[None, torch.nn.Parameter]], Dict[str, Tuple[int, ...]]):
        """
        Create params dict for the layer (the parameters are uninitialized)
        :return: dict {params_name, params}
        """
        params_dict = {"cp_a1": None, "cp_a2": None, "cp_b1": None, "cp_b2": None}
        shapes_dict = {"cp_a1": (self.num_input_dist, self.r, len(self.products)),
                       "cp_a2": (self.r, self.num_sums, len(self.products)),
                       "cp_b1": (self.num_input_dist, self.r, len(self.products)),
                       "cp_b2": (self.r, self.num_sums, len(self.products))}

        return params_dict, shapes_dict

    @property
    def clamp_value(self) -> float:
        par_tensor = list(self.params_dict.items())[0][1]
        smallest_normal = torch.finfo(par_tensor.dtype).smallest_normal

        if self.prod_exp:
            return torch.pow(torch.Tensor([smallest_normal]), 1/4).item()
        else:
            return torch.sqrt(torch.Tensor([smallest_normal])).item()

    def central_einsum(self, left_prob: torch.Tensor, right_prob: torch.Tensor) -> torch.Tensor:
        left_max = torch.max(self.left_child_log_prob, 1, keepdim=True)[0]
        left_prob = torch.exp(self.left_child_log_prob - left_max)
        right_max = torch.max(self.right_child_log_prob, 1, keepdim=True)[0]
        right_prob = torch.exp(self.right_child_log_prob - right_max)

        pa_1 = self.params_dict["cp_a1"]
        pa_2 = self.params_dict["cp_a2"]
        pb_1 = self.params_dict["cp_b1"]
        pb_2 = self.params_dict["cp_b2"]

        left_hidden = torch.einsum('bip,irp->brp', left_prob, pa_1)
        left_hidden = torch.einsum('brp,rip->bip', left_hidden, pa_2)
        right_hidden = torch.einsum('bjp,jrp->brp', right_prob, pb_1)
        right_hidden = torch.einsum('brp,rjp->bjp', right_hidden, pb_2)

        if self.prod_exp:
            hidden = left_hidden * right_hidden
            log_prob = torch.log(hidden) + left_max + right_max
        else:
            # LogEinsumExp trick, re-add the max
            log_left_hidden = torch.log(left_hidden) + left_max
            log_right_hidden = torch.log(right_hidden) + right_max
            log_prob = log_left_hidden + log_right_hidden

        return log_prob


class HCPTLoLoSharedEinsumLayer(GenericEinsumLayer): # TODO: same thing for numerical stability
    def __init__(self, graph, products, layers, k: int, prod_exp=False, r=1):
        self.r = r
        super(HCPTLoLoSharedEinsumLayer, self).__init__(graph, products, layers, prod_exp=prod_exp, k=k)

    def build_params(self) -> (Dict[str, Union[None, torch.nn.Parameter]], Dict[str, Tuple[int, ...]]):
        """
        Create params dict for the layer (the parameters are uninitialized)
        :return: dict {params_name, params}
        """
        params_dict = {"cp_a1": None, "cp_a2": None, "cp_b1": None, "cp_b2": None, "cp_d1": None, "cp_d2": None}
        shapes_dict = {"cp_a1": (self.num_input_dist, self.r),
                       "cp_a2": (self.r, self.num_sums),
                       "cp_b1": (self.num_input_dist, self.r),
                       "cp_b2": (self.r, self.num_sums),
                       "cp_d1": (len(self.products), self.r),
                       "cp_d2": (len(self.products), self.r)}

        return params_dict, shapes_dict

    @property
    def clamp_value(self) -> float:
        par_tensor = list(self.params_dict.items())[0][1]
        smallest_normal = torch.finfo(par_tensor.dtype).smallest_normal

        if self.prod_exp:
            return torch.pow(torch.Tensor([smallest_normal]), 1/4).item()
        else:
            return torch.pow(torch.Tensor([smallest_normal]), 1/3).item()

    def central_einsum(self, left_prob: torch.Tensor, right_prob: torch.Tensor) -> torch.Tensor:
        left_max = torch.max(self.left_child_log_prob, 1, keepdim=True)[0]
        left_prob = torch.exp(self.left_child_log_prob - left_max)
        right_max = torch.max(self.right_child_log_prob, 1, keepdim=True)[0]
        right_prob = torch.exp(self.right_child_log_prob - right_max)

        pa_1 = self.params_dict["cp_a1"]
        pa_2 = self.params_dict["cp_a2"]
        pb_1 = self.params_dict["cp_b1"]
        pb_2 = self.params_dict["cp_b2"]
        pd_1 = self.params_dict["cp_d1"]
        pd_2 = self.params_dict["cp_d2"]

        left_hidden = torch.einsum('bip,ir->brp', left_prob, pa_1)
        left_hidden = torch.einsum('brp,pr->brp', left_hidden, pd_1)
        left_hidden = torch.einsum('brp,ri->bip', left_hidden, pa_2)

        right_hidden = torch.einsum('bjp,jr->brp', right_prob, pb_1)
        right_hidden = torch.einsum('brp,pr->brp', right_hidden, pd_2)
        right_hidden = torch.einsum('brp,rj->bjp', right_hidden, pb_2)

        if self.prod_exp:
            hidden = left_hidden * right_hidden
            log_prob = torch.log(hidden) + left_max + right_max
        else:
            clamp_left_hidden = torch.log(left_hidden)
            clamp_right_hidden = torch.log(right_hidden)
            assert not torch.isinf(clamp_left_hidden).any()
            assert not torch.isinf(clamp_right_hidden).any()

            # LogEinsumExp trick, re-add the max
            log_left_hidden = clamp_left_hidden + left_max
            log_right_hidden = clamp_right_hidden + right_max
            log_prob = log_left_hidden + log_right_hidden

        return log_prob

#class HCPTSharedLoLoEinsumLayer(GenericEinsumLayer):
#    def __init__(self, graph, products, layers, decomposition_strategy="slice", prod_exp=False, r=1):
#        super(HCPTSharedLoLoEinsumLayer, self).__init__(graph, products, layers, decomposition_strategy, prod_exp, r)
#        self.prod_exp = prod_exp
#        self.r = r


class RescalEinsumLayer(GenericEinsumLayer):
    def __init__(self, graph, products, layers, k: int):
        super(RescalEinsumLayer, self).__init__(graph, products, layers, prod_exp=False, k=k)

    def build_params(self) -> (Dict[str, Union[None, torch.nn.Parameter]], Dict[str, Tuple[int, ...]]):
        """
        Create params dict for the layer (the parameters are uninitialized)
        :return: dict {params_name, params}
        """
        params_dict = {"params": None}
        shapes_dict = {"params": (self.num_input_dist, self.num_input_dist, self.num_sums, len(self.products))}

        return params_dict, shapes_dict

    @property
    def clamp_value(self) -> float:
        par_tensor = list(self.params_dict.items())[0][1]
        smallest_normal = torch.finfo(par_tensor.dtype).smallest_normal

        return torch.sqrt(torch.Tensor([smallest_normal])).item()

    def central_einsum(self, left_prob: torch.Tensor, right_prob: torch.Tensor) -> torch.Tensor:
        left_max = torch.max(self.left_child_log_prob, 1, keepdim=True)[0]
        left_prob = torch.exp(self.left_child_log_prob - left_max)
        right_max = torch.max(self.right_child_log_prob, 1, keepdim=True)[0]
        right_prob = torch.exp(self.right_child_log_prob - right_max)

        params = self.params_dict["params"]

        output = torch.einsum('bip,bjp,ijop->bop', left_prob, right_prob, params)
        log_prob = torch.log(output) + left_max + right_max

        return log_prob
