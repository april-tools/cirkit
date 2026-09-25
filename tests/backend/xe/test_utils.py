import pytest

pytest.importorskip("extended_einsum")

from cirkit.backend.torch.compiler import TorchCompiler
from cirkit.backend.xe.compiler import XETorchCompiler
from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.parameters import TensorParameter


def symbolic_tensor_parameters(sc: Circuit) -> list[TensorParameter]:
    parameters: list[TensorParameter] = []
    seen: set[int] = set()
    for sl in sc.layers:
        for pgraph in sl.params.values():
            for node in pgraph.nodes:
                if isinstance(node, TensorParameter) and id(node) not in seen:
                    seen.add(id(node))
                    parameters.append(node)
    return parameters


def copy_parameters(
    sc: Circuit, src_compiler: XETorchCompiler, dst_compiler: TorchCompiler
) -> None:
    """Copy the compiled parameter values of a circuit from an extended-einsum
    compiler to a torch compiler, exploiting the mapping both compilers maintain
    between symbolic tensor parameters and allocated torch parameters."""
    for sp in symbolic_tensor_parameters(sc):
        src_parameter, src_fold_idx = src_compiler.torch_compiler.state.retrieve_compiled_parameter(
            sp
        )
        dst_parameter, dst_fold_idx = dst_compiler.state.retrieve_compiled_parameter(sp)
        dst_parameter().data[dst_fold_idx] = src_parameter().data[src_fold_idx]
