import itertools

import torch

from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.pipeline import PipelineContext
from tests.symbolic.test_utils import build_multivariate_monotonic_structured_cpt_pc


def test_torch_compile_backend():
    torch.manual_seed(42)
    sc = build_multivariate_monotonic_structured_cpt_pc(num_units=2)
    ctx = PipelineContext(backend="torch-compile", semiring="lse-sum", fold=True, optimize=True)
    cc = ctx.compile(sc)
    # The jit compilation is performed in-place, i.e., the compiled circuit is still
    # a torch circuit and the mapping between symbolic and compiled circuits holds
    assert isinstance(cc, TorchCircuit)
    # pylint: disable-next=protected-access
    assert cc._compiled_call_impl is not None  # set by nn.Module.compile()
    assert ctx.get_symbolic_circuit(cc) is sc
    worlds = torch.tensor(list(itertools.product([0, 1], repeat=5)))
    scores = cc(worlds)
    assert scores.shape == (32, 1, 1)
    assert torch.all(torch.isfinite(scores))
    # The circuit must remain normalized, and symbolic operators must still apply
    int_cc = ctx.integrate(cc)
    assert torch.allclose(
        int_cc().flatten(), torch.logsumexp(scores.flatten(), dim=0), rtol=1e-5, atol=1e-6
    )
