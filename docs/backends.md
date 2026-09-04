# Backends

A symbolic circuit does not perform any computation by itself: it is compiled into an
executable computational graph by a **backend**. You can specify the used backend when constructing a
[`PipelineContext`](../api/cirkit/pipeline/index.html#cirkit.pipeline.PipelineContext),
together with backend-specific compilation flags:

```python
from cirkit.pipeline import PipelineContext

ctx = PipelineContext(backend="torch", semiring="lse-sum", fold=True, optimize=True)
circuit = ctx.compile(symbolic_circuit)
```

Regardless of the chosen backend, the symbolic circuit remains the source of truth:
symbolic operators such as
[`integrate`](../api/cirkit/symbolic/functional/index.html#cirkit.symbolic.functional.integrate),
[`multiply`](../api/cirkit/symbolic/functional/index.html#cirkit.symbolic.functional.multiply) and
[`evidence`](../api/cirkit/symbolic/functional/index.html#cirkit.symbolic.functional.evidence)
transform symbolic circuits, and the pipeline context compiles the transformed circuits
such that they share the (possibly learned) parameters with the circuits they were
derived from.

## The `torch` backend (default)

The default backend compiles symbolic circuits to
[`TorchCircuit`](../api/cirkit/backend/torch/circuits/index.html#cirkit.backend.torch.circuits.TorchCircuit)
modules, i.e., computational graphs of PyTorch layers. It supports the following flags:

| Flag       | Default         | Description                                                                 |
| ---------- | --------------- | --------------------------------------------------------------------------- |
| `semiring` | `"sum-product"` | The semiring the circuit is evaluated in, e.g., `"lse-sum"` for log-space.  |
| `fold`     | `False`         | Vectorize groups of layers that can be evaluated in parallel.               |
| `optimize` | `False`         | Fuse or shatter layers into more efficient ones.                            |

See the [compilation handbook](handbook-cirkit-torch-compiler/handbook.md) for an
in-depth explanation of how this backend works.

## The `torch-compile` backend

The `torch-compile` backend uses the same compilation strategy as the `torch` backend
(and accepts the same flags), and in addition just-in-time compiles the resulting
circuit with [`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html):

```python
ctx = PipelineContext(backend="torch-compile", semiring="lse-sum", fold=True, optimize=True)
circuit = ctx.compile(symbolic_circuit)
```
The tracing and code generation happen lazily on the __first__ evaluation of the
circuit, which is therefore expected to be much slower than the following ones.
To get the best performance we recommended to compile the whole training step including optimizer updates.