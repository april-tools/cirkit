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
[`integrate`](../../api/cirkit/symbolic/functional/index.html#cirkit.symbolic.functional.integrate),
[`multiply`](../../api/cirkit/symbolic/functional/index.html#cirkit.symbolic.functional.multiply) and
[`evidence`](../../api/cirkit/symbolic/functional/index.html#cirkit.symbolic.functional.evidence)
transform symbolic circuits, and the pipeline context compiles the transformed circuits
such that they share the (possibly learned) parameters with the circuits they were
derived from.

## The `torch` backend (default)

The default backend compiles symbolic circuits to
[`TorchCircuit`](../../api/cirkit/backend/torch/circuits/index.html#cirkit.backend.torch.circuits.TorchCircuit)
modules, i.e., computational graphs of PyTorch layers. It supports the following flags:

| Flag       | Default         | Description                                                                |
| ---------- | --------------- | -------------------------------------------------------------------------- |
| `semiring` | `"sum-product"` | The semiring the circuit is evaluated in, e.g., `"lse-sum"` for log-space. |
| `fold`     | `False`         | Vectorize groups of layers that can be evaluated in parallel.              |
| `optimize` | `False`         | Fuse or shatter layers into more efficient ones.                           |

See the [compilation handbook](../cirkit-torch-compiler/torch-compiler/index.html) for an
in-depth explanation of how this backend works.

## The `torch-compile` backend

The `torch-compile` backend uses the same compilation strategy as the `torch` backend
(and accepts the same flags), and in addition just-in-time compiles the resulting
circuit with [`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html):

```python
ctx = PipelineContext(backend="torch-compile", semiring="lse-sum", fold=True, optimize=True)
circuit = ctx.compile(symbolic_circuit)
```

The tracing and code generation happen lazily on the **first** evaluation of the
circuit, which is therefore expected to be much slower than the following ones.
To get the best performance we recommended to compile the whole training step including optimizer updates.

## The `xe-torch` backend

The `xe-torch` backend lowers circuits into
[extended-einsum](https://pypi.org/project/extended-einsum/) programs that are executed
with PyTorch. Extended-einsum is an einsum-centered tensor program compiler that folds same-shaped operations, optimizes the contraction paths of einsum
operations, and automatically rewrites programs for numerical stability. The backend
requires the optional `extended-einsum` dependency (and Python 3.12 or newer):

```shell
pip install 'libcirkit[xe]'
```

```python
ctx = PipelineContext(backend="xe-torch", semiring="lse-sum", fold=True, optimize=True)
circuit = ctx.compile(symbolic_circuit)
```

The compiled circuit is an `XETorchCircuit`, i.e., a PyTorch module that can be trained
and moved across devices as usual. It supports the following flags:

| Flag             | Default          | Description                                                                                                                                                            |
| ---------------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `semiring`       | `"lse-sum"`      | The output space of the circuit: `"lse-sum"` outputs log-values, `"sum-product"` linear-space values.                                                                   |
| `fold`           | `False`          | Fold (vectorize) same-shaped operations in the lowered programs.                                                                                                        |
| `optimize`       | `False`          | Optimize the contraction paths of einsum operations in the lowered programs.                                                                                            |
| `stability`      | `None`           | The numerical stability mode programs are evaluated with, one of `"unstable"`, `"logspace_min"`, `"logspace_max"`, `"scaled_min"`, `"scaled_max"` or `"scaled_sum"`. If None, it defaults to `"logspace_max"` for the lse-sum semiring and `"unstable"` for the sum-product semiring. |
| `fold_depth`     | `"input"`        | Whether folded operations are grouped by their depth with respect to the program inputs (`"input"`) or the program output (`"output"`).                                 |
| `scale_interval` | `3`              | How often intermediate results are re-normalized when a scaled stability mode is used.                                                                                  |
| `jit`            | `False`          | Just-in-time compile the lowered programs with `torch.compile`.                                                                                                         |

Differently from the torch backend, the semiring only determines the __output space__ of
the circuit, while __how__ the computation is evaluated (e.g., in log-space) is chosen
by the stability mode.

### How it works

The compiler (`XETorchCompiler`) embeds a
[`TorchCompiler`](../../api/cirkit/backend/torch/compiler/index.html#cirkit.backend.torch.compiler.TorchCompiler),
which is used to compile the input layers, the symbolic parameter computational graphs,
and the initializers. As a consequence, (i) every input layer, parameter node and
initializer supported by the torch backend is supported (including custom ones with a
torch compilation rule), and (ii) parameters that are shared across symbolic circuits —
e.g., as the result of symbolic operators such as
[`integrate`](../../api/cirkit/symbolic/functional/index.html#cirkit.symbolic.functional.integrate)
and
[`multiply`](../../api/cirkit/symbolic/functional/index.html#cirkit.symbolic.functional.multiply)
— are compiled into shared torch parameters, exactly as in the torch backend. In
particular, circuits derived from a trained circuit re-use its learned parameters.

The computational graph over the inner (sum and product) layers is instead lowered into
an extended-einsum program: input layers enter the program as linear-space data inputs,
and sum layer weights as parameter inputs, which the stability translations keep in
linear space (matching the log-einsum-exp trick of the torch backend in the lse-sum
semiring). Since extended-einsum programs are specialized on tensor shapes, a program is
lowered lazily for each (batch size, device) pair the circuit is evaluated with, and
then cached.

### Current limitations

- Backend-specific queries (e.g., `IntegrateQuery` and `SamplingQuery` of the torch
  backend) are not available, use the symbolic circuit operators of the pipeline
  context instead.
- In the sum-product semiring with negative sum weights, only the default `"unstable"`
  stability mode is applicable, since log-space evaluations of negative values are
  undefined.
- Because the lowered programs are shape-specialized, evaluating a circuit with many
  different batch sizes triggers one program lowering for each of batch size.
