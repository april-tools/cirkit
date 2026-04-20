# Handbook for Cirkit's Torch Compiler

## Basic Terminology

- _Symbolic Layers / Circuit_ are the abstract layers used in the normal circuit,
  they do not specify the concrete operations.
- _Compiled Layers / Circuit_ are the layers that specify the exact
  computation in the chosen computation framework (here PyTorch)
- The _Topological Order_ of nodes in a circuit is an ordering of the nodes
  from input to output.
- The _Layer Topological Order_ is an ordering that groups layers depending
  on their depth with respect to its inputs. A layer is included in the **current list**
  if all its inputs have been included in **previous lists**.

  ![type:video](./media/videos/TopoLayerOrder.mp4)

## Representing a Circuit

If you are new to cirkit, you need to understand how PCs are represented.
A graph is described by:

- the list of its nodes: `[n1,n2,n3,n4]`.
- the input dictionary, which represents the edges of the graph as
  an adjacency list:

  ```python
  {
    n1:[],
    n2:[n1],
    n3:[n1],
    n4:[n2, n3]
  }
  ```

  This adjacency list represents this graph:

  ```mermaid
  flowchart TD
    n1((n1)) --> n2((n2))
    n1 --> n3((n3))
    n2--> n4((n4))
    n3--> n4
  ```

- The outputs list which contains the nodes that are returned by the graph.
  _Of course, these nodes need to be defined in the input dictionary._

## Torch Compiler Components

Here you will find a short explanation of some important components used for
the compilation:

- `TorchParameter` are computational graphs, they are made of
  `Torch Parameter Node` objects of two types:
- Input node, which can be a randomly initialized constant or a number provided
  by an external process (such as learned parameters).
- Operation node: computations applied to parameters, e.g., Softmax.
- The compiler registries (`CompilerRegistry`) are mappings between types and
  _Rules_ (Functions). They specify how to transform a symbolic layer / parameter / initializer
  into concrete PyTorch Graph element.
  There are three types of compiler registries:
- `CompilerLayerRegistry`
- `CompilerParameterRegistry`
- `CompilerInitializerRegistry`
- The `Semiring` object stores the concrete operations to use for addition and
  multiplication as well as the domain.
- The `CompiledCircuitsMap` which stores bidirectional mappings between symbolic
  and compiled circuits.

## The Compilation Process

Compiling a symbolic circuit involves three main steps:

1. Transforming all symbolic layers / parameters / initializers to their
   compiled versions.
2. Optimizing the Torch graph by fusing or splitting certain layers according to
   optimization parameters.
3. Folding the circuit

You can dive deeper into the different processes on these pages:

- [Compilation](./compilation.md)
- [Optimization](./optimization.md)
- [Folding](./folding.md)
