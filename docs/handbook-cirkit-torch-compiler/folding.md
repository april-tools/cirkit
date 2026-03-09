# Folding Circuits

## Concept

Folding is an important concept to reduce the execution time and use the
hardware more efficiently. It consists of **identifying** which layers
could be computed together and **grouping** them as one tensor to compute
operations on.

![type:video](./media/videos/FoldingGraph.mp4)

## What can we fold ?

Different layers / parameter nodes will require different conditions to decide
if they can be folded with another layer. These conditions are defined in
the `fold_settings()` method of each module. In general, these conditions are
defined in the inheritance process for a family of modules, but they can
also be overriden for more specific restrictions.

### `TorchParameterNode`

Any class that inherits from `TorchParameterNode` and does not override
`fold_settings()` will specify that a node can only be folded with another
node if they share the exact same `config` dictionary.

### `TorchTensorParameter`

A `TorchTensorParameter` can be folded with another tensor parameter if they have the same
shape, data type and the same gradient requirements (either both `True` or both `False`)

### `TorchInnerLayer`

Any class that inherits from `TorchInnerLayer` and does not override `fold_settings()`
will specify that a layer can be folded with another layer if they share the same
`config` dictionary **and** if their parameters have the same shape.

### `TorchInputLayer`

Any class that inherits from `TorchInputLayer` and does not override `fold_settings()`
will specify that a layer can be folded with another layer if they share the same
number of inputs, `config` dictionary **and** if their parameters have the same shape.

## How to fold in Cirkit

To handle the folding procedure, the compiler calls the `_fold_circuit` function
defined in `cirkit.backend.torch.compiler.py`. This function simply retrieves
a layer-wise topological ordering of the circuit
(see [Basic Terminology](./handbook.md#basic-terminology))
and feeds it to the `build_folded_graph` function.

### Fold Group Function (`fold_group_fn`)

Fold group functions are functions that take into entry:

- A list of Torch Modules: either `TorchParameter` or `TorchLayer` depending on the
  implementation.
- The current compiler.

It returns a unique module representing the folded modules.

These functions handle the logic needed to merge multiple modules into a
single one. They are not responsible for constructing the folded graph or
finding potential folding opportunities!

#### Fold Group of Layer (`_fold_layer_group`)

As explained above, this function transforms a list of `TorchLayer` objects into
a single folded `TorchLayer`.

The algorithm works as follows:

1. Verify that **all** layers in the list are of the same type.
2. Create a dictionary `kwargs` storing the `config` of the first layer in the group.
3. If the layers are input layers, update the `scope_idx` entry of the dictionary
   to store the concatenation of each input's `scope`.
4. Otherwise, store the sum of the `num_folds` config entry of each layer in the `kwargs`'s
   `num_folds` entry.
5. Construct a list `layer_params` containing the parameter graphs of all layers
6. If there are any parameters, call the `_fold_parameter` function to obtain a single
   `TorchParameter` object.
7. Construct a list `layer_submodules` containing the submodule layers of all layers.
8. If there are any submodules, call the `_fold_layer_group` on the list.
9. Finally, call the constructor of the first layer with the kwargs values.

#### Fold Group of Parameter (`_fold_parameter_nodes_group`)

First, let's explain the `foldwise_initializer_` function. It's a function that
populates a tensor of size $(F,...)$, where the first dimension $F$ is the fold,
using a list of $F$ initializers. It is used by doing a partial function as follows:

```python
functools.partial(
  foldwise_initializer_,
  initializers=list(map(lambda p: p.initializer, group_tensors)),
)
```

Now to fold parameters, we identify three cases:

1. The parameters to fold are `TorchTensorParameter`.
   1. In this case, create a new parameter with the same config as the first in
      the list.
   2. For the `initializer` parameter, create a new partial function using the
      `foldwise_initializer_` function.
   3. Finally, update the compiler state to register the pair
      (folded parameter, fold_idx) as corresponding to the correct symbolic parameter.

      For example, if we have three compiled parameter P1, P2 and P3, mapped with
      symbolic parameter S1, S2 and S3, that we want to fold into one parameter PF,
      we would register:

      ```python
      {
        S1:(PF,1), # Previously (P1,0)
        S2:(PF,2), # Previously (P2,0)
        S3:(PF,3), # Previously (P3,0)
      }
      ```

2. The parameters to fold are `TorchPointerParameter`.
   1. We assume that all pointer parameters in the group point to the same base
      `TorchTensorParameter`.
   2. We create a new pointer parameter that collects all fold indices that
      are selected by each parameter in the group.

3. The parameters to fold are `TorchParameterOp` (like sum, product, etc.)
   1. In this case, construct a new parameter operation with the same config and
      the updated number of folds (number of elements in the group).

##### Small Explanation on `TorchPointerParameter`

`TorchPointerParameter` are objects that represent a fold slice of a larger `TorchTensorParameter`.
The slice can be:

- The full tensor.
- A single fold index.
- A list of index (potentially not contiguous).

### Fold a Parameter Graph (`_fold_parameters`)

Similar to `_fold_circuit`, we retrieve a layer-wise topological ordering of
the parameter nodes. An important fact to note is that we are treating all
parameter graphs in the group as **one large graph with several outputx**,
and it is this graph that we want to fold. This means that the final graph can
contain fold both **inside** a single parameter graph but also **between**
parameter graphs.

Once we have the layer-wise topological ordering of all the nodes in
all the graphs, we use the `build_folded_graph` function to obtain the structure
of the folded parameter graph and instantiate a new `TorchParameter` using
these informations.

### Finding Potential Fold (`group_foldable_modules`)

This function, defined in `cirkit.backend.torch.graph.foldin.py`, is responsible
for finding potential folding in a given group corresponding to a level in the graph
layer-wise topological ordering.

It works as follows:

1. Define the `_gather_fold_settings` function that retrieves all the `fold_settings()`
   properties of the module and its submodules
   (See [What Can We Fold?](#what-can-we-fold))
   This tuple also includes the type of the module, which is a fundamental condition.
2. Create a dictionary that will store a mapping between lists of conditions and
   modules that match them.
3. For each module:
   1. Retrieve the tuple of fold settings for the module.
   2. Add the module to the list corresponding to the tuple in the dictionary.
4. Returns a list containing all the grouped modules (list of lists)

This function constructs the groups simply by using a mapping of conditions.

### Main Logic (`build_folded_graph`)

This function, defined in `cirkit.backend.torch.graph.folding.py`, is the main
entry point to fold a circuit. It takes as parameters:

- `ordering`: a layer-wise topological ordering of the circuit.
- `outputs`: an iterable object of the output module in the circuit.
- `incomings_fn`: a function returning the inputs of a given module.
- `fold_group_fn`: a function that folds a given list of modules.

and returns a tuple with:

- The final, potentially folded, modules.
- The adjacency list updated with the folded modules.
- The list of modules that acts as output of the graph.
- A `FoldIndexInfo` object which stores the information necessary
  to "locate" a unfolded module within the folded circuit.
  It is basically a map between a module in the unfolded circuit
  and a pair (id_folded_module, fold_id).

#### Data Structures

The algorithm will use the following data structures:

- `fold_idx`: a dictionary that maps each unfolded module to:
  1. a `fold_id` (a natural number) pointing to the module layer it is
     associated to.
  2. a `slice_idx` (a natural number) within the output of the folded module,
     which recovers the output of the unfolded module.
- `in_fold_idx`: a dictionary mapping each folded module id to a tensor of
  indices IDX of size (F, H, 2), where F is the number of modules in the fold,
  H is the number of inputs to each fold. Each entry i,j,: of IDX is a pair
  `(fold_id, slice_idx)`, pointing to the folded module of id `fold_id` and to
  the slice `slice_idx` within that fold.
- `modules`: the list of each folded modules.
- `in_modules`: the adjacency list mapping each module to its inputs.

#### Algorithm

1. For each group of modules with the same level in the layer-wise topological ordering:
   1. Retrieve the folding group using `group_foldable_modules`.
   2. For each group:
      1. Get the folded module using the given `fold_group_fn`.
      2. Retrieve the inputs of each module in the group.
      3. For each input in the list, retrieve the corresponding folded module
         using `fold_idx`.
      4. Store the mapping between the current module and the inputs in `in_modules`.
      5. Create a new list `in_modules_idx` storing the entry of `fold_idx` for each
         input of each unfolded module. The list respect the double hierarchy
         unfolded modules -> input of unfolded modules -> tuple as it is a double
         list of tuples.
      6. Store the new mapping in `fold_idx` using the current length of `modules`
         as the current folded module id.
      7. Store `in_modules_idx` as the entry for the current module id in `in_fold_idx`.
      8. Append the current folded module to `modules`.
2. Retrieve the folded module corresponding to the unfolded `outputs` modules
   using `fold_idx` and `modules`.
3. Create a `FoldIndexInfo` object from the `modules`, `in_fold_idx` and `out_fold_idx`.
4. Return `modules`, `in_modules`, `outputs` and the fold index info.
