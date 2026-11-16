# Data-driven initialization of categorical circuits

A probabilistic circuit separates two modeling choices: its **structure** determines
which factorizations and mixtures can be represented, while its **parameters** determine
the distribution encoded by that structure. Cirkit can initialize the parameters of an
already constructed categorical circuit from observations before gradient-based
training begins.

This is a data-dependent parameter initialization procedure. It does not learn a new
circuit structure. The circuit must already be smooth and decomposable, for example
because it was built from a Region Graph. The procedure uses the data to:

1. associate subsets of observations with layers in the fixed circuit;
2. estimate mixture weights at sum layers;
3. estimate categorical probabilities at input layers; and
4. perturb replicated tensor units to break their initial symmetry.

The resulting circuit is still a symbolic circuit. It is compiled separately with a
`PipelineContext`.

## Relation to LearnSPN

The initialization follows the parameter-estimation intuition of top-down learners such
as LearnSPN [1, 2]. LearnSPN represents a subproblem by a pair
$(V, I)$, where $V$ is a set of variables and $I$ is a population of observations.
It recursively applies two kinds of split:

- a **vertical split** partitions $V$ and creates a product node;
- a **horizontal split** partitions $I$ and creates a sum node whose weights are the
  relative sizes of the resulting populations.

When the recursion reaches a leaf, the input distribution is estimated from the
observations in its population, usually with smoothed empirical counts.

Cirkit reuses the horizontal, population-based part of this idea on a tensorized circuit
whose structure is already fixed. It does not perform independence tests, vertical
splits, stopping tests, or structural recursion. Instead, it traverses the existing
circuit from its outputs towards its inputs and propagates observation indices according
to the type of each layer.

| Layer in the fixed circuit | Population rule |
| --- | --- |
| Output layer | Receives all observation indices. |
| Sum layer | Partitions its population among its child layers using Euclidean k-means. |
| Product layer | Sends the complete population to every child layer. |
| Categorical input layer | Estimates category probabilities from its received population. |

The partition at a sum layer is over its child **layers**, not over individual tensor
units. Consequently, data-driven initialization does not alter the number of units or
the connectivity of the tensorized circuit.

## Population propagation

Let $\mathcal{D} = \{\mathbf{x}^{(n)}\}_{n=1}^{N}$ be a categorical dataset and let
$I_\ell \subseteq \{1,\ldots,N\}$ denote the population assigned to layer $\ell$.
Every output layer starts with

$$
I_\ell = \{1,\ldots,N\}.
$$

Consider a sum layer with scope $V_\ell$, population $I_\ell$, and child layers
$c_1,\ldots,c_H$. Cirkit applies Euclidean k-means to the observations restricted to
that scope,

$$
\left\{\mathbf{x}^{(n)}_{V_\ell}: n \in I_\ell\right\},
$$

and obtains child populations $I_{c_1},\ldots,I_{c_H}$. When $H=1$, no clustering is
needed and the population passes directly to the only child. If fewer than $H$
observations are available, the remaining child populations are empty.

A Hadamard or Kronecker product represents a factorization over its children. Each child
therefore receives the same population:

$$
I_c = I_\ell \qquad \text{for every child } c.
$$

Circuits are directed acyclic graphs rather than necessarily trees. If a layer is reached
from several parents, Cirkit takes the union of their incoming row indices before
processing it. An observation is counted once even if it reaches the shared layer along
multiple paths.

These rules preserve the given smooth and decomposable structure: they only decide
which observations are used to initialize each existing parameter.

## Smoothed empirical estimates

Suppose a categorical input for variable $X_j$ has $C$ categories. For a population
$I_\ell$, define

$$
n_c = \sum_{n \in I_\ell} \mathbb{1}\!\left[x_j^{(n)} = c\right],
\qquad c \in \{0,\ldots,C-1\}.
$$

Cirkit converts these counts into probabilities using

$$
\widehat{p}_c =
\frac{n_c + \alpha}
     {\sum_{d=0}^{C-1} n_d + C\alpha}.
$$

$\alpha$ is a non-negative pseudocount added to every category. An empty population is
therefore assigned a uniform distribution (including the $\alpha=0$ case, which is
handled explicitly).

For example, consider a single binary categorical variable $X \in \{0,1\}$ observed
in four data samples with values $(0,0,0,1)$. Its category counts are therefore
$n_0=3$ and $n_1=1$. With $\alpha=2$, smoothing adds two pseudocounts to each
category and gives

$$
\widehat{\mathbf{p}}
= \left(\frac{3+2}{4+2\cdot2},\frac{1+2}{4+2\cdot2}\right)
= \left(\frac{5}{8},\frac{3}{8}\right).
$$

At a sum layer with $H$ child layers $c_1,\ldots,c_H$, k-means assigns a population
$I_{c_h}$ to each child $c_h$. Let

$$
m_h = |I_{c_h}|
$$

be the number of observations assigned to child $c_h$. Cirkit initializes the mixing
probability of that branch by applying the same smoothed-count normalization used for
categorical inputs:

$$
\widehat{\pi}_h =
\frac{m_h + \alpha}
     {\sum_{r=1}^{H} m_r + H\alpha},
\qquad h \in \{1,\ldots,H\}.
$$

Thus, the initial branch probabilities sum to one, and a child with an empty population
still receives positive mass when $\alpha>0$. For example, suppose a sum layer has
$H=3$ children whose population sizes are

$$
(m_1,m_2,m_3)=(5,3,0).
$$

With $\alpha=1$, the initialized branch probabilities are

$$
\widehat{\boldsymbol{\pi}}
= \left(\frac{5+1}{8+3},\frac{3+1}{8+3},\frac{0+1}{8+3}\right)
= \left(\frac{6}{11},\frac{4}{11},\frac{1}{11}\right).
$$

Here, $(5,3,0)$ contains the numbers of observations assigned to the three children,
not observed values of a categorical variable. The third child receives no observations,
but its pseudocount still gives it an initial branch mass of $1/11$.

Each child $c_h$ of a sum layer outputs a vector of $K$ units,
$\mathbf{x}_h=(x_{h,1},\ldots,x_{h,K})$. The estimate $\widehat{\pi}_h$ assigns a total
mass to the whole child, rather than separate masses to its individual units. How Cirkit
turns this child-level estimate into weights depends on the sum-layer parameterization.

In a dense sum layer, every output unit is connected to all $K$ units of every child.
Because clustering provides no evidence for preferring one unit within a child, the
initialization divides the child's mass equally among its units:

$$
w_{k,h,j}=\frac{\widehat{\pi}_h}{K}.
$$

Consequently, the $K$ weights connecting output unit $k$ to child $h$ sum to
$\widehat{\pi}_h$. For example, if $K=3$ and two children have masses
$(\widehat{\pi}_1,\widehat{\pi}_2)=(0.75,0.25)$, each output unit initially receives the
weight vector

$$
(0.25,0.25,0.25,\;0.25/3,0.25/3,0.25/3).
$$

A mixing layer has a more restrictive connectivity pattern. Output unit $k$ connects
only to input unit $k$ of each child:

$$
y_k=\sum_{h=1}^{H} v_{k,h}x_{h,k}.
$$

With the same example, the first output unit combines only $x_{1,1}$ and $x_{2,1}$
with weights $0.75$ and $0.25$; it has no connections to units with indices $2$ or $3$.
Cirkit preserves this diagonal structure by retaining the `MixingWeightParameter`
operator. Data-driven initialization sets its $K H$ free mixing weights, but does not
replace it with an unrestricted dense matrix containing $K^2H$ independently learnable
connections.

Multichannel categorical inputs estimate one marginal per channel. Their joint input
function is a product of channel-wise categorical distributions, rather than a single
categorical distribution over all joint channel assignments.

## Tensor units and symmetry breaking

A tensorized input layer contains $K$ units, while the empirical calculation above
produces one categorical estimate for its population. Cirkit initially broadcasts that
estimate to all $K$ units. Sum-layer estimates are broadcast across their output units
in the same way.

Exact replication would give the units identical initial values and, in symmetric
architectures, identical gradients. The `noise_scale` parameter adds independent noise
to let the replicated units specialize during training. Input estimates always use
softmax: their stored logits are log probabilities plus Gaussian noise. For sum weights,
Cirkit maps the estimate to the selected activation's parameter space before perturbing it:

| Activation | Stored preactivation before noise |
| --- | --- |
| `softmax` | $\theta = \log \widehat{p}$ |
| `softplus` | $\theta = \log(\exp(\widehat{p}) - 1)$ |
| `sigmoid` | $\theta = \log \widehat{p} - \log(1-\widehat{p})$ |
| `none` or `positive-clamp` | the positive estimate itself |

For softmax, softplus and sigmoid, Gaussian noise with standard deviation
`noise_scale` is added to $\theta$. For raw positive parameters, Cirkit applies
multiplicative log-normal noise. Setting `noise_scale=0` preserves the empirical
estimate up to numerical clipping.

This construction should be understood as an initialization bias, not as a guarantee
that the empirical estimate is optimal after training. Estimated parameters can encode
sampling noise or overly specific cluster assignments. Validation, regularization and
the choice of $\alpha$ remain part of model selection.

## Parameter sharing

Parameter sharing and empirical initialization are separate choices.

With `input_sharing="none"`, each input layer estimates its marginal from the population
that reaches that layer. With `input_sharing="full"`, compatible input layers use one
shared parameter tensor and their counts are pooled:

- tabular inputs pool observations across features with identical categorical
  specifications;
- multichannel images pool pixels across spatial positions, separately for each channel.

For RGB data, the shared tensor has shape $(K,C,256)$: $K$ input units, $C$
channels and 256 categories. Sharing occurs across pixel locations, while the channel
distributions remain distinct. This representation follows the input-sharing setup used
for tensorized probabilistic circuits [3].

## Normalized and raw parameterizations

Categorical inputs use softmax probabilities, with normally initialized logits or
perturbed log empirical probabilities when initializing from data. Input logits remain
unconstrained throughout training and are never projected to positive values.
`sum_weight_param` configures only the sum weights:

- `activation="none"` adds no local activation to sum weights. Only these weights are
  projected to `POSITIVE_PARAMETER_MIN` before the first forward pass and after each
  optimizer step. Input logits are not clamped.
- `activation="positive-clamp"` constrains sum weights locally, with no
  optimizer-side projection.
- `activation="softmax"`, `"softplus"`, or `"sigmoid"` applies that activation to sum weights
  while retaining softmax inputs.

The template and initialization APIs configure sum weights through `sum_weight_param`.
Softmax sum weights give a locally normalized circuit; other activations require global
normalization. A training application can compute the log-likelihood using the current
partition function, obtained from a symbolically integrated circuit:

$$
\log p_\theta(\mathbf{x})
= \log s_\theta(\mathbf{x}) - \log Z_\theta,
\qquad
Z_\theta = \sum_{\mathbf{x}} s_\theta(\mathbf{x}).
$$

## Configuration and practical walkthrough

`image_data` and `tabular_data` run the initialization after constructing the symbolic
circuit and before returning it. The resulting symbolic circuit is then compiled in the
usual way.

The settings have the following interpretation:

| Setting | Meaning |
| --- | --- |
| `alpha` | Non-negative pseudocount added to every category or mixture branch. |
| `noise_scale` | Standard deviation of symmetry-breaking noise. |
| `seed` | Seed of a local NumPy generator used for centroids and perturbations. |

The local generator leaves the global NumPy and Torch random states unchanged.

The executable [data-driven initialization notebook](../../notebooks/data-driven-initialization.ipynb)
shows how to initialize, compile, train and evaluate a categorical probabilistic circuit
for density estimation on MNIST.

`initialize_parameters` can also be called directly on a newly constructed circuit.
It replaces symbolic parameters in place, so call it before compilation and before
constructing derived circuits that reference the old parameters.

The `image_data` template accepts every image region graph through `region_graph`:
`quad-tree-2` (default), `quad-tree-4`, `quad-graph`, `random-binary-tree`,
`poon-domingos`, `chow-liu-tree`, and `linear-tree`.

Chow-Liu learns its structure from the supplied data, independently of whether
data-driven parameter initialization is enabled. The optional `num_bins` (1 to 256)
affects only structure learning; input distributions and parameter initialization retain
all 256 pixel levels.
The optional `mi_chunk_size` must be positive and limits the samples processed at once.
Without these options, all pixel levels and all training samples are used. Chunking does
not reduce the joint-count table, whose size is quadratic in both the number of variables
and the number of bins. Both options are ignored for other region graphs.

All region graphs support single-channel images with either input-sharing mode.
Multichannel images support every region graph with `input_sharing="none"`; full sharing
requires `quad-tree-2`, `quad-tree-4`, or `quad-graph`. The template supports `cp`, `cp-t`,
and `tucker` inner layers.

## Data assumptions and scope

Data-driven initialization currently supports smooth and decomposable circuits composed
of categorical inputs, sum layers, and Hadamard or Kronecker products.

- Tabular data must have shape $(N,D)$. Different variables may have different
  category counts when parameters are not fully shared.
- Image data may have shape $(N,C,H,W)$, $(N,C\cdot H\cdot W)$, or $(N,H,W)$
  for single-channel images. Flattening is channel-first.
- Observations must be finite, integer-valued and inside the category range of each
  variable. Integer-valued floating tensors are accepted.
- For images, `num_bins` affects Chow-Liu structure learning but not parameter
  initialization, which uses the original pixel categories.
- Full sharing requires identical input types, unit counts, channel counts and category
  counts.
- Custom input parameter factories are not combined with empirical initialization,
  because initialization replaces the input parameters.

Euclidean k-means, population pooling at shared DAG nodes and perturbation of replicated
units are choices of this fixed-structure initialization procedure. They should not be
read as a complete implementation of LearnSPN. In particular, tree leaves, bagging,
independence tests and representation-learning-based clustering are outside its scope.

## References

1. Robert Gens and Pedro Domingos. **Learning the Structure of Sum-Product Networks.**
   ICML 2013, PMLR 28(3), pp. 873-880.
   [Paper](https://proceedings.mlr.press/v28/gens13.pdf).
2. Antonio Vergari, Nicola Di Mauro and Floriana Esposito. **Simplifying, Regularizing
   and Strengthening Sum-Product Network Structure Learning.** ECML PKDD 2015,
   pp. 343-358. [DOI](https://doi.org/10.1007/978-3-319-23525-7_21).
3. Gennaro Gala, Cassio de Campos, Antonio Vergari and Erik Quaeghebeur.
   **Scaling Continuous Latent Variable Models as Probabilistic Integral Circuits.**
   NeurIPS 2024, Section 3.3 and Appendix D.2.
   [Paper](https://arxiv.org/abs/2406.06494).
