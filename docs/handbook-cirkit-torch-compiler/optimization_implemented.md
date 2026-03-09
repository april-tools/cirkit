# Optimization Patterns Implemented in Cirkit

## Layers

### Tucker Fusion

#### Tucker Pattern

For this optimization, we are searching for sum layers that operate on a single
input vector (`arity=1`) and whose parameters are the output of a Kronecker product

#### Tucker Optimization

This optimization consists of rewriting the full operation in a single
einsum to avoid computing the intermediate tensor from the Kronecker
product.

The output of the Kronecker product which takes the vectors $x$ and $y$ of shape
$a$ and $b$ respectively (no batch or fold for simplicity), can be written as the
following einsum:

$$
    a,b \rightarrow ab
$$

We then proceed to flatten the output to get a vector $z$ of
size $i=a \times b$. This vector is then used in the einsum for the sum operation.
Given W, the parameter matrix, of shape $(o,i)$, the sum $Wx$ is:

$$
    i,oi \rightarrow o
$$

Now let us reshape the tensors to re-introduce the $a$ and $b$ dimensions.
The sum is written as:

$$
    ab, oab \rightarrow o
$$

We can finally substitute the output of the Kronecker product by the $x$
and $y$ vectors:

$$
    a,b,oab \rightarrow o
$$

Thus avoiding the intermediate Kronecker product.
This is exactly what the Tucker layer will compute.

A more complete explanation can be found in
[(Loconte et al., 2025)](http://arxiv.org/abs/2409.07953) in subsection 2.1.

### CP-T Fusion

#### CP-T Pattern

Detect the presence of a sum layer whose input is a Hadamard product.

#### CP-T Optimization

Replace the two layers with a single CP-T layer, which does both operations.

### Sum Collapse

#### Sum Collapse Pattern

Detect a sum layer whose only input is another sum layer.

#### Sum Collapse Optimization

In this case, we can simply merge the two node into a
single sum using matrix multiplication of the two
sums' parameters.

Indeed, if we have two sums with parameters $W_1$, $W_2$:
$$S_1=W_1X$$
$$S_2=W_2S_1$$
$$S_2=W_2W_1X$$

The final sum has weights: $W_2W_1$

### Kronecker Sum Shatter

#### Kronecker Sum Pattern

We are searching for sum layers whose parameter graph's output is a Kronecker
parameter layer.

#### Kronecker Sum Optimization

If the weight of the sum can be decomposed into a Kronecker product, we can apply the
[Kronecker Dot Trick](#kronecker-dot-trick) to avoid computing the weight matrix.
We end up with two Tensor Dot layers which will replace the original sum layer.

### Kronecker Tensor Dot Shatter

#### Kronecker Tensor Dot Pattern

We are searching for Tensor Dot layers whose weight matrix is the result of a
Kronecker product (Kronecker Parameter Node).

#### Kronecker Tensor Dot Optimization

As for the sum case, if the weight of the tensor dot can be decomposed into a
Kronecker product, we can apply the [Kronecker Dot Trick](#kronecker-dot-trick)
to avoid computing the weight matrix. We end up with two Tensor Dot layers
which will replace the sum.

## Parameters

### Log Softmax

A simple pattern which replaces consecutive Softmax -> Log nodes with a single
LogSoftmax layer.

### Reduce Sum Outer Product

#### Reduce Sum Outer Product Pattern

The pattern is looking for configurations where a reduce sum node takes the
result of an inner product node as input.

This pattern aims to reduce the memory usage by removing the need to store
the outer product matrix.

#### Reduce Sum Outer Product Optimization

The pattern replaces the two nodes by an einsum. Depending on the indices on which
we compute the sum and outer product, we will need to add a flatten layer.

For example, given matrices $A$ and $B$ of shapes $(b,i,j,l)$ and $(b,i,k,l)$.

**First case:** we want `outer_dim=2` and `reduce_dim=2`, we would compute:

1. The product: $bijl, bikl \rightarrow bijkl$ which makes a large matrix.
2. Flatten $jk$ into $f$.
3. The sum: $bifl \rightarrow bil$.

This computation can be done in a single einsum: $bijl, bikl \rightarrow bil$.

**Second case:** we want `outer_dim=2` and `reduce_dim=3`. In this configuration,
step 3 would look like:

$bifl \rightarrow bif$.

Again, this can be done in a single einsum _and_ a flatten operation:

$bijl, bikl \rightarrow bijk$ and we then flatten $jk$ into $f$.

## Tricks

### Kronecker Dot Trick

This trick comes from [(Zhang et al., 2025)](http://arxiv.org/abs/2506.12383)
(Subsection 3.1).

Given $W=A \otimes B$, the parameters of a sum or dot layer,
with $A$ of shape $(a_1,\dots,a_n)$ and $B$ of shape $(b_1,\dots,b_n)$

$$
            \begin{align*}
            (Wx)_{kl} &=((A \otimes B) x)_{kl}\\
            &= (B (A x)^{T})_{k1}
            \end{align*}
$$

We can convince ourselves that it works by developing a simple example:
First the normal Kronecker and sum / dot:

$$
        A=
        \begin{bmatrix}
        a_1 & a_2 \\
        a_3 & a_4
        \end{bmatrix}
        \text{ and }
        B=
        \begin{bmatrix}
        b_1 & b_2 \\
        b_3 & b_4
        \end{bmatrix}
$$

Then

$$
        W=A\otimes B= \begin{bmatrix}
        a_1b_1 & a_1b_2 & a_2b_1 & a_2b_2 \\
        a_1b_3 & a_1b_4 & a_2b_3 & a_2b_4 \\
        a_3b_1 & a_3b_2 & a_4b_1 & a_4b_2 \\
        a_3b_3 & a_3b_4 & a_4b_3 & a_4b_4 \\
        \end{bmatrix}
$$

So the final sum would be:

$$
    Wx=\begin{bmatrix}
    a_1b_1 & a_1b_2 & a_2b_1 & a_2b_2 \\
    a_1b_3 & a_1b_4 & a_2b_3 & a_2b_4 \\
    a_3b_1 & a_3b_2 & a_4b_1 & a_4b_2 \\
    a_3b_3 & a_3b_4 & a_4b_3 & a_4b_4 \\
    \end{bmatrix}
    \begin{bmatrix}
    x_1\\x_2\\x_3\\x_4
    \end{bmatrix}
    =
    \begin{bmatrix}
    x_1a_1b_1 + x_2a_1b_2 + x_3a_2b_1 + x_4a_2b_2\\
    x_1a_1b_3 + x_2a_1b_4 + x_3a_2b_3 +x_4a_2b_4 \\
    \dots
    \end{bmatrix}
$$

Now let's compute $Ax$ following the dot layer procedure.
Let's first reshape x:

$$
        x=\begin{bmatrix} x_1 & x_2 \\ x_3 & x_4 \end{bmatrix}
$$

Now $Ax$:

$$
        \begin{bmatrix}
        a_1 & a_2 \\
        a_3 & a_4
        \end{bmatrix}
        \begin{bmatrix}
        x_1 & x_2 \\
        x_3 & x_4
        \end{bmatrix}
        =
        \begin{bmatrix}
        a_1x_1+a_2x_3 & a_1x_2+a_2x_4 \\
        a_3x_1+a_4x_3 & a_3x_2+a_4x_4
        \end{bmatrix}
$$

And $B(Ax)^T$:

$$
        \begin{bmatrix}
        b_1 & b_2 \\
        b_3 & b_4
        \end{bmatrix}
        \begin{bmatrix}
        a_1x_1+a_2x_3 & a_3x_1+a_4x_3  \\
        a_1x_2+a_2x_4 & a_3x_2+a_4x_4
        \end{bmatrix}
        =
        \begin{bmatrix}
        x_1a_1b_1 + x_2a_1b_2 + x_3a_2b_1 + x_4a_2b_2\\
        x_1a_1b_3 + x_2a_1b_4 + x_3a_2b_3 +x_4a_2b_4 \\
        \dots
        \end{bmatrix}
$$

We get the same result using both expressions !

## References

Loconte, L., Mari, A., Gala, G., Peharz, R., Campos, C. de, Quaeghebeur, E.,
Vessio, G., & Vergari, A. (2025). What is the Relationship between Tensor
Factorizations and Circuits (and How Can We Exploit it)? (No. arXiv:2409.07953).
arXiv. <https://doi.org/10.48550/arXiv.2409.07953>

Zhang, H., Dang, M., Wang, B., Ermon, S., Peng, N., & Broeck, G. V. den. (2025).
Scaling Probabilistic Circuits via Monarch Matrices (No. arXiv:2506.12383). arXiv.
<https://doi.org/10.48550/arXiv.2506.12383>
