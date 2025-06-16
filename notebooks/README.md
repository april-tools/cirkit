# Running Notebooks


```bash
pip install ".[notebooks]"
jupyter notebook
```


# Suggested Path through Notebooks


While you can explore the notebooks in any order, we recommend you start from the **learning-a-circuit.ipynb** notebook, and proceed as in the graph below.

```mermaid
graph TD;
    A[<a href='https://github.com/april-tools/cirkit/blob/main/notebooks/learning-a-circuit.ipynb'>Learning a circuit</a>]-->B[<a href='https://github.com/april-tools/cirkit/blob/main/notebooks/compilation-options.ipynb'>Compilation Options</a>];
    A-->I[<a href='https://github.com/april-tools/cirkit/blob/main/notebooks/learning-a-gaussian-mixture-model.ipynb'>Build a GMM Layer by Layer</a>];
    A-->C[<a href='https://github.com/april-tools/cirkit/blob/main/notebooks/region-graphs-and-parametrisation.ipynb'>Region Graphs and Parametrisation</a>];
    A-->D[<a href='https://github.com/april-tools/cirkit/blob/main/notebooks/learning-a-circuit-with-pic.ipynb'>Probabilistic Integral Circuits</a>];
    A-->H[<a href='https://github.com/april-tools/cirkit/blob/main/notebooks/generative-vs-discriminative-circuit.ipynb'>Building an MNIST Classifier using Cirkit</a>];
    B-->E[<a href='https://github.com/april-tools/cirkit/blob/main/notebooks/sum-of-squares-circuits.ipynb'>Sum of Squares Circuits</a>];
    C-->E;
    B-->F[<a href='https://github.com/april-tools/cirkit/blob/main/notebooks/compression-cp-factorization.ipynb'>CP Tensor Factorization</a>];
    B-->G[<a href='https://github.com/april-tools/cirkit/blob/main/notebooks/logic-circuits.ipynb'>Logic Circuits</a>]
```

