# Running Notebooks


```bash
pip install ".[notebooks]"
jupyter notebook
```


# Suggested Path through Notebooks


While you can explore the notebooks in any order, we recommend you start from [learning a circuit](learning-a-circuit.ipynb) and proceed as in the graph below.

```mermaid
graph TD;
    A[<a href='https://github.com/april-tools/cirkit/blob/main/notebooks/learning-a-circuit.ipynb'>Learning a circuit</a>]-->B[<a href='https://github.com/april-tools/cirkit/blob/main/notebooks/compilation-options.ipynb'>Compilation Options</a>];
    A-->C[<a href='https://github.com/april-tools/cirkit/blob/main/notebooks/region-graphs-and-parametrisation.ipynb'>Region Graphs and Parametrisation</a>];
    A-->D[<a href='https://github.com/april-tools/cirkit/blob/main/notebooks/learning-a-circuit-with-pic.ipynb'>Probabilistic Integral Circuits</a>];
