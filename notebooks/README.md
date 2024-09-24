# Running notebooks


```bash
pip install ".[notebooks]"
jupyter notebook
```


# Suggested Path


While you can explore the notebooks in any order, we recommend you start from [learning a circuit](learning-a-circuit.ipynb) and proceed as in the graph below.

```mermaid
graph TD;
    A[<a href='https://github.com/april-tools/cirkit/blob/main/notebooks/learning-a-circuit.ipynb'>Learning a circuit</a>]-->B[<a href='https://github.com/april-tools/cirkit/blob/main/notebooks/compilation-options.ipynb'>Compilation Options</a>];
    A-->C[Complex inference via pipelines];
```

