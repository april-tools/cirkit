[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/release/python-3100/)
[![Documentation Status](https://readthedocs.org/projects/cirkit-docs/badge/?version=latest)](https://cirkit-docs.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/april-tools/cirkit/branch/main/graph/badge.svg?token=MLHONY840L)](https://codecov.io/gh/april-tools/cirkit)

![cirkit logo](./logo.png)


# What is Cirkit? :electric_plug:


**cirkit** is a framework for building, learning and reasoning about **probabilistic machine learning** models, such as [circuits](https://arxiv.org/abs/2409.07953) and [tensor networks](https://arxiv.org/abs/1708.00006), which are **tractable** ( ⬆️ ) and **expressive** ( ➡️ ).

![cirkit](https://github.com/user-attachments/assets/cc1c648d-6c04-4d2c-b83a-1b0939b62d35)


# Main Features

* ⚡ **Exact and Efficient Inference** : Support for tractable operations that are automatically compiled to efficient computational graphs that run on the GPU.
* **Compatible**: Seamlessly integrate your circuit with deep learning models; run on any device compatible with PyTorch.
* **Modular and Extensible**: Support for user-defined layers and parameterizations that extend the symbolic language of cirkit.
* **Templates for Common Cases** : Templates for constructing circuits by mixing layers and structures with a few lines of code.


## Project Structure :open_file_folder:


```
.
├── cirkit              Main Code
│   ├── backend         Circuits to Numerical Operations (Currently via PyTorch backend)
│   ├── symbolic        Circuits / Layers / Operators / Compilation
│   ├── templates       APIs for easy use (e.g. region graphs, data modalities)
│   └── utils
├── docs
├── notebooks           Start here: Examples
└── tests
```

## How to Install the Library

cirkit currently requires Python 3.10 and PyTorch 2.3 or above versions.
To start developing, install the virtual environment and activate it first.
```shell
virtualenv venv           # or python -m venv venv
source venv/bin/activate
```
Then install the required dependencies in [development mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html).
```shell
pip install -U pip        # update pip
pip install -e ".[dev]"
```
This will install not only the core dependencies of the library itself (e.g., PyTorch) but also additional dependencies useful for development (e.g., PyTest).
It also installs other development tools, such as Black, PyLint and MyPy.

### Additional Requirements for Jupyter Notebooks

If you want to execute the Jupyter notebooks in the ```notebooks/``` directory, then install the additional dependencies with:
```shell
pip install ".[notebooks]"
```

## Documentation 📘

For more details [see the Documentation here](https://cirkit-docs.readthedocs.io/en/latest/).


## Development 🛠️

### Build Documentation Locally

Whenever you write documentation, you can check how it would look like by building HTML pages locally.
To do so, install the dependencies for building documentation:

```shell
pip install ".[docs]"
```

Then, run the following at the root level of the repository directory.

```shell
mkdocs serve
```

After waiting a few seconds, you can then navigate the rendered documentation at the link http://127.0.0.1:8000/.

### Automatic Code Formatting

We try to follow a consistent formatting across the library.
If you want to automatically format your code, then you should run the following script.

```shell
bash scripts/format.sh
```

### Linting and Static Code Checks

Locate youself in the repository root.
Then, run the following for executing the linters and other static code checkers.

```shell
bash scripts/check.sh [--tool linting-tool] [file ...]
```
Optionally,
1. Specify `--tool` to select the linting tool to use. If no one is given, then all of them will be run, i.e., black, isort, pydocstyle, pylint, mypy.
2. Add files to lint part of the repo. If none is specified then, all tracked directiories will be checked. 

### Run Unit Tests and Check the Coverage 

Locate youself in the repository root.
Then, rn the following script.

```shell
bash scripts/coverage.sh [--FORMAT] [pytest_arg ...]
```
Optionally,
1. Use a `--FORMAT` (e.g. `--xml`) flag for exporting converage to file.
2. Pass additional args to pytest (files to test etc.).


## Papers :scroll:


If you want to learn more about the internals of cirkit, a good starting point is [What is the Relationship between Tensor Factorizations and Circuits (and How Can We Exploit it)?](https://arxiv.org/abs/2409.07953).


## Papers Implemented in Cirkit

| **Papers**          | **Links within Cirkit** |
|--------------------|------------------------|
| [Sum of Squares Circuits](https://arxiv.org/abs/2408.11778) 🆘 | [notebook](https://github.com/april-tools/cirkit/blob/main/notebooks/sum-of-squares-circuits.ipynb) |
| [Scaling Continuous Latent Variable Models as Probabilistic Integral Circuits](https://arxiv.org/abs/2406.06494) | [notebook](https://github.com/april-tools/cirkit/blob/main/notebooks/learning-a-circuit-with-pic.ipynb) |
| [What is the Relationship between Tensor Factorizations and Circuits (and How Can We Exploit it)?](https://arxiv.org/abs/2409.07953) | See [Region Graphs](https://github.com/april-tools/cirkit/blob/main/notebooks/region-graphs-and-parametrisation.ipynb) and [Folding](https://github.com/april-tools/cirkit/blob/main/notebooks/compilation-options.ipynb)|
| [Random Sum-Product Networks: A Simple and Effective Approach to Probabilistic Deep Learning](https://proceedings.mlr.press/v115/peharz20a) | See [Random Binary Tree](https://github.com/april-tools/cirkit/blob/main/notebooks/region-graphs-and-parametrisation.ipynb) |
| [Einsum Networks: Fast and Scalable Learning of Tractable Probabilistic Circuits](https://arxiv.org/abs/2004.06231) | See [Optimizing the Circuit Layers](https://github.com/april-tools/cirkit/blob/main/notebooks/compilation-options.ipynb) |


## Citation

[comment]: <> (The following bib file can be generated from the github page via the "Cite this repository" button. To update bib, simply update the CITATIONS.cff file by uploading current cff file to https://citation-file-format.github.io/ and modifying it )

If you use cirkit in your publications, please cite:

```
@software{The_APRIL_Lab_cirkit_2024,
author = {The APRIL Lab},
license = {GPL-3.0},
month = oct,
title = {{cirkit}},
url = {https://github.com/april-tools/cirkit},
version = {0.1},
year = {2024}
}
```
