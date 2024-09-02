[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8](https://img.shields.io/badge/python-3.8+-orange.svg)](https://www.python.org/downloads/release/python-380/)
[![codecov](https://codecov.io/gh/april-tools/cirkit/branch/main/graph/badge.svg?token=MLHONY840L)](https://codecov.io/gh/april-tools/cirkit)

# cirkit

## Development

### Requirements

cirkit currently requires Python 3.8 and PyTorch 2.3 or above versions.
To start developing, install the virtual environment and activate it first.
```shell
virtualenv venv           # or python -m venv venv
source venv/bin/activate
```
Then install the required dependencies in [development mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html).
```shell
pip install -U pip        # update pip
pip install -e .[dev]
```
This will install not only the core dependencies of the library itself (e.g., PyTorch) but also additional dependencies useful for development (e.g., PyTest).
It also installs other development tools, such as PyLint and MyPy.

#### Additional Requirements for Jupyter Notebooks

If you want to execute the Jupyter notebooks in the ```notebooks/``` directory, then install the additional dependencies with:
```shell
pip install .[notebooks]
```

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
bash scripts/check.sh [--verbose] [file ...]
```
Optionally,
1. Use the `--verbose` flag to enable more verbose output.
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
