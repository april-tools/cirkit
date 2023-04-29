[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8](https://img.shields.io/badge/python-3.8+-orange.svg)](https://www.python.org/downloads/release/python-380/)

# cirkit

## Development

### Requirements

cirkit currently requires Python 3.8 and PyTorch 2.0 or above versions.
To start developing, install the virtual environment and activate it first.
```shell
virtualenv venv  # or python -m venv venv
# Linux & MacOS
source venv/bin/activate
# Windows
venv\Scripts\activate
```
Then install the required dependencies in [development mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html).
```shell
pip intall .[dev]
```
This will install not only the core dependencies of the library itself (e.g., PyTorch) but also additional dependencies useful for development (e.g., PyTest).
