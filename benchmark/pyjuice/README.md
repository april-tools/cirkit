# `pyjuice` Benchmark

## Get Started
First, initiate the submodule and install the dependencies for benchmark. We use submodule for a managed `pyjuice` version.

```shell
git submodule update --init ./pyjuice
pip install -r requirements.txt  # TODO: place in pyproj
```

Then, install `pyjuice` in editable (dev) mode. The dependency list of the package is outdated so we won't use it.

```shell
pip install --no-deps -e ./pyjuice
```
