# `pyjuice` Benchmark

## Get Started
First, initiate the submodule and install the dependencies for benchmark. We use submodule for a managed `pyjuice` version.

```shell
pip install -e ../..[benchmark]  # ../.. is cirkit root
git submodule update --init ./pyjuice
```

Then, install `pyjuice` in editable (dev) mode. The dependency list of the package is outdated so we won't use it.

```shell
pip install --no-deps -e ./pyjuice
```
