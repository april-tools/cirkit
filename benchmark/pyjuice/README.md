# `pyjuice` Benchmark

## Setup
First, remember to initiate the submodule if you haven't. We use submodule for a managed `pyjuice` version dedicated for this repo.

```shell
git submodule update --init ./pyjuice-april
```

Then, install the dependencies for benchmark and the submodule. We use a separate dependency list because `pyjuice`'s is outdated.

```shell
pip install -e ../..[benchmark]  # ../.. is repo root
../install_submodule.sh pyjuice
```

## Run benchmark

The `run_pyjuice.py` runs one pass and generates raw benchmark data with given configs (including several batches). It also includes a sanity check (directly execute `python run_pyjuice.py` without args).

The `run_benchmark.py` runs the whole benchmark suite by calling `run_pyjuice.py` in a child process. In order to correctly evaluate the initialization (compilation) time, it must be put into another process and restarted every time. (For other tests, just increase the number of batches.)
