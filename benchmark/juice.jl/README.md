# Juice.jl Benchmark

To be able to run the benchmark script ```run_benchmark.jl``` install the dependencies first.
Launch ```julia```, open the package mode with ```]``` and run the following commands.

```
dev ProbabilisticCircuits.jl/
add CUDA Images MLDatasets BenchmarkTools JSON StatsBase
```

Note that ```ProbabilisticCircuits.jl/``` is the local directory of the [ProbabilisticCircuits.jl](https://github.com/Juice-jl/ProbabilisticCircuits.jl) git submodule (tag v0.4.0).

Launch the script ```run_benchmark.jl``` by specifying the CUDA device id, the model to benchmark (either RAT or HCLT) and the batch size, e.g.,
```
julia run_benchmark.jl 1 RAT 500
```

The script will measure times and memory usage of performing EVI, MAR and approximate MAP inference on the training CIFAR10 split.
Then it will save the results in a JSON file.
