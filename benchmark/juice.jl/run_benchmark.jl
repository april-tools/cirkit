using JSON
using Random
using StatsBase
using Statistics
using BenchmarkTools
using ProbabilisticCircuits
using ProbabilisticCircuits: BitsProbCircuit, CuBitsProbCircuit, loglikelihoods
using MLDatasets
using CUDA
using Images

# From https://github.com/JuliaGPU/CUDA.jl/blob/2ae53a46658f10b0be32ae61fb4e145782ed9208/test/setup.jl
macro grab_output(ex)
    quote
        mktemp() do fname, fout
            ret = nothing
            open(fname, "w") do fout
                redirect_stdout(fout) do
                    ret = $(esc(ex))

                    # NOTE: CUDA requires a 'proper' sync to flush its printf buffer
                    synchronize(context())
                end
            end
            ret, read(fname, String)
        end
    end
end


function mnist_dataset()
    # Load the MNIST dataset and preprocess it
    train_int = transpose(reshape(MNIST(UInt8, split=:train).features, 28 * 28, :));
    test_int = transpose(reshape(MNIST(UInt8, split=:test).features, 28 * 28, :));
    train_data = UInt32.(train_int)
    test_data = UInt32.(test_int)
    train_data, test_data
end

function truncate_data(data::Matrix; bits = 3)
    data .÷ 2^bits
end

function convert_to_bytes(amount, fmt)
    if fmt == "KiB"
        to_bytes = 1e3
    elseif fmt == "MiB"
        to_bytes = 1e6
    elseif fmt == "GiB"
        to_bytes = 1e9
    else
        throw(AssertionError("Invalid memory format: " * fmt))
    end
    trunc(Int64, amount * to_bytes)
end

function retrieve_stats(pc::ProbCircuit)
    # Count the number of nodes (for each type)
    # Note that the internal implementation introduce additional sum units to _balance_ the computational graph
    # So the number of computational units showed here may differ if using other implementations
    num_inputs = length(inputnodes(pc))
    num_prods = length(mulnodes(pc))
    num_sums = length(sumnodes(pc))
    #println("Total number of nodes: $(num_inputs + num_prods + num_sums)")
    #println("Number of input nodes: $(num_inputs)")
    #println("Number of product nodes: $(num_prods)")
    #println("Number of sum nodes: $(num_sums)")

    # Count _all_ the learnable parameters, so independent is set to false
    num_params = num_parameters(pc, false)
    #println("Number of learnable parameters: $(num_params)")

    Dict(
        "num_inputs" => num_inputs,
        "num_prods" => num_prods,
        "num_sums" => num_sums,
        "num_params" => num_params
    )
end

function run_benchmark(
    pc::CuBitsProbCircuit, train_data, mis_train_data; batch_size = 128, samples = 100, burnin = 5,
    benchmark_mar = false, benchmark_map = false)
    # Select a batch of data only
    indices = StatsBase.sample(1:size(train_data, 1), batch_size, replace = false)
    train_data = train_data[indices, :]
    if benchmark_mar || benchmark_map
        mis_train_data = mis_train_data[indices, :]
    end

    # Benchmark feed-forward pass time (on CPU)
    #trial = @benchmark loglikelihoods($pc, $train_data; batch_size=$batch_size) samples=samples
    # Benchmark feed-forward pass time (on GPU)
    total_samples = burnin + samples
    evi_trial = @benchmark (CUDA.@sync loglikelihoods($pc, $train_data; batch_size=$batch_size)) samples=total_samples
    #dump(trial)
    #evi_trial = median(evi_trial)
    #dump(median(trial))
    #@time ll = loglikelihoods(pc, train_data; batch_size)
    #println("Test LL: $(mean(ll))")
    # Benchmark feed-forward GPU memory
    cuda_trial = @grab_output (CUDA.@time loglikelihoods(pc, train_data; batch_size=batch_size))
    cuda_alloc_memory = parse(Float64, split(split(cuda_trial[2], "GPU allocations:")[2])[1])
    cuda_alloc_memory_fmt = chop(split(split(cuda_trial[2], "GPU allocations:")[2])[2])
    evi_cuda_alloc_memory = convert_to_bytes(cuda_alloc_memory, cuda_alloc_memory_fmt)
    res = Dict(
        "evi" => Dict(
            "times" => last(evi_trial.times, length(evi_trial.times) - burnin),
            "gctimes" => last(evi_trial.gctimes, length(evi_trial.gctimes) - burnin),
            "cpu_alloc_memory" => evi_trial.memory,
            "cuda_alloc_memory" => evi_cuda_alloc_memory
        ))

    if benchmark_mar
        # Benchmark MAR inference (on GPU)
        mar_trial = @benchmark (CUDA.@sync loglikelihoods($pc, $mis_train_data; batch_size=$batch_size)) samples=total_samples
        #mar_trial = median(mar_trial)
        cuda_trial = @grab_output (CUDA.@time loglikelihoods(pc, mis_train_data; batch_size=batch_size))
        cuda_alloc_memory = parse(Float64, split(split(cuda_trial[2], "GPU allocations:")[2])[1])
        cuda_alloc_memory_fmt = chop(split(split(cuda_trial[2], "GPU allocations:")[2])[2])
        mar_cuda_alloc_memory = convert_to_bytes(cuda_alloc_memory, cuda_alloc_memory_fmt)
        res["mar"] = Dict(
            "times" => last(mar_trial.times, length(mar_trial.times) - burnin),
            "gctimes" => last(mar_trial.gctimes, length(mar_trial.gctimes) - burnin),
            "cpu_alloc_memory" => mar_trial.memory,
            "cuda_alloc_memory" => mar_cuda_alloc_memory
        )
    end

    if benchmark_map
        # Benchmark MAP inference (on GPU)
        map_trial = @benchmark (CUDA.@sync MAP($pc, $mis_train_data; batch_size=$batch_size)) samples=total_samples
        #map_trial = median(map_trial)
        cuda_trial = @grab_output (CUDA.@time MAP(pc, mis_train_data; batch_size=batch_size))
        cuda_alloc_memory = parse(Float64, split(split(cuda_trial[2], "GPU allocations:")[2])[1])
        cuda_alloc_memory_fmt = chop(split(split(cuda_trial[2], "GPU allocations:")[2])[2])
        map_cuda_alloc_memory = convert_to_bytes(cuda_alloc_memory, cuda_alloc_memory_fmt)
        res["map"] = Dict(
            "times" => last(map_trial.times, length(map_trial.times) - burnin),
            "gctimes" => last(map_trial.gctimes, length(map_trial.gctimes) - burnin),
            "cpu_alloc_memory" => map_trial.memory,
            "cuda_alloc_memory" => map_cuda_alloc_memory
        )
    end

    res
end

function run_benchmark_rat(
        train_data, mis_train_data;
        batch_size = 128, num_nodes_region = 2, num_nodes_leaf = 2, rg_depth = 1, rg_replicas = 1)
    num_features = size(train_data, 2)

    # Only categorical and binomial distributions are available
    #input_func = RAT_InputFunc(Binomial, 256) 
    input_func = RAT_InputFunc(Categorical, 256)

    # Instantiate a RAT-SPN with the given parameters
    Random.seed!(42)
    pc = RAT(num_features; input_func, num_nodes_region, num_nodes_leaf, rg_depth, rg_replicas, balance_childs_parents=false)
    init_parameters(pc; perturbation = 0.5)
    stats = retrieve_stats(pc)

    # Layerize the circuit, move to GPU and run benchmarks
    pc = CuBitsProbCircuit(BitsProbCircuit(pc))
    results = run_benchmark(pc, train_data, mis_train_data; batch_size=batch_size)
    
    Dict(
        "hparams" => Dict(
            "num_nodes_region" => num_nodes_region,
            "num_nodes_leaf" => num_nodes_leaf,
            "rg_depth" => rg_depth,
            "rg_replicas" => rg_replicas
        ),
        "stats" => stats,
        "results" => results
    )
end

function run_benchmark_hclt(train_data, mis_train_data, truncated_data; batch_size = 100, latents = 8)
    # Instantiate a HCLT with the given parameters
    Random.seed!(42)
    pc = hclt(truncated_data, latents, num_cats = 256, pseudocount = 0.1, input_type = Categorical)
    init_parameters(pc; perturbation = 0.5)
    stats = retrieve_stats(pc)

    # Layerize the circuit, move to GPU and run benchmarks
    pc = CuBitsProbCircuit(BitsProbCircuit(pc));
    results = run_benchmark(pc, train_data, mis_train_data; batch_size=batch_size)

    Dict(
        "hparams" => Dict(
            "latents" => latents
        ),
        "stats" => stats,
        "results" => results
    )
end

# Install the following Julia dependencies
# add ProbabilisticCircuits.jl CUDA Images MLDatasets BenchmarkTools JSON StatsBase
# 
# Executes a benchmark on HCLTs on cuda:1 using batch size 128
# julia run_benchmark.jl 1 HCLT 128
#
function main()
    # Get arguments
    if length(ARGS) < 3
        println("Specify CUDA device id (e.g., 1), model name (either RAT or HCLT), and batch size (e.g., 128)")
        exit(1)
    end
    device_id = parse(UInt64, ARGS[1])
    model_name = ARGS[2]
    if !(model_name in ["RAT", "HCLT"])
        println("Specify either RAT or HCLT as model name")
        exit(1)
    end
    batch_size = parse(Int64, ARGS[3])
    device!(device_id)
    BenchmarkTools.DEFAULT_PARAMETERS.seconds = 7200

    # Load the dataset
    train_data_cpu, _ = mnist_dataset()
    train_data = cu(train_data_cpu)
    mis_train_data = Array{Union{Missing, UInt8}}(train_data)
    Random.seed!(42)
    # Construct data set with 50% missing features
    cols_to_marginalise = StatsBase.sample(1:size(train_data, 2), trunc(Int64, size(train_data, 2) / 2), replace = false)
    mis_train_data[:, cols_to_marginalise] .= missing
    mis_train_data = cu(mis_train_data)

    if model_name == "RAT"
        default_rg_depth = 9
        default_rg_replicas = 1
        hp_num_sum = [16, 32, 64, 128, 256, 512]
        results = Dict()

        # Benchmark by varying the number of sum and leaf units per region
        results["num_sum_region"] = []
        for k in hp_num_sum
            println("Benchmarking RAT with K = " * string(k))
            res = run_benchmark_rat(train_data, mis_train_data;
                batch_size = batch_size, num_nodes_region = k, num_nodes_leaf = k,
                rg_depth = default_rg_depth, rg_replicas = default_rg_replicas
            )
            push!(results["num_sum_region"], res)
        end
    else  # model_name == "HCLT"
        # Sample and preprocess a subset of the data for the Chow-Liu Tree algorithm
        filtered_data = collect(transpose(reshape(train_data_cpu, 28 * 28, :)))
        truncated_data = truncate_data(filtered_data; bits = 3)
        truncated_data = cu(truncated_data)

        hp_latents = [16, 32, 64, 128, 256, 512]
        results = Dict()

        # Benchmark by varying the support size for each latent variable
        results["latents"] = []
        for l in hp_latents
            println("Benchmarking HCLT with L = " * string(l))
            res = run_benchmark_hclt(train_data, mis_train_data, truncated_data;
                batch_size = batch_size, latents = l
            )
            push!(results["latents"], res)
        end
    end

    open("pcs-jl-" * model_name * "-trials.json", "w") do f 
        JSON.print(f, results, 4)
    end
end

main()
