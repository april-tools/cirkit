using JSON
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
    # Uncomment this when using the GPU
    #train_data = UInt32.(train_int) .+ one(UInt32);
    #test_data = UInt32.(test_int) .+ one(UInt32);
    train_data = UInt32.(train_int)
    test_data = UInt32.(test_int)
    train_data, test_data
end

function run_benchmark(train_data, test_data; batch_size = 100, num_nodes_region = 2, num_nodes_leaf = 2, rg_depth = 1, rg_replicas = 1)
    num_features = size(train_data, 2)

    # Only categorical and binomial distributions are available
    #input_func = RAT_InputFunc(Binomial, 256);  
    input_func = RAT_InputFunc(Categorical, 256);

    # Instantiate a RAT-SPN with the given parameters
    pc = RAT(num_features; input_func, num_nodes_region, num_nodes_leaf, rg_depth, rg_replicas);
    init_parameters(pc; perturbation = 0.5);

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

    # Layerize the circuit and move to GPU
    pc = CuBitsProbCircuit(BitsProbCircuit(pc));

    # Benchmark feed-forward pass time (on CPU)
    #trial = @benchmark loglikelihoods($pc, $test_data; batch_size=$batch_size) seconds=60
    # Benchmark feed-forward pass time (on GPU)
    trial = @benchmark CUDA.@sync loglikelihoods($pc, $test_data; batch_size=$batch_size) seconds=60
    #dump(trial)
    median_trial = median(trial)
    #dump(median(trial))
    #@time ll = loglikelihoods(pc, test_data; batch_size)
    #println("Test LL: $(mean(ll))")
    
    # Benchmark feed-forward GPU memory
    cuda_trial = @grab_output CUDA.@time loglikelihoods($pc, $test_data; batch_size=$batch_size)
    cuda_alloc_memory = split(cuda_trial)[11]
    cuda_alloc_memory_fmt = chop(split(cuda_trial)[12])
    if cuda_alloc_memory_fmt == "MiB"
        to_bytes = 1e3
    elseif cuda_alloc_memory_fmt == "MiB"
        to_bytes = 1e6
    elseif cuda_alloc_memory_fmt == "GiB"
        to_bytes = 1e9
    else
        throw(BoundsError("Something is wrong!"))
    end
    cuda_alloc_memory /= to_bytes
    
    Dict(
        "hparams" => Dict(
            "num_nodes_region" => num_nodes_region,
            "num_nodes_leaf" => num_nodes_leaf,
            "rg_depth" => rg_depth,
            "rg_replicas" => rg_replicas
        ),
        "stats" => Dict(
            "num_inputs" => num_inputs,
            "num_prods" => num_prods,
            "num_sums" => num_sums,
            "num_params" => num_params
        ),
        "median_time" => median_trial.time,
        "median_gctime" => median_trial.gctime,
        "cpu_alloc_memory" => median_trial.memory,
        "cuda_alloc_memory" => cuda_alloc_memory
    )
end

function main()
    # Set CUDA device
    if length(ARGS) < 1
        device!(0)  # Set to cuda:0 by default
    else
        device!(parse(UInt64, ARGS[1]))
    end

    # Load the MNIST dataset
    train_data, test_data = cu.(mnist_dataset())

    batch_size = 500
    default_num_sum = 10
    default_rg_depth = 4
    default_rg_replicas = 10

    hp_num_sum = [2, 4, 8, 16, 32]
    hp_rg_depth = [4, 5, 6, 7, 8]
    hp_rg_replicas = [2, 4, 8, 16, 32]
    results = Dict()

    # Benchmark by varying the number of sum and leaf units per region
    results["num_sum_region"] = []
    for k in hp_num_sum
        res = run_benchmark(train_data, test_data;
            batch_size = batch_size, num_nodes_region = k, num_nodes_leaf = k,
            rg_depth = default_rg_depth, rg_replicas = default_rg_replicas
        )
        push!(results["num_sum_region"], res)
    end

    # Benchmark by varying the depth
    results["rg_depth"] = []
    for d in hp_rg_depth
        res = run_benchmark(train_data, test_data;
            batch_size = batch_size, num_nodes_region = default_num_sum, num_nodes_leaf = default_num_sum,
            rg_depth = d, rg_replicas = default_rg_replicas
        )
        push!(results["rg_depth"], res)
    end

    # Benchmark by varying the number of replicas
    results["rg_replicas"] = []
    for r in hp_rg_replicas
        res = run_benchmark(train_data, test_data;
            batch_size = batch_size, num_nodes_region = default_num_sum, num_nodes_leaf = default_num_sum,
            rg_depth = default_rg_depth, rg_replicas = r
        )
        push!(results["rg_replicas"], res)
    end

    open("pcs-jl-trials.json", "w") do f 
        JSON.print(f, results, 4)
    end
end

main()
