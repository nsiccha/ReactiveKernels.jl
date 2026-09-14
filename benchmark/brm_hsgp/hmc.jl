# Exact motorcycle kernel through the shared authored HMC benchmark. Run each
# backend in a fresh process with identical batch sizes and CPU affinity.
using ReactiveKernels, Enzyme, DifferentiationInterface
using LinearAlgebra, Random, Serialization, Statistics, SHA, TOML, Dates
import Pkg
const BACKEND_NAME = Symbol(get(ENV, "HMC_BACKEND", "native"))
BACKEND_NAME in (:native, :reactant) || error("HMC_BACKEND must be native or reactant")
BACKEND_NAME === :reactant && (@eval import Reactant)
include(joinpath(@__DIR__, "..", "..", "examples", "brm_hsgp.jl"))
include(joinpath(@__DIR__, "..", "sampler_transpiler", "hmc_benchmark.jl"))
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))

function main()
    bundle = ENV["BRM_BUNDLE"]
    output = ENV["HMC_OUTPUT"]
    batch_sizes = parse.(Int, split(get(ENV, "HMC_BATCHES", "4,100,1000"), ','))
    frames = split(get(ENV, "HMC_FRAMES", "noncentered,partial"), ',')
    rounds = parse(Int, get(ENV, "HMC_ROUNDS", "9"))
    steps = parse(Int, get(ENV, "HMC_STEPS", "16"))
    stepsize = parse(Float64, get(ENV, "HMC_STEPSIZE", "0.03"))
    BLAS.set_num_threads(1)
    hashfile(path) = bytes2hex(sha256(read(path)))
    hashes = Dict("noncentered.jls"=>"9f0b2513dd9762d54360c26548322e6349fa4e834691d0a92ae210d951435c50",
        "partial.jls"=>"ad667cb50ddf28faff727cc73bdf22556ddaff54d5384dd4337556a22fca9247",
        "centeredness.tsv"=>"e882c5b7a906275bbff370291c686ba8284272fdf7aa5102bccba3323468e289")
    for (file, hash) in hashes
        hashfile(joinpath(bundle, file)) == hash || error("bundle hash mismatch: $file")
    end
    rows = split.(readlines(joinpath(bundle, "centeredness.tsv"))[2:end], '\t')
    selected = vcat(parse.(Float64, getindex.(rows, 2)), parse.(Float64, getindex.(rows, 3)))
    data = BRMHSGPExample.motorcycle_data(joinpath(ROOT, "examples", "data", "mcycle.csv"))
    model_prepare_seconds = @elapsed density = BRMHSGPExample.prepare_model(data)
    backend = AutoEnzyme(; mode=Enzyme.Reverse, function_annotation=Enzyme.Const)
    make_rng(i) = BACKEND_NAME === :native ? Xoshiro(91+i) :
        Reactant.ReactantRNG(Reactant.to_rarray(UInt64[91+i, 77]))
    packages = Dict{String,Any}()
    for (_, info) in Pkg.dependencies()
        info.name in ("ReactiveKernels", "Reactant", "Enzyme", "DifferentiationInterface",
            "MutatingFunctions", "OutputSignatures", "LogExpFunctions") || continue
        entry = Dict{String,Any}("version"=>string(info.version), "source"=>something(info.source, ""))
        if info.source !== nothing && isdir(joinpath(info.source, ".git")) ||
                info.source !== nothing && isfile(joinpath(info.source, ".git"))
            entry["git_head"] = strip(read(`git -C $(info.source) rev-parse HEAD`, String))
        end
        packages[info.name] = entry
    end
    source_files = ["examples/brm_hsgp.jl", "benchmark/brm_hsgp/hmc.jl",
        "benchmark/sampler_transpiler/hmc_benchmark.jl",
        "benchmark/sampler_transpiler/density_callbacks.jl",
        "benchmark/sampler_transpiler/position_multinomial_hmc_kernel.jl",
        "benchmark/nuts_kernel_authoring_fixture_b.jl"]
    receipt = Dict{String,Any}("schema"=>"brm-hsgp-hmc-v1",
        "generated_at_utc"=>string(now(UTC)), "backend"=>string(BACKEND_NAME),
        "rk_head"=>strip(read(`git -C $ROOT rev-parse HEAD`, String)),
        "rk_worktree_status"=>read(`git -C $ROOT status --short`, String),
        "source_sha256"=>Dict(f=>hashfile(joinpath(ROOT, f)) for f in source_files),
        "bundle_sha256"=>hashes, "packages"=>packages,
        "julia_version"=>string(VERSION), "julia_threads"=>Threads.nthreads(),
        "blas_threads"=>BLAS.get_num_threads(), "cpu_model"=>Sys.cpu_info()[1].model,
        "cpu_affinity"=>match(r"Cpus_allowed_list:\s*([^\n]+)", read("/proc/self/status", String)).captures[1],
        "loadavg_start"=>strip(read("/proc/loadavg", String)),
        "model_prepare_seconds"=>model_prepare_seconds,
        "methodology"=>"Same authored multinomial HMC, fixed L and step size, matched batch sizes. Fixed diagonal mass = inverse coordinate variance of the supplied 10000-draw frame. Start at supplied column 5000. Data-only basis prepared once; q active, centeredness held by sampler context. Native uses prepared Enzyme; Reactant compiles the HMC batch including AD and RNG, synchronizes once. No adaptation or history; throughput is not ESS. Julia bytes exclude native runtime/device allocations. RNG construction and result validation outside timing; state-copy/output wrapper inside.",
        "frames"=>Dict{String,Any}())
    save() = open(output, "w") do io
        receipt["loadavg_latest"] = strip(read("/proc/loadavg", String))
        TOML.print(io, receipt; sorted=true)
    end
    for frame in frames
        frame in ("noncentered", "partial") || error("unknown frame $frame")
        posterior = deserialize(joinpath(bundle, "$frame.jls")).posterior_position
        q = copy(posterior[:, 5000])
        c = frame == "noncentered" ? zeros(40) : selected
        variance = vec(var(posterior; dims=2))
        all(v -> isfinite(v) && v > 0, variance) || error("invalid metric variance")
        metric = Diagonal(inv.(variance))
        ad_prepare_seconds = @elapsed ad = prepare_ad(density, backend, q, c; active=:q)
        first_gradient_seconds = @elapsed value, gradient = ad_value_and_gradient(ad, q, c)
        all(isfinite, gradient) && isfinite(value) || error("invalid initial gradient")
        frame_receipt = Dict{String,Any}("initial_position"=>q, "centeredness"=>c,
            "metric_diagonal"=>diag(metric), "initial_logdensity"=>value,
            "ad_prepare_seconds"=>ad_prepare_seconds,
            "first_native_gradient_seconds"=>first_gradient_seconds,
            "results"=>Dict{String,Any}())
        receipt["frames"][frame] = frame_receipt
        for transitions in batch_sizes
            println("HMC begin backend=$BACKEND_NAME frame=$frame T=$transitions L=$steps"); flush(stdout)
            result = HMCBenchmark.benchmark_hmc(density, ad, q, make_rng;
                backend=BACKEND_NAME, transitions, steps, stepsize, rounds, metric, context=(c,))
            frame_receipt["results"][string(transitions)] = result
            save()
            println("HMC result backend=$BACKEND_NAME frame=$frame T=$transitions ",
                "us_per_transition=", result["median_us_per_transition"],
                " prepare_s=", result["prepare_seconds"],
                " first_s=", result["first_execution_seconds"]); flush(stdout)
        end
    end
    save()
    println("HMC_BENCHMARK_DONE $output")
end
main()
