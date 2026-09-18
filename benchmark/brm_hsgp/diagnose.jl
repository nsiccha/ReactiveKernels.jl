using ReactiveKernels, Reactant, MutatingFunctions, BenchmarkTools
using DifferentiationInterface: AutoEnzyme
using LinearAlgebra, Statistics, Serialization, TOML, SHA, Profile
import Enzyme

include(joinpath(@__DIR__, "..", "..", "examples", "brm_hsgp.jl"))
const BACKEND = AutoEnzyme(; mode=Enzyme.Reverse, function_annotation=Enzyme.Const)

function measure(f)
    f()
    trial = @benchmark $f() seconds=2 samples=100000 evals=1
    Dict("median_ns"=>median(trial).time, "minimum_ns"=>minimum(trial).time,
        "bytes"=>trial.memory, "allocations"=>trial.allocs, "samples"=>length(trial))
end

# Same input/output shapes as the model, but negligible mathematical work.
# This is a control for the synchronous call boundary, not an empty/no-op call.
cheap_primal(q, c) = sum(abs2, q) + sum(c)
cheap_value_gradient(q, c) = (cheap_primal(q, c), 2 .* q)

function profile_native(f, output)
    f()
    Profile.clear()
    Profile.@profile for _ in 1:100000
        f()
    end
    open(joinpath(output, "native-enzyme-profile.txt"), "w") do io
        Profile.print(io; format=:flat, sortedby=:count, mincount=5, C=true)
    end
    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate=1.0 begin
        for _ in 1:20
            f()
        end
    end
    allocations = Profile.Allocs.fetch().allocs
    open(joinpath(output, "native-enzyme-allocations.txt"), "w") do io
        for a in allocations
            println(io, "allocation ", a.size, " bytes; type ", a.type)
            for frame in a.stacktrace
                println(io, "  ", frame)
            end
        end
    end
    Dict("calls"=>20, "observed_allocations"=>length(allocations),
        "observed_bytes"=>sum(a.size for a in allocations))
end

function main(bundle, output)
    mkpath(output)
    BLAS.set_num_threads(1)
    data = BRMHSGPExample.motorcycle_data(joinpath(@__DIR__, "..", "..", "examples", "data", "mcycle.csv"))
    source = deserialize(joinpath(bundle, "noncentered.jls")).posterior_position
    q, c = copy(source[:,5000]), zeros(40)
    kernel = BRMHSGPExample.prepare_model(data)
    prepared = prepare_ad(kernel, BACKEND, q, c; active=:q)
    gradient = similar(q)
    native = () -> ad_value_and_gradient!(prepared, gradient, q, c)
    reference_value, reference_gradient = native()
    reference_gradient = copy(reference_gradient)
    rq, rc = Reactant.to_rarray(q), Reactant.to_rarray(c)
    primal = Reactant.@compile sync=true kernel(rq, rc)
    compiled = compile_ad_value_and_gradient(prepared, rq, rc)
    control_primal = Reactant.@compile sync=true cheap_primal(rq, rc)
    control_gradient = Reactant.@compile sync=true cheap_value_gradient(rq, rc)
    affinity = match(r"Cpus_allowed_list:\s*([^\n]+)", read("/proc/self/status", String)).captures[1]
    result = Dict{String,Any}("julia_version"=>string(VERSION), "cpu_affinity"=>affinity,
        "source_sha256"=>bytes2hex(sha256(read(@__FILE__))),
        "kernel_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__, "..", "..", "examples", "brm_hsgp.jl")))),
        "posterior_column"=>5000, "blas_threads"=>BLAS.get_num_threads())
    result["dynamic_centeredness"] = Dict(
        "native_primal"=>measure(() -> kernel(q, c)),
        "native_value_gradient"=>measure(native),
        "reactant_primal"=>measure(() -> primal(rq, rc)),
        "reactant_value_gradient"=>measure(() -> compiled(rq, rc)))
    result["cheap_control"] = Dict(
        "reactant_primal"=>measure(() -> control_primal(rq, rc)),
        "reactant_value_gradient"=>measure(() -> control_gradient(rq, rc)))
    result["native_allocation_profile"] = profile_native(native, output)
    println("dynamic and call controls complete"); flush(stdout)
    open(joinpath(output,"receipt.toml"),"w") do io
        TOML.print(io, result; sorted=true)
    end

    # Hold the same NCP point fixed, but bind c too. This measures specialization
    # without changing the model or the public live-centeredness example.
    fixed = prepare(BRMHSGPExample.model;
        have=(:q, :c, :x, :y, :modes, :half_width), want=:posterior,
        bound=(; data..., modes=collect(1.0:20.0), half_width=1.5, c))
    fixed_ad = prepare_ad(fixed, BACKEND, q; active=:q)
    fixed_gradient = similar(q)
    fv, fg = ad_value_and_gradient!(fixed_ad, fixed_gradient, q)
    @assert isapprox(fv, reference_value; atol=2e-9, rtol=2e-10)
    @assert isapprox(fg, reference_gradient; atol=2e-9, rtol=2e-10)
    fixed_primal = Reactant.@compile sync=true fixed(rq)
    fixed_compiled = compile_ad_value_and_gradient(fixed_ad, rq)
    cv, cg = fixed_compiled(rq)
    @assert isapprox(Float64(cv), reference_value; atol=2e-9, rtol=2e-10)
    @assert isapprox(Array(cg), reference_gradient; atol=2e-9, rtol=2e-10)
    result["bound_centeredness"] = Dict(
        "native_primal"=>measure(() -> fixed(q)),
        "native_value_gradient"=>measure(() -> ad_value_and_gradient!(fixed_ad, fixed_gradient, q)),
        "reactant_primal"=>measure(() -> fixed_primal(rq)),
        "reactant_value_gradient"=>measure(() -> fixed_compiled(rq)))
    write(joinpath(output, "dynamic-primal.mlir"), String(Reactant.@code_hlo optimize=true kernel(rq, rc)))
    write(joinpath(output, "bound-primal.mlir"), String(Reactant.@code_hlo optimize=true fixed(rq)))
    println("bound-centeredness control complete"); flush(stdout)

    # A separate primal control measures RK's reusable-buffer execution path.
    nonallocating = prepare_nonallocating(kernel)
    @assert isapprox(nonallocating(q,c), reference_value; atol=2e-9, rtol=2e-10)
    result["reusable_buffer_primal"] = measure(() -> nonallocating(q,c))
    open(joinpath(output,"receipt.toml"),"w") do io
        TOML.print(io, result; sorted=true)
    end
    println("diagnostics complete")
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 || error("usage: diagnose.jl BUNDLE_DIR OUTPUT_DIR")
    main(ARGS...)
end
