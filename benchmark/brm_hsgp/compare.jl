using ReactiveKernels, Reactant, BenchmarkTools, BridgeStan
using BayesianRegressionModels, Turing, StanBlocks, WarmupHMC
using DifferentiationInterface: AutoEnzyme
using Distributions, LinearAlgebra, Serialization, SHA, TOML, Statistics, Test
import Enzyme, LogDensityProblems
import Pkg

include(joinpath(@__DIR__, "..", "..", "examples", "brm_hsgp.jl"))
const DP = Turing.DynamicPPL
const LDP = LogDensityProblems
const BACKEND = AutoEnzyme(; mode=Enzyme.Reverse, function_annotation=Enzyme.Const)
const TRANSPORT_BACKEND = AutoEnzyme(;
    mode=Enzyme.set_runtime_activity(Enzyme.Reverse), function_annotation=Enzyme.Const)
const BUNDLE_HASHES = Dict(
    "noncentered.jls"=>"9f0b2513dd9762d54360c26548322e6349fa4e834691d0a92ae210d951435c50",
    "partial.jls"=>"ad667cb50ddf28faff727cc73bdf22556ddaff54d5384dd4337556a22fca9247",
    "centeredness.tsv"=>"e882c5b7a906275bbff370291c686ba8284272fdf7aa5102bccba3323468e289")

# Load the two literal @brm definitions from the independently maintained
# reproduction. No BRM density or derivative is reimplemented in this runner.
module BRMSource
using BayesianRegressionModels, Distributions
const source_path = joinpath(dirname(dirname(pathof(BayesianRegressionModels))),
    "research", "adaptive_centering", "reproduce.jl")
const source = read(source_path, String)
const model_source = split(split(source, "# BEGIN ADAPTIVE MOTORCYCLE MODEL\n")[2],
    "# END ADAPTIVE MOTORCYCLE MODEL")[1]
include_string(@__MODULE__, model_source, source_path)
end

function coordinate_names(c=zeros(40))
    mu_weights = any(!iszero, c[1:20]) ? "beta_partial" : "beta_raw"
    sigma_weights = any(!iszero, c[21:40]) ? "beta_partial" : "beta_raw"
    vcat(["hsgp_x_rho_iso", "hsgp_x_sigma"],
        ["hsgp_x_$(mu_weights).$j" for j in 1:20],
        ["hsgp_log_sigma_x_rho_iso", "hsgp_log_sigma_x_sigma"],
        ["hsgp_log_sigma_x_$(sigma_weights).$j" for j in 1:20])
end

function native_permutation(density, c)
    ranges = DP.get_all_ranges_and_transforms(density)
    mu_weights = any(!iszero, c[1:20]) ? DP.@varname(term_mu_1.beta_partial) : DP.@varname(term_mu_1.beta_raw)
    sigma_weights = any(!iszero, c[21:40]) ? DP.@varname(term_sigma_1.beta_partial) : DP.@varname(term_sigma_1.beta_raw)
    vcat(collect(ranges[DP.@varname(term_mu_1.rho)].range),
        collect(ranges[DP.@varname(term_mu_1.sigma)].range),
        collect(ranges[mu_weights].range),
        collect(ranges[DP.@varname(term_sigma_1.rho)].range),
        collect(ranges[DP.@varname(term_sigma_1.sigma)].range),
        collect(ranges[sigma_weights].range))
end

function references(data, c, output, label; native_enabled=true)
    model_data = (; data..., c_mu=c[1:20], c_sigma=c[21:40])
    brmi = BRMSource.MOTORCYCLE_PARTIAL(model_data)
    sb_seconds = @elapsed begin
        sb = SBBRMI(brmi; mod=@__MODULE__)
        stan = StanBlocks.stan_instantiate(sb.model;
            path=joinpath(output, "motorcycle-$label.stan"))
    end
    stan_names = BridgeStan.param_unc_names(stan.model)
    ps = [only(findall(==(name), stan_names)) for name in coordinate_names(c)]
    @assert sort(ps) == collect(1:44)
    native_enabled || return (; stan, ps, sb_seconds, native=nothing)
    native_seconds = @elapsed begin
        tb = TuringBRMI(brmi)
        initial(cs) = any(!iszero, cs) ?
            (; rho=0.2, sigma=0.5, beta_partial=zeros(20)) :
            (; rho=0.2, sigma=0.5, beta_raw=zeros(20))
        parameters = (; term_mu_1=initial(c[1:20]), term_sigma_1=initial(c[21:40]))
        vi = DP.VarInfo(tb.model, DP.InitFromParams(parameters), DP.LinkAll())
        density = DP.LogDensityFunction(tb.model, DP.getlogjoint_internal, vi)
        native = adaptive_centering_problem(tb, density, TRANSPORT_BACKEND)
    end
    pn = native_permutation(density, c)
    @assert sort(pn) == collect(1:44)
    (; stan, ps, native, density, pn, sb_seconds, native_seconds,
       turing_source_sha256=bytes2hex(sha256(turing_model_source(tb))))
end

function selected_centeredness(bundle)
    rows = split.(readlines(joinpath(bundle, "centeredness.tsv"))[2:end], '\t')
    @assert parse.(Int, getindex.(rows, 1)) == 1:20
    vcat(parse.(Float64, getindex.(rows, 2)), parse.(Float64, getindex.(rows, 3)))
end

function transport(q, c_from, c_to)
    result = copy(q)
    for (offset, coffset) in ((0, 0), (22, 20)), j in 1:20
        logs = q[offset+2] + q[offset+1]/2 + log(2pi)/4 -
            exp(2q[offset+1]) * (j*pi/3)^2 / 4
        result[offset+2+j] *= exp((c_to[coffset+j]-c_from[coffset+j])*logs)
    end
    result
end

function adversarial_points(c)
    points = Vector{Float64}[]
    for (i, log_rho) in enumerate((-6.0, -4.0, -2.0, -0.5, 0.0, 0.2)),
        log_sd in (-12.0, -2.0, 0.7)
        q = 0.03 .* sin.((1:44) .* (i + 0.5))
        q[1], q[23] = log_rho, log_rho - 0.1
        q[2], q[24] = log_sd, -2.5
        push!(points, transport(q, zeros(40), c))
    end
    reduce(hcat, points)
end

function check_points(compiled, primal, refs, points, c; label)
    rc = Reactant.to_rarray(c)
    max_value_sb = max_gradient_sb = max_value_native = max_gradient_native = 0.0
    max_scaled_gradient_sb = max_scaled_gradient_native = 0.0
    qsb, qnative = zeros(44), zeros(44)
    for (i, q) in enumerate(eachcol(points))
        qsb[refs.ps] .= q
        sv, sg = BridgeStan.log_density_gradient(refs.stan.model, qsb;
            propto=false, jacobian=true)
        rq = Reactant.to_rarray(collect(q))
        rv, rg = compiled(rq, rc)
        v, g = Float64(rv), Array(rg)
        pv = Float64(primal(rq, rc))
        @assert isapprox(pv, sv; atol=2e-8, rtol=2e-10) "$label point $i compiled primal mismatch"
        @assert all(isfinite, (sv, v)) && all(isfinite, g) &&
            all(isfinite, sg) "$label point $i non-finite"
        @assert isapprox(v, sv; atol=2e-8, rtol=2e-10) "$label point $i Stan value: $v != $sv"
        es = maximum(abs.(g .- sg[refs.ps]))
        rs = maximum(abs.(g .- sg[refs.ps]) ./ (1 .+ abs.(sg[refs.ps])))
        @assert rs < 2e-7 "$label point $i Stan gradient scaled error: $rs"
        max_value_sb = max(max_value_sb, abs(v-sv))
        max_gradient_sb = max(max_gradient_sb, es)
        max_scaled_gradient_sb = max(max_scaled_gradient_sb, rs)
        if refs.native !== nothing
            qnative[refs.pn] .= q
            nv, ng = LDP.logdensity_and_gradient(refs.native, qnative)
            # Keep the generated DynamicPPL target as a separate value control.
            dv = LDP.logdensity(refs.density, qnative)
            @assert all(isfinite, (nv, dv)) && all(isfinite, ng)
            @assert isapprox(nv, dv; atol=2e-8, rtol=2e-10)
            @assert isapprox(v, nv; atol=2e-8, rtol=2e-10) "$label point $i native value: $v != $nv"
            en = maximum(abs.(g .- ng[refs.pn]))
            rn = maximum(abs.(g .- ng[refs.pn]) ./ (1 .+ abs.(ng[refs.pn])))
            @assert rn < 2e-7 "$label point $i native gradient scaled error: $rn"
            max_value_native = max(max_value_native, abs(v-nv))
            max_gradient_native = max(max_gradient_native, en)
            max_scaled_gradient_native = max(max_scaled_gradient_native, rn)
        end
    end
    result = Dict{String,Any}("points"=>size(points,2), "stan_value_max_abs"=>max_value_sb,
        "stan_gradient_max_abs"=>max_gradient_sb, "stan_gradient_max_scaled"=>max_scaled_gradient_sb,
        "native_verified"=>refs.native !== nothing)
    if refs.native !== nothing
        merge!(result, Dict("native_value_max_abs"=>max_value_native,
            "native_gradient_max_abs"=>max_gradient_native,
            "native_gradient_max_scaled"=>max_scaled_gradient_native))
    end
    result
end

function measurement(f)
    f()
    trial = @benchmark $f() seconds=1 samples=1000 evals=1
    Dict("median_ns"=>median(trial).time, "minimum_ns"=>minimum(trial).time,
        "bytes"=>trial.memory, "allocations"=>trial.allocs, "samples"=>length(trial))
end

function runtimes(compiled, primal, kernel, prepared, refs, q, c)
    rq, rc = Reactant.to_rarray(q), Reactant.to_rarray(c)
    qs = q[invperm(refs.ps)]
    resident = measurement(() -> compiled(rq, rc)) # sync=true is the public default
    host = measurement() do
        v, g = compiled(Reactant.to_rarray(q), rc)
        Float64(v), Array(g)
    end
    sb = measurement(() -> BridgeStan.log_density_gradient(refs.stan.model, qs;
        propto=false, jacobian=true))
    grad_buffer = similar(q)
    native_ad = measurement(() -> ad_value_and_gradient!(prepared, grad_buffer, q, c))
    native_primal = measurement(() -> kernel(q, c))
    reactant_primal = measurement(() -> primal(rq, rc))
    result = Dict{String,Any}("reactant_resident"=>resident, "reactant_host"=>host, "stanblocks"=>sb,
        "rk_native_value_gradient"=>native_ad, "rk_native_primal"=>native_primal,
        "reactant_primal"=>reactant_primal,
        "resident_over_stan"=>resident["median_ns"]/sb["median_ns"],
        "host_over_stan"=>host["median_ns"]/sb["median_ns"])
    if refs.native !== nothing
        qn = q[invperm(refs.pn)]
        native = measurement(() -> LDP.logdensity_and_gradient(refs.native, qn))
        result["native_turing"] = native
        result["native_over_stan"] = native["median_ns"]/sb["median_ns"]
    end
    result
end

function main(bundle, output; limit=typemax(Int), native_enabled=true)
    mkpath(output)
    limit > 0 || error("point limit must be positive")
    for (file, expected) in BUNDLE_HASHES
        bytes2hex(sha256(read(joinpath(bundle,file)))) == expected || error("bundle hash mismatch: $file")
    end
    BLAS.set_num_threads(1)
    data = BRMHSGPExample.motorcycle_data(joinpath(@__DIR__, "..", "..", "examples", "data", "mcycle.csv"))
    prep_seconds = @elapsed kernel = BRMHSGPExample.prepare_model(data)
    q = zeros(44); q[[1,23]] .= -2
    c = zeros(40)
    ad_seconds = @elapsed prepared = prepare_ad(kernel, BACKEND, q, c; active=:q)
    native_first_seconds = @elapsed ad_value_and_gradient!(prepared, similar(q), q, c)
    rq, rc = Reactant.to_rarray(q), Reactant.to_rarray(c)
    primal_compile_seconds = @elapsed primal = Reactant.@compile sync=true kernel(rq, rc)
    compile_seconds = @elapsed compiled = compile_ad_value_and_gradient(prepared, rq, rc)
    first_seconds = @elapsed compiled(rq, rc)
    result = Dict{String,Any}("kernel_preparation_seconds"=>prep_seconds,
        "ad_preparation_seconds"=>ad_seconds, "reactant_compilation_seconds"=>compile_seconds,
        "reactant_primal_compilation_seconds"=>primal_compile_seconds,
        "native_ad_first_execution_seconds"=>native_first_seconds,
        "reactant_first_execution_seconds"=>first_seconds,
        "julia_version"=>string(VERSION), "data_sha256"=>BRMHSGPExample.MCYCLE_SHA256,
        "brm_model_source_sha256"=>bytes2hex(sha256(BRMSource.model_source)),
        "coordinate_names"=>coordinate_names(), "blas_threads"=>BLAS.get_num_threads(),
        "bundle_hashes"=>BUNDLE_HASHES,
        "posterior_provenance"=>TOML.parsefile(joinpath(bundle,"provenance.toml")),
        "benchmark_source_sha256"=>bytes2hex(sha256(read(@__FILE__))),
        "kernel_source_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"..","..","examples","brm_hsgp.jl")))))
    relevant = Set(("ReactiveKernels", "BayesianRegressionModels", "StanBlocks", "WarmupHMC",
        "Reactant", "Reactant_jll", "Enzyme", "Enzyme_jll", "DifferentiationInterface",
        "DynamicPPL", "Turing", "Distributions", "BridgeStan", "MutatingFunctions",
        "OutputSignatures", "LogDensityProblems", "BenchmarkTools"))
    result["packages"] = [Dict("name"=>p.name, "version"=>string(p.version),
        "tree_hash"=>string(p.tree_hash), "source"=>p.source,
        "git_sha"=>ispath(joinpath(p.source,".git")) ?
            strip(read(`git -C $(p.source) rev-parse HEAD`,String)) : "")
        for p in sort(collect(values(Pkg.dependencies())); by=p->p.name) if p.name in relevant]
    selected = selected_centeredness(bundle)
    ncp = deserialize(joinpath(bundle, "noncentered.jls")).posterior_position
    partial = deserialize(joinpath(bundle, "partial.jls")).posterior_position
    @assert size(ncp) == size(partial) == (44, 10000)
    # Draw matrices are in BridgeStan order, validated by reference metadata.
    for (label, controls, source, source_c) in (
        ("noncentered", zeros(40), ncp, zeros(40)),
        ("selected_partial", selected, partial, selected),
        ("mixed", repeat([0.0, 0.25, 0.7, 1.0],10), ncp, zeros(40)),
        ("centered", ones(40), ncp, zeros(40)))
        println("prepare_reference\t", label); flush(stdout)
        refs = references(data, controls, output, label; native_enabled)
        n = min(size(source,2), limit)
        points = reduce(hcat, (transport(source[refs.ps,i], source_c, controls) for i in 1:n))
        println("compare\t", label, "\t", n); flush(stdout)
        row = check_points(compiled, primal, refs, points, controls; label)
        row["adversarial"] = check_points(compiled, primal, refs,
            adversarial_points(controls), controls; label=label * " adversarial")
        row["stan_preparation_seconds"] = refs.sb_seconds
        row["stan_permutation"] = refs.ps
        row["stan_coordinate_names"] = coordinate_names(controls)
        row["source_coordinate_names"] = coordinate_names(source_c)
        if native_enabled
            row["native_preparation_seconds"] = refs.native_seconds
            row["native_turing_source_sha256"] = refs.turing_source_sha256
            row["native_permutation"] = refs.pn
        end
        row["centeredness"] = controls
        row["runtime"] = runtimes(compiled, primal, kernel, prepared, refs, points[:,cld(n,2)], controls)
        result[label] = row
        open(joinpath(output,"receipt.toml"), "w") do io
            TOML.print(io, result; sorted=true)
        end
        println("verified\t", label, "\t", row); flush(stdout)
    end
    result
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 2 || error("usage: compare.jl BUNDLE_DIR OUTPUT_DIR [POINT_LIMIT]")
    main(ARGS[1], ARGS[2]; limit=length(ARGS)>2 ? parse(Int,ARGS[3]) : typemax(Int),
        native_enabled=get(ENV,"RK_HSGP_NATIVE","1") == "1")
end
