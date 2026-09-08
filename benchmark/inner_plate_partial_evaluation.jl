# Focused acceptance/measurement driver. Run in a consumer environment containing
# RK, its distribution package, BenchmarkTools, SpecialFunctions, and Enzyme;
# add Reactant for RK_INNER_PE_BACKEND=reactant. Output belongs in task scratch.
const load_started = time()
using ReactiveKernels, BenchmarkTools, SpecialFunctions, TOML, SHA, Test, Pkg
import ReactiveKernelsDistributionKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources: binomial
using DifferentiationInterface: AutoEnzyme
import Enzyme

const backend_name = get(ENV, "RK_INNER_PE_BACKEND", "native")
backend_name in ("native", "reactant") || error("unknown backend: $backend_name")
reactant_loaded() = any(id -> id.name == "Reactant", keys(Base.loaded_modules))
if backend_name == "reactant"
    @eval using Reactant
    @assert Base.get_extension(ReactiveKernels, :ReactiveKernelsReactantExt) !== nothing
end
backend_name == "native" && reactant_loaded() && error(
    "native-only measurements require a fresh Julia process without Reactant loaded")
const package_load_s = time() - load_started
const ad_backend = AutoEnzyme(; mode = Enzyme.Reverse)
const root = dirname(@__DIR__)
include("_repro_guard.jl")
const candidate_sha = _require_clean_detached_candidate(root)
realpath(dirname(dirname(pathof(ReactiveKernels)))) == realpath(root) ||
    error("benchmark driver and loaded ReactiveKernels must use the same exact checkout")
realpath(dirname(dirname(pathof(ReactiveKernelsDistributionKernels)))) ==
    realpath(joinpath(root, "packages", "ReactiveKernelsDistributionKernels")) ||
    error("distribution source must come from the same exact candidate checkout")

function dependency_receipts()
    Dict(string(uuid) => Dict(
        "name" => info.name, "version" => string(info.version),
        "tree_hash" => something(info.tree_hash, ""),
        "git_revision" => something(info.git_revision, ""),
        "source" => info.source,
        "path_sha" => info.is_tracking_path ?
            _require_clean_detached_candidate(info.source) : "")
        for (uuid, info) in Pkg.dependencies())
end

@kernel normalizer(q::Vector{Float64}, observed::Vector{Int}, trials::Vector{Int}) = begin
    parameter::Float64 = sum(q)
    pointwise = plate(observed, trials, parameter) do y, n, theta
        log_choose::Float64 = loggamma(n + 1.0) - loggamma(y + 1.0) - loggamma(n - y + 1.0)
        result::Float64 = log_choose * theta
        result
    end
    total::Float64 = sum(pointwise)
end

@kernel binomial_plate(q::Vector{Float64}, observed::Vector{Int}, trials::Vector{Int}) = begin
    probability::Float64 = 1 / (1 + exp(-sum(q)))
    pointwise = plate(observed, trials, probability) do y, n, p
        binomial(n, p).logpdf(y)
    end
    total::Float64 = sum(pointwise)
end

@kernel cheap(q::Vector{Float64}, data::Vector{Float64}, a::Float64, b::Float64) = begin
    parameter::Float64 = sum(q)
    pointwise = plate(data, a, b, parameter) do x, scale, offset, theta
        transformed::Float64 = scale * x + offset
        result::Float64 = transformed * theta
        result
    end
    total::Float64 = sum(pointwise)
end

# Consume the existing source authority and surgical example unchanged, without
# loading every unrelated PPL example as part of this bounded acceptance driver.
module ReactiveKernelsPPLExamples
using ReactiveKernels: KernelSpec, PreparedKernel
include("../packages/ReactiveKernelsPPLExamples/src/_ppl_source_authority.jl")
include("../packages/ReactiveKernelsPPLExamples/src/surgical.jl")
end

invoke_kernel(kernel, args) = kernel(args...)
invoke_gradient(prepared, buffer, args) = ad_value_and_gradient!(prepared, buffer, args...)
function measure(f, args...)
    trial = @benchmark $f($args...) seconds=0.2 samples=1000 evals=1
    result = minimum(trial)
    (; ns = result.time, bytes = result.memory, allocations = result.allocs)
end

function dump_inner(label, p)
    println("INNER_PLAN ", label)
    for r in p.recipes
        r.op isa ReactiveKernels._AuthoredPlateOp || continue
        println("plate ", only(r.outputs).name, " inputs=", map(v -> v.name, r.inputs))
        for cell in plate_body(r).recipes
            println("  ", map(v -> v.name, cell.outputs), " <- ",
                    map(v -> v.name, cell.inputs), " source=", cell.source)
        end
    end
end

# These are operand-byte footprints, not hardware DRAM counters. Axes-only
# operands retain storage but cause no scalar loads in the generated loop.
function used_inputs(p)
    used = Set{Int}()
    for r in p.recipes
        if r.op isa ReactiveKernels._AuthoredPlateOp
            inner = plate_body(r)
            needed = Set(canon_id(inner.graph, v.id)
                         for cell in inner.recipes for v in cell.inputs)
            union!(needed, (canon_id(inner.graph, v.id) for v in inner.want))
            for (v, outer) in zip(inner.have, r.inputs)
                canon_id(inner.graph, v.id) in needed &&
                    push!(used, canon_id(p.graph, outer.id))
            end
        else
            union!(used, (canon_id(p.graph, v.id) for v in r.inputs))
        end
    end
    used
end

function run_case(label, spec, q, data; active = :q, want = :total)
    backend_name == "native" && reactant_loaded() && error(
        "Reactant was loaded into the native-only measurement process")
    reference_value = nothing
    reference_gradient = nothing
    results = Dict[]
    original = plan(spec; want)
    original_used = used_inputs(original)
    dump_inner(label * " original", original)
    for bound in (false, true)
        args = bound ? (q,) : (q, values(data)...)
        prep = @timed prepare(spec; want, bound = bound ? data : NamedTuple())
        kernel = prep.value
        first_call = @timed kernel(args...)
        ad_prep = @timed prepare_ad(kernel, ad_backend, args...; active)
        prepared = ad_prep.value
        gradient = similar(q)
        first_ad = @timed ad_value_and_gradient!(prepared, gradient, args...)
        value = first_call.value
        if !bound
            reference_value, reference_gradient = value, copy(gradient)
        else
            @test value ≈ reference_value rtol=2e-12 atol=1e-10
            @test gradient ≈ reference_gradient rtol=2e-11 atol=1e-10
            dump_inner(label * " residual", kernel.plan)
        end
        constants = [r for r in kernel.plan.recipes if r.op isa ReactiveKernels._BoundConstant]
        caches = [r for r in constants if startswith(String(only(r.outputs).name), "bound_plate_")]
        cache_bytes = sum(sizeof(r.op.value) for r in caches if r.op.value isa AbstractArray; init=0)
        residual_used = used_inputs(kernel.plan)
        eliminated = sum(sizeof(value) for (name, value) in pairs(data)
            if value isa AbstractArray && canon_id(original.graph, port(spec, name).id) in original_used &&
               !(canon_id(original.graph, port(spec, name).id) in residual_used); init=0)
        row = Dict{String,Any}(
            "case" => label, "bound" => bound, "backend" => backend_name,
            "host_context" => reactant_loaded() ? "reactant_loaded" : "native_only",
            "prepare_s" => prep.time, "prepare_bytes" => prep.bytes,
            "native_first_call_s" => first_call.time,
            "ad_prepare_s" => ad_prep.time, "ad_first_call_s" => first_ad.time,
            "value" => value, "gradient" => copy(gradient),
            "cache_array_bytes" => cache_bytes, "raw_array_bytes_eliminated" => eliminated,
            "net_array_operand_bytes" => cache_bytes - eliminated,
            "retained_constant_bytes" => Base.summarysize(Tuple(r.op.value for r in constants)))
        if backend_name == "native"
            primal = measure(invoke_kernel, kernel, args)
            derivative = measure(invoke_gradient, prepared, gradient, args)
        else
            traced = map(Reactant.to_rarray, args)
            compilation = @timed Reactant.compile(kernel, traced; sync=true)
            ad_compilation = @timed compile_ad_value_and_gradient(prepared, traced...; sync=true)
            compiled, compiled_ad = compilation.value, ad_compilation.value
            @test Float64(compiled(traced...)) ≈ value rtol=2e-11 atol=1e-9
            rv, rg = compiled_ad(traced...)
            @test Float64(rv) ≈ value rtol=2e-11 atol=1e-9
            @test Array(rg) ≈ gradient rtol=2e-10 atol=1e-9
            row["reactant_compile_s"] = compilation.time
            row["reactant_ad_compile_s"] = ad_compilation.time
            primal = measure(invoke_kernel, compiled, traced)
            derivative = measure(invoke_kernel, compiled_ad, traced)
        end
        for (prefix, measurement) in (("primal", primal), ("value_gradient", derivative))
            for (key, metric) in pairs(measurement)
                row[prefix * "_" * String(key)] = metric
            end
        end
        println("MEASUREMENT ", row)
        push!(results, row)
    end
    results
end

function main()
    dependencies = dependency_receipts()
    rows = Dict[]
    sizes = parse.(Int, split(get(ENV, "RK_INNER_PE_SIZES", "12,4096,100000"), ','))
    for n in sizes
        observed = [mod(i, 7) for i in 1:n]
        trials = [20 + mod(i, 13) for i in 1:n]
        data = collect(range(0.1, 2.0; length=n))
        append!(rows, run_case("normalizer/$n", normalizer, [0.3], (; observed, trials)))
        append!(rows, run_case("binomial/$n", binomial_plate, [0.3], (; observed, trials)))
        append!(rows, run_case("affine/$n", cheap, [0.3], (; data, a=1.2, b=0.4)))
    end
    surgical = ReactiveKernelsPPLExamples.SurgicalExample.evaluate_surgical_source()
    @test surgical.source == strip(ReactiveKernelsPPLExamples.SurgicalExample.SURGICAL_SOURCE, '\n')
    append!(rows, Base.invokelatest(run_case, "surgical/12", surgical.model,
        surgical.inputs.q, (; successes=surgical.inputs.successes, totals=surgical.inputs.totals);
        active=:unconstrained, want=:posterior))
    dependency_receipts() == dependencies || error(
        "dependency identities changed during the measurement process")
    output = Dict("julia" => string(VERSION), "package_load_s" => package_load_s,
        "candidate_sha" => candidate_sha, "dependencies" => dependencies,
        "threads" => Threads.nthreads(), "cpu" => Sys.CPU_NAME,
        "source_sha256" => bytes2hex(sha256(read(@__FILE__))), "rows" => rows)
    path = get(ENV, "RK_INNER_PE_OUTPUT", "")
    isempty(path) || open(io -> TOML.print(io, output), path; write=true)
    println("INNER_PLATE_PE_COMPLETE rows=", length(rows))
end

main()
