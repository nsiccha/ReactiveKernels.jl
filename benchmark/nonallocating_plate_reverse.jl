# Synthetic grouped Gaussian and matrix-regression controls. Run in an
# environment with this checkout, its PPL package, Enzyme and BenchmarkTools.
# RK_COMPOSE_SIZES selects runtime observation counts (default: 2000,100000).
using ReactiveKernels, ReactiveKernelsPPL, MutatingFunctions
using DifferentiationInterface, Enzyme, LinearAlgebra, BenchmarkTools, Statistics
BLAS.set_num_threads(1)
const cost_failures = Ref(0)
macro phase_time(ex)
    quote
        Base.Experimental.@force_compile
        Base.cumulative_compile_timing(true)
        local before = Base.cumulative_compile_time_ns()
        local measurement
        local compilation
        try
            measurement = @timed $(esc(ex))
        finally
            compilation = Base.cumulative_compile_time_ns() .- before
            Base.cumulative_compile_timing(false)
        end
        (; measurement..., compile_time = first(compilation) / 1e9,
           recompile_time = last(compilation) / 1e9)
    end
end
module ComposeCostModels end
const models = (
    grouped = quote
        a ~ Normal(0, 5)
        b ~ Normal(0, 1)
        sd ~ Exponential(1)
        sigma ~ Exponential(1)
        z[1:J] .~ Normal.(0, 1)
        mu = a .+ b .* x .+ sd .* z[g]
        y .~ Normal.(mu, sigma)
    end,
    regression = quote
        a ~ Normal(0, 5)
        sigma ~ Exponential(1)
        beta[axes(X, 2)] .~ Normal.(0, 1)
        mu = a .+ X * beta
        y .~ Normal.(mu, sigma)
    end,
)
function run_costs(name, N)
    data = if name === :grouped
        (; y = [sin(i) for i in 1:N], x = [cos(i) for i in 1:N],
           g = [mod1(i, 50) for i in 1:N], J = 50)
    else
        (; y = [sin(i) for i in 1:N], X = [cos(i + j) for i in 1:N, j in 1:8])
    end
    lowering = @phase_time begin
        plan = lower_rkppl(getproperty(models, name), keys(data);
                          mod = ComposeCostModels, conditioned = keys(data))
        bound = bind_data(plan, data)
        built = build_kernel(bound)
        built, bound
    end
    built, bound = lowering.value
    u = [0.1 * sin(i) for i in 1:built.layout.total]
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    println((stage = :build, case = name, N, seconds = lowering.time,
             bytes = lowering.bytes, compile_seconds = lowering.compile_time))
    flush(stdout)
    reference_value = nothing
    reference_gradient = nothing
    variants = []
    for nonallocating in (false, true), enabled in (false, true)
        try
            preparation = @phase_time with_plate_reverse(enabled) do
                prepare_sampler(built, bound, u; backend, nonallocating)
            end
            query = preparation.value
            gradient = similar(u)
            first_gradient = @phase_time sampler_value_and_gradient!(query, gradient, u)
            push!(variants, (; nonallocating, enabled, query, gradient))
            if reference_value === nothing
                reference_value = first(first_gradient.value)
                reference_gradient = copy(gradient)
            else
                @assert isapprox(first(first_gradient.value), reference_value; rtol = 1e-12)
                @assert isapprox(gradient, reference_gradient; rtol = 1e-9, atol = 1e-10)
            end
            second_gradient = @phase_time sampler_value_and_gradient!(query, gradient, u)
            trial = @benchmark sampler_value_and_gradient!($query, $gradient, $u) samples=30 evals=1 seconds=1
            best = minimum(trial)
            println((stage = :gradient, case = name, N, nonallocating, enabled,
                preparation_seconds = preparation.time, preparation_bytes = preparation.bytes,
                preparation_compile_seconds = preparation.compile_time,
                first_seconds = first_gradient.time, first_compile_seconds = first_gradient.compile_time,
                second_seconds = second_gradient.time, ns = best.time, bytes = best.memory,
                allocations = best.allocs, value = first(first_gradient.value),
                gradient_sum = sum(gradient), report = plate_reverse_report(query.kernel)))
            if N == 2_000
                point, contexts = ReactiveKernels._ad_prepared_arguments(
                    query.ad, (u,), NamedTuple())
                program_findings = ReactiveKernels._code_findings(query.ad.call,
                    ReactiveKernels._ad_call_argument_types(point, contexts);
                    recurse = ReactiveKernels._code_recursion(Val(:all)))
                println((stage = :program_inference_audit, case = name,
                         nonallocating, enabled, findings = length(program_findings)))
                foreach(finding -> println(repr(finding)), program_findings)
                @assert isempty(program_findings)
                findings = ReactiveKernels._code_findings(sampler_value_and_gradient!,
                    (typeof(query), typeof(gradient), typeof(u));
                    recurse = ReactiveKernels._code_recursion(Val(:all)))
                println((stage = :inference_audit, case = name, nonallocating,
                         enabled, findings = length(findings)))
                foreach(finding -> println(repr(finding)), findings)
                hot_findings = ReactiveKernels._code_findings(ReactiveKernelsPPL._sampler_gradient!,
                    (typeof(query), typeof(gradient), typeof(u));
                    recurse = ReactiveKernels._code_recursion(Val(:all)))
                println((stage = :hot_inference_audit, case = name, nonallocating,
                         enabled, findings = length(hot_findings)))
                foreach(finding -> println(repr(finding)), hot_findings)
            end
        catch error
            cost_failures[] += 1
            println((stage = :error, case = name, N, nonallocating, enabled, type = typeof(error)))
            showerror(stdout, error, catch_backtrace()); println()
        end
        flush(stdout)
    end
    # Every variant is warm before these rounds. Keep all samples' summaries
    # and rotate/reverse execution order so comparisons can be paired by round.
    if length(variants) == 4
        for round in 1:12
            order = circshift(collect(eachindex(variants)), round - 1)
            iseven(round) && reverse!(order)
            for index in order
                variant = variants[index]
                query, gradient = variant.query, variant.gradient
                trial = @benchmark sampler_value_and_gradient!($query, $gradient, $u) samples=30 evals=1 seconds=1
                estimate = median(trial)
                println((stage = :paired_gradient, case = name, N, round,
                    nonallocating = variant.nonallocating, enabled = variant.enabled,
                    ns = estimate.time, bytes = estimate.memory, allocations = estimate.allocs,
                    samples_ns = trial.times))
            end
        end
    end
    flush(stdout)
end
println((julia = VERSION, enzyme = pkgversion(Enzyme), rk = pkgversion(ReactiveKernels),
         rk_path = pathof(ReactiveKernels), bounds = Base.JLOptions().check_bounds,
         enzyme_path = pathof(Enzyme),
         library = Enzyme.Compiler.Enzyme_jll.libEnzyme_path))
for N in parse.(Int, split(get(ENV, "RK_COMPOSE_SIZES", "2000,100000"), ',')), name in keys(models)
    run_costs(name, N)
end
println((stage = :complete, failures = cost_failures[]))
exit(cost_failures[] == 0 ? 0 : 1)
