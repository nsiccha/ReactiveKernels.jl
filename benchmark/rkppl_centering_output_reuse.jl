# Compare the current centering adapter with its former output-copy recipe in
# one process. Run in the optional centering-test consumer environment. The
# cheap target isolates centering cost; these numbers are not sampler timings.
using ReactiveKernels, ReactiveKernelsPPL, WarmupHMC, Enzyme, Test, Statistics
using DifferentiationInterface: AutoEnzyme
import LogDensityProblems as LDP

const PPL = ReactiveKernelsPPL
const WH = WarmupHMC
const EXT = Base.get_extension(PPL, :ReactiveKernelsPPLWarmupHMCExt)

const MODEL = @rkppl begin
    phi[1:N] .~ Normal.(0, 1)
    z[1:N] .~ Normal.(0, 1)
    energy = sum(phi .* phi) / N
    location = energy .+ offsets
    log_scale = energy / 2
    total = location .+ exp(log_scale) .* z
    y .~ Normal.(total, se)
end

struct Quadratic
    dimension::Int
end
LDP.dimension(p::Quadratic) = p.dimension
LDP.capabilities(::Type{Quadratic}) = LDP.LogDensityOrder{1}()
LDP.logdensity(::Quadratic, x) = -sum(abs2, x) / 2
LDP.logdensity_and_gradient(p::Quadratic, x) = (LDP.logdensity(p, x), -x)

# Historical baseline: normalize every array by copying all of its elements.
copied_values(value::Real) = Float64[value]
copied_values(value::AbstractArray{<:Real}) = collect(Float64, vec(value))
function copied_kernel(built, plan, name)
    spec = ReactiveKernels._copy_cell_factor_spec(built.spec)
    output = value!(spec.graph, PPL._CENTERING_VALUE, Vector{Float64})
    spec.ports[PPL._CENTERING_VALUE] = output
    push!(spec.port_order, PPL._CENTERING_VALUE)
    add!(spec.graph, (spec[name],) => output, copied_values)
    Base.invokelatest(PPL._prepare_cached, plan, spec;
        have = PPL._query_have(plan), want = output,
        bound = PPL._query_bound(plan), on_error = nothing)
end

function elapsed_per_call(f, repetitions)
    (@elapsed for _ in 1:repetitions
        f()
    end) / repetitions
end

# Rotate order so the two versions share the same host-load/GC conditions.
function measurements(fs::NamedTuple; repetitions = 100)
    foreach(f -> (f(); f()), fs)
    bytes = map(f -> @allocated(f()), fs)
    samples = [Float64[] for _ in fs]
    for round in 1:7, step in 1:length(fs)
        i = mod1(step + round - 1, length(fs))
        push!(samples[i], elapsed_per_call(fs[i], repetitions))
    end
    NamedTuple{keys(fs)}(Tuple((; median_ns = median(samples[i]) * 1e9,
        bytes = bytes[i]) for i in 1:length(fs)))
end

function compare(n, copied, reused, records, u)
    before = EXT._centering_reparametrization(copied, records)
    after = EXT._centering_reparametrization(reused, records)
    target = Quadratic(length(u))
    before_problem = WH.ReparametrizedProblem(before, target, AutoEnzyme())
    after_problem = WH.ReparametrizedProblem(after, target, AutoEnzyme())
    destination = copy(u)
    for point in (u, u .+ 0.01, view(hcat(u, u .+ 0.01), :, 2))
        @test map(k -> k(point), copied) == map(k -> k(point), reused)
        @test WH.with_logabsdet_jacobian!(copy(point), before, point) ==
            WH.with_logabsdet_jacobian!(copy(point), after, point)
        old_value, old_gradient = LDP.logdensity_and_gradient(before_problem, point)
        new_value, new_gradient = LDP.logdensity_and_gradient(after_problem, point)
        @test new_value == old_value
        @test new_gradient ≈ old_gradient
    end
    result = measurements((
        copied_endpoints = () -> map(k -> k(u), copied),
        reused_endpoints = () -> map(k -> k(u), reused),
        copied_map = () -> WH.with_logabsdet_jacobian!(destination, before, u),
        reused_map = () -> WH.with_logabsdet_jacobian!(destination, after, u),
        copied_gradient = () -> LDP.logdensity_and_gradient(before_problem, u),
        reused_gradient = () -> LDP.logdensity_and_gradient(after_problem, u));
        repetitions = n < 4096 ? 1000 : 100)
    println((; n, result...))
    flush(stdout)
end

println((; julia = VERSION, rk = pkgversion(ReactiveKernels),
    ppl = pkgversion(PPL), wh = pkgversion(WH), enzyme = pkgversion(Enzyme),
    enzyme_path = pathof(Enzyme),
    libenzyme = Enzyme.Compiler.Enzyme_jll.libEnzyme_path))
flush(stdout)

@testset "centering output reuse versus copying" begin
    built = nothing
    for n in (16, 256, 4096)
        plan = MODEL(; N = n, offsets = collect(range(-0.3, 0.4; length = n)),
            se = 0.7) | (; y = zeros(n))
        built = built === nothing ? build_kernel(plan) : built
        u = 0.1 .* sin.(1:assign_layout(plan).total)
        cells = (z = (location = :location, log_scale = :log_scale, source = 0.7),)
        prepared = PPL._centering_cells(built, plan, cells; u0 = u, on_error = nothing)
        copied = map(name -> copied_kernel(built, plan, name), (:location, :log_scale))
        compare(n, copied, prepared.kernels, prepared.records, u)
    end
end
