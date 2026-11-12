# Optional inference integration suite, like test/smc/runtests.jl. Run in a
# consumer environment containing RKPPL, WarmupHMC, LogDensityProblems,
# DifferentiationInterface, Enzyme and Test.
module AdaptiveCenteringTests

using Test, ReactiveKernels, ReactiveKernelsPPL, Enzyme, WarmupHMC
using DifferentiationInterface: AutoEnzyme
import LogDensityProblems as LDP

const BACKEND = AutoEnzyme()
const EXT = Base.get_extension(ReactiveKernelsPPL, :ReactiveKernelsPPLWarmupHMCExt)

const MODEL = @rkppl begin
    mu ~ Normal(0, mu_sd)
    sigma ~ Exponential(sigma_scale)
    z[1:N] .~ Normal.(0, 1)
    t ~ Cauchy(0, 1)
    z_location = mu .+ offsets
    log_scale = log(sigma)
    t_location = mu
    total = z_location .+ sigma .* z .+ t
    y .~ Normal.(total, se)
end

function setup(n = 4; built = nothing)
    plan = MODEL(; N = n, offsets = n == 0 ? Float64[] : collect(range(-0.3, 0.4; length = n)),
        mu_sd = 2.0, sigma_scale = 1.2, se = 0.7) | (; y = fill(0.2, n))
    built = built === nothing ? build_kernel(plan) : built
    layout = assign_layout(plan)
    u = 0.1 .* sin.(1:layout.total)
    problem = prepare_sampler(built, plan, u; backend = BACKEND)
    (; built, plan, layout, u, problem)
end

function map_for(built, plan, cells; u0 = nothing)
    p = ReactiveKernelsPPL._centering_cells(built, plan, cells;
        u0, on_error = nothing, locality = :error)
    EXT._centering_reparametrization(p.kernels, p.records), p
end

function finite_gradient(f, x)
    h = 1e-5
    [(f(x .+ h .* (eachindex(x) .== j)) -
      f(x .- h .* (eachindex(x) .== j))) / (2h) for j in eachindex(x)]
end

function emission_shape(kernel)
    heads = Dict{Symbol,Int}()
    function visit(ex)
        ex isa Expr || return
        heads[ex.head] = get(heads, ex.head, 0) + 1
        foreach(visit, ex.args)
    end
    visit(code_expr(kernel))
    (; heads, callables = length(kernel.callables))
end

@testset "graph-backed affine cells" begin
    @test EXT !== nothing
    s = setup()
    (; built, plan, u, problem) = s
    cells = (z = (location = :z_location, log_scale = :log_scale),
        t = (location = :t_location, log_scale = :log_scale))
    rp = adaptive_centering_problem(built, plan, problem, BACKEND; cells)
    @test LDP.logdensity(rp, u) == LDP.logdensity(problem, u)
    @test LDP.logdensity_and_gradient(rp, u) == LDP.logdensity_and_gradient(problem, u)
    ir, p = map_for(built, plan, cells)
    @test isconcretetype(eltype(ir.pairs))
    @test length(p.kernels) == 3
    @test @inferred(last(first(ir.pairs)).args[1](u)) isa Float64
    column = view(hcat(u, 2u), :, 1)
    @test @inferred(last(first(ir.pairs)).args[1](column)) ==
        last(first(ir.pairs)).args[1](u)
    @test LDP.logdensity(rp, column) == LDP.logdensity(rp, u)

    entries = Dict(e.name => e for e in s.layout.entries)
    zi = entries[:z].offset .+ (0:entries[:z].size-1)
    ti = entries[:t].offset
    mu, sigma = u[entries[:mu].offset], exp(u[entries[:sigma].offset])
    loc = mu .+ collect(range(-0.3, 0.4; length = length(zi)))
    for (target, source) in ((0.0, 0.25), (0.0, 1.0), (1.0, 0.0), (0.7, 0.2))
        chosen = (z = merge(cells.z, (; target, source)),
            t = merge(cells.t, (; target, source)))
        ir, _ = map_for(built, plan, chosen)
        expected = copy(u)
        expected[zi] .= target .* loc .+ (u[zi] .- source .* loc) .* sigma^(target - source)
        expected[ti] = target * mu + (u[ti] - source * mu) * sigma^(target - source)
        # The indexed in-place map writes its selected coordinates only.
        jac, mapped = WarmupHMC.with_logabsdet_jacobian!(copy(u), ir, u)
        @test mapped ≈ expected
        @test jac ≈ (length(zi) + 1) * (target - source) * log(sigma)
        ijac, back = WarmupHMC.with_logabsdet_jacobian!(copy(mapped),
            WarmupHMC.InverseFunctions.inverse(ir), mapped)
        @test back ≈ u
        @test ijac ≈ -jac
        rp = adaptive_centering_problem(built, plan, problem, BACKEND; cells = chosen)
        @test LDP.logdensity(rp, u) ≈ LDP.logdensity(problem, expected) + jac
        value, gradient = LDP.logdensity_and_gradient(rp, u)
        @test value ≈ LDP.logdensity(rp, u)
        @test gradient ≈ finite_gradient(x -> LDP.logdensity(rp, x), u) rtol = 2e-7 atol = 2e-7
        @test LDP.logdensity_and_gradient(deepcopy(rp), u) == (value, gradient)
    end

    empty = adaptive_centering_problem(built, plan, problem, BACKEND; cells = NamedTuple())
    @test LDP.logdensity(empty, u) == LDP.logdensity(problem, u)
    @test LDP.logdensity_and_gradient(empty, u) == LDP.logdensity_and_gradient(problem, u)
    for bad in ((missing = cells.z,), (sigma = cells.z,),
            (z = (location = :z, log_scale = :log_scale),),
            (z = (location = :total, log_scale = :log_scale),),
            (z = merge(cells.z, (; source = 1.1)),),
            (z = merge(cells.z, (; location = :missing)),),
            (t = cells.z,))
        @test_throws ContractValidationError adaptive_centering_problem(
            built, plan, problem, BACKEND; cells = bad)
    end
    extreme = copy(u)
    extreme[entries[:sigma].offset] = -1000.0
    ir, _ = map_for(built, plan, cells)
    jac, mapped = WarmupHMC.with_logabsdet_jacobian!(copy(extreme), ir, extreme)
    @test mapped == extreme
    @test jac == 0.0
end

@testset "binding sizes stay runtime data" begin
    small = setup(3)
    large = setup(23; built = small.built)
    cells = (z = (location = :z_location, log_scale = :log_scale, source = 0.5),)
    ir1, p1 = map_for(small.built, small.plan, cells)
    ir2, p2 = map_for(large.built, large.plan, cells)
    @test typeof(ir1) == typeof(ir2)
    @test length(ir1.pairs) == 3
    @test length(ir2.pairs) == 23
    # Generated local names differ across preparations. Check body/control-flow
    # size without rewriting the displayed code, and exercise the new binding.
    @test map(emission_shape, p1.kernels) == map(emission_shape, p2.kernels)
    mu_index = only(e.offset for e in large.layout.entries if e.name === :mu)
    @test last(first(ir2.pairs)).args[1](large.u) ≈ large.u[mu_index] - 0.3
    @test last(last(ir2.pairs)).args[1](large.u) ≈ large.u[mu_index] + 0.4
    zero = setup(0; built = small.built)
    ir0, _ = map_for(zero.built, zero.plan, cells)
    @test isempty(ir0.pairs)
end

const SPATIAL = @rkppl begin
    mu ~ Normal(0, mu_sd)
    sigma ~ Exponential(sigma_scale)
    rho ~ Beta(1, 1)
    phi[1:N] ~ ICAR(node1, node2)
    phi_sum = sum(phi)
    phi_sum ~ Normal(0, sum_sd)
    theta[1:N] .~ Normal.(0, 1)
    theta_location = mu .+ sigma .* sqrt(rho * inv_scale) .* phi
    theta_log_scale = log(sigma) + log1p(-rho) / 2
    phi_location = 0.0
    phi_log_scale = log(sigma) + log(rho * inv_scale) / 2
    total = theta_location .+ exp(theta_log_scale) .* theta
    y .~ Normal.(total, se)
end

const MATRIX = @rkppl begin
    mu ~ Normal(0, mu_sd)
    z[1:M, 1:N] .~ Normal.(0, 1)
    location = mu .+ offsets
    log_scale = log.(scales)
    total = location .+ scales .* z
    y .~ Normal.(total, se)
end

@testset "rectangular cells retain linear coordinate order" begin
    offsets = [0.1 0.3 0.5; 0.2 0.4 0.6]
    scales = [0.7 1.1 1.5; 0.9 1.3 1.7]
    plan = MATRIX(; M = 2, N = 3, offsets, scales, mu_sd = 2.0, se = 0.8) |
        (; y = zeros(2, 3))
    built = build_kernel(plan)
    u = 0.1 .* cos.(1:built.layout.total)
    cells = (z = (location = :location, log_scale = :log_scale, source = 1.0),)
    ir, _ = map_for(built, plan, cells; u0 = u)
    entries = Dict(e.name => e for e in built.layout.entries)
    zi = entries[:z].offset .+ (0:5)
    expected = copy(u)
    expected[zi] .= (u[zi] .- u[entries[:mu].offset] .- vec(offsets)) ./ vec(scales)
    jac, mapped = WarmupHMC.with_logabsdet_jacobian!(copy(u), ir, u)
    @test mapped ≈ expected
    @test jac ≈ -sum(log, scales)
end

@testset "a graph-derived spatial cell preserves the original density" begin
    plan = SPATIAL(; N = 4, node1 = [1, 2, 3], node2 = [2, 3, 4],
        inv_scale = 0.8, sum_sd = 0.4, mu_sd = 3.0, sigma_scale = 1.0,
        se = 0.3) | (; y = [0.2, 0.4, -0.3, 0.1])
    built = build_kernel(plan)
    u = 0.1 .* cos.(1:built.layout.total)
    problem = prepare_sampler(built, plan, u; backend = BACKEND)
    theta = (location = :theta_location, log_scale = :theta_log_scale, source = 0.6)
    phi = (location = :phi_location, log_scale = :phi_log_scale, source = 0.4)
    # Each real block can be transformed, regardless of its prior family.
    for cells in ((; theta), (; phi))
        ir, _ = map_for(built, plan, cells; u0 = u)
        jac, mapped = WarmupHMC.with_logabsdet_jacobian!(copy(u), ir, u)
        rp = adaptive_centering_problem(built, plan, problem, BACKEND; cells, u0 = u)
        value, gradient = LDP.logdensity_and_gradient(rp, u)
        @test value ≈ LDP.logdensity(problem, mapped) + jac
        @test gradient ≈ finite_gradient(x -> LDP.logdensity(rp, x), u) rtol = 2e-7 atol = 2e-7
        _, back = WarmupHMC.with_logabsdet_jacobian!(copy(mapped),
            WarmupHMC.InverseFunctions.inverse(ir), mapped)
        @test back ≈ u
    end
    # theta_location depends on phi. Independent source-vector accessors do
    # not implement the triangular transport needed to transform both.
    @test_throws ContractValidationError adaptive_centering_problem(
        built, plan, problem, BACKEND; cells = (; theta, phi), u0 = u)
end

end
