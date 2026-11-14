# Optional inference integration suite, like test/smc/runtests.jl. Run in a
# consumer environment containing RKPPL, WarmupHMC, LogDensityProblems,
# DifferentiationInterface, Enzyme and Test.
module AdaptiveCenteringTests

using Test, ReactiveKernels, ReactiveKernelsPPL, Enzyme, WarmupHMC
using ReactiveKernelsPPL: adaptive_centering_problem
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
        u0, on_error = nothing)
    EXT._centering_reparametrization(p.kernel, p.records), p
end

function finite_gradient(f, x)
    h = 1e-5
    [(f(x .+ h .* (eachindex(x) .== j)) -
      f(x .- h .* (eachindex(x) .== j))) / (2h) for j in eachindex(x)]
end

# Primal-only observation of the actual prepared cuts. AD tests use the
# unmodified kernels and the backend's ordinary reverse path.
struct CountedKernel{K}
    kernel::K
    calls::Base.RefValue{Int}
end
(c::CountedKernel)(x) = (c.calls[] += 1; c.kernel(x))

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

# Stand-in for BayesianRegressionModels, which exports a different
# `adaptive_centering_problem(model, problem, ad_backend; ...)`. A consumer
# loading both packages must still call BRM's name unqualified
# (snag rkppl-export-ada-38f23f30).
module SameNameExporter
export adaptive_centering_problem
adaptive_centering_problem(model, problem, ad_backend) = :same_name_exporter
end

module SameNameConsumer
using ReactiveKernelsPPL
using ..SameNameExporter
call_unqualified() = adaptive_centering_problem(nothing, nothing, nothing)
end

@testset "a package exporting the same name loads beside RKPPL" begin
    @test !Base.isexported(ReactiveKernelsPPL, :adaptive_centering_problem)
    @static if isdefined(Base, :ispublic)
        @test Base.ispublic(ReactiveKernelsPPL, :adaptive_centering_problem)
    end
    @test SameNameConsumer.call_unqualified() === :same_name_exporter
    s = setup()
    cells = (z = (location = :z_location, log_scale = :log_scale),)
    rp = ReactiveKernelsPPL.adaptive_centering_problem(s.built, s.plan, s.problem,
        BACKEND; cells)
    @test LDP.logdensity(rp, s.u) == LDP.logdensity(s.problem, s.u)
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
    @test length(p.kernel(u).values) == 3
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
    @test emission_shape(p1.kernel) == emission_shape(p2.kernel)
    mu_index = only(e.offset for e in large.layout.entries if e.name === :mu)
    @test last(first(ir2.pairs)).args[1](large.u) ≈ large.u[mu_index] - 0.3
    @test last(last(ir2.pairs)).args[1](large.u) ≈ large.u[mu_index] + 0.4
    zero = setup(0; built = small.built)
    ir0, _ = map_for(zero.built, zero.plan, cells)
    @test isempty(ir0.pairs)
end

@testset "constant affine quantities" begin
    s = setup()
    (; built, plan, u, problem) = s
    entries = Dict(e.name => e for e in s.layout.entries)
    zi = entries[:z].offset .+ (0:entries[:z].size-1)
    ti = entries[:t].offset
    mu = u[entries[:mu].offset]
    ls = u[entries[:sigma].offset]
    # Exercise both constant fields, nonzero constants, and mixtures with
    # graph queries in the same indexed map (one concrete pair type).
    for (location, log_scale, loc, scale) in
            ((0.0, :log_scale, 0.0, ls),
             (:z_location, log(1.4), mu .+ collect(range(-0.3, 0.4; length = length(zi))), log(1.4)),
             (0.3f0, 0, Float64(0.3f0), 0.0),
             (1//4, -0.2, 0.25, -0.2))
        for (target, source) in ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (0.7, 0.2))
            cells = (z = (; location, log_scale, target, source),
                t = (; location = :t_location, log_scale = :log_scale, target, source))
            ir, p = map_for(built, plan, cells; u0 = u)
            @test isconcretetype(eltype(ir.pairs))
            @test @inferred(last(first(ir.pairs)).args[1](u)) isa Float64
            expected = copy(u)
            expected[zi] .= target .* loc .+ (u[zi] .- source .* loc) .* exp(scale * (target - source))
            expected[ti] = target * mu + (u[ti] - source * mu) * exp(ls * (target - source))
            jac, mapped = WarmupHMC.with_logabsdet_jacobian!(copy(u), ir, u)
            @test mapped ≈ expected
            @test jac ≈ (length(zi) * scale + ls) * (target - source)
            ijac, back = WarmupHMC.with_logabsdet_jacobian!(copy(mapped),
                WarmupHMC.InverseFunctions.inverse(ir), mapped)
            @test back ≈ u
            @test ijac ≈ -jac
            rp = ReactiveKernelsPPL.adaptive_centering_problem(built, plan, problem,
                BACKEND; cells, u0 = u)
            value, gradient = LDP.logdensity_and_gradient(rp, u)
            @test value ≈ LDP.logdensity(problem, expected) + jac
            @test gradient ≈ finite_gradient(x -> LDP.logdensity(rp, x), u) rtol = 2e-7 atol = 2e-7
            @test LDP.logdensity_and_gradient(deepcopy(rp), u) == (value, gradient)
            column = view(hcat(u, 2u), :, 1)
            @test LDP.logdensity_and_gradient(rp, column) == (value, gradient)
        end
    end
    # Configuration numbers stay runtime values; changing them or a latent's
    # bound length does not create a new accessor program.
    cells = (z = (location = 0.0, log_scale = 0.2, source = 0.5),)
    ir1, p1 = map_for(built, plan, cells; u0 = u)
    other = (z = (location = 1.3, log_scale = -0.7, source = 0.5),)
    ir2, p2 = map_for(built, plan, other; u0 = u)
    @test typeof(ir1) == typeof(ir2)
    @test typeof(p1.kernel) == typeof(p2.kernel)
    @test last(first(ir1.pairs)).args[1](u) == 0.0
    @test last(first(ir2.pairs)).args[1](u) == 1.3
    equal_constants = (z = (location = 0.0, log_scale = 0.0, source = 0.5),)
    ir_equal, p_equal = map_for(built, plan, equal_constants; u0 = u)
    @test typeof(ir_equal) == typeof(ir1)
    @test typeof(p_equal.kernel) == typeof(p1.kernel)
    for n in (0, 23)
        resized = setup(n; built)
        ir, p = map_for(resized.built, resized.plan, cells; u0 = resized.u)
        @test typeof(ir) == typeof(ir1)
        @test length(ir.pairs) == n
        @test typeof(p.kernel) == typeof(p1.kernel)
    end
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
    ir, prepared = map_for(built, plan, cells; u0 = u)
    entries = Dict(e.name => e for e in built.layout.entries)
    zi = entries[:z].offset .+ (0:5)
    expected = copy(u)
    expected[zi] .= (u[zi] .- u[entries[:mu].offset] .- vec(offsets)) ./ vec(scales)
    jac, mapped = WarmupHMC.with_logabsdet_jacobian!(copy(u), ir, u)
    @test mapped ≈ expected
    @test jac ≈ -sum(log, scales)

    # Array endpoints may borrow graph storage, including bound cached values.
    # Maps and reverse AD must only read it, and a later evaluation must not
    # change an earlier result or the caller's input/data.
    inputs = (copy(u), copy(offsets), copy(scales))
    endpoints = prepared.kernel(u).values
    snapshots = map(copy, endpoints)
    problem = prepare_sampler(built, plan, u; backend = BACKEND)
    rp = adaptive_centering_problem(built, plan, problem, BACKEND; cells, u0 = u)
    first_result = LDP.logdensity_and_gradient(rp, u)
    first_gradient = copy(last(first_result))
    for point in (u, u .+ 0.07, view(hcat(u, u .+ 0.07), :, 2))
        value, gradient = LDP.logdensity_and_gradient(rp, point)
        @test value ≈ LDP.logdensity(rp, point)
        @test gradient ≈ finite_gradient(x -> LDP.logdensity(rp, x), point) rtol = 2e-7 atol = 2e-7
        _, forward = WarmupHMC.with_logabsdet_jacobian!(copy(point), ir, point)
        _, inverse = WarmupHMC.with_logabsdet_jacobian!(copy(forward),
            WarmupHMC.InverseFunctions.inverse(ir), forward)
        @test inverse ≈ point
        @test endpoints == snapshots
        @test (u, offsets, scales) == inputs
        @test last(first_result) == first_gradient
    end
    @test LDP.logdensity_and_gradient(rp, u) == first_result
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
    # theta_location reads phi, so theta is centred about phi's model value:
    # phi maps first, then theta's endpoints are evaluated on the mapped phi.
    entries = Dict(e.name => e for e in built.layout.entries)
    phii = entries[:phi].offset .+ (0:3)
    thetai = entries[:theta].offset .+ (0:3)
    function reference(x)
        c = constrain(built.layout, x)
        phi_ls = log(c.sigma) + log(c.rho * 0.8) / 2
        theta_ls = log(c.sigma) + log1p(-c.rho) / 2
        y = copy(x)
        y[phii] .= x[phii] .* exp(-0.4 * phi_ls)
        loc = c.mu .+ c.sigma .* sqrt(c.rho * 0.8) .* y[phii]
        y[thetai] .= (x[thetai] .- 0.6 .* loc) .* exp(-0.6 * theta_ls)
        y, -4 * (0.4 * phi_ls + 0.6 * theta_ls)
    end
    for cells in ((; theta, phi), (; phi, theta))
        ir, p = map_for(built, plan, cells; u0 = u)
        @test first.(ir.pairs) == [phii; thetai]
        @test [last(q).args[1].stage for q in ir.pairs] == [fill(1, 4); fill(2, 4)]
        expected, expected_jac = reference(u)
        jac, mapped = WarmupHMC.with_logabsdet_jacobian!(copy(u), ir, u)
        @test mapped ≈ expected
        @test jac ≈ expected_jac
        ijac, back = WarmupHMC.with_logabsdet_jacobian!(copy(mapped),
            WarmupHMC.InverseFunctions.inverse(ir), mapped)
        @test back ≈ u
        @test ijac ≈ -jac
        rp = adaptive_centering_problem(built, plan, problem, BACKEND; cells, u0 = u)
        value, gradient = LDP.logdensity_and_gradient(rp, u)
        @test value ≈ LDP.logdensity(problem, expected) + expected_jac
        reference_density(x) = (r = reference(x); LDP.logdensity(problem, r[1]) + r[2])
        @test gradient ≈ finite_gradient(reference_density, u) rtol = 2e-7 atol = 2e-7
    end
end

# A nested two-level hierarchy (snag adaptive-affine-d6c2fa03). The unit level
# is declared before the group level it is centred on, so the packed order is
# the reverse of the dependency order.
const NESTED = @rkppl begin
    mu ~ Normal(0, mu_sd)
    tau_g ~ Exponential(tau_g_scale)
    tau_u ~ Exponential(tau_u_scale)
    z_u[1:U] .~ Normal.(0, 1)
    z_g[1:G] .~ Normal.(0, 1)
    unit_location = mu .+ tau_g .* z_g[group]
    log_tau_g = log(tau_g)
    log_tau_u = log(tau_u)
    unit_mean = sum(z_u) / U
    theta = unit_location .+ tau_u .* z_u
    y .~ Normal.(theta, sigma)
end

function nested_setup(G = 3, per = 2; built = nothing)
    group = repeat(1:G; inner = per)
    U = length(group)
    plan = NESTED(; G, U, group, mu_sd = 2.0, tau_g_scale = 1.0,
        tau_u_scale = 0.5, sigma = 0.3) | (; y = collect(range(-0.5, 0.7; length = U)))
    built = built === nothing ? build_kernel(plan) : built
    # A reused build keeps its own exemplar layout; this binding's sizes come
    # from the plan.
    layout = assign_layout(plan)
    u = 0.1 .* sin.(1:layout.total)
    problem = prepare_sampler(built, plan, u; backend = BACKEND)
    entries = Dict(e.name => e for e in layout.entries)
    gi = entries[:z_g].offset .+ (0:G-1)
    ui = entries[:z_u].offset .+ (0:U-1)
    (; built, plan, layout, u, problem, group, G, U, gi, ui)
end

nested_cells(tg, sg, tu, su) = (
    z_g = (location = :mu, log_scale = :log_tau_g, target = tg, source = sg),
    z_u = (location = :unit_location, log_scale = :log_tau_u, target = tu, source = su))

# The written-out composition: the group level about mu, then the unit level
# about mu + tau_g * z_g[group] at the group level's model values.
function nested_reference(s, x, (tg, sg, tu, su))
    c = constrain(s.layout, x)
    y = copy(x)
    y[s.gi] .= tg .* c.mu .+ (x[s.gi] .- sg .* c.mu) .* c.tau_g^(tg - sg)
    loc = c.mu .+ c.tau_g .* y[s.gi][s.group]
    y[s.ui] .= tu .* loc .+ (x[s.ui] .- su .* loc) .* c.tau_u^(tu - su)
    y, s.G * (tg - sg) * log(c.tau_g) + s.U * (tu - su) * log(c.tau_u)
end

@testset "nested cells centre on their parents' model values" begin
    s = nested_setup()
    (; built, plan, u, problem) = s
    for centering in ((0.0, 0.5, 0.0, 0.5), (0.0, 1.0, 0.0, 1.0), (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0), (0.0, 0.3, 0.0, 0.8), (1.0, 0.2, 0.5, 0.0))
        cells = nested_cells(centering...)
        ir, p = map_for(built, plan, cells; u0 = u)
        @test isconcretetype(eltype(ir.pairs))
        @test first.(ir.pairs) == [s.gi; s.ui]
        @test [last(q).args[1].stage for q in ir.pairs] == [fill(1, s.G); fill(2, s.U)]
        expected, expected_jac = nested_reference(s, u, centering)
        jac, mapped = WarmupHMC.with_logabsdet_jacobian!(copy(u), ir, u)
        @test mapped ≈ expected
        @test jac ≈ expected_jac
        ijac, back = WarmupHMC.with_logabsdet_jacobian!(copy(mapped),
            WarmupHMC.InverseFunctions.inverse(ir), mapped)
        @test back ≈ u
        @test ijac ≈ -jac
        rp = adaptive_centering_problem(built, plan, problem, BACKEND; cells, u0 = u)
        reference_density(x) = (r = nested_reference(s, x, centering);
            LDP.logdensity(problem, r[1]) + r[2])
        for point in (u, u .+ 0.05, view(hcat(u, u .+ 0.05), :, 2))
            value, gradient = LDP.logdensity_and_gradient(rp, point)
            @test value ≈ reference_density(point)
            @test gradient ≈ finite_gradient(reference_density, point) rtol = 2e-7 atol = 2e-7
        end
        @test LDP.logdensity_and_gradient(deepcopy(rp), u) == LDP.logdensity_and_gradient(rp, u)

        # Re-centering the group level alone keeps the unit coordinates of a
        # model point: they are relative to the group level's model values.
        moved = (centering[1], 1 - centering[2], centering[3], centering[4])
        ir_moved, _ = map_for(built, plan, nested_cells(moved...); u0 = u)
        _, recentered = WarmupHMC.with_logabsdet_jacobian!(copy(mapped),
            WarmupHMC.InverseFunctions.inverse(ir_moved), mapped)
        @test recentered[s.ui] ≈ u[s.ui]
        @test nested_reference(s, recentered, moved)[1] ≈ mapped
    end

    # The joint kernel runs once per level on a map and on its inverse.
    ir, p = map_for(built, plan, nested_cells(0.0, 0.6, 0.0, 0.4); u0 = u)
    @test length(p.kernel(u).values) == 4
    counted = CountedKernel(p.kernel, Ref(0))
    observed = EXT._centering_reparametrization(counted, p.records)
    _, mapped = WarmupHMC.with_logabsdet_jacobian!(copy(u), observed, u)
    @test counted.calls[] == 2
    counted.calls[] = 0
    _, back = WarmupHMC.with_logabsdet_jacobian!(copy(mapped),
        WarmupHMC.InverseFunctions.inverse(observed), mapped)
    @test counted.calls[] == 2
    @test back ≈ u

    # Level structure is graph structure: a larger binding reuses the program.
    large = nested_setup(5, 4; built)
    ir_large, p_large = map_for(large.built, large.plan, nested_cells(0.0, 0.6, 0.0, 0.4);
        u0 = large.u)
    @test typeof(ir_large) == typeof(ir)
    @test emission_shape(p_large.kernel) == emission_shape(p.kernel)
    expected, _ = nested_reference(large, large.u, (0.0, 0.6, 0.0, 0.4))
    @test last(WarmupHMC.with_logabsdet_jacobian!(copy(large.u), ir_large, large.u)) ≈ expected
end

@testset "nested levels follow log-scale reads; own reads and cycles refuse" begin
    s = nested_setup()
    (; built, plan, u, problem) = s
    # z_g's log-scale reads z_u, so z_u is level 1 and z_g level 2; here the
    # dependency order matches the packed order.
    cells = (z_g = (location = :mu, log_scale = :unit_mean, source = 0.5),
        z_u = (location = :mu, log_scale = :log_tau_u, source = 0.5))
    ir, _ = map_for(built, plan, cells; u0 = u)
    @test first.(ir.pairs) == [s.ui; s.gi]
    @test [last(q).args[1].stage for q in ir.pairs] == [fill(1, s.U); fill(2, s.G)]
    c = constrain(s.layout, u)
    expected = copy(u)
    expected[s.ui] .= (u[s.ui] .- 0.5 * c.mu) .* c.tau_u^-0.5
    unit_mean = sum(expected[s.ui]) / s.U
    expected[s.gi] .= (u[s.gi] .- 0.5 * c.mu) .* exp(-0.5 * unit_mean)
    jac, mapped = WarmupHMC.with_logabsdet_jacobian!(copy(u), ir, u)
    @test mapped ≈ expected
    @test jac ≈ -0.5 * (s.U * log(c.tau_u) + s.G * unit_mean)
    _, back = WarmupHMC.with_logabsdet_jacobian!(copy(mapped),
        WarmupHMC.InverseFunctions.inverse(ir), mapped)
    @test back ≈ u

    message(cells) = try
        adaptive_centering_problem(built, plan, problem, BACKEND; cells, u0 = u)
        ""
    catch err
        err isa ContractValidationError || rethrow()
        sprint(showerror, err)
    end
    # theta reads z_u itself (and z_g); unit_mean reads z_u.
    @test occursin("`z_u.location` (`theta`) reads `z_u` itself",
        message((z_u = (location = :theta, log_scale = :log_tau_u),)))
    @test occursin("`z_u.log_scale` (`unit_mean`) reads `z_u` itself",
        message((z_u = (location = 0.0, log_scale = :unit_mean),)))
    @test occursin("cells `z_g` → `z_u` → `z_g` read each other's latents",
        message((z_g = (location = :unit_mean, log_scale = :log_tau_g),
            z_u = (location = :unit_location, log_scale = :log_tau_u))))
end

const SHARED = @rkppl begin
    phi[1:N] .~ Normal.(0, 1)
    z[1:N] .~ Normal.(0, 1)
    location = sum(phi) / N
    array_location = location .+ offsets
    log_scale = sum(phi .* phi) / (2N)
    total = array_location .+ exp(log_scale) .* z
    y .~ Normal.(total, se)
end

function shared_setup(n = 4; built = nothing)
    offsets = collect(range(-0.3, 0.4; length = n))
    plan = SHARED(; N = n, offsets, se = 0.7) | (; y = fill(0.2, n))
    built = built === nothing ? build_kernel(plan) : built
    layout = assign_layout(plan)
    u = 0.1 .* sin.(1:layout.total)
    problem = prepare_sampler(built, plan, u; backend = BACKEND)
    (; built, plan, layout, u, problem, offsets)
end

@testset "one joint evaluation across cells and affine roles" begin
    s = setup()
    # Reverse keyword order and reuse the same endpoint at both positions.
    # Both cells depend on unselected mu; the selected coordinates are invariant.
    cells = (t = (location = :t_location, log_scale = :t_location, source = 0.4),
        z = (location = :z_location, log_scale = :t_location, source = 0.7))
    ir, p = map_for(s.built, s.plan, cells; u0 = s.u)
    @test issorted(first.(ir.pairs))
    @test length(p.kernel(s.u).values) == 2
    counted = CountedKernel(p.kernel, Ref(0))
    observed = EXT._centering_reparametrization(counted, p.records)
    jac, mapped = WarmupHMC.with_logabsdet_jacobian!(copy(s.u), observed, s.u)
    @test counted.calls[] == 1
    counted.calls[] = 0
    ijac, back = WarmupHMC.with_logabsdet_jacobian!(copy(mapped),
        WarmupHMC.InverseFunctions.inverse(observed), mapped)
    @test counted.calls[] == 1
    @test back ≈ s.u
    @test ijac ≈ -jac

    rp = WarmupHMC.ReparametrizedProblem(ir, s.problem, BACKEND)
    for point in (s.u, s.u .+ 0.03, view(hcat(s.u, s.u .+ 0.03), :, 2))
        value, gradient = LDP.logdensity_and_gradient(rp, point)
        @test value ≈ LDP.logdensity(rp, point)
        @test gradient ≈ finite_gradient(x -> LDP.logdensity(rp, x), point) rtol = 2e-7 atol = 2e-7
    end

    # Observe the public retrospective scorer with primal counters only.
    # Its supplied gradients came from the unmodified ordinary-AD target.
    positions = hcat(s.u, s.u .+ 0.01, s.u .- 0.02, s.u .+ 0.03)
    gradients = hcat((last(LDP.logdensity_and_gradient(rp, x))
        for x in eachcol(positions))...)
    counted.calls[] = 0
    scored = WarmupHMC.candidate_scoring_losses(
        WarmupHMC.ReparametrizedProblem(observed, s.problem, BACKEND), positions, gradients)
    @test counted.calls[] == size(positions, 2)
    reference = WarmupHMC.candidate_scoring_losses(rp, positions, gradients)
    @test isequal(scored, reference)

    identity_records = map(c -> merge(c, (; target = 0.7, source = 0.7)), p.records)
    identity = EXT._centering_reparametrization(counted, identity_records)
    counted.calls[] = 0
    for mapping in (identity, WarmupHMC.InverseFunctions.inverse(identity))
        j, point = WarmupHMC.with_logabsdet_jacobian!(copy(s.u), mapping, s.u)
        @test point == s.u
        @test j == 0.0
    end
    @test counted.calls[] == 0
end

@testset "shared graph reductions run once per vector" begin
    shapes = []
    types = []
    built = nothing
    for n in (4, 128)
        s = shared_setup(n; built)
        built = s.built
        entries = Dict(e.name => e for e in s.layout.entries)
        pi = entries[:phi].offset .+ (0:n-1)
        zi = entries[:z].offset .+ (0:n-1)
        for node in (:location, :array_location)
            cells = (z = (location = node, log_scale = :log_scale, source = 0.7),)
            ir, p = map_for(s.built, s.plan, cells; u0 = s.u)
            push!(types, typeof(ir))
            push!(shapes, emission_shape(p.kernel))
            @test @inferred(last(first(ir.pairs)).args[1](s.u)) isa Float64
            counted = CountedKernel(p.kernel, Ref(0))
            observed = EXT._centering_reparametrization(counted, p.records)
            jac, mapped = WarmupHMC.with_logabsdet_jacobian!(copy(s.u), observed, s.u)
            @test counted.calls[] == 1
            counted.calls[] = 0
            ijac, back = WarmupHMC.with_logabsdet_jacobian!(copy(mapped),
                WarmupHMC.InverseFunctions.inverse(observed), mapped)
            @test counted.calls[] == 1
            @test back ≈ s.u
            @test ijac ≈ -jac
            function reference(x)
                loc = sum(x[pi]) / n .+ (node === :location ? zeros(n) : s.offsets)
                ls = sum(abs2, x[pi]) / (2n)
                expected = copy(x)
                expected[zi] .= (x[zi] .- 0.7 .* loc) .* exp(-0.7 * ls)
                expected, -0.7 * n * ls
            end
            expected, expected_jac = reference(s.u)
            @test mapped ≈ expected
            @test jac ≈ expected_jac
            rp = adaptive_centering_problem(s.built, s.plan, s.problem, BACKEND;
                cells, u0 = s.u)
            value, gradient = LDP.logdensity_and_gradient(rp, s.u)
            @test value ≈ LDP.logdensity(s.problem, expected) + expected_jac
            if n == 4
                function reference_density(x)
                    y, j = reference(x)
                    LDP.logdensity(s.problem, y) + j
                end
                @test gradient ≈ finite_gradient(reference_density, s.u) rtol = 2e-7 atol = 2e-7
            end
            column = view(hcat(s.u, 2s.u), :, 1)
            column_value, column_gradient = LDP.logdensity_and_gradient(rp, column)
            @test column_value == value
            # Reverse accumulation can round differently for a column view.
            @test column_gradient ≈ gradient rtol = 1e-13 atol = 1e-13
        end
    end
    @test types[1] == types[3]
    @test types[2] == types[4]
    @test shapes[1] == shapes[3]
    @test shapes[2] == shapes[4]
end

# ── Markov chains: `markov_chains` and WarmupHMC 0.3.1's `MarkovChain` ─────
#
# A marked linear-Gaussian recurrence is lowered in its states and gets one
# `MarkovChain`; a recurrence per `@plate` index is one chain restarting at each
# index. Oracles are independent: the walk's density written out, and each
# state's location and step SD from the stated law.

# A non-centred random walk with a sampled step size: re-spelled in its states.
const WALK = @rkppl begin
    log_q ~ Normal(-1, 0.5)
    noise ~ Exponential(1)
    @scan begin
        w[1] ~ Normal(0, 2)
        for t in 2:T
            e ~ Normal(0, 1)
            w[t] = w[t - 1] + exp(log_q) * e
        end
    end
    y .~ Normal.(w, noise)
end

# Per series, a centred AR(1) about a mean from a fixed start: the chain's
# states are `h[2:end]`, and each series' first state reads the fixed seed.
const AR_CELLS = @rkppl begin
    rho ~ Uniform(-1, 1)
    mu ~ Normal(0, 1)
    s ~ Exponential(1)
    noise ~ Exponential(1)
    @plate for g in eachindex(y)
        @scan begin
            h[1] = 0.0
            for t in 2:length(y[g])
                h[t] ~ Normal(rho * h[t - 1] + (1 - rho) * mu, s)
            end
        end
        y[g] .~ Normal.(h, noise)
    end
end

# The reporter's spelling: a non-centred walk per station with its own step
# size, from a fixed start.
const WANDER = @rkppl begin
    wander_sd[1:G] .~ Exponential.(0.5)
    noise ~ Exponential(1)
    @plate for j in eachindex(y)
        @scan begin
            wander[1] = 0.0
            for t in 2:length(y[j])
                wander_innovation ~ Normal(0, 1)
                wander[t] = wander[t - 1] + wander_sd[j] * wander_innovation
            end
        end
        y[j] .~ Normal.(wander, noise)
    end
end

chain_series(lengths; seed = 3) =
    [cumsum(0.4 .* sin.(seed .+ (1:n)) .+ 0.1) for n in lengths]

function chain_setup(plan)
    built = build_kernel(plan)
    u = 0.1 .* cos.(1:built.layout.total)
    problem = prepare_sampler(built, plan, u; backend = BACKEND)
    rp = adaptive_centering_problem(built, plan, problem, BACKEND)
    (; built, plan, u, problem, rp, ir = rp.reparametrizer)
end

# Log densities written out (this suite does not load Distributions).
normal_lpdf(m, s, x) = -0.5 * ((x - m) / s)^2 - log(s) - 0.5 * log(2pi)
exponential_lpdf(scale, x) = -log(scale) - x / scale

# Map source `x` through the chain's innovation frame (phi, c) = (1, 0):
# w_k = m_k + x_k * sd_k, the series restarting where `restart[k]`.
function innovation_walk(x, idx, coef, drift, sd, restart)
    w = similar(x, length(idx))
    for k in eachindex(idx)
        m = restart[k] ? drift[k] : coef[k] * w[k - 1] + drift[k]
        w[k] = m + x[idx[k]] * sd[k]
    end
    w
end

@testset "markov_chains lowers a non-centred walk in its states" begin
    T = 6
    y = chain_series([T])[1]
    plan = markov_chains(WALK(; T), :w) | (; y)
    built = build_kernel(plan)
    # The packed block holds the states (the sampled seed, then each step), not
    # the innovations; a scan with step locals keeps the `_ppl_scan_z_` name.
    @test coordinate_names(built.layout) ==
        [:log_q, :noise, (Symbol("_ppl_scan_z_w.$t") for t in 1:T)...]
    sampler = prepare_sampler(built, plan, zeros(built.layout.total); backend = BACKEND)
    oracle(u) = begin
        c = constrain(built.layout, u)
        q = exp(c.log_q)
        w = c._ppl_scan_z_w
        normal_lpdf(-1, 0.5, c.log_q) + exponential_lpdf(1, c.noise) +
            normal_lpdf(0, 2, w[1]) +
            sum(normal_lpdf(w[t - 1], q, w[t]) for t in 2:T) +
            sum(normal_lpdf.(w, c.noise, y)) + logjac(built.layout, u)
    end
    for k in 1:3
        u = 0.3 .* sin.(k .* (1:built.layout.total))
        value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
        @test value ≈ oracle(u) rtol = 1e-12
        @test gradient ≈ finite_gradient(oracle, u) rtol = 1e-6 atol = 1e-8
    end
    # The model as written packs the innovations: the same coordinates read as
    # states give another density.
    plain = WALK(; T) | (; y)
    plain_sampler = prepare_sampler(build_kernel(plain), plain, zeros(built.layout.total);
        backend = BACKEND)
    u = 0.3 .* sin.(1:built.layout.total)
    @test !(plain_sampler(u) ≈ oracle(u))
end

@testset "one MarkovChain per marked recurrence" begin
    T = 5
    walk = chain_setup(markov_chains(WALK(; T), :w) | (; y = chain_series([T])[1]))
    lengths = [4, 1, 3]
    ys = chain_series(lengths)
    ar = chain_setup(markov_chains(AR_CELLS(), :h) | (; y = ys))
    wander = chain_setup(markov_chains(WANDER(; G = 3), :wander) | (; y = ys))
    for (case, f) in ((walk, function (c)
                # the seed w[1] ~ Normal(0, 2), then steps of SD exp(log_q)
                coef = [0.0; ones(T - 1)]
                drift = zeros(T)
                sd = [2.0; fill(exp(c.log_q), T - 1)]
                restart = [true; falses(T - 1)]
                (; idx = collect(3:2 + T), coef, drift, sd, restart)
            end),
            (ar, function (c)
                # per series h[2:end]: the first reads the fixed seed 0
                coef, drift, sd, restart = Float64[], Float64[], Float64[], Bool[]
                for n in lengths, t in 2:n
                    push!(coef, t == 2 ? 0.0 : c.rho)
                    push!(drift, (1 - c.rho) * c.mu + (t == 2 ? c.rho * 0.0 : 0.0))
                    push!(sd, c.s)
                    push!(restart, t == 2)
                end
                (; idx = collect(5:4 + length(coef)), coef, drift, sd, restart)
            end),
            (wander, function (c)
                coef, drift, sd, restart = Float64[], Float64[], Float64[], Bool[]
                for (j, n) in enumerate(lengths), t in 2:n
                    push!(coef, t == 2 ? 0.0 : 1.0)
                    push!(drift, 0.0)
                    push!(sd, c.wander_sd[j])
                    push!(restart, t == 2)
                end
                (; idx = collect(5:4 + length(coef)), coef, drift, sd, restart)
            end))
        ir = case.ir
        @test length(ir.chains) == 1
        chain = only(ir.chains)
        x = 0.2 .* cos.(1:length(case.u)) .+ 0.1
        c = constrain(case.built.layout, x)
        expected = f(c)
        @test chain.states == expected.idx
        # At the default controls the chain is the model as written.
        @test LDP.logdensity(case.rp, x) == LDP.logdensity(case.problem, x)
        # The innovation frame: each state is its location plus x times its SD.
        chain.phi .= 1.0
        chain.c .= 0.0
        ljac, y = ir(x)
        @test y[expected.idx] ≈ innovation_walk(x, expected.idx, expected.coef,
            expected.drift, expected.sd, expected.restart) rtol = 1e-12
        @test ljac ≈ sum(log, expected.sd) rtol = 1e-12
        # The gradient through the chain's accessors (Enzyme over the frame kernel).
        lp, g = LDP.logdensity_and_gradient(case.rp, x)
        @test lp ≈ LDP.logdensity(case.rp, x)
        @test g ≈ finite_gradient(z -> LDP.logdensity(case.rp, z), x) rtol = 1e-5 atol = 1e-7
        chain.phi .= 0.0
        chain.c .= 1.0
    end
    # A model without marked recurrences gets no chains.
    (; built, plan, problem) = setup()
    @test isempty(adaptive_centering_problem(built, plan, problem, BACKEND;
        cells = (z = (location = :z_location, log_scale = :log_scale),)).reparametrizer.chains)
end

@testset "markov_chains refusals" begin
    err(f) = try
        f()
        "no error"
    catch e
        sprint(showerror, e)
    end
    y = chain_series([4])[1]
    @test occursin("already lowered", err(() -> markov_chains(WALK(; T = 4) | (; y), :w)))
    @test occursin("no top-level `@scan`", err(() -> markov_chains(WALK(; T = 4), :q) | (; y)))
    @test occursin("integrated out", err(() -> markov_chains(marginalize(WALK(; T = 4), :w), :w)))
    nonlinear = RKPPLModel(quote
        s ~ Exponential(1)
        @scan begin
            w[1] ~ Normal(0, 1)
            for t in 2:T
                w[t] ~ Normal(tanh(w[t - 1]), s)
            end
        end
        y .~ Normal.(w, 1)
    end, @__MODULE__)
    msg = err(() -> markov_chains(nonlinear(; T = 4), :w) | (; y))
    @test occursin("markov_chains(`w`)", msg) && occursin("is not linear", msg)
end

end
