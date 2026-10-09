using DifferentiationInterface
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# `prepare_sampler(...; retain)`: the presets named in `retain` come from the
# gradient's own primal sweep. Each case compares them against the separate
# primal query at the same point, with the value and gradient against the
# ordinary sampler query.

const _RETAINED_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

# Subject-level random effects; the Normal likelihood sums its pointwise plate.
_retained_normal = @rkppl begin
    lcl ~ Normal(0, 1)
    lv ~ Normal(0, 1)
    omega ~ Exponential(1)
    sigma ~ Exponential(1)
    z[levels(g)] .~ Normal.(0, 1)
    cl = exp.(lcl .+ omega .* z[g])
    mu = dose .* exp.(cl .* nt) ./ exp(lv)
    y .~ Normal.(mu, sigma)
end

# A log-link Poisson response: its sampler density uses the whole-vector sum,
# so the pointwise plate is computed only because it is retained.
_retained_poisson = @rkppl begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    omega ~ Exponential(1)
    z[levels(g)] .~ Normal.(0, 1)
    eta = a .+ b .* t .+ omega .* z[g]
    c .~ Poisson.(exp.(eta))
end

# One array of observations per subject in a dotted @plate cell (nested
# group and observation plates): `:pointwise` holds one array per subject.
_retained_ragged = @rkppl begin
    lcl ~ Normal(0, 1)
    lv ~ Normal(0, 1)
    omega ~ Exponential(1)
    sigma ~ Exponential(1)
    z[axes(ys, 1)] .~ Normal.(0, 1)
    @plate for i in eachindex(ys)
        cl = exp(lcl + omega * z[i])
        ys[i] .~ Normal.(doses[i] .* exp.(cl .* nts[i]) ./ exp(lv), sigma)
    end
end

_retained_prior_only = @rkppl begin
    a ~ Normal(0, 1)
    s ~ Exponential(1)
end

function _retained_data()
    g = repeat(1:5, inner = 4)
    t = repeat([0.5, 2.0, 6.0, 12.0], 5)
    dose = fill(100.0, 20)
    z = [0.3, -0.5, 1.1, 0.0, -1.2]
    y = dose .* exp.(-exp.(-1.0 .+ 0.3 .* z[g]) .* t) ./ exp(2.0) .+
        0.1 .* sin.(1:20)
    c = [0, 1, 3, 2, 1, 0, 4, 6, 2, 2, 1, 0, 5, 3, 3, 1, 0, 2, 2, 7]
    (; g, t, nt = -t, dose, y, c)
end

function _check_retained(plan, retain)
    built = build_kernel(plan)
    n = built.layout.total
    u = [0.1 * sin(k) for k in 1:n]
    q = prepare_sampler(built, plan, u; backend = _RETAINED_BACKEND)
    qr = prepare_sampler(built, plan, u; backend = _RETAINED_BACKEND, retain)
    queries = map(preset -> prepare_query(built, plan, preset), retain)
    g, gr = zeros(n), zeros(n)
    previous = nothing
    for point in (u, u .+ 0.05, u .- 0.1)
        value, _ = sampler_value_and_gradient!(q, g, point)
        value_r, returned, retained = sampler_value_gradient_and_retained!(qr, gr, point)
        @test returned === gr
        @test value_r == value
        @test gr ≈ g rtol = 1e-12 atol = 1e-12
        @test keys(retained) == retain
        for (preset, query) in zip(retain, queries)
            @test retained[preset] == Base.invokelatest(query, point)
        end
        # A later call returns fresh values and leaves earlier ones untouched.
        if previous !== nothing
            snapshot, stored = previous
            @test stored == snapshot
        end
        previous = (deepcopy(retained), retained)
        # The retaining preparation still answers the ordinary call.
        value_o, _ = sampler_value_and_gradient!(qr, gr, point)
        @test value_o == value
        @test gr ≈ g rtol = 1e-12 atol = 1e-12
    end
    q, qr
end

@testset "prepare_sampler retains presets from the gradient sweep" begin
    d = _retained_data()
    data = (; g = d.g, nt = d.nt, dose = d.dose)

    @testset "Normal response: pointwise and likelihood" begin
        plan = _retained_normal(; data...) | (; y = d.y)
        _check_retained(plan, (:pointwise,))
        _, qr = _check_retained(plan, (:pointwise, :likelihood, :prior))
        n = qr.layout.total
        _, _, retained = sampler_value_gradient_and_retained!(qr, zeros(n), zeros(n))
        @test sum(sum, values(retained.pointwise)) ≈ retained.likelihood
        @test size(retained.pointwise.y) == size(d.y)
    end

    @testset "one array of observations per subject" begin
        rows = [findall(==(i), d.g) for i in 1:5]
        plan = _retained_ragged(; nts = [d.nt[r] for r in rows],
                                 doses = [d.dose[r] for r in rows]) |
               (; ys = [d.y[r] for r in rows])
        _, qr = _check_retained(plan, (:pointwise,))
        n = qr.layout.total
        _, _, retained = sampler_value_gradient_and_retained!(qr, zeros(n), zeros(n))
        @test length(retained.pointwise.ys) == 5
        @test all(k -> length(retained.pointwise.ys[k]) == length(rows[k]), 1:5)
    end

    @testset "Poisson whole-vector likelihood" begin
        plan = _retained_poisson(; g = d.g, t = d.t) | (; c = d.c)
        _check_retained(plan, (:pointwise, :likelihood))
    end

    @testset "prior-only model retains an empty pointwise NamedTuple" begin
        plan = bind_data(lower_rkppl(_retained_prior_only, Dict{Symbol,Any}()),
                         Dict{Symbol,Any}())
        _, qr = _check_retained(plan, (:pointwise,))
        _, _, retained = sampler_value_gradient_and_retained!(qr, zeros(2), [0.2, -0.3])
        @test retained.pointwise == NamedTuple()
    end

    @testset "retain names other presets" begin
        plan = _retained_normal(; data...) | (; y = d.y)
        built = build_kernel(plan)
        u = zeros(built.layout.total)
        # refused: the sampler density is already the gradient's value, so
        # retaining it would return the same number twice.
        @test_throws ArgumentError prepare_sampler(built, plan, u;
            backend = _RETAINED_BACKEND, retain = (:sampler,))
        # refused: an unknown preset names no WANT node (a typo must not plan
        # another cut; `workflow_wants`).
        @test_throws ArgumentError prepare_sampler(built, plan, u;
            backend = _RETAINED_BACKEND, retain = (:pointwse,))
        q = prepare_sampler(built, plan, u; backend = _RETAINED_BACKEND)
        # refused: a query prepared without `retain` has nothing retained.
        @test_throws ArgumentError sampler_value_gradient_and_retained!(
            q, similar(u), u)
    end
end
