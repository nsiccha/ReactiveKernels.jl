# The non-allocating sampler (`prepare_sampler`'s default, `nonallocating =
# :auto`, and `nonallocating = true`): the sampler density and its reverse
# gradient prepared over `prepare_nonallocating`. The query owns the buffers
# its operations write into, so an evaluation reuses them instead of
# allocating observation-length intermediates. Values and gradients equal the
# ordinary sampler's (`nonallocating = false`). Runs through
# `test/run_nonallocating_integration.jl`, whose environment pins the
# unregistered MutatingFunctions.
using ReactiveKernels
using ReactiveKernelsPPL
using MutatingFunctions
using DifferentiationInterface: AutoEnzyme
import Enzyme
using LinearAlgebra: I
using Test

const NA_SAMPLER_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

# A Gaussian regression whose location is a matrix-vector product: the
# ordinary kernel allocates the product on every evaluation.
_na_regression = @rkppl begin
    alpha ~ Normal(0, 10)
    beta[axes(X, 2)] .~ Normal.(0, 10)
    sigma ~ HalfCauchy(2.5)
    mu = alpha .+ X * beta
    y .~ Normal.(mu, sigma)
end

# A one-step error recurrence through `@scan`.
_na_recurrence = @rkppl begin
    mu ~ Normal(0, 10)
    phi ~ Normal(0, 2)
    theta ~ Normal(0, 2)
    sigma ~ HalfCauchy(2.5)
    @scan begin
        nu[1] = mu + phi * mu
        err[1] = y[1] - nu[1]
        for t in 2:T
            nu[t] = mu + phi * y[t - 1] + theta * err[t - 1]
            err[t] = y[t] - nu[t]
        end
    end
    y .~ Normal.(nu, sigma)
end

# Per-group latents in a plate.
_na_groups = @rkppl begin
    mu ~ Normal(0, 5)
    tau ~ HalfCauchy(5)
    @plate for j in eachindex(y)
        theta[j] ~ Normal(mu, tau)
        y[j] ~ Normal(theta[j], sigma[j])
    end
end

# Interval-bounded scalar coefficients beside three data columns: RKPPL
# gathers their constrained values into one coefficient vector,
# `vcat([a], [b], [c], [d])`, for a design-matrix product.
_na_affine = @rkppl begin
    a ~ Uniform(-5, 5)
    b ~ Uniform(-5, 5)
    c ~ Uniform(-5, 5)
    d ~ Uniform(-5, 5)
    rate = a .+ b .* x1 .+ c .* x2 .+ d .* x3
    n .~ Poisson.(exp.(rate))
end

# An interval-bounded coefficient vector: its log-Jacobian broadcasts over the
# negated unconstrained values, `log1pexp.(-u)`.
_na_bounded = @rkppl begin
    alpha ~ Normal(0, 10)
    beta[axes(X, 2)] .~ Uniform.(-5, 5)
    sigma ~ HalfCauchy(2.5)
    mu = alpha .+ X * beta
    y .~ Normal.(mu, sigma)
end

# A simplex: its stick-breaking transform returns the weights and their
# log-Jacobian together.
_na_simplex = @rkppl begin
    w ~ Dirichlet(4, 1.5)
    z .~ Categorical(w)
end

# An ordered vector: its transform is a prefix sum of exponentials.
_na_ordered = @rkppl begin
    m ~ Ordered(Normal(0, 3), 2)
    sd ~ HalfNormal(1)
    v .~ Normal.(m[1] .+ m[2] .* t, sd)
end

# A Gaussian process: the density factors the covariance and solves against
# the factor.
_na_process = @rkppl begin
    ell ~ Gamma(4, 0.5)
    amp ~ HalfNormal(1.5)
    noise ~ HalfNormal(0.7)
    C = amp^2 .* exp.(-D ./ (2 * ell^2)) + noise * I
    r[axes(D, 1)] ~ MvNormal(zeros(n), C)
end

const _NA_X = [sin(0.7i + j) for i in 1:60, j in 1:3]
const _NA_Y = [0.5 + 0.8 * _NA_X[i, 1] - 0.3 * _NA_X[i, 2] + 0.2 * cos(1.3i) for i in 1:60]
const _NA_SERIES = [sin(0.3t) + 0.1 * cos(2.1t) for t in 1:80]
const _NA_COUNTS = [round(Int, exp(0.3 + 0.5 * _NA_X[i, 1] - 0.2 * _NA_X[i, 2])) for i in 1:60]
const _NA_LABELS = [1 + mod(3i + i ÷ 5, 4) for i in 1:40]
const _NA_T = [0.1i for i in 1:30]
const _NA_V = [0.2 + 0.7 * _NA_T[i] + 0.3 * sin(1.7i) for i in 1:30]
const _NA_POSITIONS = [0.37i + 0.2 * sin(i) for i in 1:9]
const _NA_D = [(a - b)^2 for a in _NA_POSITIONS, b in _NA_POSITIONS]
const _NA_R = [sin(0.8x) + 0.1 * cos(5x) for x in _NA_POSITIONS]

_na_sampler_cases() = (
    ("matrix-vector regression", _na_regression(; X = _NA_X) | (; y = _NA_Y)),
    ("scan recurrence", _na_recurrence(; T = length(_NA_SERIES)) | (; y = _NA_SERIES)),
    ("group plate", _na_groups(; sigma = [1.5, 0.8, 2.0, 1.1, 0.9]) |
        (; y = [0.4, -1.2, 2.3, 0.1, 0.7])),
    ("scalar coefficients", _na_affine(; x1 = _NA_X[:, 1], x2 = _NA_X[:, 2],
        x3 = _NA_X[:, 3]) | (; n = _NA_COUNTS)),
    ("interval-bounded coefficients", _na_bounded(; X = _NA_X) | (; y = _NA_Y)),
    ("simplex", _na_simplex() | (; z = _NA_LABELS)),
    ("ordered vector", _na_ordered(; t = _NA_T) | (; v = _NA_V)),
    ("Gaussian process", _na_process(; D = _NA_D, n = length(_NA_R)) | (; r = _NA_R)),
)

_na_density_bytes(q, u) = (q(u); q(u); @allocated q(u))
_na_gradient_bytes(q, g, u) = (sampler_value_and_gradient!(q, g, u);
    sampler_value_and_gradient!(q, g, u); @allocated sampler_value_and_gradient!(q, g, u))

@testset "non-allocating sampler equals the ordinary sampler" begin
    for (label, plan) in _na_sampler_cases()
        @testset "$label" begin
            built = build_kernel(plan)
            n = built.layout.total
            us = [[0.1 * sin(k + s) for k in 1:n] for s in 0:2]
            q = prepare_sampler(built, plan, us[1]; backend = NA_SAMPLER_BACKEND, warm = true,
                nonallocating = false)
            qna = prepare_sampler(built, plan, us[1]; backend = NA_SAMPLER_BACKEND,
                warm = true, nonallocating = true)
            @test qna.kernel isa NonAllocatingKernel
            # Several points in turn: the reused buffers carry nothing from
            # one evaluation to the next.
            for u in us
                g, gna = zeros(n), zeros(n)
                value, _ = sampler_value_and_gradient!(q, g, u)
                value_na, returned = sampler_value_and_gradient!(qna, gna, u)
                @test returned === gna
                @test value_na ≈ value rtol = 1e-12
                @test gna ≈ g rtol = 1e-10 atol = 1e-12
                @test qna(u) ≈ q(u) rtol = 1e-12
                @test qna(u) ≈ value_na rtol = 1e-12
            end
            @test qna(Float32.(us[2])) == qna(Vector{Float64}(Float32.(us[2])))
        end
    end
end

# A vector parameter stacked on a matrix parameter, `vcat(permutedims(c), D)`,
# as a class mixture's ordered first row of means over free rows (snag
# `rkppl-vcat-of-pa-decf6725`): native Enzyme reverse refused the
# non-allocating gradient. An ordered vector, a plain vector and its adjoint.
# An ordered vector is a fresh `Vector`, so its row is a `Matrix` beside the
# parameter block's reshaped view: Base's `vcat` of these, the step's result
# for a buffer of another shape, is refused by registered Enzyme (0.13.219)
# even where it never runs (`IllegalTypeAnalysisException` in `_typed_vcat`).
# The Enzyme dogfood line repairs it (Enzyme decision `0prqqvj`).
# `rejectable` pins that rejection broken for the ordered vector; the plain
# vector's row (a reshaped view like the block) and its adjoint must
# differentiate everywhere.
function _na_gradient_or_rejected(q, g, u; rejectable::Bool)
    try
        sampler_value_and_gradient!(q, g, u)
    catch err
        rejectable && err isa Enzyme.Compiler.IllegalTypeAnalysisException || rethrow()
        nothing
    end
end
@kernel _na_column_score(a, b, C) = begin
    x = [a, b]
    total = -0.5 * sum((x .- C) .^ 2)
    return total
end
_na_stacked_ordered = @rkppl begin
    c ~ Ordered(Normal(0, 1), K)
    D[1:1, 1:K] .~ Normal.(0, 1)
    C = vcat(permutedims(c), D)
    @plate for i in eachindex(a)
        a[i] ~ LogDensity(_na_column_score, b[i], C)
    end
end
_na_stacked_vector = @rkppl begin
    c[1:K] .~ Normal.(0, 1)
    D[1:1, 1:K] .~ Normal.(0, 1)
    C = vcat(permutedims(c), D)
    @plate for i in eachindex(a)
        a[i] ~ LogDensity(_na_column_score, b[i], C)
    end
end
_na_stacked_adjoint = @rkppl begin
    c[1:K] .~ Normal.(0, 1)
    D[1:1, 1:K] .~ Normal.(0, 1)
    C = vcat(c', D)
    @plate for i in eachindex(a)
        a[i] ~ LogDensity(_na_column_score, b[i], C)
    end
end

@testset "a vector stacked on a matrix parameter differentiates natively" begin
    for (label, model, rejectable) in (("ordered vector", _na_stacked_ordered, true),
                                       ("vector", _na_stacked_vector, false),
                                       ("adjoint", _na_stacked_adjoint, false))
        @testset "$label" begin
            plan = model(; b = [0.1, 0.2, 0.3], K = 2) | (; a = [0.0, 0.5, -0.2])
            built = build_kernel(plan)
            n = built.layout.total
            us = [[0.1 * sin(k + s) for k in 1:n] for s in 0:2]
            q = prepare_sampler(built, plan, us[1]; backend = NA_SAMPLER_BACKEND,
                nonallocating = false)
            qna = prepare_sampler(built, plan, us[1]; backend = NA_SAMPLER_BACKEND,
                nonallocating = true)
            @test qna.kernel isa NonAllocatingKernel
            for u in us
                g, gna = zeros(n), zeros(n)
                value, _ = sampler_value_and_gradient!(q, g, u)
                result = _na_gradient_or_rejected(qna, gna, u; rejectable)
                if result === nothing
                    @test_broken result !== nothing
                else
                    value_na, _ = result
                    @test value_na ≈ value rtol = 1e-12
                    @test gna ≈ g rtol = 1e-10 atol = 1e-12
                end
            end
        end
    end
end

@testset "non-allocating sampler density reuses its buffers" begin
    plan = first(_na_sampler_cases())[2]
    built = build_kernel(plan)
    u = [0.1 * sin(k) for k in 1:built.layout.total]
    q = prepare_sampler(built, plan, u; backend = NA_SAMPLER_BACKEND, nonallocating = false)
    qna = prepare_sampler(built, plan, u; backend = NA_SAMPLER_BACKEND, nonallocating = true)
    ordinary = Base.invokelatest(_na_density_bytes, q, u)
    reused = Base.invokelatest(_na_density_bytes, qna, u)
    # The ordinary density allocates the product (one 60-element vector; the
    # location and likelihood plates fuse into the total); the non-allocating
    # one reuses its buffer and keeps no observation-length vector.
    @test ordinary >= 60 * sizeof(Float64)
    @test reused < ordinary
    @test reused < 60 * sizeof(Float64)
end

@testset "a warmed non-allocating density and gradient allocate nothing" begin
    # Each shape of `_na_sampler_cases`: a design-matrix product, a scan
    # trajectory after its initial values, a ranged gather into a group plate,
    # a coefficient vector of scalar parameters, an interval transform's
    # log-Jacobian, the simplex and ordered transforms, and an `MvNormal`
    # covariance's factor and solve. The density is measured on the kernel:
    # the query's own call returns its value through a world-age barrier
    # (`SamplerQuery`), which boxes the number it returns. The simplex,
    # ordered and `MvNormal` destinations are RKPPL's own `apply!!` methods
    # (`src/mutating.jl`).
    @test hasmethod(MutatingFunctions.apply!!, Tuple{Vector{Float64},
        typeof(ReactiveKernelsPPL._ordered_constrain_kernel), Vector{Float64}, Int, Int})
    for (label, plan) in _na_sampler_cases()
        @testset "$label" begin
            built = build_kernel(plan)
            u = [0.1 * sin(k) for k in 1:built.layout.total]
            q = prepare_sampler(built, plan, u; backend = NA_SAMPLER_BACKEND,
                warm = true, nonallocating = true)
            @test Base.invokelatest(_na_density_bytes, q.kernel, u) == 0
            bytes = Base.invokelatest(_na_gradient_bytes, q, zeros(length(u)), u)
            if label == "Gaussian process"
                # A factored sampled covariance differentiates the ordinary
                # program (decision `0tqz3lo`), so its gradient allocates what
                # the ordinary gradient does, and no more.
                ordinary = prepare_sampler(built, plan, u; backend = NA_SAMPLER_BACKEND,
                    warm = true, nonallocating = false)
                @test 0 < bytes <= Base.invokelatest(_na_gradient_bytes, ordinary,
                    zeros(length(u)), u)
            else
                @test bytes == 0
            end
        end
    end
end

@testset "a bound number's value keeps the non-allocating sampler's type" begin
    # The user's direction on numeric values (`ODLPlaus:precompile` decision
    # `2026-11-06T01-09-46-212-0ikeidi`): another value is a runtime input, so
    # it reuses the compiled density and gradient. `z[idx]` gives the gather its
    # index-run metadata, a tuple of arrays and numbers.
    source = @rkppl begin
        z[1:z_length] .~ LogNormal.(mu, sd)
        y .~ Normal.(z[idx], noise)
    end
    function density(u, d)
        lp = 0.0
        for ui in u
            lp += -(ui - d.mu)^2 / (2d.sd^2) - log(d.sd) - log(2pi) / 2
        end
        for (j, i) in enumerate(d.idx)
            lp += -(d.y[j] - exp(u[i]))^2 / (2d.noise^2) - log(d.noise) - log(2pi) / 2
        end
        lp
    end
    function density_gradient(u, d)
        g = [-(ui - d.mu) / d.sd^2 for ui in u]
        for (j, i) in enumerate(d.idx)
            g[i] += (d.y[j] - exp(u[i])) * exp(u[i]) / d.noise^2
        end
        g
    end
    # Both schedules of the gather: flat, and grouped from a mean run of 8. The
    # schedule is the one finite class a binding may change a body by, so each
    # keeps one type across values and row counts on its own.
    for idx_of in (n -> repeat(1:3, n), n -> repeat(1:3, inner = 8n))
        d(mu, sd, noise, n) = begin
            idx = idx_of(n)
            (; y = collect(range(0.1, 0.9; length = length(idx))), idx, z_length = 3,
               mu, sd, noise)
        end
        A = d(0.1, 0.5, 0.2, 1)
        fit = source(; A.idx, A.z_length, A.mu, A.sd, A.noise) | (; A.y)
        built = build_kernel(fit)
        u = [0.1, -0.2, 0.3]
        bindings = (A, d(0.7, 1.3, 0.4, 1), d(-0.4, 0.9, 0.3, 2), A)
        queries = map(b -> prepare_sampler(built, bind_data(fit, b), u;
            backend = NA_SAMPLER_BACKEND, nonallocating = true), bindings)
        @test allequal(typeof.(queries))
        # Each query reads its own values from the one compiled body, and they
        # stay its own when the others run between its calls.
        for _ in 1:2, (b, q) in zip(bindings, queries)
            g = similar(u)
            value, _ = sampler_value_and_gradient!(q, g, u)
            @test value ≈ density(u, b)
            @test q(u) ≈ density(u, b)
            @test g ≈ density_gradient(u, b)
        end
        # Queries of one type never share a slot, neither the density's nor the
        # gradient's.
        slots(k) = filter(slot -> slot isa Base.RefValue, collect(k.caches))
        @test all(queries[i].kernel.caches !== queries[j].kernel.caches
                  for i in eachindex(queries), j in eachindex(queries) if i != j)
        @test all(a !== b for i in eachindex(queries), j in eachindex(queries) if i != j
                  for a in slots(queries[i].kernel), b in slots(queries[j].kernel))
    end
end

@testset "non-allocating programs keep one type across bound numbers" begin
    # Bound prior and response scales in the other shapes of this file.
    regression = @rkppl begin
        alpha ~ Normal(0, prior_scale)
        beta[axes(X, 2)] .~ Normal.(0, prior_scale)
        mu = alpha .+ X * beta
        y .~ Normal.(mu, sigma)
    end
    recurrence = @rkppl begin
        mu ~ Normal(0, prior_scale)
        phi ~ Normal(0, 2)
        sigma ~ HalfCauchy(2.5)
        @scan begin
            nu[1] = mu + phi * mu
            for t in 2:T
                nu[t] = mu + phi * y[t - 1]
            end
        end
        y .~ Normal.(nu, sigma)
    end
    groups = @rkppl begin
        mu ~ Normal(0, prior_scale)
        tau ~ HalfCauchy(5)
        @plate for j in eachindex(y)
            theta[j] ~ Normal(mu, tau)
            y[j] ~ Normal(theta[j], sigma[j])
        end
    end
    cases = (
        (regression, (; X = _NA_X), (; y = _NA_Y),
         (; prior_scale = 10.0, sigma = 0.7), (; prior_scale = 3.0, sigma = 1.4)),
        (recurrence, (; T = length(_NA_SERIES)), (; y = _NA_SERIES),
         (; prior_scale = 10.0), (; prior_scale = 3.0)),
        (groups, (; sigma = [1.5, 0.8, 2.0, 1.1, 0.9]), (; y = [0.4, -1.2, 2.3, 0.1, 0.7]),
         (; prior_scale = 5.0), (; prior_scale = 2.0)),
    )
    for (source, fixed, observed, A, B) in cases
        fit = source(; fixed..., A...) | observed
        built = build_kernel(fit)
        u = [0.1 * sin(k) for k in 1:built.layout.total]
        bound(v) = bind_data(fit, (; fixed..., observed..., v...))
        qA = prepare_sampler(built, bound(A), u; backend = NA_SAMPLER_BACKEND,
            nonallocating = true)
        qB = prepare_sampler(built, bound(B), u; backend = NA_SAMPLER_BACKEND,
            nonallocating = true)
        @test typeof(qA) === typeof(qB)
        ordinary = prepare_sampler(built, bound(B), u; backend = NA_SAMPLER_BACKEND,
            nonallocating = false)
        g, gna = similar(u), similar(u)
        value, _ = sampler_value_and_gradient!(ordinary, g, u)
        value_na, _ = sampler_value_and_gradient!(qB, gna, u)
        @test value_na ≈ value rtol = 1e-12
        @test gna ≈ g rtol = 1e-10 atol = 1e-12
        @test qB(u) ≈ value rtol = 1e-12
        # The two values are a real difference, and qA keeps its own.
        gA = similar(u)
        @test first(sampler_value_and_gradient!(qA, gA, u)) ≉ value
    end
end

@testset "non-allocating sampler options" begin
    plan = first(_na_sampler_cases())[2]
    built = build_kernel(plan)
    u = [0.1 * sin(k) for k in 1:built.layout.total]
    # Eligibility for native forward batching excludes non-allocating kernels:
    # the preparation stays the ordinary reverse one and agrees.
    qf = prepare_sampler(built, plan, u; backend = AutoEnzyme(),
        nonallocating = true, native_forward_batch = true)
    q = prepare_sampler(built, plan, u; backend = NA_SAMPLER_BACKEND, nonallocating = false)
    g, gf = zeros(length(u)), zeros(length(u))
    @test first(sampler_value_and_gradient!(qf, gf, u)) ≈
        first(sampler_value_and_gradient!(q, g, u)) rtol = 1e-12
    @test gf ≈ g rtol = 1e-10 atol = 1e-12
    # Retained outputs over a non-allocating kernel are not built in
    # ReactiveKernels yet (capability gap, not a refused shape).
    @test_broken (prepare_sampler(built, plan, u; backend = NA_SAMPLER_BACKEND,
        nonallocating = true, retain = (:pointwise,)); true)
end

# The default, `nonallocating = :auto` (decision `0nxki1e`): the non-allocating
# sampler with reverse-mode `AutoEnzyme`, the ordinary one where the request
# needs it, and the same values either way.
@testset "the sampler is non-allocating by default" begin
    plan = first(_na_sampler_cases())[2]
    built = build_kernel(plan)
    u = [0.1 * sin(k) for k in 1:built.layout.total]
    ordinary = prepare_sampler(built, plan, u; backend = NA_SAMPLER_BACKEND,
        nonallocating = false)
    @test !(ordinary.kernel isa NonAllocatingKernel)
    g = zeros(length(u))
    value, _ = sampler_value_and_gradient!(ordinary, g, u)
    agrees(q) = (gq = zeros(length(u)); vq = first(sampler_value_and_gradient!(q, gq, u));
        isapprox(vq, value; rtol = 1e-12) && isapprox(gq, g; rtol = 1e-10, atol = 1e-12))
    for backend in (NA_SAMPLER_BACKEND, AutoEnzyme())
        q = prepare_sampler(built, plan, u; backend)
        @test q.kernel isa NonAllocatingKernel
        @test typeof(q) === typeof(prepare_sampler(built, plan, u; backend,
            nonallocating = :auto))
        @test agrees(q)
    end
    # The requests that keep the ordinary sampler under `:auto`.
    retained = prepare_sampler(built, plan, u; backend = NA_SAMPLER_BACKEND,
        retain = (:pointwise,))
    @test !(retained.kernel isa NonAllocatingKernel)
    @test agrees(retained)
    batched = prepare_sampler(built, plan, u; backend = AutoEnzyme(),
        native_forward_batch = true)
    @test !(batched.kernel isa NonAllocatingKernel)
    @test agrees(batched)
    forward = prepare_sampler(built, plan, u; backend = AutoEnzyme(; mode = Enzyme.Forward))
    @test !(forward.kernel isa NonAllocatingKernel)
    @test agrees(forward)
    # The opt-in plate reverse differentiates the ordinary program; its own
    # gradients are `test_plate_reverse.jl`'s. (A destructuring assignment's
    # program: `test_destructuring.jl`.)
    reversed = ReactiveKernels.with_plate_reverse() do
        prepare_sampler(built, plan, u; backend = NA_SAMPLER_BACKEND)
    end
    @test !(reversed.kernel isa NonAllocatingKernel)
    @test_throws ArgumentError prepare_sampler(built, plan, u;
        backend = NA_SAMPLER_BACKEND, nonallocating = :always)
end
