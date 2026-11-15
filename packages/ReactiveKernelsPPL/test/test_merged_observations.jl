module MergedObservationTests
using Test, ReactiveKernels, ReactiveKernelsPPL, Distributions, LinearAlgebra
import DifferentiationInterface as DI
import Enzyme
import Random

# An observed vector merged with other values (data augmentation): with the
# vector conditioned, each of its entries is an observation (`likelihood`,
# `:pointwise.<name>`), and every other entry of the merge is a density
# factor (`prior`). The oracle is the elementwise density of the merged
# vector, split by where each entry comes from.

structure(spec) = [(e.kind, e.depth) for e in recipe_inventory(spec)
    if e.kind !== :ordinary]

const AUGMENTED = @rkppl begin
    mu ~ Normal(0, 1)
    s ~ Exponential(1)
    below[1:K] .~ restricted.(Flat.(), -Inf, limit)
    value = vcat(measured, below)[order]
    value .~ Normal.(mu .+ shift, s)
end

# `N` rows, every third below its limit.
function augmented_data(N; seed = 1)
    rng = Random.Xoshiro(seed)
    y = randn(rng, N)
    below = [i for i in 1:N if i % 3 == 0]
    seen = setdiff(1:N, below)
    inputs = (; K = length(below), limit = y[below] .+ 0.3, shift = 0.2 .* randn(rng, N),
        order = invperm(vcat(seen, below)))
    return (; inputs, measured = y[seen], seen, below)
end

augmented_plan(d) = AUGMENTED(; d.inputs...) | (; d.measured)

function augmented_oracle(built, d)
    (; shift, order) = d.inputs
    function (u)
        c = constrain(built.layout, u)
        value = vcat(d.measured, c.below)[order]
        terms = logpdf.(Normal.(c.mu .+ shift, c.s), value)
        prior = logpdf(Normal(0, 1), c.mu) + logpdf(Exponential(1), c.s) +
            sum(terms[d.below])
        likelihood = sum(terms[d.seen])
        (; total = prior + likelihood + logjac(built.layout, u), prior, likelihood,
            pointwise = terms[d.seen])
    end
end

function check_split(built, plan, oracle, observed; points = 3, seed = 1)
    n = built.layout.total
    rng = Random.Xoshiro(seed)
    sampler = prepare_sampler(built, plan, zeros(n);
        backend = DI.AutoEnzyme(; mode = Enzyme.Reverse))
    q(name) = (f = prepare_query(built, plan, name); u -> Base.invokelatest(f, u))
    for _ in 1:points
        u = 0.4 .* randn(rng, n)
        expected = oracle(u)
        value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
        @test value ≈ expected.total rtol = 1e-10
        @test q(:likelihood)(u) ≈ expected.likelihood rtol = 1e-10
        @test q(:prior)(u) ≈ expected.prior rtol = 1e-10
        pw = q(:pointwise)(u)
        @test keys(pw) == observed
        @test reduce(vcat, (getproperty(pw, k) for k in observed)) ≈ expected.pointwise rtol = 1e-10
        h = cbrt(eps(Float64))
        fd = map(1:n) do i
            up, dn = copy(u), copy(u)
            up[i] += h
            dn[i] -= h
            (oracle(up).total - oracle(dn).total) / (2h)
        end
        @test gradient ≈ fd rtol = 2e-6 atol = 2e-7
    end
end

@testset "a conditioned vector merged with bounded latents is observed entry by entry" begin
    d = augmented_data(10)
    plan = augmented_plan(d)
    built = build_kernel(plan)
    @test coordinate_names(built.layout) == [:mu, :s, Symbol("below.1"), Symbol("below.2"),
        Symbol("below.3")]
    check_split(built, plan, augmented_oracle(built, d), (:measured,))
    # Bound as an input instead, the vector is not conditioned: the same
    # density, all of it a factor (no automatic conditioning).
    input = bind_data(AUGMENTED(; d.inputs..., d.measured))
    ibuilt = build_kernel(input)
    u = 0.3 .* cos.(1:built.layout.total)
    q(b, p, name) = Base.invokelatest(prepare_query(b, p, name), u)
    @test q(ibuilt, input, :sampler) ≈ q(built, plan, :sampler) rtol = 1e-12
    @test q(ibuilt, input, :likelihood) == 0
    @test isempty(q(ibuilt, input, :pointwise))
    # One program as rows and latents grow.
    shape(d) = structure(build_kernel(augmented_plan(d)).spec)
    @test shape(d) == shape(augmented_data(31; seed = 4))
end

const AUGMENTED_FORMS = (
    # No index: the parts in order.
    unordered = (@rkppl(begin
        mu ~ Normal(0, 1)
        below[1:K] .~ restricted.(Flat.(), -Inf, limit)
        value = [measured; below]
        value .~ Normal.(mu, 0.5)
    end), (d, i) -> (; K = length(d.below), limit = d.limit), (:measured,)),
    # The observed vector after a latent part, read through a broadcast.
    later = (@rkppl(begin
        mu ~ Normal(0, 1)
        gap[1:K] .~ restricted.(Flat.(), 0, Inf)
        value = vcat(limit .- gap, measured)[order]
        value .~ Normal.(mu, 0.5)
    end), (d, i) -> (; K = length(d.below), limit = d.limit,
        order = invperm(vcat(d.below, d.seen))), (:measured,)),
    # Two observed vectors around a latent part.
    two = (@rkppl(begin
        mu ~ Normal(0, 1)
        below[1:K] .~ restricted.(Flat.(), -Inf, limit)
        value = vcat(a, below, b)[order]
        value .~ Normal.(mu, 0.5)
    end), (d, i) -> (; K = length(d.below), limit = d.limit,
        order = invperm(vcat(d.seen[1:i], d.below, d.seen[(i + 1):end]))), (:a, :b)),
)

@testset "merge forms: no index, a later part, two observed vectors" begin
    rng = Random.Xoshiro(3)
    N = 9
    y = randn(rng, N)
    below = [2, 5, 9]
    seen = setdiff(1:N, below)
    d = (; below, seen, limit = y[below] .+ 0.4)
    for (name, (model, inputs, observed)) in pairs(AUGMENTED_FORMS)
        i = 2
        data = observed == (:a, :b) ? (; a = y[seen[1:i]], b = y[seen[(i + 1):end]]) :
            (; measured = y[seen])
        plan = model(; inputs(d, i)...) | data
        built = build_kernel(plan)
        # Where the observed and the latent entries sit in the merge.
        at_seen, at_below = name === :unordered ?
            (1:length(seen), (length(seen) + 1):N) : (seen, below)
        oracle = function (u)
            c = constrain(built.layout, u)
            latent = name === :later ? d.limit .- c.gap : c.below
            value = zeros(eltype(u), N)
            value[at_seen] = y[seen]
            value[at_below] = latent
            terms = logpdf.(Normal(c.mu, 0.5), value)
            prior = logpdf(Normal(0, 1), c.mu) + sum(terms[at_below])
            (; total = prior + sum(terms[at_seen]) + logjac(built.layout, u), prior,
                likelihood = sum(terms[at_seen]), pointwise = terms[at_seen])
        end
        @testset "$name" begin
            check_split(built, plan, oracle, observed; points = 2)
        end
    end
end

@testset "merges the split does not cover are refused by name" begin
    d = augmented_data(7)
    refused(model, data) = try
        build_kernel(model(; d.inputs...) | data)
        nothing
    catch err
        err
    end
    message(err) = err isa SurfaceLoweringError ? sprint(showerror, err) : repr(err)
    # A part reading the observed vector.
    err = refused(@rkppl(begin
        mu ~ Normal(0, 1)
        below[1:K] .~ restricted.(Flat.(), -Inf, limit)
        value = vcat(measured, below .+ sum(measured))[order]
        value .~ Normal.(mu .+ shift, 1)
    end), (; d.measured))
    # refused: an observation cannot depend on a sampled value (each observed entry is observed once, whole).
    @test occursin("its part `below .+ sum(measured)` reads observed data", message(err))
    # The observed vector observed twice.
    err = refused(@rkppl(begin
        mu ~ Normal(0, 1)
        below[1:K] .~ restricted.(Flat.(), -Inf, limit)
        value = vcat(measured, below)[order]
        value .~ Normal.(mu .+ shift, 1)
        measured .~ Normal.(0, 1)
    end), (; d.measured))
    # refused: each observed entry is observed once.
    @test occursin("`measured` is observed by a sampling statement of its own too", message(err))
    # An argument whose shape lowering cannot tell.
    err = refused(@rkppl(begin
        mu ~ Normal(0, 1)
        below[1:K] .~ restricted.(Flat.(), -Inf, limit)
        value = vcat(measured, below)[order]
        value .~ Normal.(mu .+ reverse(shift), 1)
    end), (; d.measured))
    # refused: a gather of a value of unknown shape could index a number.
    @test occursin("cannot tell whether its argument `mu .+ reverse(shift)`", message(err))
    # A definition reading observed data and a sampled value in another form
    # names both reads.
    err = refused(@rkppl(begin
        mu ~ Normal(0, 1)
        below[1:K] .~ restricted.(Flat.(), -Inf, limit)
        value = sort(vcat(measured, below))
        value .~ Normal.(mu .+ shift, 1)
    end), (; d.measured))
    # refused: an observation does not depend on a sampled value (P9).
    @test occursin("reads the observed data `measured` and the sampled value(s) `below`",
        message(err))
    # An index that is not a permutation of the merged entries.
    bad = merge(d.inputs, (; order = vcat(d.inputs.order[2:end], d.inputs.order[2])))
    err = try
        bind_data(AUGMENTED(; bad...) | (; d.measured))
        nothing
    catch e
        e
    end
    # refused: a repeated or missing position would double-count or drop an observation.
    @test err isa ContractValidationError
    @test occursin("needs `order` to be a permutation", sprint(showerror, err))
end

include(joinpath(pkgdir(ReactiveKernels), "test", "jax_python.jl"))
include("jax_ppl_data.jl")

@testset "jax_source of a merged observation" begin
    d = augmented_data(10)
    plan = augmented_plan(d)
    built = build_kernel(plan)
    src = jax_source(built, plan)
    @test occursin("UNTRANSLATED = {}", string(src))
    if !rk_jax_available()
        @info "RK_JAX_PYTHON is not set: JAX parity of a merged observation skipped"
        @test_skip false
    else
        query = prepare_query(built, plan, :sampler)
        u = 0.2 .* randn(Random.Xoshiro(7), built.layout.total)
        r = rk_jax_evaluate(src, "log_density", (u, jax_raw_data(plan, src)...);
            grad = true, prepare = true)
        @test r.value ≈ Base.invokelatest(query, u) rtol = 1e-10
        reference = rk_fd_gradient(u -> Base.invokelatest(query, u), u)
        @test maximum(abs.(r.grad .- reference) ./ max.(1, abs.(reference))) < 1e-5
    end
end

end # module
