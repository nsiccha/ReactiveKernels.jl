using DifferentiationInterface, Enzyme, ReactiveKernels, ReactiveKernelsPPL, Test

# A dotted plate cell gathering a shared value by one index vector per index,
# `theta[rater[j]]`: each item is graded by its own subset of raters, and the
# cell observes that item's grades entry by entry. The cell reads the
# gathered entries of `theta` in place, so a density or gradient allocates the
# same for any number of items, and the cumulative-logit terms evaluate one
# log-logistic each.

const _CIG_DIRECT = @rkppl begin
    theta[1:nrater] .~ Normal.(0, 2)
    @plate for j in eachindex(nlevel)
        score[j] .~ Ordinal.(Cumulative(), LogitLink(), theta[rater[j]],
            Ref(cuts[j]), disc[j])
    end
end

const _CIG_LOCAL = @rkppl begin
    theta[1:nrater] .~ Normal.(0, 2)
    @plate for j in eachindex(nlevel)
        ability = theta[rater[j]]
        score[j] .~ Ordinal.(Cumulative(), LogitLink(), ability, Ref(cuts[j]), disc[j])
    end
end

const _CIG_NORMAL = @rkppl begin
    theta[1:nrater] .~ Normal.(0, 2)
    sigma ~ Exponential(1)
    @plate for j in eachindex(nlevel)
        y[j] .~ Normal.(theta[rater[j]], sigma)
    end
end

# Deterministic synthetic items: 2-4 levels, ordered cutpoints, a positive
# discrimination, each item graded by a varying subset of raters (item 3 by
# none).
function _cig_data(nitem, nrater)
    nlevel = [2 + mod(j, 3) for j in 1:nitem]
    rater = [j == 3 ? Int[] : [r for r in 1:nrater if mod(3j + 5r, 7) != 0]
        for j in 1:nitem]
    cuts = [collect(range(-1.5 + 0.1 * mod(j, 4); step = 1.3, length = nlevel[j] - 1))
        for j in 1:nitem]
    disc = [0.6 + 0.35 * mod(j, 4) for j in 1:nitem]
    score = [[1 + mod(j + 2r, nlevel[j]) for r in rater[j]] for j in 1:nitem]
    y = [[0.2 * mod(j + r, 5) - 0.4 for r in rater[j]] for j in 1:nitem]
    return (; nrater, nlevel, rater, cuts, disc, score, y)
end

# Exact 256-bit density and gradient of the ordinal model.
function _cig_ordinal_oracle(u, d)
    setprecision(BigFloat, 256) do
        theta = big.(u)
        F(z) = inv(1 + exp(-z))
        value = sum(t -> -log(2 * big(pi)) / 2 - log(big(2)) - t^2 / 8, theta)
        gradient = -theta ./ 4
        for j in eachindex(d.nlevel), (k, r) in enumerate(d.rater[j])
            c, s, K = big.(d.cuts[j]), big(d.disc[j]), d.nlevel[j]
            g = d.score[j][k]
            a = g == K ? big(0) : s * (c[g] - theta[r])
            b = g == 1 ? big(0) : s * (c[g-1] - theta[r])
            hi = g == K ? big(1) : F(a)
            lo = g == 1 ? big(0) : F(b)
            # `hi - lo` as `(e^a - e^b) / ((1 + e^a)(1 + e^b))`, which never
            # subtracts two values near one: BigFloat's exponent range keeps
            # it exact hundreds of units beyond the cutpoints.
            p = g == 1 ? hi : g == K ? inv(1 + exp(b)) :
                (exp(a) - exp(b)) / ((1 + exp(a)) * (1 + exp(b)))
            value += log(p)
            gradient[r] += s * (hi + lo - 1)
        end
        Float64(value), Float64.(gradient)
    end
end

function _cig_prepare(model, d; observe = (; score = d.score))
    bound = model(; nrater = d.nrater, nlevel = d.nlevel, rater = d.rater,
        cuts = d.cuts, disc = d.disc) | observe
    built = build_kernel(bound)
    u = collect(range(-1.2, 0.9; length = built.layout.total))
    sampler = prepare_sampler(built, bound, u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    gradient = similar(u)
    sampler(u)
    sampler_value_and_gradient!(sampler, gradient, u)
    return (; bound, built, sampler, u, gradient)
end

@testset "dotted cell gathers by one index vector per index: values and gradients" begin
    d = _cig_data(9, 6)
    saved = deepcopy(d)
    direct = _cig_prepare(_CIG_DIRECT, d)
    named = _cig_prepare(_CIG_LOCAL, d)
    # The generated cell reads the gather in place; the authored spelling
    # appears in the model only.
    source = string(kernel_expr(direct.bound, direct.built.layout))
    @test occursin("_ppl_cell_gather(theta", source)
    for fx in (direct, named), shift in (0.0, 0.35)
        u = fx.u .+ shift
        value, gradient = sampler_value_and_gradient!(fx.sampler, similar(u), u)
        ref_value, ref_gradient = _cig_ordinal_oracle(u, d)
        @test value ≈ ref_value rtol = 1e-13
        @test gradient ≈ ref_gradient rtol = 1e-12
        @test fx.sampler(u) ≈ ref_value rtol = 1e-13
        pw = Base.invokelatest(prepare_query(fx.built, fx.bound, :pointwise), u)
        @test length.(pw.score) == length.(d.score)
    end
    @test isequal(d, saved)
end

@testset "dotted cell gathers by one index vector per index: allocations do not grow with items" begin
    measured = map((6, 48)) do nitem
        d = _cig_data(nitem, 6)
        map((_CIG_DIRECT, _CIG_LOCAL)) do model
            fx = _cig_prepare(model, d)
            (; density = (@allocations fx.sampler(fx.u)),
               gradient = (@allocations sampler_value_and_gradient!(
                   fx.sampler, fx.gradient, fx.u)))
        end
    end
    for k in 1:2
        @test measured[1][k].density == measured[2][k].density
        @test measured[1][k].gradient == measured[2][k].gradient
    end
    normal = map((6, 48)) do nitem
        d = _cig_data(nitem, 6)
        fx = _cig_prepare(_CIG_NORMAL, d; observe = (; y = d.y))
        value, gradient = sampler_value_and_gradient!(fx.sampler, fx.gradient, fx.u)
        (; theta, sigma) = constrain(fx.built.layout, fx.u)
        # Exponential(1) prior on sigma, plus the log-Jacobian of its log
        # transform.
        reference = sum(t -> -log(2pi) / 2 - log(2) - t^2 / 8, theta) +
            (-sigma + log(sigma)) +
            sum(sum((-log(2pi) / 2 - log(sigma) - (v - theta[r])^2 / (2sigma^2))
                for (v, r) in zip(d.y[j], d.rater[j]); init = 0.0) for j in 1:nitem)
        (; value, reference, density = (@allocations fx.sampler(fx.u)),
           gradient = (@allocations sampler_value_and_gradient!(
               fx.sampler, fx.gradient, fx.u)))
    end
    @test all(m -> m.value ≈ m.reference, normal)
    @test normal[1].density == normal[2].density
    @test normal[1].gradient == normal[2].gradient
end

@testset "cumulative logit cells stay exact and finite far beyond the cutpoints" begin
    # Abilities hundreds of units from the cutpoints put every log-logistic
    # argument beyond the range where exp overflows; each term evaluates
    # only its stable branch.
    d = _cig_data(5, 4)
    fx = _cig_prepare(_CIG_DIRECT, d)
    for u in ([-400.0, 350.0, -2.0, 1.0], [800.0, -800.0, 40.0, -40.0])
        value, gradient = sampler_value_and_gradient!(fx.sampler, similar(u), u)
        ref_value, ref_gradient = _cig_ordinal_oracle(u, d)
        @test isfinite(value)
        @test all(isfinite, gradient)
        @test value ≈ ref_value rtol = 1e-13
        @test gradient ≈ ref_gradient rtol = 1e-12
    end
end
