include("fixtures/latent_prior_inputs.jl")

@testset "latent prior inputs retain their independent domain" begin
    for prior in (:nested, :named, :data), iterator in (:eachindex, :literal),
            (n, nobs) in ((1, 4), (6, 4), (15, 11))
        _check_latent_prior(_latent_prior_fixture(n, nobs; prior, iterator))
    end
    # An empty prior loop still has active shared scale coordinates, and
    # contributes no cell density beside its independent observations.
    for prior in (:nested, :named), iterator in (:eachindex, :literal)
        _check_latent_prior(_latent_prior_fixture(0, 4; prior, iterator))
    end
end

@testset "mapped latent priors retain gather bounds" begin
    ast = quote
        scale[1:2] .~ Exponential.(1)
        @plate for j in eachindex(index)
            z[j] ~ Normal(0, scale[index[j]])
        end
        y .~ Normal.(sum(z), 1)
    end
    bad = (; index=[1, 2, 3, 1, 2, 1], y=zeros(4))
    # refused: a mapped prior cell selects scale[3] outside its declared
    # two-element domain (standard Julia indexing safety).
    @test_throws "holds indices outside 1:2" bind_data(
        lower_rkppl(ast, bad; conditioned=(:y,)), bad)
end

@testset "prior reads do not waive a column's likelihood dimensions" begin
    fx = _latent_prior_shared_fixture()
    saved = deepcopy(fx.inputs)
    @test fx.bound.n_obs == 10
    @test fx.built.layout.total == 6
    sampler = prepare_sampler(fx.built, fx.bound, fx.u;
        backend=AutoEnzyme(; mode=Enzyme.Reverse))
    value, gradient = sampler_value_and_gradient!(sampler, similar(fx.u), fx.u)
    @test value ≈ fx.oracle(fx.u)
    h = cbrt(eps(Float64))
    reference = map(eachindex(fx.u)) do i
        hi, lo = copy(fx.u), copy(fx.u)
        hi[i] += h; lo[i] -= h
        (fx.oracle(hi) - fx.oracle(lo))/(2h)
    end
    @test gradient ≈ reference rtol=1e-5 atol=1e-7
    @test fx.inputs == saved
    bad = merge(fx.inputs, (; y1=zeros(4)))
    # refused: the same x is used elementwise by the likelihood, where six
    # entries cannot broadcast with four (standard Julia array semantics).
    @test_throws "[x] column length 6 ≠ n_obs 4" bind_data(
        lower_rkppl(fx.ast, bad; conditioned=(:y1, :y2)), bad)
end
