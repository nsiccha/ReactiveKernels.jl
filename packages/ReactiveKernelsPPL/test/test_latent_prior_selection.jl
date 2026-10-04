include("fixtures/latent_prior_selection.jl")

@testset "selected prior reads retain other input consumers" begin
    fx = _prior_selection_fixture(6, 4)
    ast = Expr(:block, fx.ast.args..., :(y1 .~ Normal.(x, 1)))
    inputs = merge(fx.inputs, (; y1=zeros(9)))
    saved = deepcopy(inputs)
    bound = bind_data(lower_rkppl(ast, inputs; conditioned=(:y, :y1)), inputs)
    built = build_kernel(bound)
    @test bound.n_obs == 13
    @test built.layout.total == length(fx.u)
    sampler = prepare_sampler(built, bound, fx.u;
        backend=AutoEnzyme(; mode=Enzyme.Reverse))
    baseline = prepare_sampler(fx.built, fx.bound, fx.u;
        backend=AutoEnzyme(; mode=Enzyme.Reverse))
    value, gradient = sampler_value_and_gradient!(sampler, similar(fx.u), fx.u)
    _, expected = sampler_value_and_gradient!(baseline, similar(fx.u), fx.u)
    @test value ≈ fx.oracle(fx.u) + sum(logpdf.(Normal.(inputs.x, 1), inputs.y1))
    @test gradient ≈ expected
    @test inputs == saved
    bad = merge(inputs, (; y1=zeros(4)))
    # refused: the whole x likelihood still pairs nine x entries with y1.
    @test_throws "[x] column length 9 ≠ n_obs 4" bind_data(
        lower_rkppl(ast, bad; conditioned=(:y, :y1)), bad)
end

@testset "indexed prior columns retain Julia bounds" begin
    for iterator in (:eachindex, :literal, :axes)
        fx = _prior_selection_fixture(6, 4; iterator)
        for len in (0, 1, 5)
            inputs = merge(fx.inputs, (; x=zeros(len)))
            # refused: x[j] reads all six authored indices; an indexed
            # singleton does not broadcast as a shared scalar.
            @test_throws "BoundsError" begin
                bound = bind_data(lower_rkppl(fx.ast, inputs; conditioned=(:y,)), inputs)
                built = build_kernel(bound)
                query = prepare_query(built, bound, :sampler)
                Base.invokelatest(query, fx.u)
            end
        end
    end
    fx = _prior_selection_fixture(6, 4; prior=:mapped)
    inputs = merge(fx.inputs, (; index=[1,2,9,1,2,1,9,9,9]))
    # refused: a selected mapped index lies outside the two-scale domain.
    @test_throws "holds indices outside 1:2" bind_data(
        lower_rkppl(fx.ast, inputs; conditioned=(:y,)), inputs)
end

@testset "latent priors select their authored cells" begin
    for prior in (:mapped, :data, :active, :derived),
            iterator in (:eachindex, :literal, :axes),
            (n, nobs) in ((0, 4), (1, 4), (6, 4), (15, 11))
        _check_prior_selection(_prior_selection_fixture(n, nobs; prior, iterator))
    end
end
