using Distributions, ReactiveKernelsPPL, Test

@testset "catalogue spellings are ordinary quantity names" begin
    data = (; x = [-0.4, 0.2, 0.7], y = [0.1, -0.2, 0.3])
    before = deepcopy(data)
    # USER 1cmodra (names) removes reservations that exist only for the
    # statistical catalogue. These programs use no catalogue construct.
    for name in (:dummy, :mm, :gr, :spline, :spline_basis, :hsgp, :hsgp_basis, :r2d2)
        prior_model = quote
            $name ~ Normal(0, 1)
            y .~ Normal.($name, 1)
        end
        prior_plan = lower_rkppl(prior_model, data; conditioned = (:y,))
        prior_bound = bind_data(prior_plan, data)
        prior_built = build_kernel(prior_bound)
        @test coordinate_names(prior_built.layout) == [name]
        values = NamedTuple{(name,)}((0.3,))
        u = unconstrain(prior_built.layout, values)
        @test Base.invokelatest(prepare_query(prior_built, prior_bound, :sampler), u) ≈
            logpdf(Normal(), 0.3) + sum(logpdf.(Normal(0.3, 1), data.y))
        assigned_model = quote
            a ~ Normal(0, 1)
            $name = a .+ x
            y .~ Normal.($name, 1)
        end
        assigned_plan = lower_rkppl(assigned_model, data; conditioned = (:y,))
        assigned_bound = bind_data(assigned_plan, data)
        assigned_built = build_kernel(assigned_bound)
        @test coordinate_names(assigned_built.layout) == [:a]
        u = unconstrain(assigned_built.layout, (; a = 0.3))
        @test Base.invokelatest(prepare_query(assigned_built, assigned_bound, :sampler), u) ≈
            logpdf(Normal(), 0.3) + sum(logpdf.(Normal.(0.3 .+ data.x, 1), data.y))
    end
    @test isequal(data, before)
end
