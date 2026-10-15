using Test, ReactiveKernelsPPL, ReactiveKernels, DifferentiationInterface, Enzyme
using Distributions: Normal, logpdf

@testset "metadata proves singleton broadcasting without changing density roles" begin
    model = @rkppl begin
        b[1:1] .~ Normal.(0, 1)
        mu = b .+ x
        y .~ Normal.(mu, 1)
    end
    for n in (0, 1, 5)
        x, y = [0.1i for i in 1:n], [-0.2i for i in 1:n]
        saved = deepcopy((x, y))
        plan = model(; x) | (; y)
        built = build_kernel(plan)
        u, gradient = [0.3], zeros(1)
        sampler = prepare_sampler(built, plan, u;
            backend=AutoEnzyme(; mode=Enzyme.Reverse))
        value, _ = sampler_value_and_gradient!(sampler, gradient, u)
        likelihood = sum(logpdf.(Normal(), y .- (u .+ x)))
        @test value ≈ logpdf(Normal(), only(u)) + likelihood
        @test gradient ≈ [-only(u) + sum(y .- (u .+ x))]
        # Raw query kernels are constructed inside this compiled test body;
        # enter their new world as required by prepare_query's contract.
        @test Base.invokelatest(prepare_query(built, plan, :prior), u) ≈ logpdf(Normal(), only(u))
        @test Base.invokelatest(prepare_query(built, plan, :likelihood), u) ≈ likelihood
        @test Base.invokelatest(prepare_query(built, plan, :pointwise), u).y ≈
            logpdf.(Normal(), y .- (u .+ x))
        @test (x, y) == saved
    end
end

@testset "unknown numerical facts retain sampled dependence" begin
    PPL, OS = ReactiveKernelsPPL, ReactiveKernels.OutputSignatures
    states = Dict(:opaque=>PPL._PointwiseValue(OS.ValueInfo(), false),
        :live=>PPL._PointwiseValue(OS.vector_info(Float64, 3), true))
    # The unknown left operand stops numeric analysis before the live right
    # operand. The fallback must still carry the whole expression's dependence.
    expression = :(opaque .+ live)
    emitted, state = PPL._pointwise_rhs(expression, states, Dict(), Dict(), keys(states))
    @test emitted === expression
    @test OS.value_rank(state.info) === nothing
    @test state.live
end
