module SampledValueIndexingTests
using ReactiveKernels, ReactiveKernelsPPL, Distributions, DifferentiationInterface, Enzyme, Test

const BACKEND = AutoEnzyme(; mode=Enzyme.Reverse)

@kernel vector_scales(r2, phi, scale_a, scale_b) = begin
    tau = [scale_a * sqrt(phi[1] * r2 / (1-r2)),
        scale_b * sqrt(phi[2] * r2 / (1-r2))]
    return tau
end

function simplex_case(; n=nothing, composed=false, alpha=nothing)
    concentration = alpha === nothing ? :([1.2, 1.2]) : :alpha
    value = composed ? :(vector_scales(r2, phi, scale_a, scale_b)) :
        :([scale_a * sqrt(phi[1] * r2 / (1-r2)),
            scale_b * sqrt(phi[2] * r2 / (1-r2))])
    response = n === nothing ? :(y ~ Normal(sum(tau), 1.0)) :
        :(y .~ Normal.(sum(tau), 1.0))
    ast = quote
        r2 ~ Beta(1.2, 1.8)
        phi ~ Dirichlet($concentration)
        scale_a ~ Exponential(0.8)
        scale_b ~ Exponential(0.9)
        tau = $value
        $response
    end
    data = (; y=n === nothing ? 0.2 : 0.2 .* cos.(1:n))
    alpha === nothing || (data = merge(data, (; alpha)))
    plan = lower_rkppl(ast, Tuple(keys(data)); mod=@__MODULE__, conditioned=(:y,))
    bound = bind_data(plan, data)
    return bound, build_kernel(bound), data
end

function node(built, data, name)
    Base.invokelatest(prepare, built.spec; have=(:unconstrained, keys(data)...),
        want=name, bound=data)
end

function reference(layout, data, u)
    p = constrain(layout, u)
    tau = [p.scale_a * sqrt(p.phi[1] * p.r2 / (1-p.r2)),
        p.scale_b * sqrt(p.phi[2] * p.r2 / (1-p.r2))]
    alpha = haskey(data, :alpha) ? data.alpha : [1.2, 1.2]
    prior = logpdf(Beta(1.2, 1.8), p.r2) + logpdf(Dirichlet(alpha), p.phi) +
        logpdf(Exponential(0.8), p.scale_a) + logpdf(Exponential(0.9), p.scale_b)
    likelihood = sum(logpdf.(Normal(sum(tau), 1.0), data.y))
    return (; tau, prior, likelihood, posterior=prior+likelihood+logjac(layout, u))
end

function differences(f, u)
    h = cbrt(eps(Float64))
    [begin
        up, down = copy(u), copy(u)
        up[i] += h
        down[i] -= h
        (f(up)-f(down))/(2h)
    end for i in eachindex(u)]
end

@testset "sampled values compose with indexing inside array expressions" begin
    for n in (nothing, 0, 3, 9), alpha in (nothing, [1.2, 1.5, 0.8])
        bound, built, data = simplex_case(; n, alpha)
        original = deepcopy(data)
        # The transparent function spelling is a supported density control.
        # The reported scalar-response form also exposes the same tau port.
        other, composed, _ = simplex_case(; n, alpha, composed=true)
        @test coordinate_names(built.layout) == coordinate_names(composed.layout)
        tau = node(built, data, :tau)
        control_tau = n === nothing ? node(composed, data, :tau) : nothing
        sampler = prepare_sampler(built, bound, zeros(built.layout.total); backend=BACKEND)
        control = prepare_sampler(composed, other, zeros(composed.layout.total); backend=BACKEND)
        replayed = ReactiveKernelsPPL._eval_kernel_def(kernel_expr(bound, built.layout))
        replay = prepare_query((; spec=replayed, layout=built.layout), bound, :sampler)
        for shift in (0.0, -0.3)
            u = [0.2sin(i)+shift for i in 1:built.layout.total]
            saved = copy(u)
            expected = reference(built.layout, data, u)
            @test Base.invokelatest(tau, u) ≈ expected.tau
            if control_tau !== nothing
                @test Base.invokelatest(control_tau, u) ≈ expected.tau
            end
            @test Base.invokelatest(node(built, data, :prior), u) ≈ expected.prior
            @test Base.invokelatest(node(built, data, :likelihood), u) ≈ expected.likelihood
            @test Base.invokelatest(replay, u) ≈ expected.posterior
            value, grad = sampler_value_and_gradient!(sampler, similar(u), u)
            other_value, other_grad = sampler_value_and_gradient!(control, similar(u), u)
            @test value ≈ expected.posterior
            @test grad ≈ differences(v -> reference(built.layout, data, v).posterior, u) rtol=1e-5 atol=1e-7
            @test value ≈ other_value
            @test grad ≈ other_grad
            @test u == saved
        end
        @test data == original
    end
end

@testset "positional reads of other model values in vector literals" begin
    # Both a sampled ordered value and nested ordinary indexing use the
    # same model-value collector; this is not a Dirichlet special case.
    ast = quote
        c ~ Ordered(Normal(0, 1), 3)
        a ~ Normal(0, 1)
        alias = c[:]
        values = [c[1], alias[3], [a, c[2]][1]]
        y ~ Normal(sum(values), 1.0)
    end
    data = (; y=0.2)
    bound = bind_data(lower_rkppl(ast, (:y,); mod=@__MODULE__, conditioned=(:y,)), data)
    built = build_kernel(bound)
    u = [0.2sin(i) for i in 1:built.layout.total]
    p = constrain(built.layout, u)
    @test Base.invokelatest(node(built, data, :values), u) ≈ [p.c[1], p.c[3], p.a]
    oracle = v -> begin
        p = constrain(built.layout, v)
        sum(logpdf.(Normal(), p.c)) + logpdf(Normal(), p.a) +
            logpdf(Normal(p.c[1]+p.c[3]+p.a, 1.0), data.y) + logjac(built.layout, v)
    end
    sampler = prepare_sampler(built, bound, u; backend=BACKEND)
    value, grad = sampler_value_and_gradient!(sampler, similar(u), u)
    @test value ≈ oracle(u)
    @test grad ≈ differences(oracle, u) rtol=1e-5 atol=1e-7
end
end
