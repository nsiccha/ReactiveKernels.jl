using DifferentiationInterface, Distributions, Enzyme, ReactiveKernels, ReactiveKernelsPPL, Test

function _latent_prior_fixture(n, nobs; prior=:nested, iterator=:eachindex)
    index = Int[isodd(i) ? 2 : 1 for i in 1:n]
    frequency = Float64[0.03i for i in 1:n]
    data = (; index, frequency, scale_data=[0.8, 1.3],
        y=Float64[-0.2 + 0.04i for i in 1:nobs])
    range = iterator === :literal ? :(1:$n) : :(eachindex(index))
    head = prior === :data ? quote scale = scale_data end :
        quote scale[1:2] .~ Exponential.(1.0) end
    body = prior === :nested ? quote
        @plate for j in $range
            z[j] ~ Normal(0, scale[index[j]])
        end
    end : quote
        prior_scale = exp.(frequency) .* scale[index]
        @plate for j in $range
            z[j] ~ Normal(0, prior_scale[j])
        end
    end
    ast = Expr(:block, head.args..., body.args..., :(y .~ Normal.(sum(z), 1.0)))
    inputs = prior === :data ? data : (; data.index, data.frequency, data.y)
    bound = bind_data(lower_rkppl(ast, inputs; conditioned=(:y,)), inputs)
    built = build_kernel(bound)
    z = Float64[0.07i - 0.2 for i in 1:n]
    values = prior === :data ? (; z) : (; scale=[0.8, 1.3], z)
    u = unconstrain(built.layout, values)
    pointwise = v -> logpdf.(Normal(sum(constrain(built.layout, v).z; init=0.0), 1), data.y)
    oracle = v -> begin
        p = constrain(built.layout, v)
        scale = prior === :data ? data.scale_data : p.scale
        sd = prior === :nested ? scale[index] : exp.(frequency) .* scale[index]
        lp = sum(logpdf.(Normal.(0, sd), p.z); init=0.0) + sum(pointwise(v); init=0.0)
        prior === :data ? lp : lp + sum(logpdf.(Exponential(1), scale)) + sum(log, scale)
    end
    return (; inputs, bound, built, u, oracle, pointwise, prior, n)
end

function _latent_prior_shared_fixture()
    ast = quote
        @plate for j in eachindex(index)
            z[j] ~ Normal(x[j], 1)
        end
        y1 .~ Normal.(x, 1)
        y2 .~ Normal.(sum(z), 1)
    end
    inputs = (; index=collect(1:6), x=collect(range(-0.2, 0.3; length=6)),
        y1=zeros(6), y2=zeros(4))
    bound = bind_data(lower_rkppl(ast, inputs; conditioned=(:y1, :y2)), inputs)
    built = build_kernel(bound)
    u = zeros(6)
    oracle = v -> sum(logpdf.(Normal.(inputs.x, 1), v)) +
        sum(logpdf.(Normal.(inputs.x, 1), inputs.y1)) +
        sum(logpdf.(Normal(sum(v), 1), inputs.y2))
    return (; ast, inputs, bound, built, u, oracle)
end

function _check_latent_prior(fx)
    saved = deepcopy(fx.inputs)
    @test fx.bound.n_obs == length(fx.inputs.y)
    @test fx.built.layout.total == fx.n + (fx.prior === :data ? 0 : 2)
    sampler = prepare_sampler(fx.built, fx.bound, fx.u;
        backend=AutoEnzyme(; mode=Enzyme.Reverse))
    for shift in (0.0, 0.03)
        u = fx.u .+ shift
        original = copy(u)
        value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
        @test value ≈ fx.oracle(u)
        h = cbrt(eps(Float64))
        reference = map(eachindex(u)) do i
            hi, lo = copy(u), copy(u)
            hi[i] += h; lo[i] -= h
            (fx.oracle(hi) - fx.oracle(lo)) / (2h)
        end
        @test gradient ≈ reference rtol=1e-5 atol=1e-7
        pw = Base.invokelatest(prepare_query(fx.built, fx.bound, :pointwise), u)
        @test pw.y ≈ fx.pointwise(u)
        @test u == original
    end
    @test fx.inputs == saved
    return sampler
end
