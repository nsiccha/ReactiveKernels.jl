using DifferentiationInterface, Distributions, Enzyme, ReactiveKernels, ReactiveKernelsPPL, Test

function _prior_selection_fixture(n, nobs; prior=:data, iterator=:eachindex)
    domain = iterator === :axes ? zeros(2, n) : zeros(n)
    x = Float64[-0.4 + 0.05i for i in 1:n+3]
    # Unselected mapped indices are deliberately outside the scale domain.
    index = [Int[isodd(i) ? 2 : 1 for i in 1:n]; 9; 9; 9]
    y = Float64[-0.2 + 0.03i for i in 1:nobs]
    inputs = prior === :data ? (; domain, x, y) :
        prior === :mapped ? (; domain, index, y) : (; domain, y)
    range = iterator === :literal ? :(1:$n) :
        iterator === :axes ? :(axes(domain, 2)) : :(eachindex(domain))
    declarations = prior === :data ? Any[] :
        prior === :mapped ? Any[:(scale[1:2] .~ Exponential.(1))] :
        Any[:(x[1:$(n+3)] .~ Normal.(0, 1))]
    definitions = prior === :derived ? Any[:(mu = 2 .* x)] : Any[]
    mean = prior === :mapped ? :(scale[index[j]]) :
        prior === :derived ? :(mu[j]) : :(x[j])
    ast = quote
        anchor ~ Normal(0, 1)
        @plate for j in $range
            z[j] ~ Normal($mean, 1)
        end
        y .~ Normal.(anchor + sum(z), 1)
    end
    ast = Expr(:block, declarations..., definitions..., ast.args...)
    bound = bind_data(lower_rkppl(ast, inputs; conditioned=(:y,)), inputs)
    built = build_kernel(bound)
    z = Float64[0.04i - 0.2 for i in 1:n]
    values = prior === :data ? (; anchor=0.1, z) :
        prior === :mapped ? (; anchor=0.1, scale=[0.8, 1.3], z) :
        (; anchor=0.1, x, z)
    u = unconstrain(built.layout, values)
    pointwise(v) = begin
        p = constrain(built.layout, v)
        logpdf.(Normal(p.anchor + sum(p.z; init=0.0), 1), y)
    end
    oracle(v) = begin
        p = constrain(built.layout, v)
        mu = prior === :data ? x[1:n] :
            prior === :mapped ? p.scale[index[1:n]] :
            prior === :derived ? 2 .* p.x[1:n] : p.x[1:n]
        lp = logpdf(Normal(0, 1), p.anchor) +
            sum(logpdf.(Normal.(mu, 1), p.z); init=0.0) + sum(pointwise(v); init=0.0)
        prior === :data ? lp : prior === :mapped ?
            lp + sum(logpdf.(Exponential(1), p.scale)) + sum(log, p.scale) :
            lp + sum(logpdf.(Normal(0, 1), p.x))
    end
    return (; ast, inputs, bound, built, u, oracle, pointwise, n, prior, iterator)
end

function _check_prior_selection(fx)
    saved = deepcopy(fx.inputs)
    @test fx.bound.n_obs == length(fx.inputs.y)
    extra = fx.prior === :data ? 0 : fx.prior === :mapped ? 2 : fx.n + 3
    @test fx.built.layout.total == 1 + fx.n + extra
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
