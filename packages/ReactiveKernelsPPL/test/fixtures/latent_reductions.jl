using DifferentiationInterface, Distributions, Enzyme, ReactiveKernels, ReactiveKernelsPPL, Statistics, Test

function _latent_reduction_fixture(fn, n; position=:inline,
        iterator=:eachindex, n1=2, n2=3)
    # Responses deliberately have different axes, independent of the latent
    # iterator.
    data = (; domain=collect(1:n), y1=[0.1i for i in 1:n1],
        y2=[0.2 + 0.1i for i in 1:n2], x=[-0.5 + 0.3i for i in 1:n2])
    reduction = Expr(:call, fn, :z)
    range = iterator === :literal ? :(1:$n) : :(eachindex(domain))
    ast = quote
        @plate for i in $range
            z[i] ~ Normal(0, 1)
        end
        y1 .~ Normal.(0, 1)
    end
    if position === :indexed
        append!(ast.args, (quote
            @plate for i in 1:$n2
                y2[i] ~ Normal(z[i], 1)
            end
        end).args)
    elseif position === :named
        push!(ast.args, :(location = $reduction))
        push!(ast.args, :(y2 .~ Normal.(location, 1)))
    elseif position === :derived
        push!(ast.args, :(location = x .+ $reduction))
        push!(ast.args, :(y2 .~ Normal.(location, 1)))
    else
        push!(ast.args, :(y2 .~ Normal.($reduction, 1)))
    end
    # x is a response covariate only in the derived-expression case.
    inputs = iterator === :literal ? (; data.y1, data.y2) :
        (; data.domain, data.y1, data.y2)
    position === :derived && (inputs = merge(inputs, (; data.x)))
    plan = lower_rkppl(ast, inputs; conditioned=(:y1, :y2))
    bound = bind_data(plan, inputs)
    built = build_kernel(bound)
    u = unconstrain(built.layout, (; z=Float64[0.2 + 0.07i for i in 1:n]))
    reducer = getfield(Statistics, fn)
    pointwise = v -> begin
        # Normal latents have real support, so v itself is the reference z.
        z = v
        location = position === :indexed ? z[1:n2] : reducer(z)
        mu = position === :derived ? data.x .+ location : location
        (; y1=logpdf.(Normal(), data.y1),
            y2=logpdf.(Normal.(mu, 1), data.y2))
    end
    oracle = v -> begin
        z = v
        pw = pointwise(v)
        sum(logpdf.(Normal(), z); init=0.0) + sum(pw.y1) + sum(pw.y2)
    end
    return (; inputs, bound, built, u, pointwise, oracle)
end

function _check_latent_reduction(fx)
    saved = deepcopy(fx.inputs)
    @test fx.bound.n_obs == length(fx.inputs.y1) + length(fx.inputs.y2)
    @test fx.built.layout.total == length(fx.u)
    sampler = prepare_sampler(fx.built, fx.bound, fx.u;
        backend=AutoEnzyme(; mode=Enzyme.Reverse))
    for shift in (0.0, 0.03)
        u = fx.u .+ shift
        original = copy(u)
        value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
        @test value ≈ fx.oracle(u)
        h = cbrt(eps(Float64))
        reference = similar(u)
        for i in eachindex(u)
            hi, lo = copy(u), copy(u)
            hi[i] += h; lo[i] -= h
            reference[i] = (fx.oracle(hi) - fx.oracle(lo)) / (2h)
        end
        @test gradient ≈ reference rtol=1e-5 atol=1e-7
        pw = Base.invokelatest(prepare_query(fx.built, fx.bound, :pointwise), u)
        @test pw.y1 ≈ fx.pointwise(u).y1
        @test pw.y2 ≈ fx.pointwise(u).y2
        @test u == original
    end
    @test fx.inputs == saved
end
