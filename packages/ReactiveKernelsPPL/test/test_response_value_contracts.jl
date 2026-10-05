using DifferentiationInterface, Distributions, Enzyme
using ReactiveKernels, ReactiveKernelsPPL, Test

# All controls supply complete response arrays, including ordinary empty
# arrays. Priors and value computations are authored in the same model.
function _rvc_build(ast, data; conditioned = (:y,))
    bound = bind_data(lower_rkppl(ast, keys(data); conditioned), data)
    built = build_kernel(bound)
    u = [0.15cos(i) for i in 1:built.layout.total]
    return (; ast, data, bound, built, u)
end

function _rvc_beta(n; singleton = false, form = :bare)
    p = singleton ? 1 : n
    location = form === :bare ? :theta : :shape
    ast = quote
        m ~ Normal(0, 1)
        @plate for j in eachindex(domain)
            theta[j] ~ LogNormal(m, 0.4)
        end
    end
    form === :alias && push!(ast.args, :(shape = theta))
    form === :computed && push!(ast.args, :(shape = exp.(theta)))
    push!(ast.args, :(y .~ Beta.($location, 2.0)))
    fx = _rvc_build(ast, Dict{Symbol,Any}(:y => [0.2 + 0.1mod(i, 5) for i in 1:n],
        :domain => collect(1:p)))
    return (; fx..., form)
end

function _rvc_beta_reference(fx, u)
    th = constrain(fx.built.layout, u)
    shape = fx.form === :computed ? exp.(th.theta) : th.theta
    lp = sum(logpdf.(Beta.(shape, 2.0), fx.data[:y]); init = 0.0)
    return lp + logpdf(Normal(), th.m) +
        sum(logpdf.(LogNormal(th.m, 0.4), th.theta); init = 0.0) +
        logjac(fx.built.layout, u)
end

function _rvc_scan(n; singleton = false, family = :stopping, alias = false)
    hi = singleton || n == 0 ? 1 : :T
    ast = quote
        phi ~ Normal(0, 1)
    end
    if family === :cumulative
        push!(ast.args, :(c ~ Ordered(Normal(0, 1), 2)))
    else
        push!(ast.args, :(c[1:2] .~ Normal.(0, 1)))
    end
    append!(ast.args, (quote
        @scan begin
            h[1] = phi
            for t in 2:$hi
                h[t] = 0.6 * h[t-1] + phi
            end
        end
    end).args)
    alias && push!(ast.args, :(location = h))
    loc = alias ? :location : :h
    law = family === :cumulative ? :(Cumulative()) : :(StoppingRatio())
    push!(ast.args, :(y .~ Ordinal.($law, LogitLink(), $loc, Ref(c), 1.0)))
    fx = _rvc_build(ast, Dict{Symbol,Any}(:y => [mod1(i, 3) for i in 1:n]))
    return (; fx..., singleton, family)
end

function _rvc_scan_reference(fx, u)
    th = constrain(fx.built.layout, u)
    h = th.phi
    lp = 0.0
    for y in fx.data[:y]
        F(j) = cdf(Logistic(), th.c[j] - h)
        if fx.family === :cumulative
            lp += log((y == 3 ? 1.0 : F(y)) - (y == 1 ? 0.0 : F(y-1)))
        else
            lp += sum((log1p(-F(j)) for j in 1:y-1); init = 0.0)
            y == 3 || (lp += log(F(y)))
        end
        fx.singleton || (h = 0.6h + th.phi)
    end
    return lp + logpdf(Normal(), th.phi) + sum(logpdf.(Normal(), th.c)) +
        logjac(fx.built.layout, u)
end

function _rvc_effects(n; rows = n, cols = 2)
    ast = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        c[1:2] .~ Normal.(0, 1)
        y .~ Ordinal.(StoppingRatio(), LogitLink(), eta, Ref(c), 1.0, eachrow(E))
    end
    data = Dict{Symbol,Any}(:y => [mod1(i, 3) for i in 1:n],
        :x => [0.2sin(i) for i in 1:n],
        :E => [0.1cos(i+j) for i in 1:rows, j in 1:cols])
    return _rvc_build(ast, data)
end

function _rvc_effects_reference(fx, u)
    th = constrain(fx.built.layout, u)
    eta = th.a .+ th.b .* fx.data[:x]
    lp = 0.0
    for i in eachindex(eta)
        y = fx.data[:y][i]
        F(j) = cdf(Logistic(), th.c[j] - eta[i] - fx.data[:E][i,j])
        lp += sum((log1p(-F(j)) for j in 1:y-1); init = 0.0)
        y == 3 || (lp += log(F(y)))
    end
    return lp + logpdf(Normal(), th.a) + logpdf(Normal(), th.b) +
        sum(logpdf.(Normal(), th.c)) + logjac(fx.built.layout, u)
end

function _rvc_native(fx, reference)
    before = deepcopy(fx.data)
    kernel = prepare_query(fx.built, fx.bound, :sampler)
    @test Base.invokelatest(kernel, fx.u) ≈ reference(fx, fx.u) rtol = 1e-11
    sampler = prepare_sampler(fx.built, fx.bound, fx.u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    value, grad = sampler_value_and_gradient!(sampler, similar(fx.u), fx.u)
    @test value ≈ reference(fx, fx.u) rtol = 1e-11
    step = cbrt(eps(Float64))
    fd = map(eachindex(fx.u)) do i
        up, dn = copy(fx.u), copy(fx.u)
        up[i] += step
        dn[i] -= step
        (reference(fx, up) - reference(fx, dn)) / (2step)
    end
    @test grad ≈ fd rtol = 1e-5 atol = 1e-7
    @test isequal(fx.data, before)
    return (; sampler, grad)
end

@testset "response value contracts: density and ordinary reverse" begin
    for n in (0, 1, 4, 8), form in (:bare, :alias, :computed)
        _rvc_native(_rvc_beta(n; form), _rvc_beta_reference)
    end
    for family in (:stopping, :cumulative), n in (0, 1, 4, 8)
        _rvc_native(_rvc_scan(n; family, alias = true), _rvc_scan_reference)
    end
    for n in (0, 1, 4, 8)
        _rvc_native(_rvc_effects(n), _rvc_effects_reference)
    end
end

@testset "response value contracts: singleton values and matrix dimensions" begin
    for n in (0, 4), form in (:bare, :alias, :computed)
        _rvc_native(_rvc_beta(n; singleton = true, form), _rvc_beta_reference)
    end
    for family in (:stopping, :cumulative), n in (0, 4)
        _rvc_native(_rvc_scan(n; singleton = true, family), _rvc_scan_reference)
    end
    for (rows, cols) in ((4, 1), (4, 3), (1, 2), (3, 2), (5, 2))
        error = rows in (1, 4) ? DimensionMismatch : ContractValidationError
        @test_throws error _rvc_effects(4; rows, cols)
    end
end

@testset "response value contracts: row roles belong to each reader" begin
    ast = quote
        a ~ Normal(0, 1)
        c[1:2] .~ Normal.(0, 1)
        y .~ Ordinal.(StoppingRatio(), LogitLink(), a, Ref(c), 1.0, eachrow(E))
        z .~ Normal.(E, 1.0)
    end
    E = [0.1cos(i+j) for i in 1:4, j in 1:2]
    data = Dict{Symbol,Any}(:y => [1, 2, 3, 2], :z => 0.2 .* E, :E => E)
    fx = _rvc_build(ast, data; conditioned = (:y, :z))
    @test ReactiveKernelsPPL._response_rows(fx.bound, fx.bound.responses[1]) == 4
    @test ReactiveKernelsPPL._response_rows(fx.bound, fx.bound.responses[2]) == 8
    function reference(fx, u)
        th = constrain(fx.built.layout, u)
        lp = logpdf(Normal(), th.a) + sum(logpdf.(Normal(), th.c)) +
            logjac(fx.built.layout, u)
        for i in eachindex(fx.data[:y])
            y = fx.data[:y][i]
            F(j) = cdf(Logistic(), th.c[j] - th.a - fx.data[:E][i,j])
            lp += sum((log1p(-F(j)) for j in 1:y-1); init = 0.0)
            y == 3 || (lp += log(F(y)))
        end
        return lp + sum(logpdf.(Normal.(fx.data[:E], 1.0), fx.data[:z]))
    end
    _rvc_native(fx, reference)
end
