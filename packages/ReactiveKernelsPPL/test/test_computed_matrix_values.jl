using ReactiveKernels, ReactiveKernelsPPL, Test
using DifferentiationInterface: AutoEnzyme
import Enzyme

# Public synthetic source: matrix construction and a whole-array leaf
# compose before the observation gather. No statistical emitter is involved.
module ComputedMatrixValueModels
@inline read_rows(v, indices) = v[indices]
@inline matvec(X, coefficients) = X * coefficients
end

function _cm_fixture(mode, intercept, n, nobs)
    x = [0.2 * cos(i) for i in 1:n]
    indices = nobs == 0 ? Int[] : [mod1(2i, n) for i in 1:nobs]
    y = [0.3 * sin(i) for i in 1:nobs]
    X = intercept ? hcat(ones(n), x) : hcat(x)
    ast = quote b ~ Normal(0, 1) end
    intercept && push!(ast.args, :(b0 ~ Normal(0, 1)))
    matrix = intercept ? :(hcat(ones(length(x)), x)) : :(hcat(x))
    mode === :bound || mode === :inline || push!(ast.args, :(X = $matrix))
    push!(ast.args, intercept ? :(coefficients = [b0, b]) : :(coefficients = [b]))
    if mode === :inline
        push!(ast.args, :(a = matvec($matrix, coefficients)))
    elseif mode === :matvec
        push!(ast.args, :(a = matvec(X, coefficients)))
    elseif mode === :alias
        append!(ast.args, [:(M = X), :(a = matvec(M, coefficients))])
    else
        push!(ast.args, :(a = X * coefficients))
    end
    append!(ast.args, [:(mu = read_rows(a, indices)), :(y .~ Normal.(mu, 1))])
    data = mode === :bound ? (; X, indices, y) : (; x, indices, y)
    return (; ast, data, x, X, indices, y, mode, intercept)
end

function _cm_build(fx; ast = fx.ast)
    plan = lower_rkppl(ast, fx.data; mod = ComputedMatrixValueModels,
        conditioned = (:y,))
    bound = bind_data(plan, fx.data)
    return (; plan, bound, built = build_kernel(bound))
end

function _cm_oracle(fx, layout, u)
    q = Dict(zip(coordinate_names(layout), u))
    resid = fx.y .- (get(q, :b0, 0.0) .+ q[:b] .* fx.x[fx.indices])
    prior = -0.5 * length(u) * log(2pi) - 0.5 * sum(abs2, u)
    likelihood = -0.5 * length(resid) * log(2pi) - 0.5 * sum(abs2, resid)
    gradient = [name === :b ? -q[:b] + sum(fx.x[fx.indices] .* resid) :
        -q[:b0] + sum(resid) for name in coordinate_names(layout)]
    return (; prior, likelihood, value = prior + likelihood, gradient)
end

@testset "computed matrices retain whole-value axes and ordinary reverse" begin
    for mode in (:product, :matvec, :bound, :inline, :alias),
            intercept in (false, true), (n, nobs) in ((0, 0), (1, 3), (3, 3), (3, 7))
        @testset "$mode / intercept=$intercept / rows=$n,$nobs" begin
            fx = _cm_fixture(mode, intercept, n, nobs)
            original = deepcopy(fx.data)
            f = _cm_build(fx)
            @test f.bound.n_obs == nobs
            names = coordinate_names(f.built.layout)
            @test Set(names) == (intercept ? Set((:b0, :b)) : Set((:b,)))
            if mode in (:product, :matvec)
                @test f.bound.columns[:X] == fx.X
            end
            snapshots = deepcopy(f.bound.columns)
            u0 = zeros(length(names))
            sampler = prepare_sampler(f.built, f.bound, u0;
                backend = AutoEnzyme(; mode = Enzyme.Reverse))
            for u in (u0, fill(0.13, length(names)),
                    length(names) == 1 ? [-0.2] : [-0.2, 0.3])
                ref = _cm_oracle(fx, f.built.layout, u)
                value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
                @test value ≈ ref.value atol = 1e-12 rtol = 1e-12
                @test gradient ≈ ref.gradient atol = 1e-12 rtol = 1e-12
                @test isequal(fx.data, original)
                @test isequal(f.bound.columns, snapshots)
            end
            # The complete printed program replays through the public API.
            source = sprint(io -> Base.show_unquoted(io, fx.ast))
            replay = _cm_build(fx; ast = Meta.parse(source))
            query = prepare_query(replay.built, replay.bound, :sampler)
            @test Base.invokelatest(query, u0) ≈ _cm_oracle(fx, replay.built.layout, u0).value
        end
    end
end

@testset "matrix recipe sources keep independent observation consumers" begin
    fx = _cm_fixture(:matvec, true, 3, 3)
    ast = deepcopy(fx.ast)
    append!(ast.args, [:(mu2 = b .* x), :(y2 .~ Normal.(mu2, 1))])
    data = merge(fx.data, (; y2 = reverse(fx.y)))
    plan = lower_rkppl(ast, data; mod = ComputedMatrixValueModels,
        conditioned = (:y, :y2))
    bound = bind_data(plan, data)
    built = build_kernel(bound)
    u = [0.15, -0.2]
    q = Dict(zip(coordinate_names(built.layout), u))
    ref = _cm_oracle(fx, built.layout, u)
    resid = data.y2 .- q[:b] .* data.x
    value = ref.value - 0.5 * length(resid) * log(2pi) - 0.5 * sum(abs2, resid)
    gradient = ref.gradient .+ [name === :b ? sum(data.x .* resid) : 0.0
        for name in coordinate_names(built.layout)]
    snapshots = deepcopy(data)
    sampler = prepare_sampler(built, bound, u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    actual, grad = sampler_value_and_gradient!(sampler, similar(u), u)
    @test actual ≈ value
    @test grad ≈ gradient
    @test isequal(data, snapshots)
end
