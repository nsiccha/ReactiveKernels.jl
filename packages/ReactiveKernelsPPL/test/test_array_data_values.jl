using DifferentiationInterface
using Distributions
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# Synthetic whole-value composition: the helper is a leaf identity, with
# no hidden model loop. All observation iteration belongs to the RK plate.
module ArrayDataValueModels
passthrough(v) = v
covariance_factor() = [1.0 0.0; 0.2 0.8]
end

function _adv_model(kind, route)
    declaration, value = if kind === :vector
        :(z[levels(g)] .~ Normal.(0, 1)), :z
    elseif kind === :column
        :(Z[levels(g), 1:2] .~ Normal.(0, 1)), :(Z[:, 1])
    else
        :(begin F = covariance_factor();
            eachrow(Z[levels(g), 1:2]) .~ MvNormalCholesky(zeros(2), F) end),
            :(Z[:, 1])
    end
    expr = :(a .+ $value .+ b .* gx)
    use = if route === :named
        :(begin v = $expr; r = passthrough(v) end)
    elseif route === :inline
        :(r = passthrough($expr))
    elseif route === :alias
        :(begin v = $expr; w = v; r = passthrough(w) end)
    else
        :(r = $expr)
    end
    ast = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        sigma ~ Exponential(1)
    end
    append!(ast.args, declaration.head === :block ? declaration.args : Any[declaration])
    append!(ast.args, use.head === :block ? use.args : Any[use])
    push!(ast.args, :(y .~ Normal.(r[g], sigma)))
    return ast
end

function _adv_data(K, n)
    Dict(:y => [0.2 * sin(i) for i in 1:n],
        :g => [mod1(i, K) for i in 1:n],
        :gx => [0.3 * cos(i) for i in 1:K])
end

function _adv_build(kind, route, K = 3, n = 7)
    data = _adv_data(K, n)
    plan = lower_rkppl(_adv_model(kind, route), keys(data);
        mod = ArrayDataValueModels)
    bound = bind_data(plan, data)
    return (; plan, bound, built = build_kernel(bound), data)
end

function _adv_reference(kind, th, data)
    z = kind === :vector ? th.z : th.Z[:, 1]
    v = th.a .+ z .+ th.b .* data[:gx]
    lik = sum(logpdf.(Normal.(v[data[:g]], th.sigma), data[:y]))
    array_prior = if kind === :vector
        sum(logpdf.(Normal(0, 1), th.z))
    elseif kind === :column
        sum(logpdf.(Normal(0, 1), th.Z))
    else
        F = [1.0 0.0; 0.2 0.8]
        sum(logpdf(MvNormal(zeros(2), F * F'), row) for row in eachrow(th.Z))
    end
    prior = logpdf(Normal(0, 1), th.a) + logpdf(Normal(0, 1), th.b) +
        logpdf(Exponential(1), th.sigma) + array_prior
    return (; v, lik, prior)
end

function _adv_findiff(f, u; h = cbrt(eps(Float64)))
    map(eachindex(u)) do i
        up, dn = copy(u), copy(u)
        up[i] += h
        dn[i] -= h
        (f(up) - f(dn)) / (2h)
    end
end

@testset "array values compose with whole-value data" begin
    for kind in (:vector, :column, :centered),
            route in (:named, :inline, :alias, :gather)
        fx = _adv_build(kind, route)
        @test fx.bound.n_obs == 7
        @test length(fx.bound.columns[:gx]) == 3
        @test :gx in first(ReactiveKernelsPPL._model_level_inputs(fx.plan,
            Set(keys(fx.data))))
        u = [0.25 * cos(i) for i in 1:fx.built.layout.total]
        th = constrain(fx.built.layout, u)
        ref = _adv_reference(kind, th, fx.data)
        query(node) = Base.invokelatest(prepare_query(fx.built, fx.bound, node), u)
        @test query(:likelihood) ≈ ref.lik
        @test query(:prior) ≈ ref.prior
        @test query(:log_jacobian) ≈ logjac(fx.built.layout, u)
        q = prepare_sampler(fx.built, fx.bound, u;
            backend = AutoEnzyme(; mode = Enzyme.Reverse))
        gx = copy(fx.data[:gx])
        val, grad = sampler_value_and_gradient!(q, similar(u), u)
        @test val ≈ ref.lik + ref.prior + logjac(fx.built.layout, u)
        @test grad ≈ _adv_findiff(q, u) rtol = 1e-5 atol = 1e-7
        @test fx.data[:gx] == gx
    end
end

@testset "whole-value array composition retains observation alignment" begin
    for kind in (:vector, :column),
            extra in (:(begin mu = r[g] .+ gx; y .~ Normal.(mu, sigma) end),
                :(y .~ weighted.(Normal.(r[g], sigma), gx)),
                :(begin y .~ Normal.(r[g], sigma); y2 .~ Normal.(gx, sigma) end),
                :(begin y .~ Normal.(r[g], sigma);
                    u = gx .+ 1; y2 .~ Normal.(u, sigma) end))
        ast = _adv_model(kind, :named)
        pop!(ast.args)
        append!(ast.args, extra.head === :block ? extra.args : Any[extra])
        # Refused: any observation read retains the data's observation
        # axis; a bare declared array cannot broadcast over that axis
        # (standard Julia array-value contract, rkppl-use §2/§3).
        # capability: one whole array also read per observation; check its broadcast alignment at bind (P3, P10a 0dejlw1) (todo `1qlbn5b`)
        @test_broken (lower_rkppl(ast,
            (:y, :y2, :g, :gx); mod = ArrayDataValueModels); true)
    end
end

@testset "whole-value array composition grows without graph unrolling" begin
    counts = Int[]
    for (K, n) in ((2, 5), (5, 17))
        fx = _adv_build(:vector, :named, K, n)
        @test fx.bound.n_obs == n
        @test fx.built.layout.total == K + 3
        push!(counts, length(fx.built.spec.graph.recipes))
    end
    @test counts[1] == counts[2]
end
