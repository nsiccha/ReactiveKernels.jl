using DifferentiationInterface
using Distributions
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Reactant
using Test

# Axis-2 trio (XLA leg): Reactant compiled-primal + compiled-gradient
# parity for panel plates — native vs XLA at an unconstrained probe
# (the bernoulli-links precedent). Covers Int lanes (Poisson), Bool
# lanes (Bernoulli), the Gamma rate precompute, and the Gaussian
# panel with sampled scale (vector mu → default pipeline per the
# narrowed §7n lane rule; the ladder-1 pin is reserved for
# scalar-mu + sampled-scale legs, of which this file has none).

const _KR_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

function _kernel_reactant(prog::Expr, cols::Dict{Symbol,AbstractVector},
        dims::Dict{Symbol,Int}, u; ladder1::Bool = false,
        data = (:dose, :obs, :t))
    bound = bind_data(lower_rkppl(prog, data), cols; dims)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    return Base.invokelatest(_kernel_reactant_measure, built, bound, post_q,
        u; ladder1)
end

function _kernel_reactant_measure(built, bound, post_q, u; ladder1::Bool = false)
    native = post_q(u)
    compiled = Reactant.@compile post_q(Reactant.to_rarray(u))
    primal = Float64(compiled(Reactant.to_rarray(u)))
    q = prepare_sampler(built, bound, u; backend = _KR_BACKEND)
    g = similar(u, Float64)
    val, _ = sampler_value_and_gradient!(q, g, u)
    if ladder1
        # Narrowed lane rule (robust baseline §7n, provisional): only
        # scalar-mu + sampled-scale legs miscompile the gradient under
        # the default pipeline (off by exactly n-1 on log-σ; primal
        # exact; vector-mu legs are exact) — pin
        # `optimize = :only_enzyme` for the correctness assertion and
        # `@test_broken` the default pipeline (it turns suite-red the
        # day upstream fixes the pass — then drop both).
        cad = compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u);
            optimize = :only_enzyme)
        rval, rgrad = cad(Reactant.to_rarray(u))
        cad_default =
            compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u))
        _, rgrad_default = cad_default(Reactant.to_rarray(u))
        return (; native, primal, val, g, rval = Float64(rval),
            rgrad = Array(rgrad), rgrad_default = Array(rgrad_default))
    end
    cad = compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u))
    rval, rgrad = cad(Reactant.to_rarray(u))
    return (; native, primal, val, g, rval = Float64(rval),
        rgrad = Array(rgrad))
end

const _KR_T = [0.5, 1.0, 2.0, 0.5, 1.0, 2.0]
const _KR_DOSE = [1.0, 2.0]
const _KR_DIMS = Dict{Symbol,Int}(:kernel_nsub_pred => 2, :kernel_T_pred => 3)

function _kr_progs()
    pois = quote
        b0 ~ Normal(0.0, 1.0)
        pred ~ plate(t, dose, obs; subjects = kernel_nsub_pred) do ts, d, yy
            mu = exp.(b0 .* d .* ts)
            yy .~ Poisson.(mu)
            mu
        end
    end
    bern = quote
        b0 ~ Normal(0.0, 1.0)
        pred ~ plate(t, dose, obs; subjects = kernel_nsub_pred) do ts, d, yy
            eta = b0 .* d .* ts
            p = 1 ./ (1 .+ exp.(-eta))
            yy .~ Bernoulli.(p)
            p
        end
    end
    gam = quote
        b0 ~ Normal(0.0, 1.0)
        alpha ~ Exponential(1.0)
        pred ~ plate(t, dose, obs; subjects = kernel_nsub_pred) do ts, d, yy
            mu = exp.(b0 .* d .* ts)
            sc = mu ./ alpha
            yy .~ Gamma.(alpha, sc)
            mu
        end
    end
    gau = quote
        b0 ~ Normal(0.0, 1.0)
        sigma ~ Exponential(1.0)
        pred ~ plate(t, dose, obs; subjects = kernel_nsub_pred) do ts, d, yy
            mu = (b0 .* d) .* ts
            yy .~ Normal.(mu, sigma)
            mu
        end
    end
    return [
        ("poisson", pois,
            Dict{Symbol,AbstractVector}(:t => _KR_T, :dose => _KR_DOSE,
                :obs => [1, 0, 2, 3, 1, 0]),
            Dict(:b0 => 0.25), false),
        ("bernoulli", bern,
            Dict{Symbol,AbstractVector}(:t => _KR_T, :dose => _KR_DOSE,
                :obs => Bool[1, 0, 1, 1, 0, 0]),
            Dict(:b0 => 0.5), false),
        ("gamma", gam,
            Dict{Symbol,AbstractVector}(:t => _KR_T, :dose => _KR_DOSE,
                :obs => [0.5, 1.2, 2.1, 0.8, 1.5, 2.5]),
            Dict(:b0 => 0.25, :alpha => 2.0), false),
        # Gaussian panel with sampled scale: vector mu
        # (mu = (b0 .* d) .* ts) → default pipeline per the narrowed
        # §7n lane rule (matrix-a V2/M5/V6: vector-mu legs exact).
        ("gaussian", gau,
            Dict{Symbol,AbstractVector}(:t => _KR_T, :dose => _KR_DOSE,
                :obs => [0.5, 1.2, 2.1, 0.8, 1.5, 2.5]),
            Dict(:b0 => 0.5, :sigma => 2.0), false),
    ]
end

@testset "axis2 panel plates under Reactant" begin
    for (name, prog, cols, fixed, ladder1) in _kr_progs()
        @testset "$name" begin
            bound = bind_data(lower_rkppl(prog, (:dose, :obs, :t)),
                cols; dims = _KR_DIMS)
            names = coordinate_names(build_kernel(bound).layout)
            u = [n === :b0 ? fixed[n] : log(fixed[n]) for n in names]
            fx = _kernel_reactant(prog, cols, _KR_DIMS, u; ladder1)
            @test fx.primal ≈ fx.native rtol = 1e-9
            @test fx.val ≈ fx.native rtol = 1e-12
            @test fx.rval ≈ fx.native rtol = 1e-9
            @test fx.rgrad ≈ fx.g rtol = 1e-8
            ladder1 &&
                (@test_broken fx.rgrad_default ≈ fx.g rtol = 1e-8)
        end
    end
end

@testset "axis3 joint-3 two plates under Reactant" begin
    # Joint primary (same fixtures as the kernel value test): vector mu
    # + sampled sigma → default pipeline (narrowed §7n).
    prog = quote
        b0 ~ Normal(0.0, 1.0)
        sigma ~ Exponential(1.0)
        pred1 ~ plate(x1, y1; subjects = kernel_nsub_pred1) do xx1, yy1
            mu1 = b0 .* xx1
            yy1 .~ Normal.(mu1, sigma)
            mu1
        end
        pred2 ~ plate(x2, y2; subjects = kernel_nsub_pred2) do xx2, yy2
            mu2 = exp.(b0 .* xx2)
            yy2 .~ Poisson.(mu2)
            mu2
        end
    end
    cols = Dict{Symbol,AbstractVector}(
        :x1 => [0.5, 1.0, 1.5, 2.0], :y1 => [0.4, 1.1, 1.4, 2.2],
        :x2 => [0.5, 1.0, 1.5], :y2 => [1, 2, 3])
    dims = Dict{Symbol,Int}(:kernel_nsub_pred1 => 4, :kernel_nsub_pred2 => 3)
    data = (:x1, :y1, :x2, :y2)
    bound = bind_data(lower_rkppl(prog, data), cols; dims)
    names = coordinate_names(build_kernel(bound).layout)
    u = [n === :b0 ? 0.5 : log(1.5) for n in names]
    fx = _kernel_reactant(prog, cols, dims, u; data)
    @test fx.primal ≈ fx.native rtol = 1e-9
    @test fx.val ≈ fx.native rtol = 1e-12
    @test fx.rval ≈ fx.native rtol = 1e-9
    @test fx.rgrad ≈ fx.g rtol = 1e-8
    @test fx.primal ≈ -11.969630233025292 rtol = 1e-9
end

@testset "axis1 ranef grouped plate under Reactant" begin
    # Trio XLA leg for ranef-in-plate: tiny radon (2 counties, 4
    # obs, natural [1] spelling) — vector mu (per-obs county gather)
    # + sampled sigma → default pipeline per the narrowed §7n lane
    # rule (the gaussian-panel precedent).
    prog = Meta.parse("""begin
        r ~ varying_effect(county_id, [1])
        mu_alpha ~ Normal(0.0, 10.0)
        sigma_y ~ HalfNormal(1.0)
        alpha = mu_alpha .+ r
        cy = linear_pk_schedule(obs = (:county_idx, :time),
            dose = (:dsubj, :dtime, :damt))
        @plate radon for s in 1:2
            aa = alpha[county_idx]
            yy .~ Normal.(aa, sigma_y)
            aa
        end
    end""")
    cols = Dict{Symbol,AbstractVector}(
        :county_id => [1, 2], :county_idx => [1, 1, 2, 2],
        :time => [1.0, 2.0, 3.0, 4.0],
        :dsubj => Int[], :dtime => Float64[], :damt => Float64[],
        :yy => [0.5, 1.1, -0.3, 0.8])
    data = (:county_id, :county_idx, :time, :dsubj, :dtime, :damt, :yy)
    dims = Dict{Symbol,Int}()
    bound = bind_data(lower_rkppl(prog, data), cols; dims)
    names = coordinate_names(build_kernel(bound).layout)
    @test length(names) == 5
    u = map(names) do n
        s = String(n)
        startswith(s, "tau_") && return log(1.2)
        startswith(s, "z_flat_") &&
            return 0.1 * parse(Int, split(s, ".")[2])
        n === :mu_alpha && return 0.2
        n === :sigma_y && return log(0.9)
        error("unexpected coordinate $n")
    end
    fx = _kernel_reactant(prog, cols, dims, u; data)
    @test fx.primal ≈ fx.native rtol = 1e-9
    @test fx.val ≈ fx.native rtol = 1e-12
    @test fx.rval ≈ fx.native rtol = 1e-9
    @test fx.rgrad ≈ fx.g rtol = 1e-8
end
