# Von-Mises response (SB `brm_von_mises_lpdf` mirror): surface
# admission, value parity vs Distributions.jl oracles (exact +
# circular; literal / Gamma-sampled / per-observation / predictor-fed
# kappa), Enzyme-vs-findiff gradients, Reactant/XLA value+grad, O(1)
# emission, and VM1/VM2/VM3 SB parity vs the peer lane's BridgeStan
# numbers (brief 2026-09-26T16-05-40-553-6m5ln2 on
# BayesianRegressionModels:rk:parity-fam-vonmises). (`_findiff_grad` /
# `_GEN_BACKEND` come from test_generator.jl, included first.)
using DifferentiationInterface
using Distributions: VonMises, Normal, Gamma, LogNormal, logpdf
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Reactant
using SpecialFunctions
using Test

# Lower + bind + build + query a VM program; return
# `(bound, built, kern, layout)`.
function _vm_query(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    kern = prepare_query(built, bound, :sampler)
    return bound, built, kern, built.layout
end

# Posterior at a constrained probe (world-age-safe call).
_vm_posterior(kern, lay, q::NamedTuple) =
    Base.invokelatest(kern, unconstrain(lay, q))

# Scalar von-Mises log-density (Distributions.jl oracle; SB matches
# the in-support value operation-for-operation).
_vm_ref(y::Real, mu::Real, kap::Real) = logpdf(VonMises(mu, kap), y)

# Circular oracle on the BRM `CircularVonMises` path (wrap mu into
# the interval, wrap y into the moving support, Distributions base) —
# independent of the kernel's rem-based spelling.
function _vm_circ_ref(y::Real, mu::Real, kap::Real, lo::Real, hi::Real)
    wm = lo + mod(mu - lo, hi - lo)
    r = (wm - pi) + mod(y - (wm - pi), 2pi)
    return logpdf(VonMises(wm, kap), r)
end

const _VM_X = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]
const _VM_Y = [0.3, -1.1, 2.0, -2.0, 0.5, 1.1]
_vm_cols() = Dict{Symbol,AbstractVector}(:y => copy(_VM_Y), :x => copy(_VM_X))

@testset "vm surface admission" begin
    @testset "literal kappa, exact head" begin
        plan = lower_rkppl(quote
                mu = a .+ b .* x
                y .~ VonMises.(mu, 1.7)
            end, (:y, :x))
        r = only(plan.responses)
        @test r.family === VonMisesFam
        @test r.link === IdentityLink
        @test r.predictor === :mu
        @test r.scale === 1.7
        @test r.weights === nothing
        @test r.evidence.kind === :none
        @test r.trials === nothing
        @test r.range === nothing
        @test r.interval === nothing
    end
    @testset "Gamma-sampled kappa, circular head" begin
        plan = lower_rkppl(quote
                kappa ~ Gamma(2.0, 0.1)
                mu = a .+ b .* x
                y .~ CircularVonMises.(mu, kappa, -pi, pi)
            end, (:y, :x))
        r = only(plan.responses)
        @test (r.family, r.scale, r.interval) ===
            (VonMisesFam, :kappa, (-Float64(pi), Float64(pi)))
        @test only(plan.parameters).family === :gamma
    end
    @testset "per-observation kappa column" begin
        plan = lower_rkppl(quote
                mu = a .+ b .* x
                y .~ VonMises.(mu, kappac)
            end, (:y, :x, :kappac))
        @test only(plan.responses).scale === :kappac
    end
    @testset "log-link predictor kappa admitted" begin
        plan = lower_rkppl(quote
                mu = a .+ b .* x
                lk = c .+ d .* x
                y .~ CircularVonMises.(mu, exp.(lk), -pi, pi)
            end, (:y, :x))
        @test only(plan.responses).scale == ScalePredictorRef(:lk, LogLink)
    end
    @testset "non-log predictor kappa deferred" begin
        @test_throws ContractValidationError lower_rkppl(quote
                mu = a .+ b .* x
                lk = c .+ d .* x
                y .~ VonMises.(mu, lk)
            end, (:y, :x))
    end
    @testset "shifted-interval literals" begin
        plan = lower_rkppl(quote
                mu = a .+ b .* x
                y .~ CircularVonMises.(mu, 1.7, 0.0, 6.283185307179586)
            end, (:y, :x))
        @test only(plan.responses).interval === (0.0, 6.283185307179586)
    end
end

@testset "vm value parity" begin
    @testset "literal kappa, exact" begin
        _, _, kern, lay = _vm_query(quote
                mu = a .+ b .* x
                y .~ VonMises.(mu, 1.7)
            end, _vm_cols())
        q = (mu = [0.5, -0.25],)
        got = _vm_posterior(kern, lay, q)
        mu = q.mu[1] .+ q.mu[2] .* _VM_X
        want = sum(_vm_ref(y, m, 1.7) for (y, m) in zip(_VM_Y, mu)) +
            logpdf(Normal(0, 1), q.mu[1]) + logpdf(Normal(0, 1), q.mu[2])
        @test got ≈ want rtol = 1e-12
    end
    @testset "Gamma-sampled kappa, circular" begin
        _, _, kern, lay = _vm_query(quote
                kappa ~ Gamma(2.0, 0.1)
                mu = a .+ b .* x
                y .~ CircularVonMises.(mu, kappa, -pi, pi)
            end, _vm_cols())
        q = (mu = [0.5, -0.25], kappa = 2.0)
        got = _vm_posterior(kern, lay, q)
        mu = q.mu[1] .+ q.mu[2] .* _VM_X
        want = sum(_vm_circ_ref(y, m, q.kappa, -Float64(pi), Float64(pi))
            for (y, m) in zip(_VM_Y, mu)) +
            logpdf(Normal(0, 1), q.mu[1]) + logpdf(Normal(0, 1), q.mu[2]) +
            logpdf(Gamma(2.0, 0.1), q.kappa) +
            logjac(lay, unconstrain(lay, q))
        @test got ≈ want rtol = 1e-12
    end
    @testset "per-observation kappa column" begin
        cols = _vm_cols()
        cols[:kappac] = [0.5, 1.5, 2.5, 1.0, 2.0, 0.8]
        _, _, kern, lay = _vm_query(quote
                mu = a .+ b .* x
                y .~ VonMises.(mu, kappac)
            end, cols)
        q = (mu = [0.5, -0.25],)
        got = _vm_posterior(kern, lay, q)
        mu = q.mu[1] .+ q.mu[2] .* _VM_X
        want = sum(_vm_ref(y, m, k)
            for (y, m, k) in zip(_VM_Y, mu, cols[:kappac])) +
            logpdf(Normal(0, 1), q.mu[1]) + logpdf(Normal(0, 1), q.mu[2])
        @test got ≈ want rtol = 1e-12
    end
    @testset "intercept-only log-kappa submodel" begin
        _, _, kern, lay = _vm_query(quote
                mu = a .+ b .* x
                lk = c
                y .~ CircularVonMises.(mu, exp.(lk), -pi, pi)
            end, _vm_cols())
        q = (mu = [0.5, -0.25], lk = [0.3])
        got = _vm_posterior(kern, lay, q)
        mu = q.mu[1] .+ q.mu[2] .* _VM_X
        kap = exp(q.lk[1])
        want = sum(_vm_circ_ref(y, m, kap, -Float64(pi), Float64(pi))
            for (y, m) in zip(_VM_Y, mu)) +
            logpdf(Normal(0, 1), q.mu[1]) + logpdf(Normal(0, 1), q.mu[2]) +
            logpdf(Normal(0, 1), q.lk[1])
        @test got ≈ want rtol = 1e-12
    end
end

# One Enzyme-vs-findiff gradient check at a constrained probe (no oracle
# needed).
function _vm_enzyme_check(prog::Expr, cols::Dict{Symbol,AbstractVector},
        q::NamedTuple)
    bound, built, kern, lay = _vm_query(prog, cols)
    u = unconstrain(lay, q)
    prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    _, _ = sampler_value_and_gradient!(prep, g, u)
    @test all(isfinite, g)
    @test isapprox(g, _findiff_grad(w -> Base.invokelatest(kern, w), u);
        rtol = 1e-5, atol = 1e-7)
    return g
end

@testset "vm Enzyme gradients" begin
    @testset "literal kappa, exact" begin
        _vm_enzyme_check(quote
                mu = a .+ b .* x
                y .~ VonMises.(mu, 1.7)
            end, _vm_cols(), (mu = [0.5, -0.25],))
    end
    @testset "Gamma-sampled kappa, circular" begin
        _vm_enzyme_check(quote
                kappa ~ Gamma(2.0, 0.1)
                mu = a .+ b .* x
                y .~ CircularVonMises.(mu, kappa, -pi, pi)
            end, _vm_cols(), (mu = [0.5, -0.25], kappa = 2.0))
    end
    @testset "intercept-only log-kappa submodel" begin
        _vm_enzyme_check(quote
                mu = a .+ b .* x
                lk = c
                y .~ CircularVonMises.(mu, exp.(lk), -pi, pi)
            end, _vm_cols(), (mu = [0.5, -0.25], lk = [0.3]))
    end
end

# Statement-head histogram of the generated kernel (the joint-parity
# O(1) pattern): the VM plate must not unroll over observations.
function _vm_statement_heads(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    def = ReactiveKernelsPPL.kernel_expr(bound, assign_layout(bound))
    heads = Dict{String,Int}()
    for st in def.args[2].args
        st isa Expr || continue
        heads[string(st.head)] = get(heads, string(st.head), 0) + 1
    end
    return heads
end

@testset "vm emission is O(1) in n_obs" begin
    prog = quote
        kappa ~ Gamma(2.0, 0.1)
        mu = a .+ b .* x
        y .~ CircularVonMises.(mu, kappa, -pi, pi)
    end
    h6 = _vm_statement_heads(prog, _vm_cols())
    h12 = _vm_statement_heads(prog, Dict{Symbol,AbstractVector}(
        :y => vcat(_VM_Y, _VM_Y), :x => vcat(_VM_X, _VM_X)))
    @test h6 == h12
end

# Reactant/XLA value+grad parity at an unconstrained probe (no oracle —
# native vs compiled), plus the traced program size.
function _vm_reactant(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_vm_reactant_measure, built, bound, post_q, u)
end

function _vm_reactant_measure(built, bound, post_q, u)
    hlo = repr(Reactant.@code_hlo optimize = false post_q(Reactant.to_rarray(u)))
    native = post_q(u)
    compiled = Reactant.@compile post_q(Reactant.to_rarray(u))
    primal = Float64(compiled(Reactant.to_rarray(u)))
    q = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    val, _ = sampler_value_and_gradient!(q, g, u)
    cad = compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u))
    rval, rgrad = cad(Reactant.to_rarray(u))
    return (; lines = count(==('\n'), hlo), native, primal, val, g,
        rval = Float64(rval), rgrad = Array(rgrad))
end

# Upstream XLA gap (the leveled-Reactant pin precedent): `besseli` has
# no method for a traced scalar, so every live-kappa program fails at
# trace time with `MethodError: no method matching
# besseli(::Int64, ::TracedRNumber{Float64})`. Reactant ships the
# `enzymexla.special.besseli` MLIR op but no Julia-side wiring, and
# SpecialFunctions offers no `besseli(::Real, ::Number)` fallback
# (contrast `loggamma`/`digamma(::Number)`, which trace). A
# `scalar_derivative_rule` cannot bridge it either: rule calls trace
# through their cuts' source ops, and the primal cut calls `besseli`.
# The signature below is exactly that gap; anything else rethrows
# loudly. Minimal repro: `h(x) = (y = Reactant.@allowscalar x[1];
# SpecialFunctions.besseli(0, y))` under `Reactant.@compile`.
_vm_is_upstream_gap(e) =
    e isa MethodError && e.f === SpecialFunctions.besseli &&
    length(e.args) == 2 && e.args[2] isa Reactant.TracedRNumber

# Live-kappa programs blocked by the gap above. The `@test_broken true`
# at the end of a pinned prog's body FIRES (Unexpected Pass) once
# upstream wires besseli — then drop the name here and the try/catch.
const _VM_UPSTREAM_PINNED = ("Gamma-sampled kappa, circular",
    "intercept-only log-kappa submodel")

@testset "vm under Reactant" begin
    progs = [
        ("literal kappa, exact", quote
            mu = a .+ b .* x
            y .~ VonMises.(mu, 1.7)
        end, _vm_cols()),
        ("literal kappa, circular", quote
            mu = a .+ b .* x
            y .~ CircularVonMises.(mu, 1.7, -pi, pi)
        end, _vm_cols()),
        ("Gamma-sampled kappa, circular", quote
            kappa ~ Gamma(2.0, 0.1)
            mu = a .+ b .* x
            y .~ CircularVonMises.(mu, kappa, -pi, pi)
        end, _vm_cols()),
        ("intercept-only log-kappa submodel", quote
            mu = a .+ b .* x
            lk = c
            y .~ CircularVonMises.(mu, exp.(lk), -pi, pi)
        end, _vm_cols()),
    ]
    for (name, prog, cols) in progs
        @testset "$name" begin
            try
                fx = _vm_reactant(prog, cols)
                @test fx.primal ≈ fx.native rtol = 1e-9
                @test fx.val ≈ fx.native rtol = 1e-12
                @test fx.rval ≈ fx.native rtol = 1e-9
                @test fx.rgrad ≈ fx.g rtol = 1e-8
                if name in _VM_UPSTREAM_PINNED
                    # Self-firing pin: errors (Unexpected Pass) once
                    # upstream wires besseli, forcing removal of the
                    # try/catch.
                    @test_broken true
                end
            catch e
                _vm_is_upstream_gap(e) || rethrow()
                name in _VM_UPSTREAM_PINNED || rethrow()
                # Known upstream besseli gap (above): pinned, not passing.
                @test_broken false
            end
        end
    end
    @testset "traced program is O(1) in n_obs" begin
        _, prog, _ = progs[2]
        small = _vm_reactant(prog, _vm_cols())
        large = _vm_reactant(prog, Dict{Symbol,AbstractVector}(
            :y => vcat(_VM_Y, _VM_Y), :x => vcat(_VM_X, _VM_X)))
        @test small.lines == large.lines
    end
end

# VM1/VM2/VM3 parity probes (N=4 fixed literal data, no RNG — adopted
# from the peer lane's brief
# 2026-09-26T16-05-40-553-6m5ln2 on
# BayesianRegressionModels:rk:parity-fam-vonmises).
const _VM_SB_X = [-1.0, -0.25, 0.5, 1.0]
const _VM1_Y = [-2.8, -0.4, 1.1, 2.9]
const _VM2_Y = [-2.0, 0.3, 1.5, -1.2]
const _VM3_Y = [0.5, 2.0, 6.0, 1.0]

# SB parity vs the peer lane's BridgeStan numbers (brief
# 2026-09-26T16-05-40-553-6m5ln2, BRM 59b357e, StanBlocks 24578c3,
# BridgeStan 2.9.0, Julia 1.10.11): full posterior at u_unc,
# propto=false, jacobian=true, BridgeStan AD grads. RK layout order
# matches SB declaration order by coordinate name (SB pins below are
# in SB u-order).
_vm_sb_vec(names, pairs) = [Dict(pairs)[n] for n in names]

@testset "vm SB parity" begin
    @testset "VM1 circular demand + log-kappa submodel" begin
        # SB: mu ~ 1 + x; log(kappa) ~ 1; all coefs std_normal;
        # y ~ CircularVonMises(mu, kappa; interval=(-pi, pi));
        # u = [b0, b1, c0] = [0.5, -0.25, 0.3].
        prog = quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            c ~ Normal(0, 1)
            mu = a .+ b .* x
            lk = c
            y .~ CircularVonMises.(mu, exp.(lk), -pi, pi)
        end
        cols = Dict{Symbol,AbstractVector}(:y => copy(_VM1_Y),
            :x => copy(_VM_SB_X))
        bound, built, kern, lay = _vm_query(prog, cols)
        names = coordinate_names(lay)
        u = _vm_sb_vec(names, [Symbol("mu.Intercept") => 0.5,
            Symbol("mu.x") => -0.25, Symbol("lk.Intercept") => 0.3])
        @test abs(Base.invokelatest(kern, u) - (-12.60537740226801)) < 1e-12
        prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        @test all(isfinite, g)
        want = _vm_sb_vec(names, [Symbol("mu.Intercept") => 0.4606828295664177,
            Symbol("mu.x") => 1.0755814718931511,
            Symbol("lk.Intercept") => -3.9533067855122037])
        @test maximum(abs.(g .- want)) < 1e-10
    end
    @testset "VM2 exact + sampled kappa" begin
        # SB: mu ~ 1 + x; k ~ LogNormal(0, 1); coefs std_normal;
        # y ~ VonMises(mu, k); u = [b0, b1, logk] = [0.5, -0.25, 0.2].
        prog = quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            k ~ LogNormal(0, 1)
            mu = a .+ b .* x
            y .~ VonMises.(mu, k)
        end
        cols = Dict{Symbol,AbstractVector}(:y => copy(_VM2_Y),
            :x => copy(_VM_SB_X))
        bound, built, kern, lay = _vm_query(prog, cols)
        names = coordinate_names(lay)
        u = _vm_sb_vec(names, [Symbol("mu.Intercept") => 0.5,
            Symbol("mu.x") => -0.25, :k => 0.2])
        @test abs(Base.invokelatest(kern, u) - (-10.932237952776804)) < 1e-12
        prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        @test all(isfinite, g)
        want = _vm_sb_vec(names, [Symbol("mu.Intercept") => -1.3935808349508612,
            Symbol("mu.x") => 0.1339126643988171,
            :k => -2.0129577913934957])
        @test maximum(abs.(g .- want)) < 1e-10
    end
    @testset "VM3 circular (0, 2pi) + literal kappa" begin
        # SB: mu ~ 1 + x; coefs std_normal;
        # y ~ CircularVonMises(mu, 1.7; interval=(0, 2pi));
        # u = [b0, b1] = [1.0, -0.5].
        prog = quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            mu = a .+ b .* x
            y .~ CircularVonMises.(mu, 1.7, 0.0, 6.283185307179586)
        end
        cols = Dict{Symbol,AbstractVector}(:y => copy(_VM3_Y),
            :x => copy(_VM_SB_X))
        bound, built, kern, lay = _vm_query(prog, cols)
        names = coordinate_names(lay)
        u = _vm_sb_vec(names, [Symbol("mu.Intercept") => 1.0,
            Symbol("mu.x") => -0.5])
        @test abs(Base.invokelatest(kern, u) - (-7.934564762332648)) < 1e-12
        prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        @test all(isfinite, g)
        want = _vm_sb_vec(names, [Symbol("mu.Intercept") => -1.7708419435702396,
            Symbol("mu.x") => 1.6892237819376545])
        @test maximum(abs.(g .- want)) < 1e-10
    end
end
