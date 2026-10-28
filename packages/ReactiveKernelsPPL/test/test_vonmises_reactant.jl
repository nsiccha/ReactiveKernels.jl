# Compiled (Reactant) checks for the cases in test_vonmises.jl.
using Reactant

# Reactant/XLA value+grad parity at an unconstrained probe (no oracle —
# native vs compiled), plus the traced program size.
function _vm_reactant(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols); conditioned = keys(cols))
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
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            mu = a .+ b .* x
            y .~ VonMises.(mu, 1.7)
        end, _vm_cols()),
        ("literal kappa, circular", quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            mu = a .+ b .* x
            y .~ CircularVonMises.(mu, 1.7, -pi, pi)
        end, _vm_cols()),
        ("Gamma-sampled kappa, circular", quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            kappa ~ Gamma(2.0, 0.1)
            mu = a .+ b .* x
            y .~ CircularVonMises.(mu, kappa, -pi, pi)
        end, _vm_cols()),
        ("intercept-only log-kappa submodel", quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            c ~ Normal(0, 1)
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
