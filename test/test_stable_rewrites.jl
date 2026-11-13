# Stable algebraic rewrites (`src/stable_rewrites.jl`): a prepared program
# computes numerically fragile compositions in their stable form by default,
# and `simplify = false` keeps the literal arithmetic (snag
# `rk-stable-algebr-e6eb8f42`, user decision
# `ReactiveKernels:snag.rk-stable-algebr-e6eb8f42/decisions/2026-11-08T04-13-58-283-17d8927`,
# option 1). The literal `log(exp(u))` is `Inf` below u = -745 and its
# ordinary reverse is `0 * Inf = NaN` beyond either edge; the oracle for every
# value and derivative below is the stable mathematics in BigFloat.
module StableRewritesTest

import Enzyme
using ReactiveKernels
using ReactiveKernels: LogExpFunctions
using ReactiveKernelsDistributionKernels: DistributionKernelSources
using Test

const RK = ReactiveKernels
const log1pexp = LogExpFunctions.log1pexp
const logistic = LogExpFunctions.logistic
const logit = LogExpFunctions.logit
const normal = DistributionKernelSources.normal
const bernoulli = DistributionKernelSources.bernoulli

_reverse(k, args...) = Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(k), Enzyme.Active,
                                       map(Enzyme.Active, args)...)[1]
# The ordinary reverse gradient of `k` in its vector argument.
function _vgradient(k, x::Vector{Float64})
    dx = zero(x)
    Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(k), Enzyme.Active,
                    Enzyme.Duplicated(copy(x), dx))
    dx
end
# The ordinary reverse derivative of `k` in its scalar first argument, the
# other arguments held constant.
_scalar_reverse(k, u::Float64, rest...) = Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(k),
    Enzyme.Active, Enzyme.Active(u), map(Enzyme.Const, rest)...)[1][1]
_code(k) = string(code_expr(k))
# Relative agreement with a BigFloat oracle, absolute near zero.
_close(x, oracle; rtol = 1e-13, atol = 1e-300) =
    isfinite(x) && abs(x - Float64(oracle)) <= max(atol, rtol * abs(Float64(oracle)))

@kernel stable_roundtrip(u::Float64) = begin
    s = exp(u)
    r = log(s)
    out = log1pexp(-r)
    return out
end

@testset "log(exp(u)) across statements: finite where the literal is not" begin
    k = prepare(stable_roundtrip)
    literal = prepare(stable_roundtrip; simplify = false)
    @test occursin("_rk_log_of", _code(k))
    @test !occursin("Base.log(", _code(k))
    # The inner exponential nothing reads any more is gone, so its reverse
    # cannot form 0 * Inf either.
    @test !occursin("_native_exp(", _code(k))
    @test occursin("Base.log(", _code(literal)) && !occursin("_rk_", _code(literal))
    for u in (-800.0, -746.0, -700.0, -1.5, 0.0, 2.5, 700.0, 709.0, 710.0, 800.0)
        U = big(u)
        @test _close(k(u), log1p(exp(-U)))
        @test _close(_reverse(k, u)[1], -1 / (1 + exp(U)))
    end
    # The literal program is the reported hole, kept under `simplify = false`.
    @test isnan(_reverse(literal, 800.0)[1]) && isnan(_reverse(literal, -800.0)[1])
    @test literal(-800.0) == Inf
    # Where the literal is finite the two agree to within its rounding.
    for u in (-30.0, -1.0, 0.0, 3.0, 40.0)
        @test k(u) ≈ literal(u) rtol = 1e-14
        @test _reverse(k, u)[1] ≈ _reverse(literal, u)[1] rtol = 1e-14
    end
end

@testset "prepare_ad and the nested setting" begin
    backend = RK.DifferentiationInterface.AutoEnzyme(mode = Enzyme.Reverse)
    p = prepare_ad(stable_roundtrip, backend, 0.0; active = :u, want = :out)
    for u in (800.0, -800.0)
        value, gradient = ad_value_and_gradient(p, u)
        @test isfinite(value) && isfinite(gradient)
    end
    literal = prepare_ad(stable_roundtrip, backend, 0.0; active = :u, want = :out,
                         simplify = false)
    @test isnan(last(ad_value_and_gradient(literal, 800.0)))
    # An enclosing setting reaches the preparations made inside it.
    inner = RK._with_simplify(false) do
        prepare(stable_roundtrip)
    end
    @test !occursin("_rk_", _code(inner))
    @test_throws ArgumentError prepare(stable_roundtrip; simplify = :yes)
    # A cache keeps one kernel per setting.
    cache = PreparationCache()
    a = prepare!(cache, stable_roundtrip)
    b = prepare!(cache, stable_roundtrip; simplify = false)
    @test occursin("_rk_", _code(a)) && !occursin("_rk_", _code(b))
    @test prepare!(cache, stable_roundtrip) === a
end

# Each rule, through a prepared kernel, against its BigFloat oracle where the
# literal composition overflows, underflows or loses its digits.
@kernel stable_log_logistic(x::Float64) = begin
    p = logistic(x)
    out = log(p)
    return out
end
@kernel stable_log1m_logistic(x::Float64) = begin
    p = logistic(x)
    out = log1p(-p)
    return out
end
@kernel stable_log_one_minus(x::Float64) = begin
    p = logistic(x)
    out = log(1 - p)
    return out
end
@kernel stable_logit(x::Float64) = begin
    p = logistic(x)
    out = logit(p)
    return out
end
@kernel stable_log_odds(x::Float64) = begin
    p = logistic(x)
    out = log(p) - log1p(-p)
    return out
end
@kernel stable_log_one_plus(x::Float64) = begin
    out = log(1 + x)
    return out
end
@kernel stable_expm1(x::Float64) = begin
    out = exp(x) - 1
    return out
end
@kernel stable_softplus(x::Float64) = begin
    out = log(1 + exp(x))
    return out
end
@kernel stable_softplus1p(x::Float64) = begin
    out = log1p(exp(x))
    return out
end
@kernel stable_logaddexp(a::Float64, b::Float64) = begin
    out = log(exp(a) + exp(b))
    return out
end
@kernel stable_logsumexp(x::Vector{Float64}) = begin
    out = log(sum(exp.(x)))
    return out
end
@kernel stable_inv(u::Float64) = begin
    s = exp(u)
    out = inv(s)
    return out
end
@kernel stable_div(u::Float64, x::Float64) = begin
    ell = exp(u)
    out = cos(x / ell)
    return out
end
@kernel stable_square(u::Float64) = begin
    s = exp(u)
    out = s^2
    return out
end
@kernel stable_sqrt(u::Float64) = begin
    s = exp(u)
    out = sqrt(s)
    return out
end
@kernel stable_sigmoid(u::Float64) = begin
    p = 1 / (1 + exp(-u))
    out = log(p) + log1p(-p)
    return out
end
@kernel stable_exp_log(x::Float64) = begin
    s = abs2(x)
    out = exp(log(s))
    return out
end

@testset "the rule table" begin
    k = prepare(stable_log_logistic)
    for x in (-800.0, -40.0, 0.0, 40.0)
        X = big(x)
        @test _close(k(x), -log1p(exp(-X)))
        @test _close(_reverse(k, x)[1], 1 / (1 + exp(X)); atol = 1e-300)
    end
    k = prepare(stable_log1m_logistic)
    for x in (-40.0, 0.0, 40.0, 800.0)
        X = big(x)
        @test _close(k(x), -log1p(exp(X)))
        @test _close(_reverse(k, x)[1], -1 / (1 + exp(-X)))
    end
    k = prepare(stable_log_one_minus)
    @test _close(k(800.0), -800) && _close(k(40.0), -log1p(exp(big(40.0))))
    @test _close(prepare(stable_logit)(40.0), 40) && _close(prepare(stable_logit)(-800.0), -800)
    k = prepare(stable_log_odds)
    @test _close(k(40.0), 40) && _close(_reverse(k, 40.0)[1], 1)
    k = prepare(stable_log_one_plus)
    @test _close(k(1e-20), log1p(big(1e-20))) && _close(_reverse(k, 1e-20)[1], 1 / (1 + big(1e-20)))
    @test_throws DomainError k(-2.0)
    k = prepare(stable_expm1)
    @test _close(k(1e-20), expm1(big(1e-20)))
    for s in (prepare(stable_softplus), prepare(stable_softplus1p))
        @test _close(s(800.0), 800) && _close(_reverse(s, 800.0)[1], 1)
        @test _close(s(-40.0), log1p(exp(big(-40.0))))
    end
    k = prepare(stable_logaddexp)
    @test _close(k(800.0, 800.0), 800 + log(big(2)))
    @test all(g -> g == 0.5, _reverse(k, 800.0, 800.0))     # the literal's tie derivative
    @test all(g -> g == 0.5, _reverse(k, 1.0, 1.0))
    @test _close(k(800.0, -800.0), 800)
    k = prepare(stable_logsumexp)
    @test _close(k([800.0, 1.0]), 800 + log1p(exp(big(1.0) - 800)))
    @test _close(k([1.0, 2.0, 3.0]), log(sum(exp.(big.([1.0, 2.0, 3.0])))))
    @test k(Float64[]) == -Inf
    @test _vgradient(k, [2.0, 2.0]) == [0.5, 0.5]
    @test _vgradient(k, [800.0, 1.0]) ≈ [1.0, 0.0] atol = 1e-300
    k = prepare(stable_inv)
    @test _close(_reverse(k, -400.0)[1], -exp(big(400.0)))
    k = prepare(stable_div)
    @test isfinite(_reverse(k, -400.0, 1.0)[1])
    @test isinf(_reverse(prepare(stable_div; simplify = false), -400.0, 1.0)[1])
    k = prepare(stable_square)
    @test _close(k(-400.0), exp(big(-800.0))) && _close(_reverse(k, -400.0)[1], 2exp(big(-800.0)))
    k = prepare(stable_sqrt)
    @test _close(k(-800.0), exp(big(-400.0)))
    k = prepare(stable_sigmoid)
    for u in (-800.0, 0.0, 800.0)
        U = big(u)
        @test _close(k(u), -log1p(exp(-U)) - log1p(exp(U)))
    end
    k = prepare(stable_exp_log)
    @test k(3.0) === 9.0 && k(0.0) === 0.0
end

@kernel stable_broadcast(eta::Vector{Float64}) = begin
    rate = exp.(eta)
    out = sum(log.(rate))
    return out
end

@kernel stable_cell(eta::Vector{Float64}) = begin
    cells = plate(eta) do e
        s = exp(e)
        -log(s)
    end
    out = sum(cells)
    return out
end

@kernel stable_row_scale(eta::Vector{Float64}, y::Vector{Float64}) = begin
    lp = plate(eta, y) do e, yi
        normal(0.0, exp(e)).logpdf(yi)
    end
    total = sum(lp)
    return total
end

@kernel stable_shared_scale(u::Float64, y::Vector{Float64}) = begin
    sigma = exp(u)
    lp = plate(y) do yi
        normal(0.0, sigma).logpdf(yi)
    end
    total = sum(lp)
    return total
end

@testset "dotted calls and plate cells" begin
    k = prepare(stable_broadcast)
    @test _close(k([800.0, -800.0, 1.5]), 1.5)
    @test _vgradient(k, [800.0, -800.0, 1.5]) == [1.0, 1.0, 1.0]
    @test !occursin("Base.log", _code(k))
    # A cell's own composition of its argument.
    k = prepare(stable_cell)
    @test _close(k([800.0, -800.0, 1.0]), -1)
    @test _vgradient(k, [800.0, -800.0, 1.0]) == [-1.0, -1.0, -1.0]
    # A per-row scale computed in the cell, read by the law's log(scale).
    k = prepare(stable_row_scale)
    literal = prepare(stable_row_scale; simplify = false)
    @test occursin("_rk_log_of", _code(k))
    eta, y = [-0.3, 0.2, 1.1], [0.3, -0.2, 1.1]
    @test k(eta, y) ≈ literal(eta, y) rtol = 1e-14
    dk = zero(eta); dl = zero(eta)
    Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(k), Enzyme.Active,
                    Enzyme.Duplicated(copy(eta), dk), Enzyme.Const(y))
    Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(literal), Enzyme.Active,
                    Enzyme.Duplicated(copy(eta), dl), Enzyme.Const(y))
    @test dk ≈ dl rtol = 1e-13
    # A law's hoisted log(scale) reads the coordinate the scale was computed
    # from, across the plate boundary.
    k = prepare(stable_shared_scale)
    literal = prepare(stable_shared_scale; simplify = false)
    @test occursin("_rk_log_of", _code(k)) && !occursin("_rk_", _code(literal))
    y = [0.3, -0.2, 1.1]
    for u in (-2.0, 0.0, 0.7)
        @test k(u, y) ≈ literal(u, y) rtol = 1e-14
        @test _scalar_reverse(k, u, y) ≈ _scalar_reverse(literal, u, y) rtol = 1e-13
    end
end

# The probability form of a logistic regression (`reactivekernels-use` §4b)
# against the direct logit form, where the probability loses its digits.
@kernel stable_bernoulli_probability(eta::Vector{Float64}, y::Vector{Bool}) = begin
    lp = plate(eta, y) do e, yi
        bernoulli(logistic(e)).logpdf(yi)
    end
    total = sum(lp)
    return total
end
@kernel stable_bernoulli_logit(eta::Vector{Float64}, y::Vector{Bool}) = begin
    lp = plate(eta, y) do e, yi
        bernoulli(; logit = e).logpdf(yi)
    end
    total = sum(lp)
    return total
end

@testset "bernoulli(logistic(η)) reads η" begin
    k = prepare(stable_bernoulli_probability)
    direct = prepare(stable_bernoulli_logit)
    literal = prepare(stable_bernoulli_probability; simplify = false)
    for (eta, y) in (([20.0], [false]), ([40.0], [false]), ([-40.0], [true]),
                     ([0.3, -1.2, 2.0], [true, false, true]))
        @test k(eta, y) ≈ direct(eta, y) rtol = 1e-15
    end
    # The written probability form loses the tail: η = 20, y = false.
    @test !isapprox(literal([20.0], [false]), direct([20.0], [false]); rtol = 1e-12)
end

# The other preparations that compile their own programs: a compiled reactive
# state, `plate(spec; batched)` and position batching (`vectorize`). Each
# computes the stable form `prepare` computes and keeps the literal under
# `simplify = false` (todo `1vfxr65`).
@kernel stable_one_statement(u::Float64) = begin
    out::Float64 = log1pexp(-log(exp(u)))
end

const _EDGES = (-800.0, 710.0, 800.0)
_edge_value(u) = log1p(exp(-big(u)))
_edge_derivative(u) = -1 / (1 + exp(big(u)))

# The reactive getter of `want` at `u`.
function _reactive_value(program, want, u)
    state = program(u)
    get!(state, statevalue(program, want))
end
# The derivative in `u` of the one recipe the getter of `want` calls: the
# operation in its callables table, differentiated alone. Differentiating
# through constructing the state stores the program's constant graph into
# differentiable memory, which registered Enzyme on Julia 1.13 refuses
# (`EnzymeRuntimeActivityError`); that is not what this checks.
function _reactive_derivative(program, want, u)
    getter = program.getters[RK._slot_index(statevalue(program, want))]
    _scalar_reverse(only(getter.callables), u)
end

@testset "a compiled reactive state rewrites each recipe's own source" begin
    program = prepare_reactive(stable_one_statement)
    literal = prepare_reactive(stable_one_statement; simplify = false)
    out = stable_one_statement.out
    for u in _EDGES
        @test _close(_reactive_value(program, out, u), _edge_value(u))
        @test _close(_reactive_derivative(program, out, u), _edge_derivative(u))
        @test _reactive_value(program, out, u) == prepare(stable_one_statement)(u)
    end
    @test _reactive_value(literal, out, -800.0) == Inf
    @test isnan(_reactive_derivative(literal, out, 800.0))
    # The getters call the recipe through the table either way; the table
    # holds the rewritten source.
    getter(p) = p.getters[RK._slot_index(statevalue(p, out))]
    @test getter(program).ast == getter(literal).ast
    @test only(getter(program).callables) isa RK._StableRecipeOp
    @test !(only(getter(literal).callables) isa RK._StableRecipeOp)
    # Every value of a reactive state is a slot `freeze!` can set, so a
    # composition across two values is computed as written: `r = log(s)` reads
    # `s`'s slot.
    roundtrip = prepare_reactive(stable_roundtrip)
    state = roundtrip(800.0)
    @test get!(state, statevalue(roundtrip, stable_roundtrip.r)) == Inf
    freeze!(state, statevalue(roundtrip, stable_roundtrip.s), 2.0)
    @test get!(state, statevalue(roundtrip, stable_roundtrip.r)) == log(2.0)
    @test all(op -> !(op isa RK._StableRecipeOp),
              first(roundtrip.getters).callables)
end

# `plate(spec; batched)` reduces into a value of the want's declared type.
@kernel stable_batched_chain(u::Float64, c::Float64) = begin
    s::Float64 = exp(u)
    r::Float64 = log(s)
    out::Float64 = log1pexp(-r) + c
end
@kernel stable_batched_scale(y::Float64, u::Float64) = begin
    sigma::Float64 = exp(u)
    log_sigma::Float64 = log(sigma)
    lp::Float64 = -log_sigma - (y / sigma)^2 / 2
end

# The statements of `k`'s program before its loop, and the loop's body.
function _around_loop(k)
    body = code_expr(k).args[2].args
    at = findfirst(s -> s isa Expr && s.head === :for, body)
    body[1:(at - 1)], body[at].args[2]
end

@testset "plate(spec; batched): the stable forms, hoisted values still hoisted" begin
    k = plate(stable_batched_chain; have = (:u, :c), want = :out, batched = :u)
    literal = plate(stable_batched_chain; have = (:u, :c), want = :out, batched = :u,
                    simplify = false)
    collected = plate(stable_batched_chain; have = (:u, :c), want = :out, batched = :u,
                      reduce = nothing)
    us = collect(_EDGES)
    @test all(_close.(collected(us, 0.0), _edge_value.(us)))
    @test _close(k(us, 0.0), sum(_edge_value, us))
    du = zero(us)
    Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(k), Enzyme.Active,
                    Enzyme.Duplicated(copy(us), du), Enzyme.Const(0.0))
    @test all(_close.(du, _edge_derivative.(us)))
    @test literal([-800.0], 0.0) == Inf
    @test occursin("_rk_log_of", _code(k)) && !occursin("_rk_", _code(literal))
    for u in (-30.0, -1.0, 0.0, 3.0, 40.0)
        @test k([u, u + 1], 0.5) ≈ literal([u, u + 1], 0.5) rtol = 1e-14
    end
    # A shared scale is computed once, above the loop. A composition inside
    # that region is rewritten; one the loop reads per element keeps reading
    # the hoisted value, so no exponential runs per element.
    k = plate(stable_batched_scale; have = (:y, :u), want = :lp, batched = :y)
    literal = plate(stable_batched_scale; have = (:y, :u), want = :lp, batched = :y,
                    simplify = false)
    before, loop = _around_loop(k)
    @test occursin("_rk_log_of", string(before))
    @test occursin("_native_exp", string(before))
    @test !occursin("exp", string(loop)) && !occursin("_rk_", string(loop))
    ys = [0.3, -1.2, 2.0]
    @test _close(k(ys, 800.0), -3 * big(800.0))
    @test literal(ys, 800.0) == -Inf
    @test k(ys, 0.7) ≈ literal(ys, 0.7) rtol = 1e-14
end

@testset "vectorize: each part of the position driver" begin
    scalar = prepare(stable_roundtrip)
    v = vectorize(scalar; batched = :u)
    literal = vectorize(scalar; batched = :u, simplify = false)
    us = collect(_EDGES)
    @test v isa RK.GraphReplicatedKernel
    @test all(_close.(v(us), _edge_value.(us)))
    du = zero(us)
    Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(x -> sum(v(x))), Enzyme.Active,
                    Enzyme.Duplicated(copy(us), du))
    @test all(_close.(du, _edge_derivative.(us)))
    @test v(us) == map(scalar, us)
    @test occursin("_rk_log_of", _code(v)) && !occursin("_rk_", _code(literal))
    @test literal([-800.0]) == [Inf]
    # The setting reaches `prepare_batched`'s scalar kernel and driver alike.
    @test prepare_batched(stable_roundtrip; batched = :u, simplify = false)([-800.0]) ==
          [Inf]
    # A shared scale: the prefix computes it once and its own `log` stably; the
    # per-position residual reads the hoisted value.
    s = vectorize(prepare(stable_batched_scale); batched = :y)
    ys = [0.3, -1.2, 2.0]
    @test all(_close.(s(ys, 800.0), -big(800.0)))
end

@kernel stable_guarded(x::Float64) = begin
    out = x > 0 ? log(exp(sqrt(x))) : 0.0
    return out
end
@kernel stable_unproven(x::Float64) = begin
    out = exp(log(x))
    return out
end
@kernel stable_matrix(A::Matrix{Float64}) = begin
    out = log(exp(A))
    return out
end
@kernel stable_float32(u::Float32) = begin
    s = exp(u)
    out = log(s)
    return out
end
nonconst_exp = exp

@testset "semantics the rewrites keep" begin
    # Lazy branches stay lazy: the arm is rewritten inside the arm.
    k = prepare(stable_guarded)
    @test k(-1.0) === 0.0 && k(4.0) == 2.0
    @test _close(k(1.0e6), 1000)
    # exp(log(x)) of an unproven x keeps log's DomainError.
    @test !occursin("_rk_", _code(prepare(stable_unproven)))
    @test_throws DomainError prepare(stable_unproven)(-1.0)
    # A non-Real operand runs the literal composition.
    A = [0.2 0.1; 0.1 0.3]
    @test prepare(stable_matrix)(A) == prepare(stable_matrix; simplify = false)(A)
    # Result types are the literal's.
    @test prepare(stable_float32)(1.5f0) isa Float32
    # A callee bound to a non-constant global may change: never matched.
    @test RK._rw_resolve(GlobalRef(@__MODULE__, :nonconst_exp), @__MODULE__) === nothing
    @test RK._rw_resolve(GlobalRef(Base, :exp), @__MODULE__) === exp
end

@testset "helpers: stable methods for Real, the literal otherwise" begin
    @test RK._rk_log_of(exp, 3) === 3.0
    @test RK._rk_log_of(exp, 2.5f0) === 2.5f0
    @test RK._rk_pow_of(exp, 1.0f0, Val(2)) isa Float32
    @test RK._rk_log_one_plus(1.0, 1.0f-3) isa Float64
    @test RK._rk_log_one_plus(1, 1.0f-3) isa Float32
    @test RK._rk_logsumexp([Inf, 1.0]) == Inf
    @test isnan(RK._rk_logsumexp([NaN, 1.0]))
    @test RK._rk_logsumexp([-Inf, -Inf]) == -Inf
    A = [0.2 0.1; 0.1 0.3]
    @test RK._rk_log_of(exp, A) == log(exp(A))
    @test RK._rk_pow_of(exp, A, Val(2)) == exp(A)^2
end

# A kernel of a log-magnitude at an exponential (snag
# `measured-the-rem-97a93f3f`): a half-Student prior on positive values
# `exp.(eta)` reads `eta` and computes its `log(nu)` and `log(scale)` once.
const student_t = DistributionKernelSources.student_t
const standard_student_t = DistributionKernelSources.standard_student_t
const student_log_kernel = DistributionKernelSources._student_log_kernel

@kernel stable_half_student(eta::Vector{Float64}, nu::Float64, location::Float64,
                            scale::Float64) = begin
    lp = plate(eta) do e
        student_t(nu, location, scale).logpdf(exp(e))
    end
    total = sum(lp)
    return total
end
@kernel stable_student_at_exp(u::Float64, nu::Float64) = begin
    total = standard_student_t(nu).logpdf(exp(u))
    return total
end
# Each cell's own `nu`: no parameter part can run before the cells.
@kernel stable_half_student_cell_nu(eta::Vector{Float64}, nus::Vector{Float64}) = begin
    lp = plate(eta, nus) do e, nu
        student_t(nu, 0.0, 1.0).logpdf(exp(e))
    end
    total = sum(lp)
    return total
end
@kernel stable_student_dotted(eta::Vector{Float64}, nu::Float64) = begin
    lp = student_log_kernel.(nu, exp.(eta))
    total = sum(lp)
    return total
end

# d/de of the kernel at z = exp(e) / s, in BigFloat.
_half_student_slope(e, nu, s) =
    (z2 = exp(2big(e)) / big(s)^2; -(big(nu) + 1) * z2 / (big(nu) + z2))
function _eta_gradient(k, eta, rest...)
    d = zero(eta)
    Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(k), Enzyme.Active,
                    Enzyme.Duplicated(copy(eta), d), map(Enzyme.Const, rest)...)
    d
end

@testset "a Student-t kernel at an exponential reads its logarithm" begin
    k = prepare(stable_half_student)
    literal = prepare(stable_half_student; simplify = false)
    code = _code(k)
    @test occursin("_rk_at_exp_std(", code) && occursin("#stable_scale#", code)
    @test !occursin("_student_log_kernel(", code)
    mid = [-1.0, 0.0, 0.5, 2.0]
    for nu in (1.0, 4.0, Inf), (location, scale) in ((0.0, 1.0), (0.0, 2.5), (0.7, 1.0))
        args = (nu, location, scale)
        @test k(mid, args...) ≈ literal(mid, args...) rtol = 1e-13
        @test _eta_gradient(k, mid, args...) ≈ _eta_gradient(literal, mid, args...) rtol = 1e-12
    end
    # Far tails: every derivative is the stable one, to rounding.
    eta = [-700.0, -300.0, -20.0, 0.0, 20.0, 300.0, 700.0]
    for nu in (1.0, 4.0), scale in (1.0, 2.5)
        g = _eta_gradient(k, eta, nu, 0.0, scale)
        @test all(i -> _close(g[i], _half_student_slope(eta[i], nu, scale); atol = 1e-290),
                  eachindex(eta))
    end
    # Beyond exp's range the rewritten kernel is finite, the literal is not.
    k = prepare(stable_student_at_exp)
    literal = prepare(stable_student_at_exp; simplify = false)
    @test occursin("_rk_at_exp1(", _code(k))
    for nu in (1.0, 3.0)
        @test isfinite(k(800.0, nu)) && literal(800.0, nu) == -Inf
        @test _close(_reverse(k, 800.0, nu)[1], _half_student_slope(800.0, nu, 1.0))
        @test isnan(_reverse(literal, 800.0, nu)[1])
        for u in (-800.0, -20.0, 0.3, 20.0, 400.0)
            @test k(u, nu) ≈ literal(u, nu) rtol = 1e-14
        end
    end
    @test k(0.3, Inf) ≈ literal(0.3, Inf) rtol = 1e-14
    # A negative nu raises the kernel's own error.
    @test_throws DomainError k(0.3, -1.0)
    @test_throws DomainError prepare(stable_half_student)(mid, -1.0, 0.0, 1.0)
    # Per-cell parameters: the literal kernel, whose own work costs less there.
    k = prepare(stable_half_student_cell_nu)
    @test !occursin("_rk_at_exp", _code(k))
    @test k(mid, [1.0, 2.0, 3.0, 4.0]) ≈
          prepare(stable_half_student_cell_nu; simplify = false)(mid, [1.0, 2.0, 3.0, 4.0]) rtol = 1e-13
    # A dotted call computes its part once.
    k = prepare(stable_student_dotted)
    literal = prepare(stable_student_dotted; simplify = false)
    @test occursin("_rk_at_exp_part", _code(k))
    for nu in (1.0, 4.0)
        @test k(mid, nu) ≈ literal(mid, nu) rtol = 1e-13
        @test _eta_gradient(k, mid, nu) ≈ _eta_gradient(literal, mid, nu) rtol = 1e-12
    end
end

@testset "log-magnitude helpers: the literal unless the form applies" begin
    f = student_log_kernel
    @test RK._rk_at_exp_std1(f, exp, 0.3, 0.0, 2.0, 3.0) ≈ f(3.0, exp(0.3) / 2) rtol = 1e-15
    @test RK._rk_at_exp_std1(f, exp, 0.3, 0.5, 2.0, 3.0) === f(3.0, (exp(0.3) - 0.5) / 2)
    @test RK._rk_at_exp1(f, exp, 0.5f0, 3.0f0) isa Float32
    @test RK._rk_at_exp_part(f, exp, [1.0, 2.0]) === nothing
    p = RK._rk_at_exp_std_part(f, exp, 0.0, 1.0, -1.0)
    @test !p.ok
    @test_throws DomainError RK._rk_at_exp_std(f, exp, 0.3, 0.0, 1.0, p, -1.0)
    @test RK._rk_at_exp_std(f, exp, 0.3, 0.0, 1.0, nothing, 2.0) === f(2.0, exp(0.3) / 1.0)
end

# Per-cell definitions: a name each of two loops assigns once in its cell is
# that cell's definition; one a cell reads before assigning it is carried
# from the previous cell and never is.
@testset "a plate cell's own definitions" begin
    cells(body) = Expr(:for, Expr(:(=), :i, Expr(:call, GlobalRef(RK, :_plate_cells),
        Expr(:call, GlobalRef(Base, :CartesianIndices), :ax))), body)
    own = quote
        y = x[i]
        s = exp(y)
        t = t + log(s)
    end
    ast = Expr(:->, Expr(:tuple, :x, :ax),
               Expr(:block, :(t = 0.0), cells(own), cells(copy(own)), :t))
    @test count("_rk_log_of", string(RK._stable_rewrites(ast, ()))) == 2
    carried = quote
        a = exp(b)
        b = x[i]
        t = t + log(a)
    end
    ast = Expr(:->, Expr(:tuple, :x, :ax),
               Expr(:block, :(t = 0.0), :(b = 0.0), cells(carried), :t))
    @test !occursin("_rk_log_of", string(RK._stable_rewrites(ast, ())))
    # A loop that is not a plate has no prelude: nothing is hoisted out of it,
    # literal operands included.
    ast = :((y, n) -> begin
        t = 0.0
        for k in 1:n
            t = t + y / exp(2.0) + student_log_kernel(3.0, exp(1.5))
        end
        t
    end)
    rewritten = RK._stable_rewrites(ast, ())
    @test !occursin("#stable_scale#", string(rewritten))
    @test eval(rewritten)(1.0, 3) ≈ 3 * (exp(-2.0) + student_log_kernel(3.0, exp(1.5)))
end

end # module
