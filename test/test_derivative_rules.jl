# Derivative rules generated from one pure-math graph (src/derivative_rules.jl)
# and the generic Enzyme adapter (ext/ReactiveKernelsEnzymeExt.jl). The graph
# authors the primal and its named partials; every cut, the callable, and both
# Enzyme directions are derived from it. No derivative code is written here.
using DifferentiationInterface: AutoEnzyme, derivative, gradient
using LinearAlgebra: dot
import ChainRulesCore
using ChainRulesCore: NoTangent, ZeroTangent
import Enzyme
using Enzyme: Active, BatchDuplicated, Const, Duplicated, Forward,
    ForwardWithPrimal, Reverse
using ReactiveKernels
using ReactiveKernels: derivative_cut
using Test

module DerivativeRuleTestGraphs
using ReactiveKernels

# Two real inputs, one primal, one named partial per input.
@kernel two_input_graph(a::Float64, b::Float64) = begin
    s::Float64 = a + b
    e::Float64 = exp(a * b)
    y::Float64 = e + log(s)
    dy_da::Float64 = b * e + 1 / s
    dy_db::Float64 = a * e + 1 / s
    return y, dy_da, dy_db
end

# A vector port: not a scalar rule.
@kernel vector_graph(v::Vector{Float64}) = begin
    y::Float64 = sum(v)
    dy_dv::Vector{Float64} = one.(v)
    return y, dy_dv
end

# One input, so the callable has exactly one cut.
@kernel one_input_graph(x::Float64) = begin
    y::Float64 = x^3
    dy_dx::Float64 = 3 * x^2
    return y, dy_dx
end

# A rule bound as a global constant, the way a package publishes one.
const two_input = scalar_derivative_rule(
    two_input_graph; primal = :y, partials = (a = :dy_da, b = :dy_db),
    name = :two_input)
end # module

const _DRG = DerivativeRuleTestGraphs
_two_input_reference(a, b) = (
    y = exp(a * b) + log(a + b),
    dy_da = b * exp(a * b) + 1 / (a + b),
    dy_db = a * exp(a * b) + 1 / (a + b),
)

@testset "scalar derivative rule: one graph, activity-selected cuts" begin
    rule = scalar_derivative_rule(
        _DRG.two_input_graph; primal = :y, partials = (a = :dy_da, b = :dy_db),
        name = :two_input)
    @test rule isa ScalarDerivativeRule
    @test sprint(show, rule) == "ScalarDerivativeRule(:two_input; inputs = (:a, :b))"

    a, b = 0.3, 0.7
    ref = _two_input_reference(a, b)
    @test rule(a, b) == ref.y
    @test_throws MethodError rule(a)
    @test derivative_cut(rule, (true, false), a, b) == (ref.y, ref.dy_da)
    @test derivative_cut(rule, (false, true), a, b) == (ref.y, ref.dy_db)
    @test derivative_cut(rule, (true, true), a, b) == (ref.y, ref.dy_da, ref.dy_db)
    @test_throws ArgumentError derivative_cut(rule, (false, false), a, b)

    # The activity pattern is a compile-time property of the call site, so the
    # selected cut is one concretely typed prepared function.
    cut_a(r, x, y) = derivative_cut(r, (true, false), x, y)
    cut_ab(r, x, y) = derivative_cut(r, (true, true), x, y)
    @test @inferred(cut_a(rule, a, b)) == (ref.y, ref.dy_da)
    @test @inferred(cut_ab(rule, a, b)) == (ref.y, ref.dy_da, ref.dy_db)
    @test @inferred(rule(a, b)) == ref.y

    # The callable is generic over the scalar type: the mathematics lives in
    # the graph's operation table, not in a typed signature.
    @test rule(0.3f0, 0.7f0) isa Float32
    @test rule(0.3f0, 0.7f0) ≈ ref.y

    # Every cut prunes what it does not need: the primal cut has no partial
    # recipe, and the single-partial cuts do not compute the other partial.
    spec = _DRG.two_input_graph
    primal_plan = plan(spec; have = (:a, :b), want = :y)
    a_plan = plan(spec; have = (:a, :b), want = (:y, :dy_da))
    full_plan = plan(spec; have = (:a, :b), want = (:y, :dy_da, :dy_db))
    @test length(primal_plan.recipes) < length(a_plan.recipes) <
        length(full_plan.recipes)

    one = scalar_derivative_rule(
        _DRG.one_input_graph; primal = :y, partials = (x = :dy_dx,), name = :cube)
    @test one(2.0) == 8.0
    @test derivative_cut(one, (true,), 2.0) == (8.0, 12.0)
end

@testset "scalar derivative rule: authoring errors are explicit" begin
    g = _DRG.two_input_graph
    @test_throws ArgumentError scalar_derivative_rule(
        g; primal = :y, partials = (b = :dy_db, a = :dy_da))          # order
    @test_throws ArgumentError scalar_derivative_rule(
        g; primal = :y, partials = (a = :dy_da,))                     # missing
    @test_throws ArgumentError scalar_derivative_rule(
        g; primal = :nope, partials = (a = :dy_da, b = :dy_db))       # primal
    @test_throws ArgumentError scalar_derivative_rule(
        g; primal = :y, partials = (a = :y, b = :dy_db))              # = primal
    @test_throws ArgumentError scalar_derivative_rule(
        g; primal = :y, partials = (a = :dy_da, b = :missing_port))   # partial
    @test_throws ArgumentError scalar_derivative_rule(
        _DRG.vector_graph; primal = :y, partials = (v = :dy_dv,))     # not scalar
end

@testset "generated Enzyme adapter: reverse, forward, batch, guards, DI" begin
    @test Base.get_extension(ReactiveKernels, :ReactiveKernelsEnzymeExt) !== nothing
    rule = _DRG.two_input
    a, b = 0.3, 0.7
    ref = _two_input_reference(a, b)
    ref_b(x) = _two_input_reference(x, b)

    # The rule is an immutable value whose fields are singleton operation
    # tables, so a closure capturing it is a plain value: no activity
    # annotation is needed, whether the rule is a global constant (the way a
    # package publishes one) or a captured local.
    @test isbits(rule)
    @test only(Enzyme.gradient(Reverse, x -> _DRG.two_input(x, 0.7), a)) ≈ ref.dy_da
    @test only(Enzyme.gradient(Reverse, x -> x > 0 ? _DRG.two_input(x, 0.7) : -Inf, a)) ≈
        ref.dy_da

    # Reverse: both active, one active (the other Const), nested use.
    @test collect(Enzyme.gradient(Reverse, Const(rule), a, b)) ≈ [ref.dy_da, ref.dy_db]
    @test only(Enzyme.gradient(Reverse, x -> rule(x, b), a)) ≈ ref.dy_da
    @test only(Enzyme.gradient(Reverse, x -> rule(a, x), b)) ≈ ref.dy_db
    @test only(Enzyme.gradient(Reverse, x -> rule(x, 2x), a)) ≈
        _two_input_reference(a, 2a).dy_da + 2 * _two_input_reference(a, 2a).dy_db
    # A lazily evaluated guard around the rule keeps its branch semantics.
    @test only(Enzyme.gradient(Reverse, x -> x > 0 ? rule(x, b) : -Inf, a)) ≈
        ref.dy_da
    @test only(Enzyme.gradient(Reverse, x -> x > 0 ? rule(x, b) : -Inf, -a)) == 0.0
    # An inactive call contributes nothing; a broadcast maps the rule per element.
    @test only(Enzyme.gradient(Reverse, x -> rule(0.5, b) + x, a)) ≈ 1.0
    v = [0.2, 0.3, 0.4]
    @test only(Enzyme.gradient(Reverse, w -> sum(rule.(w, b)), v)) ≈
        [ref_b(x).dy_da for x in v]
    # Non-inlined callers differentiated together, one inside a loop (the shape
    # of a guarded density plus an observation plate).
    @noinline prior_op(x) = x > 0 ? rule(x, b) : -Inf
    @noinline cell_op(x, k) = k >= 0 ? rule(x, k + 1.0) : -Inf
    ks = (1.0, 2.0, 3.0)   # a tuple: a captured Vector would need Const(density)
    density(x) = prior_op(x) + sum(cell_op(x, ks[i]) for i in eachindex(ks))
    expected = ref.dy_da + sum(_two_input_reference(a, k + 1.0).dy_da for k in ks)
    @test only(Enzyme.gradient(Reverse, density, a)) ≈ expected

    # Forward: one direction, the primal alongside, a seeded direction on both
    # inputs, and the batch width Enzyme's forward gradient uses for several
    # scalar inputs.
    fwd = Enzyme.autodiff(Forward, Const(rule), Duplicated, Duplicated(a, 1.0), Const(b))
    @test only(fwd) ≈ ref.dy_da
    with_primal = Enzyme.autodiff(ForwardWithPrimal, Const(rule), Duplicated,
                                  Duplicated(a, 1.0), Const(b))
    @test collect(with_primal) ≈ [ref.dy_da, ref.y]
    seeded = Enzyme.autodiff(Forward, Const(rule), Duplicated,
                             Duplicated(a, 1.0), Duplicated(b, 2.0))
    @test only(seeded) ≈ ref.dy_da + 2 * ref.dy_db
    batched = Enzyme.autodiff(Forward, Const(rule), BatchDuplicated,
                              BatchDuplicated(a, (1.0, 0.0)), BatchDuplicated(b, (0.0, 1.0)))
    @test collect(only(batched)) ≈ [ref.dy_da, ref.dy_db]
    @test collect(Enzyme.gradient(Forward, Const(rule), a, b)) ≈ [ref.dy_da, ref.dy_db]

    # Adjoint consistency of the two generated directions on the same cut:
    # ⟨ȳ, J ẋ⟩ = ⟨Jᵀ ȳ, ẋ⟩.
    ẋ = (1.0, 2.0)
    ȳ = 1.5
    jvp = only(Enzyme.autodiff(Forward, Const(rule), Duplicated,
                               Duplicated(a, ẋ[1]), Duplicated(b, ẋ[2])))
    vjp = only(Enzyme.autodiff(Reverse, Const(rule), Active, Active(a), Active(b)))
    @test ȳ * jvp ≈ ȳ * (vjp[1] * ẋ[1] + vjp[2] * ẋ[2])

    # Through DifferentiationInterface, both modes.
    @test gradient(x -> rule(x, b), AutoEnzyme(; mode = Enzyme.Reverse), a) ≈ ref.dy_da
    @test derivative(x -> rule(x, b), AutoEnzyme(; mode = Enzyme.Forward), a) ≈ ref.dy_da
end

module DerivativeRuleVectorGraphs
using ReactiveKernels
# One graph with authored JVP and VJP branches (the manual-rule design example).
@kernel matvec_rule(
        A::Matrix{Float64}, x::Vector{Float64},
        A_dot::Matrix{Float64}, x_dot::Vector{Float64},
        y_bar::Vector{Float64}) = begin
    y::Vector{Float64} = A * x
    A_direction::Vector{Float64} = A_dot * x
    x_direction::Vector{Float64} = A * x_dot
    y_dot::Vector{Float64} = A_direction + x_direction
    A_bar::Matrix{Float64} = y_bar * transpose(x)
    x_bar::Vector{Float64} = transpose(A) * y_bar
    return y, y_dot, A_bar, x_bar
end
# Reverse branch only.
@kernel scale_rule(s::Float64, v::Vector{Float64}, y_bar::Vector{Float64}) = begin
    y::Vector{Float64} = s .* v
    s_bar::Float64 = sum(y_bar .* v)
    v_bar::Vector{Float64} = s .* y_bar
    return y, s_bar, v_bar
end
# The VJP reads the primal output: the staged residual is y, not x.
@kernel softmax_rule(x::Vector{Float64}, y_bar::Vector{Float64}) = begin
    e::Vector{Float64} = exp.(x .- maximum(x))
    y::Vector{Float64} = e ./ sum(e)
    x_bar::Vector{Float64} = y .* (y_bar .- sum(y_bar .* y))
    return y, x_bar
end
# A shared exp/log residual: both cotangents read the weights w.
@kernel logsumexp_rule(a::Vector{Float64}, x::Vector{Float64}, y_bar::Float64) = begin
    e::Vector{Float64} = exp.(a .* x)
    s::Float64 = sum(e)
    y::Float64 = log(s)
    w::Vector{Float64} = e ./ s
    a_bar::Vector{Float64} = y_bar .* w .* x
    x_bar::Vector{Float64} = y_bar .* w .* a
    return y, a_bar, x_bar
end
# An ODE right-hand side: `t` is a port the mathematics does not use, so its
# authored cotangent is a covector-independent constant.
@kernel decay_rhs(u::Vector{Float64}, p::Vector{Float64}, t::Float64,
        du_bar::Vector{Float64}) = begin
    du::Vector{Float64} = -p .* u
    u_bar::Vector{Float64} = -p .* du_bar
    p_bar::Vector{Float64} = -u .* du_bar
    t_bar::Float64 = 0.0
    return du, u_bar, p_bar, t_bar
end
end # module

const _DRV = DerivativeRuleVectorGraphs

@testset "vector derivative rule: one graph, authored forward and reverse branches" begin
    matvec = derivative_rule(_DRV.matvec_rule; primal = :y,
        directions = (A = :A_dot, x = :x_dot), tangent = :y_dot,
        covector = :y_bar, cotangents = (A = :A_bar, x = :x_bar), name = :matvec)
    @test matvec isa DerivativeRule
    @test has_forward_branch(matvec) && has_reverse_branch(matvec)
    @test sprint(show, matvec) ==
        "DerivativeRule(:matvec; inputs = (:A, :x), branches = forward+reverse)"
    A = [1.0 2.0; 3.0 4.0]; x = [0.5, -1.0]
    A_dot = [0.1 -0.2; 0.3 0.4]; x_dot = [-0.7, 0.2]; y_bar = [1.2, -0.4]
    @test matvec(A, x) == A * x
    y, y_dot = forward_cut(matvec, A, x, A_dot, x_dot)
    @test y == A * x && y_dot == A_dot * x + A * x_dot
    # Activity selects both the executed cotangents and the retained inputs:
    # A_bar reads x only, x_bar reads A only.
    @test reverse_residuals(matvec, Val(1)) == (false, true)
    @test reverse_residuals(matvec, Val(2)) == (true, false)
    @test reverse_residuals(matvec, Val(3)) == (true, true)
    @test reverse_cut(matvec, Val(1), nothing, x, y_bar) == (y_bar * transpose(x),)
    @test reverse_cut(matvec, Val(2), A, nothing, y_bar) == (transpose(A) * y_bar,)
    A_bar, x_bar = reverse_cut(matvec, Val(3), A, x, y_bar)
    @test A_bar == y_bar * transpose(x) && x_bar == transpose(A) * y_bar
    # Adjoint consistency of the two authored branches.
    @test dot(y_bar, y_dot) ≈ dot(A_bar, A_dot) + dot(x_bar, x_dot)
    cut3(r, A, x, yb) = reverse_cut(r, Val(3), A, x, yb)
    @test @inferred(cut3(matvec, A, x, y_bar)) == (A_bar, x_bar)
    @test @inferred(forward_cut(matvec, A, x, A_dot, x_dot)) == (y, y_dot)
    # Cuts prune: the primal plan has one recipe, the forward plan adds the
    # JVP recipes only, an x-only reverse plan has the x_bar recipe only.
    spec = _DRV.matvec_rule
    @test length(plan(spec; have = (:A, :x), want = :y).recipes) == 1
    @test length(plan(spec; have = (:A, :x, :y_bar), want = :x_bar).recipes) == 1
    @test length(plan(spec; have = (:A, :x, :A_dot, :x_dot), want = (:y, :y_dot)).recipes) == 4

    scale = derivative_rule(_DRV.scale_rule; primal = :y,
        covector = :y_bar, cotangents = (s = :s_bar, v = :v_bar), name = :scale)
    @test !has_forward_branch(scale) && has_reverse_branch(scale)
    @test scale(2.0, x) == 2.0 .* x
    @test reverse_cut(scale, Val(3), 2.0, x, y_bar) == (sum(y_bar .* x), 2.0 .* y_bar)
    @test reverse_residuals(scale, Val(1)) == (false, true)
    @test_throws ArgumentError forward_cut(scale, 2.0, x, 1.0, x)
    @test_throws ArgumentError reverse_cut(scale, Val(3), 2.0, x)
end

@testset "vector derivative rule: authoring errors are explicit" begin
    g = _DRV.matvec_rule
    @test_throws ArgumentError derivative_rule(g; primal = :y)                              # no branch
    @test_throws ArgumentError derivative_rule(g; primal = :nope, covector = :y_bar,
        cotangents = (A = :A_bar, x = :x_bar))                                              # primal
    @test_throws ArgumentError derivative_rule(g; primal = :y, directions = (A = :A_dot, x = :x_dot))  # no tangent
    @test_throws ArgumentError derivative_rule(g; primal = :y, covector = :y_bar,
        cotangents = (x = :x_bar, A = :A_bar))                                              # order
    @test_throws ArgumentError derivative_rule(g; primal = :y, covector = :y_bar,
        cotangents = (A = :A_bar,))                                                         # missing
    @test_throws ArgumentError derivative_rule(g; primal = :y, covector = :nope,
        cotangents = (A = :A_bar, x = :x_bar))                                              # covector
    @test_throws ArgumentError derivative_rule(g; primal = :y, directions = (A = :A_dot, x = :x_dot),
        tangent = :y)                                                                       # tangent = primal
    @test_throws ArgumentError derivative_rule(g; primal = :y, directions = (A = :A_dot, x = :A_dot),
        tangent = :y_dot)                                                                   # duplicate direction
end

@testset "generated Enzyme adapter for vector rules: reverse, forward, batch, DI" begin
    matvec = derivative_rule(_DRV.matvec_rule; primal = :y,
        directions = (A = :A_dot, x = :x_dot), tangent = :y_dot,
        covector = :y_bar, cotangents = (A = :A_bar, x = :x_bar), name = :matvec)
    scale = derivative_rule(_DRV.scale_rule; primal = :y,
        covector = :y_bar, cotangents = (s = :s_bar, v = :v_bar), name = :scale)
    A = [1.0 2.0; 3.0 4.0]; x = [0.5, -1.0]; w = [1.2, -0.4]
    A_dot = [0.1 -0.2; 0.3 0.4]; x_dot = [-0.7, 0.2]
    # Closures below capture arrays (A, w), so they are Const for Enzyme as usual.
    loss(A, x) = sum(matvec(A, x) .* w)
    gA, gx = Enzyme.gradient(Reverse, Const(loss), A, x)
    @test gA ≈ w * transpose(x)
    @test gx ≈ transpose(A) * w
    # One active array input: only its cotangent is computed and returned.
    @test only(Enzyme.gradient(Reverse, Const(xx -> sum(matvec(A, xx) .* w)), x)) ≈ transpose(A) * w
    @test only(Enzyme.gradient(Reverse, Const(AA -> sum(matvec(AA, x) .* w)), A)) ≈ w * transpose(x)
    # Mixed scalar (Active) and array (Duplicated) inputs, reverse branch only.
    loss2(s, v) = sum(scale(s, v) .* w)
    gs, gv = Enzyme.gradient(Reverse, Const(loss2), 2.0, x)
    @test gs ≈ sum(w .* x)
    @test gv ≈ 2.0 .* w
    # Nested use inside a larger differentiated function.
    outer(xx) = sum(matvec(A, xx) .^ 2) + sum(scale(3.0, xx))
    @test only(Enzyme.gradient(Reverse, Const(outer), x)) ≈ 2 .* (transpose(A) * (A * x)) .+ 3.0
    # Array-valued result with a batched covector (Jacobian by reverse lanes).
    @test only(Enzyme.jacobian(Reverse, Const(xx -> matvec(A, xx)), x)) ≈ A
    # Forward: directions on both inputs, one input, and batch lanes.
    fwd = Enzyme.autodiff(Forward, Const(matvec), Duplicated, Duplicated(A, A_dot), Duplicated(x, x_dot))
    @test only(fwd) ≈ A_dot * x + A * x_dot
    fwd_x = Enzyme.autodiff(Forward, Const(matvec), Duplicated, Const(A), Duplicated(x, x_dot))
    @test only(fwd_x) ≈ A * x_dot
    with_primal = Enzyme.autodiff(ForwardWithPrimal, Const(matvec), Duplicated, Duplicated(A, A_dot), Const(x))
    @test with_primal[1] ≈ A_dot * x && with_primal[2] ≈ A * x
    batched = Enzyme.autodiff(Forward, Const(matvec), BatchDuplicated,
                              BatchDuplicated(A, (A_dot, zero(A))), BatchDuplicated(x, (zero(x), x_dot)))
    @test only(batched)[1] ≈ A_dot * x && only(batched)[2] ≈ A * x_dot
    @test only(Enzyme.jacobian(Forward, Const(xx -> matvec(A, xx)), x)) ≈ A
    # No forward branch: explicit refusal.
    @test_throws ArgumentError Enzyme.autodiff(Forward, Const(scale), Duplicated, Const(2.0), Duplicated(x, x_dot))
    # Through DifferentiationInterface (Const function annotation, as RK's own AD tests use).
    backend = AutoEnzyme(; mode = Enzyme.Reverse, function_annotation = Enzyme.Const)
    @test gradient(xx -> sum(matvec(A, xx) .* w), backend, x) ≈ transpose(A) * w
end

@testset "vector derivative rule: residual compaction (cross-stage liveness)" begin
    sm = derivative_rule(_DRV.softmax_rule; primal = :y, covector = :y_bar,
        cotangents = (x = :x_bar,), name = :softmax)
    x = [0.3, -1.2, 2.0]; ȳ = [0.5, -0.25, 1.0]
    softmax(v) = exp.(v) ./ sum(exp.(v))
    # The recompute form reads x; the staged form retains the primal output.
    @test reverse_residuals(sm, Val(1)) == (true,)
    @test stage_residuals(sm, Val(1)) == (:y,)
    y, residuals = stage_primal(sm, Val(1), x)
    @test y ≈ softmax(x) && residuals[1] === y
    @test stage_reverse(sm, Val(1), residuals, ȳ) == reverse_cut(sm, Val(1), x, ȳ)
    # Cut sizes, read off the encoded bodies: the recompute form rebuilds e and y
    # from x (3 statements), the staged reverse stage is the VJP statement only.
    reverse_specs, staged_specs = typeof(sm).parameters[6], typeof(sm).parameters[7]
    @test length(reverse_specs[1][2]) == 3 && length(staged_specs[1][2][2]) == 1

    lse = derivative_rule(_DRV.logsumexp_rule; primal = :y, covector = :y_bar,
        cotangents = (a = :a_bar, x = :x_bar), name = :logsumexp)
    a = [0.4, -0.7, 1.1]; z = [1.0, 0.5, -2.0]
    # Inputs first in signature order, then the shared intermediate.
    @test stage_residuals(lse, Val(3)) == (:a, :x, :w)
    @test stage_residuals(lse, Val(1)) == (:x, :w)
    @test stage_residuals(lse, Val(2)) == (:a, :w)
    for mask in 1:3
        y, residuals = stage_primal(lse, Val(mask), a, z)
        @test y ≈ log(sum(exp.(a .* z)))
        @test all(stage_reverse(lse, Val(mask), residuals, 0.7) .≈
                  reverse_cut(lse, Val(mask), a, z, 0.7))
    end
    staged3(r, a, z) = stage_primal(r, Val(3), a, z)
    @test @inferred(staged3(lse, a, z))[1] isa Float64
    @test_throws ArgumentError stage_primal(lse, Val(4), a, z)
    @test_throws ArgumentError stage_primal(lse, Val(3), a)
    @test_throws ArgumentError stage_reverse(lse, Val(3), (a,), 0.7)
    @test_throws ArgumentError stage_residuals(lse, Val(0))
    # A rule without a reverse branch has no staged cuts.
    @kernel double_graph(v::Vector{Float64}, v_dot::Vector{Float64}) = begin
        y::Vector{Float64} = 2 .* v
        y_dot::Vector{Float64} = 2 .* v_dot
        return y, y_dot
    end
    double = derivative_rule(double_graph; primal = :y,
        directions = (v = :v_dot,), tangent = :y_dot, name = :double)
    @test_throws ArgumentError stage_primal(double, Val(1), x)
    @test_throws ArgumentError stage_residuals(double, Val(1))

    # Enzyme stages through the compacted cuts.
    w = [1.2, -0.4, 0.3]
    ysm = softmax(x)
    @test only(Enzyme.gradient(Reverse, Const(v -> sum(sm(v) .* w)), x)) ≈
        ysm .* (w .- sum(w .* ysm))
    weights = exp.(a .* z) ./ sum(exp.(a .* z))
    ga, gz = Enzyme.gradient(Reverse, Const((aa, zz) -> lse(aa, zz)), a, z)
    @test ga ≈ weights .* z && gz ≈ weights .* a
    @test only(Enzyme.gradient(Reverse, Const(zz -> lse(a, zz)), z)) ≈ weights .* a

    # A covector-independent cotangent (the authored zero for `t`) is retained
    # from the primal stage and returned by the reverse stage as it is.
    decay = derivative_rule(_DRV.decay_rhs; primal = :du, covector = :du_bar,
        cotangents = (u = :u_bar, p = :p_bar, t = :t_bar), name = :decay)
    u = [1.0, 2.0]; pp = [0.5, 1.5]; λ = [0.3, -0.8]
    @test :t_bar in stage_residuals(decay, Val(7))
    @test stage_residuals(decay, Val(3)) == (:u, :p)
    for mask in 1:7
        _, residuals = stage_primal(decay, Val(mask), u, pp, 0.4)
        @test stage_reverse(decay, Val(mask), residuals, λ) ==
            reverse_cut(decay, Val(mask), u, pp, 0.4, λ)
    end
end

@testset "generated ChainRules adapter: scalar and vector rules" begin
    @test Base.get_extension(ReactiveKernels, :ReactiveKernelsChainRulesCoreExt) !== nothing
    a, b = 0.3, 0.7
    ref = _two_input_reference(a, b)
    y, pullback = ChainRulesCore.rrule(_DRG.two_input, a, b)
    @test y == ref.y
    @test collect(pullback(1.5)[2:3]) ≈ [1.5 * ref.dy_da, 1.5 * ref.dy_db]
    @test pullback(1.5)[1] === NoTangent()
    @test collect(ChainRulesCore.frule((NoTangent(), 1.0, 2.0), _DRG.two_input, a, b)) ≈
        [ref.y, ref.dy_da + 2 * ref.dy_db]
    @test ChainRulesCore.frule((NoTangent(), ZeroTangent(), 2.0), _DRG.two_input, a, b)[2] ≈ 2 * ref.dy_db

    matvec = derivative_rule(_DRV.matvec_rule; primal = :y,
        directions = (A = :A_dot, x = :x_dot), tangent = :y_dot,
        covector = :y_bar, cotangents = (A = :A_bar, x = :x_bar), name = :matvec)
    scale = derivative_rule(_DRV.scale_rule; primal = :y,
        covector = :y_bar, cotangents = (s = :s_bar, v = :v_bar), name = :scale)
    A = [1.0 2.0; 3.0 4.0]; x = [0.5, -1.0]
    A_dot = [0.1 -0.2; 0.3 0.4]; x_dot = [-0.7, 0.2]; y_bar = [1.2, -0.4]
    y, pullback = ChainRulesCore.rrule(matvec, A, x)
    @test y == A * x
    @test pullback.residuals == (A, x)
    cot = pullback(y_bar)
    @test cot[1] === NoTangent() && cot[2] == y_bar * transpose(x) && cot[3] == transpose(A) * y_bar
    fy, fẏ = ChainRulesCore.frule((NoTangent(), A_dot, x_dot), matvec, A, x)
    @test fy == A * x && fẏ == A_dot * x + A * x_dot
    @test ChainRulesCore.frule((NoTangent(), ZeroTangent(), x_dot), matvec, A, x)[2] == A * x_dot
    @test_throws ArgumentError ChainRulesCore.frule((NoTangent(), 1.0, x_dot), scale, 2.0, x)
    sy, spullback = ChainRulesCore.rrule(scale, 2.0, x)
    @test sy == 2.0 .* x
    @test spullback(y_bar)[2] ≈ sum(y_bar .* x) && spullback(y_bar)[3] == 2.0 .* y_bar
    # Staged residuals: the softmax pullback holds the primal output, not x.
    sm = derivative_rule(_DRV.softmax_rule; primal = :y, covector = :y_bar,
        cotangents = (x = :x_bar,), name = :softmax)
    v = [0.3, -1.2, 2.0]; v̄ = [0.5, -0.25, 1.0]
    my, mpullback = ChainRulesCore.rrule(sm, v)
    @test mpullback.residuals == (my,)
    @test mpullback(v̄)[2] == only(reverse_cut(sm, Val(1), v, v̄))
end
