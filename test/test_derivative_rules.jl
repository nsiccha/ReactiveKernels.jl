# Derivative rules generated from one pure-math graph (src/derivative_rules.jl)
# and the generic Enzyme adapter (ext/ReactiveKernelsEnzymeExt.jl). The graph
# authors the primal and its named partials; every cut, the callable, and both
# Enzyme directions are derived from it. No derivative code is written here.
using DifferentiationInterface: AutoEnzyme, derivative, gradient
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
