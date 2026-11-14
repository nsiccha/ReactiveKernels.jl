module PlateReverseOnePassTests
using ReactiveKernels, ReactiveKernelsPPL, MutatingFunctions, Test
using DifferentiationInterface: AutoEnzyme
import Enzyme
const RK = ReactiveKernels

# The generated sweep's two schedules must agree. `Val(false)` adds the seed's
# cotangents into the shadows as given (two passes); `Val(true)` also returns
# the value and may add unscaled partials into a zeroed owned buffer, scaling
# it once after the loop (one pass, `_PR_WANT_VALUE`). A factored read must
# not scale another read's contributions to the same shadow.

# A hand-built program `(x, s) -> total` (as `test_plate_reverse_direct_reads.jl`
# builds them): `cells` lists one cell loop per entry, each a vector of
# densities of the read `v` (or `v1`, `v2` through two prepared aliases of x).
function program(cells; aliases = false)
    prefix = Any[:(raw = Base.broadcastable(x)),
        :(prepared = Base.Broadcast.preprocess(nothing, raw)),
        :(total = $(GlobalRef(RK, :_plate_total_seed))(Float64, 0.0))]
    aliases && push!(prefix, :(again = Base.Broadcast.preprocess(nothing, raw)))
    read(p) = Expr(:call, GlobalRef(Base.Broadcast, :_broadcast_getindex), p, :index)
    loops = Any[]
    for density in cells
        body = Expr(:block, :(index = Base.CartesianIndex(linear)), :(v1 = $(read(:prepared))))
        aliases && push!(body.args, :(v2 = $(read(:again))))
        push!(body.args, :(term = $density),
              :(total = $(GlobalRef(RK, :_plate_total_add))(total, term)))
        push!(loops, macroexpand(RK, Expr(:macrocall, GlobalRef(Base, Symbol("@simd")),
            LineNumberNode(0), Expr(:for, :(linear = Base.eachindex(x)), body))))
    end
    Expr(:function, Expr(:tuple, :x, :s), Expr(:block, prefix..., loops..., :total))
end
const MUL, ADD, SIN = GlobalRef(Base, :*), GlobalRef(Base, :+), GlobalRef(Base, :sin)
# The two loops of `program(cells)` in the arms of `if 0 < s`, as the native
# lowering puts a cell loop's direct and extruded copies: one call runs one.
function arms_program(cells)
    ast = program(cells)
    stmts = ast.args[2].args
    condition = Expr(:call, GlobalRef(Base, :<), 0.0, :s)
    ast.args[2] = Expr(:block, stmts[1:end-3]...,
        Expr(:if, condition, Expr(:block, stmts[end-2]), Expr(:block, stmts[end-1])), stmts[end])
    ast
end
# name => (program, value, ∂x, ∂s, whether a read factors through its buffer)
const PROGRAMS = [
    # s · v: the read's partial is invariant, s's is the read (deferred to a
    # per-group product through the buffer).
    "one loop" => (program([:($MUL(s, v1))]),
        (x, s) -> s * sum(x; init = 0.0), (x, s) -> fill(s, length(x)),
        (x, s) -> sum(x; init = 0.0), true),
    # Two cotangent terms reach one read. Select the ordinary per-cell sum
    # before extracting a single factored term.
    "one read with two terms" => (program([:($ADD($MUL(s, v1), $SIN(v1)))]),
        (x, s) -> s * sum(x; init = 0.0) + sum(sin, x; init = 0.0),
        (x, s) -> s .+ cos.(x), (x, s) -> sum(x; init = 0.0), false),
    # The same operand in two cell loops of one program.
    "two loops" => (program([:($MUL(s, v1)), :($SIN(v1))]),
        (x, s) -> s * sum(x; init = 0.0) + sum(sin, x; init = 0.0),
        (x, s) -> s .+ cos.(x), (x, s) -> sum(x; init = 0.0), false),
    # Two prepared aliases of the same operand in one cell.
    "two aliases" => (program([:($ADD($MUL(s, v1), $SIN(v2)))]; aliases = true),
        (x, s) -> s * sum(x; init = 0.0) + sum(sin, x; init = 0.0),
        (x, s) -> s .+ cos.(x), (x, s) -> sum(x; init = 0.0), false),
    # The same operand in two loops in exclusive arms: each still factors.
    "exclusive arms" => (arms_program([:($MUL(s, v1)), :($SIN(v1))]),
        (x, s) -> 0 < s ? s * sum(x; init = 0.0) : sum(sin, x; init = 0.0),
        (x, s) -> 0 < s ? fill(s, length(x)) : cos.(x),
        (x, s) -> 0 < s ? sum(x; init = 0.0) : 0.0, true),
]

# The cotangent of the gathered operand's parent: each cell's into its group.
function scatter(g, c, n)
    out = zeros(n)
    for (i, j) in enumerate(g)
        out[j] += c[i]
    end
    out
end

@testset "one-pass and two-pass sweeps agree: $name" for (name, (ast, value, dx, ds, factors)) in PROGRAMS
    rexpr, scalar, batched = RK._pr_program_reverse(ast, [:x, :s], RK, Dict{Symbol,Any}())
    @test scalar == [:s]
    @test batched == [:x]
    @test occursin("_pr_scale_buffer!", string(rexpr)) == factors
    primal, reverse = RK.compile(ast), RK.compile(rexpr)
    for n in (0, 3, 200), s in (-0.6, 0.0, 0.6), ȳ in (0.0, 1.7)
        # A plain operand: its own storage is the buffer.
        x = [0.4 * sin(i) for i in 1:n]
        @test primal(x, s) ≈ value(x, s)
        given = zeros(n)
        _, s̄ = reverse(x, s, ȳ, Val(false), given)
        @test given ≈ ȳ .* dx(x, s)
        @test s̄ ≈ ȳ * ds(x, s)
        owned = zeros(n)
        v, s̄ = reverse(x, s, ȳ, Val(true), owned)
        @test v ≈ value(x, s)
        @test owned ≈ given
        @test s̄ ≈ ȳ * ds(x, s)
        # A gathered operand `view(z, g)`: repeated groups, and a last group
        # no cell reads (its buffer entry stays zero).
        J = max(n ÷ 7, 1) + 1
        z = [0.3 * cos(j) for j in 1:J]
        g = [mod1(3i, J - 1) for i in 1:n]
        x = view(z, g)
        expected = scatter(g, ȳ .* dx(collect(x), s), J)
        given = zeros(J)
        _, s̄ = reverse(x, s, ȳ, Val(false), view(given, g))
        @test given ≈ expected
        @test s̄ ≈ ȳ * ds(collect(x), s)
        buffer = view(zeros(J), g)
        v, s̄ = reverse(x, s, ȳ, Val(true), buffer)
        @test v ≈ value(collect(x), s)
        @test parent(buffer) ≈ expected
        @test parent(buffer)[J] == 0.0
        @test s̄ ≈ ȳ * ds(collect(x), s)
    end
end

# Which views can take one pass, on types: an index vector may repeat groups; a
# unit range repeats none (its buffer is never shorter than its cells).
@testset "one-pass eligibility by index type" begin
    z = zeros(5)
    @test RK._pr_gather_value(typeof(view(z, [1, 2, 2])))
    @test !RK._pr_gather_value(typeof(view(z, 2:4)))
    @test !RK._pr_gather_value(typeof(view(z, Base.OneTo(3))))
    @test !RK._pr_gather_value(typeof(z))
end

# The primal picks the rule per call: one pass for a gathered operand with at
# least `_PR_ONE_PASS_MIN_CELLS` cells and a shorter parent, the two-pass body
# otherwise (an identity gather on a small plate, as RKPPL's `b[axes(n, 1)]`,
# or a parent as long as the plate). Gradients through Enzyme agree either way.
seeded(rule, x, s, ȳ) = ȳ * rule(x, s)
@testset "one pass is chosen per call: $name, $n cells, parent of $J" for
        (name, (ast, value, dx, ds, _)) in PROGRAMS[1:2],
        (n, J, onepass) in ((0, 5, false), (40, 8, false), (200, 30, true),
                           (200, 250, false), (21, 21, false))
    rexpr, scalar, batched = RK._pr_program_reverse(ast, [:x, :s], RK, Dict{Symbol,Any}())
    primal, reverse = RK.compile(ast), RK.compile(rexpr)
    rule = RK._PlateSumRule{(2,),(1,),typeof(primal),typeof(reverse)}(primal, reverse)
    z = [0.3 * cos(j) for j in 1:J]
    g = n == J ? collect(1:n) : [mod1(3i, J) for i in 1:n]
    x = view(z, g)
    @test RK._pr_rule_target(rule, (x, 0.6)) isa
          (onepass ? RK._PlateSumRuleOnePass : RK._PlateSumRuleBody)
    @test RK._pr_rule_target(rule, (collect(x), 0.6)) isa RK._PlateSumRuleBody
    @test RK._pr_rule_target(rule, (view(zeros(n + 3), 2:n+1), 0.6)) isa RK._PlateSumRuleBody
    @test rule(x, 0.6) ≈ value(x, 0.6)
    dz = zeros(J)
    cot = first(Enzyme.autodiff(Enzyme.Reverse, seeded, Enzyme.Active, Enzyme.Const(rule),
        Enzyme.Duplicated(x, view(dz, g)), Enzyme.Active(0.6), Enzyme.Const(1.7)))
    @test dz ≈ scatter(g, 1.7 .* dx(x, 0.6), J)
    @test cot[3] ≈ 1.7 * ds(x, 0.6)
end

# The original public preparation reproducer, without a diagnostic adapter or
# generator hook. Reuse each preparation with changed values and lengths.
@kernel gathered_fanout(z::Vector{Float64}, groups::Vector{Int}, s::Float64) = begin
    terms = plate(z[groups]) do v
        s * v + sin(v)
    end
    total::Float64 = sum(terms)
    return total
end

structure(x) = (1, 0)
structure(x::RK._PlateProgram) = structure(x.ast)
structure(x::RK._PlateCall) = structure(RK._program_expression(x.program))
structure(x::RK._PlateSumRule) = structure(x.program) .+ structure(x.reverse)
function structure(x::Expr)
    children = map(structure, x.args)
    (1 + sum(first, children; init = 0),
     (x.head in (:for, :while)) + sum(last, children; init = 0))
end

@testset "gathered read fan-out through public ordinary Reverse" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    z0 = [0.3 * cos(j) for j in 1:30]
    groups0 = [mod1(3i, 29) for i in 1:200]
    for enabled in (false, true)
        q, shape = with_plate_reverse(enabled) do
            k = prepare_nonallocating(gathered_fanout, z0, groups0, 0.6)
            ast = RK._program_expression(first(RK._ad_na_program(k, (1,), (z0, groups0, 0.6))))
            prepare_ad(k, backend, z0, groups0, 0.6; active = :z), structure(ast)
        end
        for (n, J) in ((0, 5), (40, 8), (200, 30), (200, 250), (2000, 71)), s in (-0.6, 0.0, 0.6)
            z = [0.2 * sin(j) for j in 1:J]
            groups = [mod1(3i, J - 1) for i in 1:n]
            before = deepcopy((z, groups))
            v, dz = ad_value_and_gradient!(q, zero(z), z, groups, s)
            @test v ≈ sum((s * z[j] + sin(z[j]) for j in groups); init = 0.0)
            @test dz ≈ scatter(groups, [s + cos(z[j]) for j in groups], J)
            @test dz[end] == 0.0
            @test (z, groups) == before
        end
        with_plate_reverse(enabled) do
            groups = [mod1(7i, 29) for i in 1:2000]
            k = prepare_nonallocating(gathered_fanout, z0, groups, 0.6)
            ast = RK._program_expression(first(RK._ad_na_program(k, (1,), (z0, groups, 0.6))))
            @test structure(ast) == shape
        end
    end
end

# End to end: a grouped model whose plate reads `sd .* z[group]` (a gathered
# view) takes one pass from `_PR_ONE_PASS_MIN_CELLS` cells when its buffer (as
# long as `z`) is shorter than the plate, and two passes otherwise.
module OnePassModels
using ReactiveKernelsPPL
const grouped = @rkppl begin
    a ~ Normal(0, 5)
    sd ~ Exponential(1)
    sigma ~ Exponential(1)
    z[1:J] .~ Normal.(0, 1)
    mu = a .+ x .+ sd .* z[group]
    y .~ Normal.(mu, sigma)
end
end

@testset "one-pass grouped gradients, $n observations in $J groups" for (n, J) in
        ((40, 10), (600, 150), (600, 700))
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    # The last group has no observation.
    group = [mod1(7i, J - 1) for i in 1:n]
    x = [0.3 * cos(i) for i in 1:n]
    y = [0.5 * sin(i) + 0.1 * group[i] / J for i in 1:n]
    plan = OnePassModels.grouped(; group, J, x) | (; y)
    built = build_kernel(plan)
    u = [0.1 * sin(i) for i in 1:built.layout.total]
    for nonallocating in (false, true)
        off = with_plate_reverse(false) do
            prepare_sampler(built, plan, u; backend, nonallocating)
        end
        on = with_plate_reverse(true) do
            prepare_sampler(built, plan, u; backend, nonallocating)
        end
        @test any(r -> last(r) == "rule", plate_reverse_report(on.kernel))
        for point in (u, u .+ 0.3, u .- 0.2)
            original = copy(point)
            g0, g1 = similar(point), similar(point)
            v0, _ = sampler_value_and_gradient!(off, g0, point)
            v1, _ = sampler_value_and_gradient!(on, g1, point)
            @test v1 ≈ v0 rtol = 1e-12
            @test g1 ≈ g0 rtol = 1e-10 atol = 1e-12
            @test point == original
        end
        # Witness of the schedule: one pass keeps a J-length buffer on its tape,
        # an allocation per gradient that the non-allocating kernel otherwise
        # does not make (todo `08gmx4k` may move it to owned storage, which then
        # needs another witness). The rule off is the control. More groups
        # than observations: two passes, no buffer.
        if nonallocating && n >= 128
            g = similar(u)
            sampler_value_and_gradient!(on, g, u)
            sampler_value_and_gradient!(off, g, u)
            @test ((@allocated sampler_value_and_gradient!(on, g, u)) >= 8J) == (J < n)
            @test (@allocated sampler_value_and_gradient!(off, g, u)) < 8J
        end
    end
end
end
