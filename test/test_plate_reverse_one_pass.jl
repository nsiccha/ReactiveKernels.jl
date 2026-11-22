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

@testset "quotient values and seeded cotangents across the floating-point range" begin
    divop = GlobalRef(Base, :/)
    function quotient_sweep(op)
        q = :($divop(v1, s))
        ast = program([op === identity ? q : Expr(:call, GlobalRef(Base, nameof(op)), q)])
        RK.compile(first(RK._pr_program_reverse(ast, [:x, :s], RK, Dict{Symbol,Any}())))
    end

    for (op, slope) in ((abs2, z -> 2z), (sin, cos), (identity, one))
        sweep = quotient_sweep(op)
        for (x, d, seed) in (([1e250], 1e100, 1.0),
                             ([0.0, 1e-313], 1e-310, 1.0),
                             ([0.0, 1e-313], -1e-310, -0.5),
                             ([2.0, -0.5], -0.75, 2.0),
                             ([0.25, 1.0], Inf, 1.0),
                             (Float32[0, 1f-38], 1f-40, 1e-100),
                             (Float64[], 0.0, 1.0))
            q = x ./ d
            # Form each product in a wider exponent range before rounding to
            # the gradient's Float64 destination. The primal uses native /.
            dy = BigFloat(seed) .* BigFloat.(slope.(q))
            gx = Float64.(dy ./ BigFloat(d))
            gd = Float64(-sum(dy .* BigFloat.(q) ./ BigFloat(d); init = big"0.0"))
            for want in (false, true)
                dx = zeros(length(x))
                value, ds = sweep(x, d, seed, Val(want), dx)
                want && @test value ≈ sum(op, q; init = 0.0)
                tolerance = eltype(x) === Float32 ? 5e-7 : 1e-13
                @test all(isapprox.(dx, gx; rtol = tolerance, atol = 0.0))
                @test isapprox(ds, gd; rtol = tolerance, atol = 0.0)
            end
        end
    end

    linear = quotient_sweep(identity)
    # Adjacent rounding boundaries where multiplication by inv(d) changes
    # whether the quotient or its divisor cotangent is finite.
    for (x, d) in ((0x1.f335084b9e1fcp+1023, 0x1.f335084b9e1fdp-1),
                   (0x1.77ee5e8d806e2p+1023, 0x1.77ee5e8d806e2p-1),
                   (0x1.8368cae763e03p+1023, 0x1.bd5e79a7c1822p-1))
        dx = zeros(1)
        value, ds = linear([x], d, 1.0, Val(true), dx)
        @test value == x / d
        @test ds == -(x / d) / d
    end
    # Mutating the same bound storage must recheck its range on the next call.
    x, dx = [1.0], zeros(1)
    d = 0x1.bd5e79a7c1822p-1
    linear(x, d, 1.0, Val(true), dx)
    x[1] = 0x1.8368cae763e03p+1023
    fill!(dx, 0)
    value, ds = linear(x, d, 1.0, Val(true), dx)
    @test value == x[1] / d
    @test ds == -value / d

    # The final shared cotangent is representable, even when an individual
    # row or the partial sum of the first two rows exceeds Float64.
    for (x, d) in (([1e100, -1e100], 1e-150), ([1.0, 1.0, -1.0], 1e-154))
        expected = Float64(-sum(BigFloat.(x ./ d)) / BigFloat(d))
        for want in (false, true)
            dx = zeros(length(x))
            value, ds = linear(x, d, 1.0, Val(want), dx)
            want && @test value ≈ sum(x ./ d)
            @test all(≈(1 / d), dx)
            @test ds ≈ expected
        end
    end

    # The original cell order cancels to a finite result even though
    # multiplying the seed into the first two numerators would overflow.
    d, seed = 1e300, floatmax(Float64)
    for want in (false, true)
        x = [2d, 2d, -2d]
        dx = zeros(3)
        value, ds = linear(x, d, seed, Val(want), dx)
        want && @test value ≈ 2.0
        @test all(≈(seed / d), dx)
        @test ds ≈ -2 * (seed / d)
    end
end

@testset "distinct incoming terms exceed the exponent range before division" begin
    divide, add, multiply, square = (GlobalRef(Base, f) for f in (:/, :+, :*, :abs2))
    ast = program([quote
        z = $divide(v1, s)
        $add($multiply(a, $square(z)), $multiply(b, z))
    end])
    append!(ast.args[1].args, (:a, :b))
    sweep = RK.compile(first(RK._pr_program_reverse(ast, [:x, :s], RK, Dict{Symbol,Any}())))
    x, d, seed = [1e300], 1e250, 1e300
    a, b = 1.0, 2.0
    q = BigFloat(only(x)) / BigFloat(d)
    dy = BigFloat(seed) * (2a * q + b)
    for want in (false, true)
        dx = zeros(1)
        value, dd = sweep(x, d, a, b, seed, Val(want), dx)
        want && @test value ≈ Float64(a * q^2 + b * q)
        @test only(dx) ≈ Float64(dy / d)
        @test dd ≈ Float64(-dy * q / d)
    end
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
# least `_PR_ONE_PASS_MIN_CELLS` cells and a shorter parent, the plate's own
# program otherwise (an identity gather on a small plate, as RKPPL's
# `b[axes(n, 1)]`, or a parent as long as the plate), which the backend
# differentiates. Gradients through Enzyme agree either way.
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
    target = RK._pr_rule_target(rule, (x, 0.6))
    @test onepass ? target isa RK._PlateSumRuleOnePass : target === primal
    @test RK._pr_rule_target(rule, (collect(x), 0.6)) === primal
    @test RK._pr_rule_target(rule, (view(zeros(n + 3), 2:n+1), 0.6)) === primal
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
            # The program a preparation builds: rules are installed for one.
            ast = RK._program_expression(first(RK._pr_collecting_marks(RK._PlateRuleMark[]) do
                RK._ad_na_program(k, (1,), (z0, groups0, 0.6))
            end))
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
            ast = RK._program_expression(first(RK._pr_collecting_marks(RK._PlateRuleMark[]) do
                RK._ad_na_program(k, (1,), (z0, groups, 0.6))
            end))
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
        # Witness of the schedule: one pass keeps a buffer as long as `z` (and
        # the scalars' cotangents) in the preparation's tape slot, which a
        # call in two passes leaves empty. More groups than observations: two
        # passes. Either way a warmed gradient no longer allocates the buffer.
        if nonallocating && n >= 128
            g = similar(u)
            sampler_value_and_gradient!(on, g, u)
            sampler_value_and_gradient!(off, g, u)
            slots = [c for caches in ReactiveKernels._ad_owned_storage(on.ad)
                     if caches isa Tuple for c in caches
                     if c isa ReactiveKernels._PlateTapeSlot]
            @test any(slot -> length(slot.tape) >= J, slots) == (J < n)
            @test all(slot -> isempty(slot.tape) || length(slot.tape) >= J, slots)
            @test (@allocated sampler_value_and_gradient!(on, g, u)) < 8J
            @test (@allocated sampler_value_and_gradient!(off, g, u)) < 8J
        end
    end
end

# A lane partition of a gathered operand is a view of that view. Its buffer,
# like the whole plate's, has one entry per entry of the gathered storage.
@testset "a partition's composed gather addresses the gathered storage" begin
    z = [0.3cos(i) for i in 1:7]
    group = [mod1(3i, 7) for i in 1:40]
    lanes = [i for i in 1:40 if isodd(i)]
    inner = view(z, group)
    x = RK._lane_view(inner, lanes)
    @test RK._pr_gather_value(typeof(x))
    @test RK._pr_gather_root(x) === z
    @test RK._pr_gather_root(inner) === z
    # A parameter block of a flat vector is storage, not a gather. (Base's
    # `view` of a view would reindex into the flat vector instead.)
    block = view(collect(1.0:20.0), 4:10)
    @test RK._pr_gather_root(Base.SubArray(block, ([1, 3, 3],))) === block
    @test RK._pr_gather_root(block) === block
    slice = zeros(length(z))
    buffer = RK._pr_regather(slice, x)
    @test parentindices(buffer) === parentindices(x)
    @test RK._pr_gather_root(buffer) === slice
    buffer[3] += 2.0
    @test slice[group[lanes[3]]] == 2.0
    @test count(!iszero, slice) == 1

    # The grouped schedule takes such a call, and its gradient is the two-pass one.
    ast = program([:(s * v1)])
    rexpr, _, _ = RK._pr_program_reverse(ast, [:x, :s], RK, Dict{Symbol,Any}())
    primal, sweep = RK.compile(ast), RK.compile(rexpr)
    rule = RK._PlateSumRule{(2,),(1,),typeof(primal),typeof(sweep)}(primal, sweep)
    long = [mod1(3i, 7) for i in 1:600]
    half = collect(1:2:600)
    @test RK._pr_rule_target(rule, (RK._lane_view(view(z, long), half), 0.6)) isa
        RK._PlateSumRuleOnePass{(2,),(1,),typeof(rule),true}
    # Few cells of long storage stay inline.
    wide = view(collect(1.0:5000.0), [7, 9, 7])
    @test RK._pr_rule_target(rule, (wide, 0.6)) === primal
    for seed in (1.0, -0.4)
        lane = RK._lane_view(view(z, long), half)
        owned = RK._pr_regather(zeros(length(z)), lane)
        value, ds = sweep(lane, 0.6, seed, Val(true), owned)
        given = zeros(length(z))
        _, ds2 = sweep(lane, 0.6, seed, Val(false), RK._pr_regather(given, lane))
        @test value ≈ 0.6sum(lane)
        @test ds ≈ ds2 ≈ seed * sum(lane)
        @test RK._pr_gather_root(owned) ≈ given
    end
end

# The schedule is chosen on operand types before the backend sees the call:
# an operand class the adapter has no buffer or cotangent for stays inline.
@testset "operand classes without a one-pass buffer keep the plate inline" begin
    ast = program([:(s * v1)])
    rexpr, _, _ = RK._pr_program_reverse(ast, [:x, :s], RK, Dict{Symbol,Any}())
    primal, sweep = RK.compile(ast), RK.compile(rexpr)
    joint = RK._PlateSumRule{(2,),(1,),typeof(primal),typeof(sweep),true}(primal, sweep)
    slot = RK._PlateTapeSlot()
    @test RK._pr_rule_target(joint, ([0.2, -0.3], 0.6, slot)) isa RK._PlateSumRuleOnePass
    @test RK._pr_rule_target(joint, ((0.2, -0.3), 0.6, slot)) isa RK._PlateSumRuleOnePass
    @test RK._pr_rule_target(joint, (Float32[0.2, -0.3], 0.6, slot)) === primal
    @test RK._pr_rule_target(joint, ([0.2 -0.3; 0.1 0.4], 0.6, slot)) === primal
    @test RK._pr_rule_target(joint, ([0.2, -0.3], 0.6f0, slot)) === primal
    @test RK._pr_rule_target(joint, ([0.2, -0.3], big"0.6", slot)) === primal
    @test !RK._pr_rule_applies(joint, ([0.2 -0.3; 0.1 0.4], 0.6, slot))
    @test RK._pr_rule_applies(joint, ([0.2, -0.3], 0.6, slot))
    # An owned scalar rule's joint cut does not pay for a buffer per row.
    owned = RK._PlateSumRule{(2,),(1,),typeof(primal),typeof(sweep),:owned}(primal, sweep)
    @test RK._pr_rule_target(owned, ([0.2, -0.3], 0.6, slot)) === primal
    grouped = view([0.1, 0.2, 0.3], [1, 3, 3, 2, 1])
    @test RK._pr_rule_target(owned, (grouped, 0.6, slot)) isa
        RK._PlateSumRuleOnePass{(2,),(1,),typeof(owned),true}
    @test !slot.declined
end

# Several gathered operands are measured together, so a cell that divides by
# a shared scale keeps its ordinary arithmetic.
ranges_fit(cells, reads, shadows, targets) = RK._pr_within_range(
    RK._pr_storage_exponents(RK._PR_NO_EXPONENTS, cells, reads, shadows), targets)

@testset "exponent ranges over several gathers" begin
    n = 200
    z, w = [0.3cos(i) for i in 1:9], [0.2sin(i) + 1 for i in 1:5]
    g, h = [mod1(2i, 9) for i in 1:n], [mod1(3i, 5) for i in 1:n]
    x = [0.1i for i in 1:n]
    cells = Base.OneTo(n)
    sz, sw = zeros(9), zeros(5)
    reads = (view(z, g), view(w, h), x)
    shadows = (view(sz, g), view(sw, h))
    measured(r = reads, s = shadows, c = cells) =
        RK._pr_storage_exponents(RK._PR_NO_EXPONENTS, c, r, s)
    # A Gaussian residual over a shared scale: three factors, one divisor.
    targets = ((3, 1, 110),)
    @test RK._pr_plain_storage(cells, reads, shadows)
    @test RK._pr_within_range(measured(), targets)
    lanes = collect(1:2:n)
    half = Base.OneTo(length(lanes))
    lane_reads = (RK._lane_view(view(z, g), lanes), RK._lane_view(view(w, h), lanes), x[lanes])
    lane_shadows = (RK._pr_regather(sz, lane_reads[1]), RK._pr_regather(sw, lane_reads[2]))
    @test RK._pr_within_range(measured(lane_reads, lane_shadows, half), targets)
    # A shadow that shares storage with an operand.
    @test !RK._pr_plain_storage(cells, reads, (view(z, g), view(sw, h)))
    @test measured(reads, (view(sz, g), view(w, h))) == RK._PR_BAD_EXPONENTS
    @test !RK._pr_within_range(measured(reads, (view(sz, g), view(w, h))), targets)
    # The measured exponents are charged against the float range: a unit of
    # 1e13 or 1e-13 fits, a value near the range's end, a subnormal one and
    # one that is not finite do not. A zero is no magnitude at all.
    for value in (1e13, 1e-13, 3e-17, 0.0)
        z[4] = value
        @test RK._pr_within_range(measured(), targets)
    end
    for value in (1e300, 1e-300, 5e-324, Inf, NaN)
        z[4] = value
        @test !RK._pr_within_range(measured(), targets)
    end
    z[4] = 0.25
    low, high = measured()
    @test low == RK._pr_low_exponent(minimum(abs, filter(!iszero, vcat(z, w, x))))
    @test high == RK._pr_high_exponent(maximum(abs, vcat(z, w, x)))
    @test RK._pr_exponents(0.0) == RK._PR_NO_EXPONENTS
    @test RK._pr_scalar_exponents((2.0, 0, true, 0.5)) == (RK._pr_low_exponent(0.5), RK._pr_high_exponent(2.0))
    @test RK._pr_scalar_exponents((2.0, big"1.0")) == RK._PR_BAD_EXPONENTS
    @test ranges_fit(cells, reads, shadows, targets)
    @test (@allocated ranges_fit(cells, reads, shadows, targets)) == 0
end

# In the normal range the plain quotients have the bits of the separate-exponent
# forms. Outside it they are those forms.
@testset "plain quotient partials in the normal range" begin
    rng = [(sin(37.0i) * 2.0^(mod(11i, 61) - 30), cos(13.0i) * 2.0^(mod(7i, 53) - 26),
            (sin(5.0i) + 1.5) * 2.0^(mod(3i, 41) - 20)) for i in 1:2000]
    for (c, n, d) in rng
        @test RK._pr_quotient((c, n), d) === RK._pr_wide_quotient((c, n), d)
        @test RK._pr_quotient((-1.0, c, n), d) === RK._pr_wide_quotient((-1.0, c, n), d)
        terms = ((c, n), (0.5, n))
        wide = RK._pr_parts_value(RK._pr_accumulate_quotient((0.0, 0), terms, c, d))
        @test RK._pr_sum_quotient(terms, c, d) === wide
        state = RK._pr_add_quotient((0.25, 0), terms, c, d)
        @test last(state) == 0
        @test RK._pr_parts_value(state) ≈
            RK._pr_parts_value(RK._pr_accumulate_quotient((0.25, 0), terms, c, d)) rtol = 4eps()
    end
    for (c, n, d) in ((1e200, 1e200, 1e150), (1e-200, 1e-200, 1e-300), (0.0, 5.0, 2.0),
                      (-0.0, 5.0, 2.0), (Inf, 1.0, 2.0), (NaN, 1.0, 2.0), (1.0, 2.0, 0.0),
                      (0.0, 1.0, 0.0), (1e300, 1e300, 1e-300))
        @test isequal(RK._pr_quotient((c, n), d), RK._pr_wide_quotient((c, n), d))
    end
    # A row outside the plain range moves the running sum to the wide form,
    # and later rows stay there.
    state = RK._pr_add_quotient((0.0, 0), ((1e200, 1e200),), 1.0, 1e150)
    @test last(state) != 0
    state = RK._pr_add_quotient(state, ((-1e200, 1e200),), 1.0, 1e150)
    @test RK._pr_parts_value(state) == 0.0
    @test RK._pr_quotient((1, 2.0), 4) === 0.5
    @test RK._pr_quotient((big"1.0", 2.0), 4.0) == big"0.5"
end

# A predicate over active values selects a side of `ifelse` and carries no
# derivative of its own.
@testset "ifelse on a predicate over active values" begin
    side = :($(GlobalRef(Base, :ifelse))(s * v1 > 0, 2 * (s * v1), -(s * v1)))
    ast = program([side])
    rexpr, scalar, batched = RK._pr_program_reverse(ast, [:x, :s], RK, Dict{Symbol,Any}())
    @test scalar == [:s] && batched == [:x]
    sweep = RK.compile(rexpr)
    x, s = [0.4, -1.2, 0.8, -0.1], 0.7
    slope = [v * s > 0 ? 2.0 : -1.0 for v in x]
    for seed in (1.0, -0.6), want in (false, true)
        dx = zeros(length(x))
        value, ds = sweep(x, s, seed, Val(want), dx)
        want && @test value ≈ sum(slope .* s .* x)
        @test dx ≈ seed .* slope .* s
        @test ds ≈ seed * sum(slope .* x)
    end
end

# A partitioned count likelihood and two crossed groupings: each grouped plate
# takes its one pass through the public non-allocating sampler, and no rule
# call is refused by the backend.
module ScheduleModels
using ReactiveKernelsPPL
const counts = @rkppl begin
    a ~ Normal(0, 5)
    b ~ Normal(0, 1)
    sd ~ Exponential(1)
    z[1:J] .~ Normal.(0, 1)
    eta = a .+ b .* x .+ sd .* z[group]
    y .~ Poisson.(exp.(eta))
end
const crossed = @rkppl begin
    a ~ Normal(0, 5)
    sd ~ Exponential(1)
    sd2 ~ Exponential(1)
    sigma ~ Exponential(1)
    z[1:J] .~ Normal.(0, 1)
    w[1:J2] .~ Normal.(0, 1)
    mu = a .+ x .+ sd .* z[group] .+ sd2 .* w[group2]
    y .~ Normal.(mu, sigma)
end
end

# The warm allocation of one gradient, measured behind a function barrier. At
# the testset's untyped call site the call is dispatched dynamically, and
# Julia 1.10 counts its boxed result (32 bytes, with the generated reverse on
# or off).
warm_gradient_bytes(q, g, u) = (sampler_value_and_gradient!(q, g, u);
    @allocated sampler_value_and_gradient!(q, g, u))

@testset "grouped one pass on a partitioned count likelihood and on crossed groups" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    n, J, J2 = 600, 20, 7
    group = [mod1(7i, J) for i in 1:n]
    group2 = [mod1(5i + i ÷ 9, J2) for i in 1:n]
    x = [0.3 * cos(i) for i in 1:n]
    counts = [i % 3 == 0 ? 0 : 1 + i % 4 for i in 1:n]
    responses = [0.5 * sin(i) + 0.1 * group[i] / J for i in 1:n]
    cases = ((ScheduleModels.counts(; group, J, x) | (; y = counts), 2),
             (ScheduleModels.crossed(; group, group2, J, J2, x) | (; y = responses), 1))
    for (plan, grouped) in cases
        built = build_kernel(plan)
        u = [0.1 * sin(i) for i in 1:built.layout.total]
        off = with_plate_reverse(false) do
            prepare_sampler(built, plan, u; backend, nonallocating = true)
        end
        on = with_plate_reverse(true) do
            prepare_sampler(built, plan, u; backend, nonallocating = true)
        end
        for point in (u, u .+ 0.3, u .- 0.2)
            g0, g1 = similar(point), similar(point)
            v0, _ = sampler_value_and_gradient!(off, g0, point)
            v1, _ = sampler_value_and_gradient!(on, g1, point)
            @test v1 ≈ v0 rtol = 1e-12
            @test g1 ≈ g0 rtol = 1e-10 atol = 1e-12
        end
        g = similar(u)
        sampler_value_and_gradient!(on, g, u)
        slots = [c for caches in ReactiveKernels._ad_owned_storage(on.ad)
                 if caches isa Tuple for c in caches
                 if c isa ReactiveKernels._PlateTapeSlot]
        # A likelihood's tape holds its groups and a few shared scalars, never its rows.
        @test count(slot -> J <= length(slot.tape) < n ÷ 4, slots) == grouped
        @test !any(slot -> slot.declined, slots)
        @test warm_gradient_bytes(on, g, u) == 0
        @test warm_gradient_bytes(off, g, u) == 0
    end
end

# A rule the backend refuses is found at preparation, and the preparation is
# made again with the plates inline.
@testset "a refused rule call makes the preparation ordinary" begin
    DI = RK.DifferentiationInterface
    slot = RK._PlateTapeSlot()
    contexts = (DI.Constant(1.0), DI.Cache((nothing, slot)))
    @test RK._pr_tape_slots!(RK._PlateTapeSlot[], contexts) == [slot]
    verified(evaluated, c = contexts) = RK._pr_verified_preparation(
        () -> :ordinary, Val(:gradient), :prepared, nothing, (), c, evaluated)
    @test verified(true) === :prepared
    slot.declined = true
    @test verified(true) === :ordinary
    @test verified(false, (DI.Constant(1.0),)) === :prepared
    # A program without AD caches marks each rule call instead, and a refusal
    # is recorded for the preparation while it evaluates.
    refused = Ref(false)
    marked(evaluated) = RK._pr_verified_preparation(
        () -> :ordinary, Val(:gradient), :prepared, nothing, (), (DI.Constant(1.0),),
        evaluated, [RK._PlateRuleMark()], refused)
    @test marked(true) === :prepared
    refused[] = true
    @test marked(true) === :ordinary
    # Without a marked call there is nothing to check.
    @test RK._pr_verified_preparation(() -> :ordinary, Val(:gradient), :prepared, nothing, (),
        (DI.Constant(1.0),), true, RK._PlateRuleMark[], refused) === :prepared
end

# A mark is the site record of a rule call in a program without AD caches. It
# selects no schedule that needs an owned tape and holds no state: every
# preparation of a program runs the same compiled code. When the backend
# annotates the call in a way the rule does not cover, the adapter records
# that for the preparation that is evaluating, each time.
gathered_call(rule, z, index, s, site) = rule(view(z, index), s, site)
@testset "a rule call without a tape slot records the backend's refusal" begin
    ast = program([:(s * v1)])
    primal = RK.compile(ast)
    sweep(active) = RK.compile(first(RK._pr_program_reverse(ast, active, RK, Dict{Symbol,Any}())))
    both, only_x = sweep([:x, :s]), sweep([:x])
    covered = RK._PlateSumRule{(2,),(1,),typeof(primal),typeof(both)}(primal, both)
    partial = RK._PlateSumRule{(),(1,),typeof(primal),typeof(only_x)}(primal, only_x)
    joint = RK._PlateSumRule{(2,),(1,),typeof(primal),typeof(both),true}(primal, both)
    z = [0.3cos(i) for i in 1:7]
    index = [mod1(3i, 7) for i in 1:600]
    mark = RK._PlateRuleMark()
    @test RK._pr_rule_target(covered, (view(z, index), 0.6, mark)) isa
        RK._PlateSumRuleOnePass{(2,),(1,),typeof(covered),true}
    # Shared cell work pays for a tape only in owned storage.
    @test RK._pr_rule_target(joint, ([0.2, -0.3], 0.6, mark)) === primal
    @test RK._pr_rule_target(joint, ([0.2, -0.3], 0.6, RK._PlateTapeSlot())) isa
        RK._PlateSumRuleOnePass
    expected_z = [0.6count(==(j), index) for j in 1:7]
    differentiate(rule, dz) = Enzyme.autodiff(Enzyme.ReverseWithPrimal, gathered_call,
        Enzyme.Active, Enzyme.Const(rule), Enzyme.Duplicated(z, dz), Enzyme.Const(index),
        Enzyme.Active(0.6), Enzyme.Const(RK._PlateRuleMark()))
    # Two preparations in turn: the second runs the code the first compiled,
    # and has its own record.
    for (rule, refused) in ((covered, false), (partial, true)), preparation in 1:2
        record = Ref(false)
        dz = zero(z)
        derivatives, value = RK._pr_recording_refusals(() -> differentiate(rule, dz), record)
        @test value ≈ 0.6sum(z[index])
        @test dz ≈ expected_z
        @test derivatives[4] ≈ sum(z[index])
        @test record[] == refused
    end
    # A preparation inside another's evaluation records for itself.
    outer, inner = Ref(false), Ref(false)
    RK._pr_recording_refusals(outer) do
        RK._pr_recording_refusals(() -> differentiate(partial, zero(z)), inner)
        @test inner[] && !outer[]
        differentiate(covered, zero(z))
        @test !outer[]
        differentiate(partial, zero(z))
    end
    @test outer[]
    # Outside a preparation the general path still gives the gradient.
    dz = zero(z)
    derivatives, value = differentiate(partial, dz)
    @test value ≈ 0.6sum(z[index]) && dz ≈ expected_z && derivatives[4] ≈ sum(z[index])
end
end
