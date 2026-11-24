module PlateReverseDirectReadsTests
using ReactiveKernels, MutatingFunctions, DifferentiationInterface, Enzyme, Test
const RK = ReactiveKernels

@kernel late_direct(x::Vector{Float64}, y::Vector{Float64}, s::Float64) = begin
    first_cells = plate(x, y) do xi, yi
        sin(s * xi + yi)
    end
    middle::Float64 = sum(first_cells)
    second_cells = plate(x) do xi
        sin(xi + middle)
    end
    total::Float64 = middle + sum(second_cells)
    return total
end

# Calls of a generated rule. The AD body also names the rule in the test that
# decides between its call and the plate's own inline program.
rule_count(x::Expr) = (Meta.isexpr(x, :call) && first(x.args) isa RK._PlateSumRule) +
    sum(rule_count, x.args; init = 0)
rule_count(x) = 0
structure(x) = (1, 0)
structure(x::RK._PlateProgram) = structure(x.ast)
structure(x::RK._PlateCall) = structure(RK._program_expression(x.program))
structure(x::RK._PlateSumRule) = structure(x.program) .+ structure(x.reverse)
function structure(x::Expr)
    children = map(structure, x.args)
    (1 + sum(first, children; init = 0),
     (x.head in (:for, :while)) + sum(last, children; init = 0))
end
# The AD body as a preparation builds it: generated rules are installed only
# for a preparation, which collects their site records.
preparing(f) = RK._pr_collecting_marks(f, RK._PlateRuleMark[])
ad_ast(k::PreparedKernel, args) = preparing(() -> RK._ad_inline_plate_programs(k, (1, 3)).ast)
ad_ast(k::NonAllocatingKernel, args) =
    preparing(() -> RK._program_expression(first(RK._ad_na_program(k, (1, 3), args))))

@testset "plate reverse preserves native direct reads at either preparation stage" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    x0, y0, s0 = [0.1 * sin(i) for i in 1:7], [0.2 * cos(i) for i in 1:7], 0.7
    shapes = Dict()
    for prepare_enabled in (false, true), nonallocating in (false, true)
        kernel = with_plate_reverse(prepare_enabled) do
            k = prepare(late_direct)
            nonallocating ? prepare_nonallocating(k, x0, y0, s0) : k
        end
        report = plate_reverse_report(kernel, (1, 3))
        all(r -> last(r) == "rule", report) || @info "Direct-read eligibility" nonallocating report
        @test length(report) == 2
        @test all(r -> last(r) == "rule", report)
        @test occursin("_plate_read", string(kernel.ast))
        @test occursin("_plate_spans_axis", string(kernel.ast))
        gradients = map((false, true)) do enabled
            with_plate_reverse(enabled) do
                ast = ad_ast(kernel, (x0, y0, s0))
                @test rule_count(ast) == (enabled ? 2 : 0)
                if enabled
                    shapes[(prepare_enabled, nonallocating)] = structure(ast)
                end
                gradient = prepare_ad(kernel, backend, x0, y0, s0; active = (:x, :s))
                point, contexts = RK._ad_prepared_arguments(
                    gradient, (x0, y0, s0), NamedTuple())
                findings = RK._code_findings(gradient.call,
                    RK._ad_call_argument_types(point, contexts);
                    recurse = RK._code_recursion(Val(:all)))
                @test isempty(findings)
                gradient
            end
        end
        # One preparation serves both the direct and extruded branches and
        # changed lengths. x spans every cell; y may broadcast from one cell.
        for n in (7, 73, 0, 11, 7), singleton in (false, true), s in (0.7, -0.4)
            x = [0.1 * cos(i) for i in 1:n]
            y = [0.2 * sin(i) for i in 1:(singleton ? 1 : n)]
            before = deepcopy((x, y))
            theta = s .* x .+ y
            middle = sum(sin, theta; init = 0.0)
            terms = sin.(x .+ middle)
            sumterms = sum(terms; init = 0.0)
            dterms = cos.(x .+ middle)
            shared = 1 + sum(dterms; init = 0.0)
            dx = shared .* s .* cos.(theta) .+ dterms
            ds = shared * sum(x .* cos.(theta); init = 0.0)
            for gradient in gradients
                value, (gx, gs) = ad_value_and_gradient(gradient, x, y, s)
                @test value ≈ middle + sumterms rtol = 1e-12 atol = 1e-12
                @test gx ≈ dx rtol = 1e-10 atol = 1e-12
                @test gs ≈ ds rtol = 1e-10 atol = 1e-12
            end
            @test (x, y) == before
        end
        longer = with_plate_reverse(prepare_enabled) do
            k = prepare(late_direct)
            nonallocating ? prepare_nonallocating(k, zeros(73), zeros(73), s0) : k
        end
        with_plate_reverse(true) do
            @test structure(ad_ast(longer, (zeros(73), zeros(73), s0))) ==
                  shapes[(prepare_enabled, nonallocating)]
        end
    end
end

# The compiler must reject a shadow write at the cell index when the primal
# read another index. These cases enforce derivative correctness, not a new
# restriction on authored kernels: unsupported programs use ordinary AD.
function direct_program(; at = :index, outside = false, rebind = false,
                          raw = :raw, callable = GlobalRef(RK, :_plate_read),
                          density = GlobalRef(Base, :sin), domain = :(Base.eachindex(x)))
    body = Expr(:block, :(index = Base.CartesianIndex(linear)))
    rebind && push!(body.args, :(index = Base.CartesianIndex(1)))
    push!(body.args, :(v = $(callable)($raw, $at)),
        :(term = $(density)(v)),
        :(total = $(GlobalRef(RK, :_plate_total_add))(total, term)))
    simd = macroexpand(RK, Expr(:macrocall, GlobalRef(Base, Symbol("@simd")),
        LineNumberNode(0), Expr(:for, :(linear = $domain), body)))
    prefix = Any[:(raw = Base.broadcastable(x)),
        :(prepared = Base.Broadcast.preprocess(nothing, raw)),
        :(total = $(GlobalRef(RK, :_plate_total_seed))(Float64, 0.0))]
    outside && push!(prefix, :(outside = $(GlobalRef(RK, :_plate_read))($raw, 1)))
    Expr(:function, Expr(:tuple, :x), Expr(:block, prefix..., simd, :total))
end
failure(ast) = try
    RK._pr_program_reverse(ast, [:x], RK, Dict{Symbol,Any}())
    nothing
catch err
    err
end

module ForeignPlateRead
_plate_read(x, i) = 2 * x[i]
end

@testset "direct read provenance cannot hide other accesses" begin
    for raw in (:raw, :x), at in (:index, :linear)
        ast = direct_program(; raw, at)
        rexpr, scalar, batched = RK._pr_program_reverse(ast, [:x], RK, Dict{Symbol,Any}())
        @test isempty(scalar)
        @test batched == [:x]
        primal, reverse = RK.compile(ast), RK.compile(rexpr)
        for x in (Float64[], [0.3, -0.2], collect(range(-0.5, 0.5; length = 73)))
            shadow = zeros(length(x))
            @test primal(x) ≈ sum(sin, x; init = 0.0)
            # The sweep returns the value first; `Val(false)`: not computed.
            @test Base.tail(reverse(x, 1.7, Val(false), shadow)) == ()
            @test shadow ≈ 1.7 .* cos.(x)
            # `Val(true)` also returns the value, into a zeroed owned shadow.
            owned = zeros(length(x))
            value, rest... = reverse(x, 1.7, Val(true), owned)
            @test rest == ()
            @test value ≈ sum(sin, x; init = 0.0)
            @test owned ≈ 1.7 .* cos.(x)
        end
        for bad in (direct_program(; raw, at = 1),
                    direct_program(; raw, outside = true),
                    direct_program(; raw, rebind = true))
            err = failure(bad)
            @test err isa RK._PRIneligible
            @test err isa RK._PRIneligible && occursin("cell index", err.reason)
        end
    end
    # Matching the callee's spelling is not proof that it is RK's owned read.
    err = failure(direct_program(; callable = GlobalRef(ForeignPlateRead, :_plate_read)))
    @test err isa RK._PRIneligible
end

@testset "a prepared active read scatters at its own cell" begin
    read = GlobalRef(Base.Broadcast, :_broadcast_getindex)
    x = [0.3, 0.7, 0.2]
    for domain in (:(Base.eachindex(x)), :(Base.CartesianIndices((Base.axes(x, 1), Base.OneTo(1)))))
        ast = direct_program(; raw = :prepared, callable = read, domain)
        expr = first(RK._pr_program_reverse(ast, [:x], RK, Dict{Symbol,Any}()))
        dx = zeros(length(x))
        RK.compile(expr)(x, 1.0, Val(false), dx)
        @test dx ≈ cos.(x)
    end
    fixed = direct_program(; raw = :prepared, at = 1, callable = read)
    # The scalar oracle adds every contribution to x[1]. A cell-index
    # scatter would write to each element once.
    @test RK.compile(fixed)(x) ≈ length(x) * sin(first(x))
    err = failure(fixed)
    @test err isa RK._PRIneligible && occursin("cell index", err.reason)
end

@testset "dense reverse checks strided, disjoint storage" begin
    original = [0.3 * sin(i) for i in 1:258]
    x = view(original, 1:256)
    dx = zeros(length(x))
    @test RK._pr_dense_reverse_cells(eachindex(x), (x,), (dx, nothing))
    @test !RK._pr_dense_reverse_cells(ones(Int, length(x)), (x,), (dx,))
    @test !RK._pr_dense_reverse_cells(eachindex(x), (x,), (dx, view(dx, :)))
    @test !RK._pr_dense_reverse_cells(eachindex(x), (view(x, ones(Int, length(x))),), (dx,))
    @test !RK._pr_dense_reverse_cells(eachindex(x), (x,), (view(dx, ones(Int, length(x))),))
    shadow = view(original, 2:257)
    @test !RK._pr_dense_reverse_cells(eachindex(x), (x,), (shadow,))
    expected = copy(original)
    for i in eachindex(x)
        expected[i+1] += 0.7 * cos(expected[i])
    end
    ast = direct_program()
    expr = first(RK._pr_program_reverse(ast, [:x], RK, Dict{Symbol,Any}()))
    @test occursin("_pr_dense_reverse_cells", string(expr))
    RK.compile(expr)(x, 0.7, Val(false), shadow)
    @test original ≈ expected

    # Separate Julia array headers may share a shifted region of the same
    # caller-owned memory. Each shadow write then changes the next cell's read.
    backing = collect(range(-0.3, 0.3; length = 257))
    expected = copy(backing)
    for i in 1:256
        expected[i+1] += 0.7 * cos(expected[i])
    end
    GC.@preserve backing begin
        input = unsafe_wrap(Array, pointer(backing), 256; own = false)
        output = unsafe_wrap(Array, pointer(backing, 2), 256; own = false)
        ids = collect(eachindex(input))
        @test !RK._pr_dense_reverse_cells(eachindex(ids), (input,), (output,))
        gather = view(input, ids)
        @test !RK._pr_span_possible(eachindex(ids), gather, view(output, ids), (gather,))
        RK.compile(expr)(input, 0.7, Val(false), output)
        @test backing ≈ expected
    end
end

# Record the guard chosen by the compiled sweep, including when its shadow
# overlaps a separate inactive input. A captured callable can hide the same
# array, so checking only the explicit arguments would miss such a read.
const dense_taken = Ref(false)
function watch_dense(args...)
    eligible = RK._pr_dense_reverse_cells(args...)
    dense_taken[] |= eligible
    eligible
end
function watch_dense_sweep(expr)
    RK._index_run_replace(expr) do node
        Meta.isexpr(node, :call) && node.args[1] === GlobalRef(RK, :_pr_dense_reverse_cells) ||
            return node
        Expr(:call, GlobalRef(@__MODULE__, :watch_dense), node.args[2:end]...)
    end
end
@testset "an inactive indexed or captured read keeps the sweep ordered" begin
    x = ones(256)
    for capture in (false, true)
        storage = [0.3sin(i) for i in 1:length(x)+1]
        weights, gradient = view(storage, 1:length(x)), view(storage, 2:length(storage))
        context = capture ? (i -> weights[i]) : weights
        ast = direct_program()
        push!(ast.args[1].args, :weights)
        read = capture ? :(weights(linear)) : :(Base.getindex(weights, linear))
        ast = RK._index_run_replace(ast) do node
            Meta.isexpr(node, :(=), 2) && node.args[1] === :term || return node
            :(term = v * $read)
        end
        known = capture ? Dict{Symbol,Any}(:weights => context) : Dict{Symbol,Any}()
        expr = first(RK._pr_program_reverse(ast, [:x], RK, known))
        sweep = RK.compile(watch_dense_sweep(expr))
        dense_taken[] = false
        expected = copy(storage)
        for i in eachindex(x)
            expected[i+1] += 0.7 * expected[i]
        end
        sweep(x, context, 0.7, Val(false), gradient)
        @test !dense_taken[]
        @test storage ≈ expected
    end
end

seeded_plate(program, x, seed, slots...) = seed * program(x, slots...)

struct ShadowIndex
    shadow::Vector{Float64}
end
Base.to_index(i::ShadowIndex) = iszero(first(i.shadow)) ? 1 : 2

struct WrappedDenseVector <: DenseVector{Float64}
    data::Vector{Float64}
end
Base.size(v::WrappedDenseVector) = size(v.data)
Base.getindex(v::WrappedDenseVector, i::Int) = v.data[i]

@testset "custom storage and read indices keep the ordered sweep" begin
    x = ones(16)
    weights = collect(Float64, eachindex(x))
    gradient = zeros(length(x))
    cells = eachindex(x)
    matrix = reshape(copy(x), :, 1)
    @test RK._pr_dense_reverse_cells(cells, (view(matrix, :, 1),), (gradient,))
    for operand in (WrappedDenseVector(gradient), vec(reshape(gradient, :, 1)),
                    reinterpret(Float64, reinterpret(UInt64, gradient)))
        @test !RK._pr_dense_reverse_cells(cells, (operand,), (gradient,))
    end
    ids = collect(cells)
    gathered, shadow = view(x, ids), view(gradient, ids)
    @test !RK._pr_span_possible(cells, gathered, shadow, (gathered, WrappedDenseVector(gradient)))

    ast = direct_program()
    append!(ast.args[1].args, (:weights, :choice))
    read = GlobalRef(RK, :_plate_read)
    ast = RK._index_run_replace(ast) do node
        Meta.isexpr(node, :(=), 2) && node.args[1] === :term || return node
        :(term = v * ($read(weights, index) + $read(weights, choice)))
    end
    expr = first(RK._pr_program_reverse(ast, [:x], RK, Dict{Symbol,Any}()))
    sweep = RK.compile(watch_dense_sweep(expr))
    for (choice, takes_dense) in ((1, true), (ShadowIndex(gradient), false))
        dense_taken[] = false
        fill!(gradient, 0.0)
        sweep(x, weights, choice, 1.0, Val(false), gradient)
        expected = weights .+ weights[1]
        takes_dense || (expected[2:end] .= weights[2:end] .+ weights[2])
        @test dense_taken[] == takes_dense
        @test gradient == expected
    end
end

@testset "a spanning broadcast read keeps an inactive fixed index" begin
    read = GlobalRef(Base.Broadcast, :_broadcast_getindex)
    ast = direct_program(; raw = :prepared, callable = read)
    push!(ast.args[1].args, :weights)
    pushfirst!(ast.args[2].args, :(prepared_weights = Base.Broadcast.preprocess(nothing, weights)))
    ast = RK._index_run_replace(ast) do node
        Meta.isexpr(node, :(=), 2) && node.args[1] === :term || return node
        :(term = v * ($read(prepared_weights, index) + $read(prepared_weights, 1)))
    end
    x = ones(16)
    weights = collect(Float64, eachindex(x))
    gradient = zeros(length(x))
    expr = first(RK._pr_program_reverse(ast, [:x], RK, Dict{Symbol,Any}()))
    RK.compile(expr)(x, weights, 1.0, Val(false), gradient)
    @test gradient == weights .+ weights[1]
end

function lazy_plate(program, x, scale, seed, overwrite)
    # The active scalar and array require the ordinary backend fallback.
    # Both leaves reference x, one through a view; a later write must not
    # change the values seen by the reverse, or duplicate its shadow storage.
    alias = view(x, :)
    cells = RK._cell_array_broadcast(Base.broadcasted(+,
        Base.broadcasted(*, scale, x), alias))
    result = seed * program(cells)
    overwrite && fill!(alias, 2.0)
    result
end

function overwritten_plate(program, x, seed)
    result = seed * program(x)
    fill!(view(x, :), 2.0)
    result
end

const exp_sweeps = Ref(0)
struct CountExpSweep{F}
    reverse::F
end
function (s::CountExpSweep)(args...)
    exp_sweeps[] += 1
    s.reverse(args...)
end
function exp_plate(program, x, overwrite, slots...)
    cells = RK._cell_array_broadcast(Base.broadcasted(exp, x))
    value = program(cells, slots...)
    overwrite && fill!(x, 2.0)
    value
end

@testset "generated reverse writes through a dense unary broadcast" begin
    ast = direct_program()
    rexpr, _, _ = RK._pr_program_reverse(ast, [:x], RK, Dict{Symbol,Any}())
    primal = RK.compile(ast)
    sweep = CountExpSweep(RK.compile(rexpr))
    rule = RK._PlateSumRule{(),(1,),typeof(primal),typeof(sweep)}(primal, sweep)
    slot = RK._PlateTapeSlot()
    for slotted in (false, true), overwrite in (false, true), n in (0, 1, 7)
        x = [cos(i) for i in 1:n]
        y = exp.(x)
        dx = fill(0.3, n)
        exp_sweeps[] = 0
        _, value = Enzyme.autodiff(Enzyme.ReverseWithPrimal, exp_plate, Enzyme.Active,
            Enzyme.Const(rule), Enzyme.Duplicated(x, dx), Enzyme.Const(overwrite),
            (slotted ? (Enzyme.Const(slot),) : ())...)
        @test value ≈ sum(sin, y; init = 0.0)
        @test dx ≈ (overwrite ? 0.0 : 0.3) .+ y .* cos.(y)
        # Only a call with a tape slot takes the one-pass rule. Without one the
        # backend differentiates the lazy broadcast's own program.
        @test exp_sweeps[] == Int(slotted)
        slotted && @test length(slot.tape) == n
    end
end

@testset "ordinary plate fallback retains values across aliased mutation" begin
    ast = direct_program()
    rexpr, _, _ = RK._pr_program_reverse(ast, [:x], RK, Dict{Symbol,Any}())
    primal, reverse = RK.compile(ast), RK.compile(rexpr)
    rule = RK._PlateSumRule{(),(1,),typeof(primal),typeof(reverse)}(primal, reverse)
    cells = RK._cell_array_broadcast(Base.broadcasted(*, 0.7, [0.2, -0.3]))
    @test RK._pr_rule_target(rule, (cells,)) === primal
    # Without a one-pass schedule the call runs the plate's own program.
    @test RK._pr_rule_target(rule, ([0.2, -0.3],)) === primal
    @test RK._pr_rule_target(rule, ((0.2, -0.3),)) === primal
    for n in (0, 1, 17), seed in (1.7, -0.4), overwrite in (false, true)
        original = [0.2cos(i) for i in 1:n]
        scale = 0.7
        expected = seed .* cos.((scale + 1) .* original)
        results = map((primal, rule)) do program
            x, dx = copy(original), fill(0.3, n)
            cot, value = Enzyme.autodiff(Enzyme.ReverseWithPrimal,
                lazy_plate, Enzyme.Active, Enzyme.Const(program),
                Enzyme.Duplicated(x, dx), Enzyme.Active(scale),
                Enzyme.Const(seed), Enzyme.Const(overwrite))
            @test value ≈ seed * sum(sin, (scale + 1) .* original; init=0.0)
            @test dx ≈ (overwrite ? 0.0 : 0.3) .+ (scale + 1) .* expected
            @test cot[3] ≈ sum(original .* expected; init=0.0)
            @test x == (overwrite ? fill(2.0, n) : original)
            (value, dx, cot[3])
        end
        @test all(isapprox.(results[1], results[2]))
    end
    # The supported array path still takes a snapshot when its input changes
    # after the primal. Moving the rule boundary must retain that contract.
    for n in (0, 1, 17), seed in (1.7, -0.4), program in (primal, rule)
        original = [0.2cos(i) for i in 1:n]
        x, dx = copy(original), zeros(n)
        _, value = Enzyme.autodiff(Enzyme.ReverseWithPrimal,
            overwritten_plate, Enzyme.Active, Enzyme.Const(program),
            Enzyme.Duplicated(x, dx), Enzyme.Const(seed))
        @test value ≈ seed * sum(sin, original; init=0.0)
        @test dx ≈ seed .* cos.(original)
        @test x == fill(2.0, n)
    end
end

@testset "tuple operands return cotangents through ordinary reverse" begin
    # Compile one runtime loop. The same source and rule also accept arrays
    # of changing length; tuple lengths are inherent in their Julia types.
    ast = direct_program()
    rexpr, scalar, batched = RK._pr_program_reverse(ast, [:x], RK, Dict{Symbol,Any}())
    @test isempty(scalar)
    @test batched == [:x]
    # @simd may wrap the runtime iteration in additional loops.
    @test last(structure(ast)) > 0
    primal, reverse = RK.compile(ast), RK.compile(rexpr)
    rule = RK._PlateSumRule{(),(1,),typeof(primal),typeof(reverse)}(primal, reverse)
    ext = Base.get_extension(RK, :ReactiveKernelsEnzymeExt)
    for n in (1, 3, 17), seed in (1.0, 1.7, -0.4)
        x = ntuple(i -> 0.2 * cos(i), n)
        expected = seed .* cos.(x)
        @test @inferred(ext._pr_reverse_return(Val(()), (Enzyme.Active(x),), (),
            Val((1,)), (collect(expected),))) == (expected,)
        gradients = map((primal, rule)) do program
            first(Enzyme.autodiff(Enzyme.Reverse, seeded_plate, Enzyme.Active,
                Enzyme.Const(program), Enzyme.Active(x), Enzyme.Const(seed)))[2]
        end
        @test collect(gradients[1]) ≈ collect(expected)
        @test collect(gradients[2]) ≈ collect(gradients[1])
        slot = RK._PlateTapeSlot()
        taped = first(Enzyme.autodiff(Enzyme.Reverse, seeded_plate, Enzyme.Active,
            Enzyme.Const(rule), Enzyme.Active(x), Enzyme.Const(seed), Enzyme.Const(slot)))[2]
        @test collect(taped) ≈ collect(expected)
        @test length(slot.tape) == n
        @test seeded_plate(rule, x, seed) ≈ seed * sum(sin, x; init = 0.0)
    end
    for n in (0, 1, 73, 7)
        x = [0.2 * cos(i) for i in 1:n]
        dx = zeros(n)
        Enzyme.autodiff(Enzyme.Reverse, seeded_plate, Enzyme.Active,
            Enzyme.Const(rule), Enzyme.Duplicated(x, dx), Enzyme.Const(1.7))
        @test dx ≈ 1.7 .* cos.(x)
        @test rule(x) ≈ sum(sin, x; init = 0.0)
    end
end

@testset "branch-specific terminal totals use the output adjoint" begin
    function arm(total, density)
        ast = direct_program(; density)
        body = RK._substitute_symbols(ast.args[2], Dict{Symbol,Any}(:total => total))
        body.args[end] = :(out = $(GlobalRef(RK, :_plate_total_value))($total))
        body
    end
    ast = Expr(:function, Expr(:tuple, :x, :choose), Expr(:block,
        Expr(:if, :choose, arm(:left_total, GlobalRef(Base, :sin)),
             arm(:right_total, GlobalRef(Base, :cos))), :out))
    rexpr, scalar, batched = RK._pr_program_reverse(ast, [:x], RK, Dict{Symbol,Any}())
    @test isempty(scalar)
    @test batched == [:x]
    primal, reverse = RK.compile(ast), RK.compile(rexpr)
    for x in (Float64[], [0.2, -0.3, 0.4]), choose in (false, true)
        shadow = zeros(length(x))
        @test primal(x, choose) ≈ sum(choose ? sin : cos, x; init = 0.0)
        @test Base.tail(reverse(x, choose, 1.7, Val(false), shadow)) == ()
        @test shadow ≈ 1.7 .* (choose ? cos.(x) : -sin.(x))
        owned = zeros(length(x))
        value, rest... = reverse(x, choose, 1.7, Val(true), owned)
        @test rest == ()
        @test value ≈ sum(choose ? sin : cos, x; init = 0.0)
        @test owned ≈ 1.7 .* (choose ? cos.(x) : -sin.(x))
    end
end

@kernel late_factored(x::Vector{Float64}, s::Float64) = begin
    first_cells = plate(x) do xi
        sin(xi)
    end
    middle::Float64 = sum(first_cells)
    second_cells = plate(x) do xi
        exp(xi) * s
    end
    total::Float64 = middle + sum(second_cells)
    return total
end

@testset "a factored plate takes the rule beside an explicit fallback" begin
    kernel = with_plate_reverse(false) do
        prepare(late_factored)
    end
    report = plate_reverse_report(kernel, (1, 2))
    @test length(report) == 2
    @test count(r -> last(r) == "rule", report) == 1
    # The generator reads the loop the factoring pass preserves, so a factored
    # total is no fallback. A cell it cannot differentiate still names why.
    @test count(r -> last(r) != "rule", report) == 1
    x, s = [0.2, -0.3, 0.5], 0.7
    ad = with_plate_reverse(true) do
        prepare_ad(kernel, AutoEnzyme(; mode = Enzyme.Reverse), x, s; active = (:x, :s))
    end
    value, (dx, ds) = ad_value_and_gradient(ad, x, s)
    @test value ≈ sum(sin, x) + s * sum(exp, x)
    @test dx ≈ cos.(x) .+ s .* exp.(x)
    @test ds ≈ sum(exp, x)
end

span_guard_bytes(cells, value, shadow, operands) =
    @allocated RK._pr_span_possible(cells, value, shadow, operands)

@testset "checking gathered parameter and tape slices uses zero warm bytes" begin
    indices = repeat(1:16, 3)
    cells = eachindex(indices)
    value = Base.SubArray(view(ones(20), 5:20), (indices,))
    shadow = Base.SubArray(view(zeros(17), 1:16), (indices,))
    operands = (value, ones(length(cells)), zeros(length(cells)))
    @test RK._pr_span_possible(cells, value, shadow, operands)
    @test span_guard_bytes(cells, value, shadow, operands) == 0
end

@testset "grouped reverse checks the live gather span before addressing its parents" begin
    expr = first(RK._pr_program_reverse(direct_program(), [:x], RK, Dict{Symbol,Any}()))
    @test occursin("_pr_span_end", string(expr))
    sweep = RK.compile(expr)
    values = [sin(i / 5) for i in 1:16]
    indices = repeat(collect(eachindex(values)), 3)
    operand = view(values, indices)
    destination = zeros(length(values))
    shadow = view(destination, indices)
    cells = eachindex(indices)
    @test RK._pr_span_possible(cells, operand, shadow, (operand,))
    @test !RK._pr_span_possible(cells, operand, operand, (operand,))
    @test !RK._pr_span_possible(cells, operand, view(destination, copy(indices)), (operand,))
    @test !RK._pr_span_possible(cells, operand, shadow, (operand, view(destination, indices)))
    @test RK._pr_span_end(indices, 1, last(cells), indices[1], length(values)) == 16
    for (edit, offset) in ((false, .2), (true, -.7), (true, .4))
        values .+= offset
        if edit
            # End the second span in its middle. Both ends still match, so a
            # check of only its endpoints would misaddress the interior cells.
            indices[23] = 11
            @test RK._pr_span_end(indices, 17, last(cells), indices[17], length(values)) < 17
        end
        fill!(destination, 0.0)
        value, = sweep(operand, 1.7, Val(true), shadow)
        expected = zeros(length(values))
        for j in indices
            expected[j] += 1.7 * cos(values[j])
        end
        @test destination ≈ expected
        @test value ≈ sum(sin, operand)
    end
end

const spanned_cells = Ref(0)
function watch_span_end(indices, lo, lastcell, start, lastparent)
    hi = RK._pr_span_end(indices, lo, lastcell, start, lastparent)
    spanned_cells[] += max(0, hi - lo + 1)
    hi
end

@testset "checked quotient gather restores its owned shadow for exceptional rows" begin
    read = GlobalRef(RK, :_plate_read)
    ast = direct_program(; raw = :x)
    append!(ast.args[1].args, (:weights, :d))
    ast = RK._index_run_replace(ast) do node
        Meta.isexpr(node, :(=), 2) && node.args[1] === :term || return node
        :(term = (v + $read(weights, index)) / d)
    end
    expr = first(RK._pr_program_reverse(ast, [:x, :d], RK, Dict{Symbol,Any}()))
    @test occursin("_pr_check_in_gather", string(expr))
    sweep = RK.compile(expr)
    z = zeros(256)
    indices = repeat(collect(eachindex(z)), 4)
    weights = ones(length(indices))
    gradient = zeros(length(z))
    x, shadow = view(z, indices), view(gradient, indices)
    @test RK._pr_check_in_gather(axes(x, 1), x, shadow, (x, weights))
    d = 0x1.77ee5e8d806e2p-1
    for last_weight in (1.0, 0x1.77ee5e8d806e2p+1023, 1.0)
        weights[end] = last_weight
        for want in (true, false)
            existing = want ? 0.0 : 0.25
            fill!(gradient, existing)
            value, ds = sweep(x, weights, d, 1.0, Val(want), shadow)
            quotients = weights ./ d
            want && @test value ≈ sum(quotients)
            @test ds ≈ Float64(-sum(BigFloat.(quotients)) / BigFloat(d))
            @test all(≈(4 / d + existing), gradient)
        end
    end
end

# A dense read of the program's constants table has the same entries at every
# call of one preparation. The reverse compiled for such a call takes the
# call's site record as its last argument: a tape slot keeps the read's
# measured exponents from its first call, and the sweep inspects no read.
@testset "a bound read's exponents are measured once for its tape slot" begin
    read = GlobalRef(RK, :_plate_read)
    ast = direct_program(; raw = :x)
    append!(ast.args[1].args, (:weights, :d))
    ast = RK._index_run_replace(ast) do node
        Meta.isexpr(node, :(=), 2) && node.args[1] === :term || return node
        :(term = (v + $read(weights, index)) / d)
    end
    plain = first(RK._pr_program_reverse(ast, [:x, :d], RK, Dict{Symbol,Any}()))
    bound = first(RK._pr_program_reverse(ast, [:x, :d], RK, Dict{Symbol,Any}(), [:weights]))
    @test last(plain.args[1].args) !== RK._PR_SITE
    @test last(bound.args[1].args) === RK._PR_SITE
    # The reverse with the loop programs it calls.
    function text(expr)
        parts = String[]
        visit(ex) = ex isa RK._PlateProgram ? (push!(parts, string(ex.ast)); visit(ex.ast)) :
                    ex isa Expr ? foreach(visit, ex.args) : nothing
        push!(parts, string(expr))
        visit(expr)
        join(parts, '\n')
    end
    @test occursin("_pr_low_exponent", text(plain))
    @test !occursin("_pr_cached_exponents", text(plain))
    @test !occursin("_pr_low_exponent", text(bound))
    @test occursin("_pr_cached_exponents", text(bound))
    sweep = RK.compile(bound)
    @test RK._pr_sited_reverse(typeof(RK._PlateCall(sweep)))
    @test !RK._pr_sited_reverse(typeof(RK._PlateCall(RK.compile(plain))))

    z = zeros(256)
    indices = repeat(collect(eachindex(z)), 4)
    gradient = zeros(length(z))
    x, shadow = view(z, indices), view(gradient, indices)
    @test RK._pr_check_in_gather(axes(x, 1), x, shadow, (x, ones(length(indices))))
    d = 0x1.77ee5e8d806e2p-1
    # Each set of weights is another binding, with a preparation and slot of
    # its own. The second set takes the exceptional sweep.
    for last_weight in (1.0, 0x1.77ee5e8d806e2p+1023)
        weights = ones(length(indices))
        weights[end] = last_weight
        slot = RK._PlateTapeSlot()
        for site in (slot, slot, nothing, RK._PlateRuleMark()), want in (true, false)
            existing = want ? 0.0 : 0.25
            fill!(gradient, existing)
            value, ds = sweep(x, weights, d, 1.0, Val(want), shadow, site)
            quotients = weights ./ d
            want && @test value ≈ sum(quotients)
            @test ds ≈ Float64(-sum(BigFloat.(quotients)) / BigFloat(d))
            @test all(≈(4 / d + existing), gradient)
        end
        @test slot.exponents == collect(RK._pr_exponents(weights))
        # The slot's record is read, never measured again; without a slot
        # every call measures.
        other = [0x1p-900]
        @test RK._pr_cached_exponents(slot, 1, other) == RK._pr_exponents(weights)
        @test RK._pr_cached_exponents(nothing, 1, other) == RK._pr_exponents(other)
        @test RK._pr_cached_exponents(RK._PlateRuleMark(), 1, other) == RK._pr_exponents(other)
    end
    # A second bound read has the next record.
    slot = RK._PlateTapeSlot()
    @test RK._pr_cached_exponents(slot, 2, [4.0, 0.0, -0.25]) == RK._pr_exponents([0.25, 4.0])
    @test slot.exponents[1:2] == [RK._PR_UNMEASURED, RK._PR_UNMEASURED]
    @test RK._pr_cached_exponents(slot, 1, [8.0]) == RK._pr_exponents(8.0)
    @test RK._pr_cached_exponents(slot, 2, [1.0]) == RK._pr_exponents([0.25, 4.0])
end

@testset "grouped spans preserve inactive direct-read indices" begin
    ast = direct_program()
    push!(ast.args[1].args, :weights)
    read = GlobalRef(RK, :_plate_read)
    ast = RK._index_run_replace(ast) do node
        Meta.isexpr(node, :(=), 2) && node.args[1] === :term || return node
        :(term = v * (2.0 * $read(weights, index) - $read(weights, 1)))
    end
    expr = first(RK._pr_program_reverse(ast, [:x], RK, Dict{Symbol,Any}()))
    expr = RK._index_run_replace(expr) do node
        Meta.isexpr(node, :call) && first(node.args) === GlobalRef(RK, :_pr_span_end) || return node
        Expr(:call, GlobalRef(@__MODULE__, :watch_span_end), node.args[2:end]...)
    end
    q = [sin(j / 7) for j in 1:16]
    idx = repeat(collect(eachindex(q)), 3)
    weights = [cos(i / 11) for i in eachindex(idx)]
    destination = zeros(length(q))
    spanned_cells[] = 0
    value, = RK.compile(expr)(view(q, idx), weights, 1.7, Val(true), view(destination, idx))
    expected_value = 0.0
    expected_gradient = zeros(length(q))
    for (i, j) in pairs(idx)
        coefficient = 2 * weights[i] - weights[1]
        expected_value += q[j] * coefficient
        expected_gradient[j] += 1.7 * coefficient
    end
    @test spanned_cells[] == length(idx)
    @test value ≈ expected_value
    @test destination ≈ expected_gradient
end

end
