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

rule_count(x::RK._PlateSumRule) = 1
rule_count(x::Expr) = sum(rule_count, x.args; init = 0)
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
ad_ast(k::PreparedKernel, args) = RK._ad_inline_plate_programs(k, (1, 3)).ast
ad_ast(k::NonAllocatingKernel, args) =
    RK._program_expression(first(RK._ad_na_program(k, (1, 3), args)))

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
                          density = GlobalRef(Base, :sin))
    body = Expr(:block, :(index = Base.CartesianIndex(linear)))
    rebind && push!(body.args, :(index = Base.CartesianIndex(1)))
    push!(body.args, :(v = $(callable)($raw, $at)),
        :(term = $(density)(v)),
        :(total = $(GlobalRef(RK, :_plate_total_add))(total, term)))
    simd = macroexpand(RK, Expr(:macrocall, GlobalRef(Base, Symbol("@simd")),
        LineNumberNode(0), Expr(:for, :(linear = Base.eachindex(x)), body)))
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

@testset "intermediate reduction adjoints remain an explicit fallback" begin
    kernel = with_plate_reverse(false) do
        prepare(late_factored)
    end
    report = plate_reverse_report(kernel, (1, 2))
    @test length(report) == 2
    @test count(r -> last(r) == "rule", report) == 1
    @test any(r -> occursin("intermediate plate total", last(r)), report)
    x, s = [0.2, -0.3, 0.5], 0.7
    ad = with_plate_reverse(true) do
        prepare_ad(kernel, AutoEnzyme(; mode = Enzyme.Reverse), x, s; active = (:x, :s))
    end
    value, (dx, ds) = ad_value_and_gradient(ad, x, s)
    @test value ≈ sum(sin, x) + s * sum(exp, x)
    @test dx ≈ cos.(x) .+ s .* exp.(x)
    @test ds ≈ sum(exp, x)
end
end
