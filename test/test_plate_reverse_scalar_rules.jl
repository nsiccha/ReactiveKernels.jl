module PlateReverseScalarRulesTests
using ReactiveKernels, MutatingFunctions, DifferentiationInterface, Enzyme, Test
using LogExpFunctions: log1pexp
const RK = ReactiveKernels

@kernel logistic_graph(r::Float64, s::Float64) = begin
    z::Float64 = r / exp(s)
    t::Float64 = tanh(z / 2)
    value::Float64 = -s - log1pexp(z) - log1pexp(-z)
    dr::Float64 = -t / exp(s)
    ds::Float64 = z * t - 1
    return value, dr, ds
end

function captured_weighted(rule)
    @kernel captured(x::Vector{Float64}, w::Vector{Float64}, s::Float64) = begin
        terms = plate(x, w) do xi, wi
            wi * rule(xi, s)
        end
        total::Float64 = sum(terms)
        return total
    end
    captured
end
const logistic = scalar_derivative_rule(logistic_graph;
    primal = :value, partials = (r = :dr, s = :ds))

@kernel weighted(x::Vector{Float64}, w::Vector{Float64}, s::Float64) = begin
    terms = plate(x, w) do xi, wi
        wi * logistic(xi, s)
    end
    total::Float64 = sum(terms)
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
    parts = map(structure, x.args)
    (1 + sum(first, parts; init = 0),
     (x.head in (:for, :while)) + sum(last, parts; init = 0))
end
ad_ast(k::PreparedKernel, args) = RK._ad_inline_plate_programs(k, (1, 2, 3)).ast
ad_ast(k::NonAllocatingKernel, args) =
    RK._program_expression(first(RK._ad_na_program(k, (1, 2, 3), args)))

@testset "owned scalar rules compose inside summed plates" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    args0 = ([0.2, -0.7, 1.1], [0.4, 1.3, 0.8], 0.3)
    for graph in (weighted, captured_weighted(logistic)), nonallocating in (false, true)
        k = prepare(graph)
        nonallocating && (k = prepare_nonallocating(k, args0...))
        @test all(r -> last(r) == "rule", plate_reverse_report(k, (1, 2, 3)))
        shapes = Ref{Any}()
        gradients = map((false, true)) do enabled
            with_plate_reverse(enabled) do
                ast = ad_ast(k, args0)
                @test rule_count(ast) == Int(enabled)
                enabled && (shapes[] = structure(ast))
                op = prepare_ad(k, backend, args0...; active = (:x, :w, :s))
                point, contexts = RK._ad_prepared_arguments(op, args0, NamedTuple())
                @test isempty(RK._code_findings(op.call,
                    RK._ad_call_argument_types(point, contexts);
                    recurse = RK._code_recursion(Val(:all))))
                op
            end
        end
        # Changed lengths, singleton broadcasting, changed shared values and
        # tails reuse the same preparation. No argument or old result may alias
        # scratch storage that a later call overwrites.
        for n in (0, 3, 73), singleton in (false, true), s in (-0.4, 0.8)
            x = [i == 2 ? -1000.0 : i == 3 ? 1000.0 : sin(i) for i in 1:n]
            w = [0.5 + abs(cos(i)) for i in 1:(singleton ? 1 : n)]
            before = deepcopy((x, w))
            z = x ./ exp(s)
            values = -s .- log1pexp.(z) .- log1pexp.(-z)
            expected = (sum(w .* values; init = 0.0),
                (-w .* tanh.(z ./ 2) ./ exp(s),
                 singleton ? [sum(values; init = 0.0)] : values,
                 sum(w .* (z .* tanh.(z ./ 2) .- 1); init = 0.0)))
            for op in gradients
                actual = ad_value_and_gradient(op, x, w, s)
                @test actual[1] ≈ expected[1] atol = 1e-12
                @test all(isapprox(a, b; atol = 1e-11, rtol = 1e-11)
                          for (a, b) in zip(actual[2], expected[2]))
                saved = deepcopy(actual)
                ad_value_and_gradient(op, x, w, s + 0.2)
                @test actual == saved
            end
            @test (x, w) == before
        end
        with_plate_reverse(true) do
            longer = prepare(graph)
            nonallocating && (longer = prepare_nonallocating(longer, zeros(73), ones(73), 0.3))
            @test structure(ad_ast(longer, (zeros(73), ones(73), 0.3))) == shapes[]
            @test last(shapes[]) > 0
        end
    end
end

@kernel positive_graph(x::Float64, s::Float64) = begin
    v::Float64 = log(x) + exp(s)
    dx::Float64 = inv(x)
    ds::Float64 = exp(s)
    return v, dx, ds
end
const positive = scalar_derivative_rule(positive_graph; primal = :v,
    partials = (x = :dx, s = :ds))

@kernel guarded(x::Vector{Float64}, s::Float64) = begin
    cells = plate(x) do xi
        xi > 0 ? positive(xi, s) : xi * s
    end
    total::Float64 = sum(cells)
    return total
end

@kernel hoisted(x::Vector{Float64}, s::Float64) = begin
    cells = plate(x) do xi
        sin(xi + positive(s, s))
    end
    total::Float64 = sum(cells)
    return total
end

@kernel power_graph(x::Float64, s::Float64) = begin
    v::Float64 = x^s
    dx::Float64 = s * x^(s - 1)
    ds::Float64 = log(x) * x^s
    return v, dx, ds
end
const power = scalar_derivative_rule(power_graph; primal = :v,
    partials = (x = :dx, s = :ds))
@kernel fixed_power(x::Vector{Float64}, s::Float64) = begin
    cells = plate(x) do xi
        power(xi, s)
    end
    total::Float64 = sum(cells)
    return total
end

# Deliberately throwing instrumentation witnesses that a predicate-only call
# never requests a partial. It is not a numerical rule used by a consumer.
unneeded_partial(x) = error("predicate requested an unused partial")
@kernel predicate_graph(x::Float64) = begin
    v::Float64 = identity(x)
    dx::Float64 = unneeded_partial(x)
    return v, dx
end
const predicate = scalar_derivative_rule(predicate_graph;
    primal = :v, partials = (x = :dx,))
@kernel predicate_only(x::Vector{Float64}, s::Float64) = begin
    cells = plate(x) do xi
        predicate(xi) > 0 ? xi * s : -xi
    end
    total::Float64 = sum(cells)
    return total
end

@testset "scalar rule activity, hoisting and lazy branches" begin
    with_plate_reverse(true) do
        backend = AutoEnzyme(; mode = Enzyme.Reverse)
        for graph in (guarded, hoisted, predicate_only)
            k = prepare(graph)
            report = plate_reverse_report(k, (1, 2))
            @test !isempty(report)
            if graph === hoisted
                @test all(r -> last(r) == "rule", report)
            end
            # Conditional kernels currently contain opaque branch callables.
            # Check their semantics through the selected route without making
            # that fallback a requirement if the generator later supports them.
            op = prepare_ad(k, backend, [-0.4, 0.7], 0.8; active = (:x, :s))
            for x in (Float64[], [-0.4, 0.7], [-1.2, -0.2], [0.3, 0.9, 1.4]), s in (0.8, 1.3)
                value, (dx, ds) = ad_value_and_gradient(op, x, s)
                if graph === guarded
                    @test value ≈ sum(v > 0 ? log(v) + exp(s) : v*s for v in x; init = 0.0)
                    @test dx ≈ [v > 0 ? inv(v) : s for v in x]
                    @test ds ≈ sum(v > 0 ? exp(s) : v for v in x; init = 0.0)
                elseif graph === hoisted
                    h = log(s) + exp(s)
                    @test value ≈ sum(sin.(x .+ h); init = 0.0)
                    @test dx ≈ cos.(x .+ h)
                    @test ds ≈ sum(cos.(x .+ h); init = 0.0) * (inv(s) + exp(s))
                else
                    @test value ≈ sum(v > 0 ? v*s : -v for v in x; init = 0.0)
                    @test dx ≈ [v > 0 ? s : -1.0 for v in x]
                    @test ds ≈ sum(v > 0 ? v : 0.0 for v in x; init = 0.0)
                end
            end
        end
        # The exponent is inactive: requesting its log(x) partial would throw
        # on these negative bases, though their value and x derivative exist.
        k = prepare(fixed_power)
        @test all(r -> last(r) == "rule", plate_reverse_report(k, (1,)))
        op = prepare_ad(k, backend, [-2.0, -3.0], 2.0; active = (:x,))
        @test ad_value_and_gradient(op, [-2.0, -3.0], 2.0) == (13.0, ([-4.0, -6.0],))
    end
end

@testset "joint scalar cuts retain the generator's lazy control flow" begin
    # Exercise the generator directly, independently of the opaque conditional
    # callable boundary above. The predicate's deliberately unusable partial
    # must remain unrequested; log(x) must only run in the selected positive arm.
    expression = :(predicate(x) > 0 ? positive(x, s) : x * s)
    sweep = RK._pr_scalar_reverse(expression, [:x, :s], @__MODULE__; seed = :seed)
    result = Expr(:tuple, sweep.result, sweep.bars[:x], sweep.bars[:s])
    f = RK.compile(Expr(:function, Expr(:tuple, :x, :s, :seed),
        Expr(:block, sweep.forward..., sweep.reverse..., result)))
    for x in (-1.2, 0.3), s in (-0.4, 0.7), seed in (1.0, -1.3)
        v, dx, ds = f(x, s, seed)
        @test v ≈ (x > 0 ? log(x) + exp(s) : x * s)
        @test dx ≈ seed * (x > 0 ? inv(x) : s)
        @test ds ≈ seed * (x > 0 ? exp(s) : x)
    end
end
end
