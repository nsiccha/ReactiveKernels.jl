using ReactiveKernels
using Enzyme
using Test

isdefined(@__MODULE__, :InnerPlateSplit) ||
    include("fixtures/inner_plate_split.jl")

# A cell recipe that reads bound data and a live value keeps its data-only
# subexpressions as split nodes (`_KernelSplit`); under `bound=` each maximal
# one becomes a recipe cached per lane (todo 14jgmez).
@testset "data-only subexpressions of mixed plate cells" begin
    C = InnerPlateSplit
    RK = ReactiveKernels
    caches(p) = filter(r -> r.op isa RK._BoundConstant &&
        startswith(String(only(r.outputs).name), "bound_plate_"), p.recipes)
    actual_code(k) = sprint(print, Base.remove_linenums!(deepcopy(code_expr(k))))
    gradient(f, q, data...) =
        Enzyme.gradient(Enzyme.Reverse, Const(f), q, map(Const, data)...)[1]

    y = [0.0, 2.0, 5.0, 1.0, 7.0]
    q = [0.3, -0.2, 0.1, 0.5, -1.0]

    @testset "split nodes of authored expressions" begin
        ports = [:y, :n, :logit]
        normalizer = :(loggamma(n + 1.0) - loggamma(y + 1.0) -
                       loggamma(n - y + 1.0) - y * log1pexp(-logit) -
                       (n - y) * log1pexp(logit))
        split = RK._kernel_split_nodes(normalizer, ports)
        @test split !== nothing
        data = only(n for n in split.nodes if n.own == [1, 2])
        @test data.expr == :(loggamma(n + 1.0) - loggamma(y + 1.0) -
                             loggamma(n - y + 1.0))
        # Its terms that read one of `y` and `n` are nodes too, for a binding
        # of that one alone.
        @test length(data.kids) == 2
        @test Set(split.nodes[k].own for k in data.kids) == Set([[1], [2]])
        @test RK._kernel_split_shape(ports, split) ==
              RK._kernel_split_shape(ports, RK._kernel_split_nodes(normalizer, ports))
        # Cheap arithmetic alone is not worth a cached value.
        @test RK._kernel_split_nodes(:(y * logit + n - 1.0), ports) === nothing
        # Branch arms and short-circuit operands are not entered.
        @test RK._kernel_split_nodes(:(logit * (logit > 0 ? log(y) : 0.0)), ports) ===
              nothing
        @test RK._kernel_split_nodes(:(logit * (logit > 0 && log(y) > 0)), ports) ===
              nothing
        # A dotted call is a node only as a whole.
        dotted = RK._kernel_split_nodes(:(sum(log.(y .+ 1.0)) * logit), ports)
        @test only(dotted.nodes).expr == :(sum(log.(y .+ 1.0)))
    end

    for (label, kernel) in (("untyped live input", C.mixed),
                            ("declared live rank", C.mixed_typed))
        @testset "inline data-only term, $label" begin
            plain = prepare(kernel; have = (:q, :y), want = :total)
            C.calls[] = 0
            expected, doubled = plain(q, y), plain(2 .* q, y)
            @test C.calls[] == 2length(y)
            C.calls[] = 0
            bound = prepare(kernel; have = (:q, :y), want = :total, bound = (; y))
            @test C.calls[] == length(y)             # once per lane, here
            @test bound(q) == expected
            @test bound(2 .* q) == doubled
            @test C.calls[] == length(y)             # never per evaluation
            @test only(caches(bound.plan)).op.value == log.(y .+ 1.0)
            @test !occursin("counted_term", actual_code(bound))
            @test gradient(bound, q) == gradient(plain, q, y)
        end
    end

    @testset "a top-level recipe computes its data-only part once" begin
        plain = prepare(C.top_level; have = (:q, :y), want = :total)
        expected, doubled = plain(q, y), plain(2 .* q, y)
        C.calls[] = 0
        bound = prepare(C.top_level; have = (:q, :y), want = :total, bound = (; y))
        @test C.calls[] == 1
        @test bound(q) == expected
        @test bound(2 .* q) == doubled
        @test C.calls[] == 1
        @test gradient(bound, q) == gradient(plain, q, y)
    end

    @testset "nested data-only nodes cache their maximal node" begin
        n = [3.0, 4.0, 9.0, 2.0, 8.0]
        plain = prepare(C.nested; have = (:q, :y, :n), want = :total)
        expected = plain(q, y, n)
        C.calls[] = 0
        bound = prepare(C.nested; have = (:q, :y, :n), want = :total, bound = (; y, n))
        @test C.calls[] == 2length(y)
        @test only(caches(bound.plan)).op.value == log.(n .+ 1.0) .- log.(y .+ 1.0)
        @test bound(q) == expected
        @test C.calls[] == 2length(y)
        @test gradient(bound, q) == gradient(plain, q, y, n)
        # With `y` alone bound, its own term is the maximal node.
        C.calls[] = 0
        partly = prepare(C.nested; have = (:q, :y, :n), want = :total, bound = (; y))
        @test C.calls[] == length(y)
        @test only(caches(partly.plan)).op.value == log.(y .+ 1.0)
        @test partly(q, n) == expected
    end

    @testset "a data-bound arm caches only its own lanes" begin
        signed = [-1.0, 2.0, 0.0, 3.0, -4.0]
        plain = prepare(C.guarded; have = (:q, :y), want = :total)
        expected = plain(q, signed)
        C.calls[] = 0
        bound = prepare(C.guarded; have = (:q, :y), want = :total, bound = (; y = signed))
        taken = signed[signed .> 0]
        @test C.calls[] == length(taken)
        @test only(caches(bound.plan)).op.value == log.(taken .+ 1.0)
        # The arms are summed separately, as for any partitioned plate.
        @test bound(q) ≈ expected
        @test C.calls[] == length(taken)
        @test gradient(bound, q) ≈ gradient(plain, q, signed)
    end

    @testset "a short-circuit operand stays per cell" begin
        plain = prepare(C.short_circuit; have = (:q, :y), want = :total)
        C.calls[] = 0
        bound = prepare(C.short_circuit; have = (:q, :y), want = :total, bound = (; y))
        @test C.calls[] == 0
        @test isempty(caches(bound.plan))
        @test bound(q) == plain(q, y)
        @test C.calls[] == 2count(>(0), q)
    end

    @testset "a live-guarded arm reads its data-only nodes from preparation" begin
        for kernel in (C.live_branch, C.live_guard)
            plain = prepare(kernel; have = (:q, :y), want = :total)
            expected, doubled = plain(q, y), plain(-3 .* q, y)
            C.calls[] = 0
            bound = prepare(kernel; have = (:q, :y), want = :total, bound = (; y))
            @test C.calls[] == length(y)             # every lane, speculatively
            @test only(caches(bound.plan)).op.value == log.(y .+ 1.0)
            @test bound(q) == expected
            @test bound(-3 .* q) == doubled
            @test C.calls[] == length(y)
            @test gradient(bound, q) == gradient(plain, q, y)
            @test !occursin("counted_term", actual_code(bound))
        end
    end

    @testset "a speculative node that throws stays in its arm" begin
        signed = [4.0, -1.0, 9.0]
        p3 = [0.5, 1.0, -0.25]
        plain = prepare(C.live_guard_domain; have = (:q, :y), want = :total)
        expected = plain(p3, signed)
        bound = prepare(C.live_guard_domain; have = (:q, :y), want = :total,
                        bound = (; y = signed))
        @test isempty(caches(bound.plan))
        C.calls[] = 0
        @test bound(p3) == expected == 0.5 * 2.0 - 0.25 * 3.0
        @test C.calls[] == count(>=(0), signed)  # per evaluation, taken lanes only
    end

    @testset "an error a live input may avoid keeps the cell" begin
        plain = prepare(C.guarded_domain; have = (:q, :y), want = :total)
        bound = prepare(C.guarded_domain; have = (:q, :y), want = :total,
                        bound = (; y = [-1.0]))
        @test isempty(caches(bound.plan))
        @test bound(Float64[]) == plain(Float64[], [-1.0]) == 0.0
        @test_throws DomainError bound([2.0])
        # Valid data is cached as usual.
        C.calls[] = 0
        valid = prepare(C.guarded_domain; have = (:q, :y), want = :total,
                        bound = (; y = [4.0, 9.0]))
        @test only(caches(valid.plan)).op.value == [2.0, 3.0]
        @test valid([1.0, 2.0]) == 8.0
        @test C.calls[] == 2
    end
end
