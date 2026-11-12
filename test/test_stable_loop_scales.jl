# A stable rewrite must keep shared work out of a plate's cell loop (user
# RKPPLBench/decisions/2026-10-13T01-17-11-614-1om0yie). The stable primal and
# ordinary reverse, including its overflow boundary, remain the authority.
module StableLoopScalesTests
using Test, ReactiveKernels, DifferentiationInterface, Enzyme
using LinearAlgebra
const RK = ReactiveKernels

@kernel shared_divisor(u::Float64, x::Vector{Float64}) = begin
    s = exp(u)
    cells = plate(x) do xi
        xi / s + (2xi) / s
    end
    total = sum(cells)
    return total
end

@kernel varying_divisor(u::Vector{Float64}, x::Vector{Float64}) = begin
    cells = plate(u, x) do ui, xi
        xi / exp(ui)
    end
    total = sum(cells)
    return total
end

@kernel guarded_divisor(u::Float64, x::Vector{Float64}) = begin
    s = exp(u)
    cells = plate(x) do xi
        xi > 0 ? xi / s : 0.0
    end
    total = sum(cells)
    return total
end

@kernel nested_divisor(u::Float64, groups::Vector{Vector{Float64}}) = begin
    s = exp(u)
    totals = plate(groups) do group
        cells = plate(group) do xi
            xi / s
        end
        sum(cells)
    end
    return sum(totals)
end

function inventory(ast)
    heads = Dict{Symbol,Int}()
    scales = Int[]
    function visit(ex, depth)
        ex isa RK._PlateProgram && return visit(ex.ast, depth)
        ex isa Expr || return
        ex.head in (:quote, :inert, :meta, :line) && return
        heads[ex.head] = get(heads, ex.head, 0) + 1
        if ex.head === :call && first(ex.args) == GlobalRef(RK, :_rk_div_by_scale)
            push!(scales, depth)
        end
        foreach(a -> visit(a, depth + (ex.head in (:for, :while))), ex.args)
    end
    visit(ast, 0)
    (; heads, scales)
end

@testset "one reciprocal exponential per evaluation, including bound data" begin
    @test Base.get_extension(RK, :ReactiveKernelsEnzymeExt) !== nothing
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    shapes = []
    for n in (0, 1, 7, 257)
        x = [0.25 + i / 1000 for i in 1:n]
        k = prepare(shared_divisor; bound = (; x))
        candidates = Base.get_extension(RK, :ReactiveKernelsMutatingFunctionsExt) === nothing ?
            (k,) : (k, prepare_nonallocating(k, 0.3))
        push!(shapes, inventory(code_expr(k)).heads)
        for candidate in candidates
            report = inventory(code_expr(candidate))
            @test report.scales == [0]
            ad = prepare_ad(candidate, backend, 0.3; active = :u)
            for u in (-2.0, 0.3, 3.0)
                expected = 3sum(x) * exp(-u)
                value, derivative = ad_value_and_gradient(ad, u)
                @test value ≈ expected
                @test derivative ≈ -expected
            end
        end
    end
    # Empty/singleton bound inputs simplify some reads. Increasing a general
    # plate from 7 to 257 cells must not replicate bodies or control regions.
    for head in (:for, :while, :if)
        @test all(s -> get(s, head, 0) == get(first(shapes), head, 0), shapes)
    end
    @test sum(values(last(shapes))) <= sum(values(shapes[end - 1]))
    # A single prepared program accepts changed lengths and values. Both
    # operands differentiate through the ordinary backend path.
    k = prepare(shared_divisor)
    for x in (Float64[], [0.25], [0.25, 0.5, 1.0])
        expected = 3sum(x) * exp(-0.3)
        dx = zero(x)
        du = Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(k), Enzyme.Active,
            Enzyme.Active(0.3), Enzyme.Duplicated(copy(x), dx))[1][1]
        @test k(0.3, x) ≈ expected
        @test du ≈ -expected
        @test dx ≈ fill(3exp(-0.3), length(x))
    end
end

@testset "stable tails, per-cell coordinates and inactive arms" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    k = prepare(shared_divisor)
    for (u, x) in ((710.0, [1e300]), (-700.0, [1e-300]))
        expected = Float64(3sum(big.(x)) * exp(-big(u)))
        ad = prepare_ad(k, backend, u, x; active = :u)
        value, derivative = ad_value_and_gradient(ad, u, x)
        @test value ≈ expected rtol = 1e-13
        @test derivative ≈ -expected rtol = 1e-13
    end
    varying = prepare(varying_divisor)
    @test isempty(inventory(code_expr(varying)).scales)
    u, x = [-0.5, 0.3, 0.7], [0.25, 0.5, 1.0]
    @test varying(u, x) ≈ sum(x .* exp.(-u))
    guarded = prepare(guarded_divisor)
    for x in (Float64[], [-1.0, -2.0])
        @test guarded(-800.0, x) == 0.0
        ad = prepare_ad(guarded, backend, -800.0, x; active = :u)
        @test ad_value_and_gradient(ad, -800.0, x) == (0.0, 0.0)
    end
    empty_ad = prepare_ad(k, backend, -800.0, Float64[]; active = :u)
    @test ad_value_and_gradient(empty_ad, -800.0, Float64[]) == (0.0, 0.0)
    nested = prepare(nested_divisor)
    for groups in (Vector{Vector{Float64}}(), [Float64[], Float64[]])
        ad = prepare_ad(nested, backend, -800.0, groups; active = :u)
        @test ad_value_and_gradient(ad, -800.0, groups) == (0.0, 0.0)
    end
    groups = [[0.25, 0.5], Float64[], [1.0]]
    nested_ad = prepare_ad(nested, backend, 0.3, groups; active = :u)
    value, derivative = ad_value_and_gradient(nested_ad, 0.3, groups)
    @test value ≈ 1.75exp(-0.3)
    @test derivative ≈ -value
    literal = prepare(shared_divisor; simplify = false)
    @test isempty(inventory(code_expr(literal)).scales)
    @test k(0.3, x) ≈ literal(0.3, x)
end

@testset "unsupported operand types keep their original dispatch" begin
    for u in (0.3f0, 0.3, Float16(0.3)), y in (0.7, 0.7 + 0.2im, [0.7, 1.2])
        scale = RK._rk_div_by_scale(exp, u)
        @test isequal(RK._rk_div_by_of(y, exp, u, scale), RK._rk_div_by_of(y, exp, u))
    end
    for u in (big"0.3", [0.1 0.0; 0.0 0.2])
        @test RK._rk_div_by_scale(exp, u) === nothing
        y = u isa Matrix ? Matrix{Float64}(I, 2, 2) : 0.7
        @test RK._rk_div_by_of(y, exp, u, nothing) == RK._rk_div_by_of(y, exp, u)
    end
end
end
