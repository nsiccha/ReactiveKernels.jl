using ReactiveKernels, Reactant, Test
using StaticArrays: SMatrix
import Enzyme

@traceable function _conditional_scalar_carry(x, n)
    acc = 0.0
    for i in one(n):n
        if x > 0
            acc = acc + log(x)
        end
    end
    acc
end
@kernel conditional_scalar_carry(q, n) = begin
    total = _conditional_scalar_carry(q[1], n)
    return total
end

@traceable function _conditional_matrix_carry(B::SMatrix{2,2}, n)
    R = SMatrix{2,2}(1.0, 0.0, 0.0, 1.0)
    i = zero(n)
    while i < n
        if (i & 1) == 0
            R = R * B
        end
        i = i + one(i)
    end
    R
end
@traceable _conditional_matrix_sum(A::SMatrix{2,2}) =
    A[1,1] + A[2,1] + A[1,2] + A[2,2]
@kernel conditional_matrix_carry(q, n) = begin
    B = SMatrix{2,2}(q[1], 0.1, 0.2, q[2])
    R = _conditional_matrix_carry(B, n)
    total = _conditional_matrix_sum(R) + _conditional_matrix_sum(B)
    return total
end

@traceable function _conditional_pair_carry(x, n)
    a, b = x, 0.0
    i = zero(n)
    while i < n
        if (i & 1) == 0
            a = a + x
            b = b + a
        elseif x < 0
            b = b - a
        end
        i = i + one(i)
    end
    a + b
end
@kernel conditional_pair_carry(q, n) = begin
    total = _conditional_pair_carry(q[1], n)
    return total
end

# A used branch expression must retain its value as well as its assignments.
@traceable function _conditional_value_and_state(x, flag)
    r = x
    y = if flag
        r = 2r
        r + 1
    else
        r = r - 1
        r - 1
    end
    r + y
end
@kernel conditional_value_and_state(q, flag) = begin
    total = _conditional_value_and_state(q[1], flag)
    return total
end

@traceable function _conditional_buffer(q)
    out = zeros(length(q))
    for i in eachindex(q)
        if q[i] > 0
            out[i] = sqrt(q[i])
        end
    end
    sum(out)
end
@kernel conditional_buffer(q) = begin
    total = _conditional_buffer(q)
    return total
end

_branch_traced(x::AbstractArray) = Reactant.to_rarray(x)
_branch_traced(x::Number) = Reactant.to_rarray(x; track_numbers=true)
_branch_host(x::Reactant.AbstractConcreteArray) = Array(x)
_branch_host(x) = Reactant.to_number(x)
function _branch_inventory(hlo)
    counts = Dict{String,Int}()
    for match in eachmatch(r"\b(?:stablehlo|chlo|func|arith|enzyme|scf|tensor)\.\w+", hlo)
        counts[match.match] = get(counts, match.match, 0) + 1
    end
    counts
end

@testset "branch-only scalar assignments stay lazy and retained" begin
    k = prepare(conditional_scalar_carry)
    gradient(q, n) = only(Enzyme.gradient(Enzyme.Reverse, v -> k(v, n), q))
    for T in (Int32, Int64)
        compiled = Reactant.@compile k(_branch_traced([0.7]), _branch_traced(T(3)))
        reverse = Reactant.@compile gradient(_branch_traced([0.7]), _branch_traced(T(3)))
        inventories = Dict{String,Int}[]
        reverse_inventories = Dict{String,Int}[]
        for n in (0, 1, 2, 5, 17), x in (0.7, -0.4)
            q, tn = [x], _branch_traced(T(n))
            reference = x > 0 ? n * log(x) : 0.0
            derivative = x > 0 ? [n / x] : [0.0]
            @test k(q, T(n)) ≈ reference
            @test Float64(compiled(_branch_traced(q), tn)) ≈ reference
            @test gradient(q, T(n)) ≈ derivative
            @test _branch_host(reverse(_branch_traced(q), tn)) ≈ derivative
            push!(inventories, _branch_inventory(repr(
                Reactant.@code_hlo optimize=true k(_branch_traced(q), tn))))
            push!(reverse_inventories, _branch_inventory(repr(
                Reactant.@code_hlo optimize=true gradient(_branch_traced(q), tn))))
        end
        @test allequal(inventories)
        @test first(inventories)["stablehlo.while"] == 1
        @test allequal(reverse_inventories)
        @test get(first(reverse_inventories), "stablehlo.while", 0) >= 1
    end
end

@testset "branch-only matrix assignments preserve wrappers and the caller" begin
    k = prepare(conditional_matrix_carry)
    for T in (Int32, Int64)
        compiled = Reactant.@compile k(_branch_traced([0.6, 0.7]), _branch_traced(T(3)))
        inventories = Dict{String,Int}[]
        for n in (0, 1, 2, 5, 17), q in ([0.6, 0.7], [0.45, 0.75])
            B = [q[1] 0.2; 0.1 q[2]]
            reference = sum(B^cld(n, 2)) + sum(B)
            tn = _branch_traced(T(n))
            @test k(q, T(n)) ≈ reference
            @test Float64(compiled(_branch_traced(q), tn)) ≈ reference
            push!(inventories, _branch_inventory(repr(
                Reactant.@code_hlo optimize=true k(_branch_traced(q), tn))))
        end
        @test allequal(inventories)
        @test first(inventories)["stablehlo.while"] == 1
        @test first(inventories)["stablehlo.if"] == 1
    end
    q, h = [0.6, 0.7], 1e-6
    gradient(v) = only(Enzyme.gradient(Enzyme.Reverse, w -> k(w, 5), v))
    differences = [(e = zeros(2); e[j] = h;
        (k(q + e, 5) - k(q - e, 5)) / (2h)) for j in 1:2]
    @test gradient(q) ≈ differences rtol=1e-8
end

@testset "multiple conditional carries retain assignment order and elseif" begin
    k = prepare(conditional_pair_carry)
    compiled = Reactant.@compile k(_branch_traced([0.7]), _branch_traced(Int64(3)))
    for n in (0, 1, 2, 5, 17), x in (0.7, -0.4)
        even, odd = cld(n, 2), fld(n, 2)
        coefficient = even + 1 + even * (even + 3) / 2 -
            (x < 0 ? odd * (odd + 3) / 2 : 0)
        @test k([x], n) ≈ coefficient * x
        @test Float64(compiled(_branch_traced([x]), _branch_traced(Int64(n)))) ≈ coefficient * x
    end
end

@testset "used conditional expressions retain both value and state" begin
    k = prepare(conditional_value_and_state)
    gradient(q, flag) = only(Enzyme.gradient(Enzyme.Reverse, v -> k(v, flag), q))
    compiled = Reactant.@compile k(_branch_traced([0.7]), _branch_traced(true))
    reverse = Reactant.@compile gradient(_branch_traced([0.7]), _branch_traced(true))
    for flag in (false, true), x in (0.7, -0.4)
        reference = flag ? 4x + 1 : 2x - 3
        derivative = [flag ? 4.0 : 2.0]
        @test k([x], flag) ≈ reference
        @test Float64(compiled(_branch_traced([x]), _branch_traced(flag))) ≈ reference
        @test gradient([x], flag) ≈ derivative
        @test _branch_host(reverse(_branch_traced([x]), _branch_traced(flag))) ≈ derivative
    end
end

@testset "conditional indexed writes preserve inactive arms and empty loops" begin
    k = prepare(conditional_buffer)
    inventories = Dict{String,Int}[]
    for n in (0, 32, 96)
        q = [isodd(i) ? 0.25 + i / 10 : -0.4 for i in 1:n]
        reference = sum(sqrt(x) for x in q if x > 0; init=0.0)
        compiled = Reactant.@compile k(_branch_traced(q))
        @test k(q) ≈ reference
        @test Float64(compiled(_branch_traced(q))) ≈ reference
        n == 0 && continue
        gradient(v) = only(Enzyme.gradient(Enzyme.Reverse, k, v))
        reverse = Reactant.@compile gradient(_branch_traced(q))
        derivative = [x > 0 ? 1 / (2sqrt(x)) : 0.0 for x in q]
        @test gradient(q) ≈ derivative
        @test _branch_host(reverse(_branch_traced(q))) ≈ derivative
        push!(inventories, _branch_inventory(repr(
            Reactant.@code_hlo optimize=true k(_branch_traced(q)))))
    end
    @test allequal(inventories)
    @test first(inventories)["stablehlo.while"] == 1
end
