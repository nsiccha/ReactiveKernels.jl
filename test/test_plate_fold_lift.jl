module PlateFoldLiftTests
using ReactiveKernels, Test
using ReactiveKernels.LogExpFunctions: logsumexp

# A plate's `sum` or `logsumexp` written inside a larger expression
# (`s^2 * sum(terms)`) streams the plate's total, as the same fold written as
# its own statement does: no cell buffer. The references are that statement
# form (the same lowering, so values agree bit for bit) and a form that reads
# a copy of the cells (`sum(copy(terms))`), which keeps the buffer.

allocated(f::F, args::Vararg{Any,N}) where {F,N} = @allocated f(args...)
streamed(kernel) = occursin("_plate_total", string(code_expr(kernel)))

@kernel nested_sum(x::Vector{Float64}, s::Float64) = begin
    terms = plate(x, Ref(s)) do xi, s
        exp(-s * xi) / (1 + xi^2)
    end
    return s^2 * sum(terms)
end
@kernel statement_sum(x::Vector{Float64}, s::Float64) = begin
    terms = plate(x, Ref(s)) do xi, s
        exp(-s * xi) / (1 + xi^2)
    end
    total = sum(terms)
    return s^2 * total
end
@kernel copied_sum(x::Vector{Float64}, s::Float64) = begin
    terms = plate(x, Ref(s)) do xi, s
        exp(-s * xi) / (1 + xi^2)
    end
    return s^2 * sum(copy(terms))
end

@kernel nested_logsumexp(x::Vector{Float64}) = begin
    terms = plate(x) do xi
        -xi^2 / 2
    end
    return 3 * logsumexp(terms) - 1
end
@kernel statement_logsumexp(x::Vector{Float64}) = begin
    terms = plate(x) do xi
        -xi^2 / 2
    end
    total = logsumexp(terms)
    return 3 * total - 1
end

# Every occurrence of one fold reads one statement, an existing one included:
# a plate keeps a single fold statement.
@kernel repeated_sum(x::Vector{Float64}) = begin
    terms = plate(x) do xi
        sin(xi)
    end
    a = 2 * sum(terms)
    b = sum(terms) / 3
    return a + b
end
@kernel repeated_reference(x::Vector{Float64}) = begin
    terms = plate(x) do xi
        sin(xi)
    end
    total = sum(terms)
    a = 2 * total
    b = total / 3
    return a + b
end
@kernel statement_and_nested(x::Vector{Float64}) = begin
    terms = plate(x) do xi
        sin(xi)
    end
    total = sum(terms)
    return 2 * total + sum(terms) / 3
end

# Two different folds of one plate, a fold with a declared type and a fold in
# a branch arm keep the expression as written.
@kernel two_folds(x::Vector{Float64}) = begin
    terms = plate(x) do xi
        -xi^2
    end
    return sum(terms) + logsumexp(terms)
end
@kernel typed_and_nested(x::Vector{Float64}) = begin
    terms = plate(x) do xi
        sin(xi)
    end
    total::Float64 = sum(terms)
    return total + 2 * sum(terms)
end
@kernel branch_fold(x::Vector{Float64}, s::Float64) = begin
    terms = plate(x) do xi
        sin(xi)
    end
    r = s > 0 ? s * sum(terms) : 0.0
    return r
end

# A nested plate inside a cell, summed inside the cell's expression.
@kernel cell_nested_sum(groups::Vector{Vector{Float64}}, s::Float64) = begin
    totals = plate(groups, Ref(s)) do g, s
        inner = plate(g, Ref(s)) do gi, s
            s * gi^2
        end
        1 + 2 * sum(inner)
    end
    return totals
end

@testset "a plate fold inside an expression streams the plate's total" begin
    x = collect(range(-2.0, 3.0; length = 37))
    nested, statement, copied =
        prepare(nested_sum), prepare(statement_sum), prepare(copied_sum)
    @test streamed(nested) && streamed(statement) && !streamed(copied)
    for xs in (x, x[1:1], Float64[])
        @test nested(xs, 0.7) === statement(xs, 0.7)
        @test nested(xs, 0.7) ≈ copied(xs, 0.7)
    end
    lse, lse_statement = prepare(nested_logsumexp), prepare(statement_logsumexp)
    @test streamed(lse)
    @test lse(x) === lse_statement(x)

    repeated, reference = prepare(repeated_sum), prepare(repeated_reference)
    @test streamed(repeated)
    @test repeated(x) === reference(x)
    both = prepare(statement_and_nested)
    @test streamed(both)
    @test both(x) === reference(x)

    # Unchanged: each still prepares and evaluates its expression.
    @test prepare(two_folds)(x) ≈ sum(-x .^ 2) + logsumexp(-x .^ 2)
    @test prepare(typed_and_nested)(x) ≈ 3 * sum(sin.(x))
    branch = prepare(branch_fold)
    @test branch(x, 0.5) ≈ 0.5 * sum(sin.(x))
    @test branch(x, -1.0) === 0.0

    groups = [x[1:5], Float64[], x[6:20]]
    cells = prepare(cell_nested_sum)
    @test cells(groups, 0.3) ≈ [1 + 2 * sum(0.3 .* g .^ 2) for g in groups]

    # No buffer of cells: the nested form allocates what the statement form
    # does (nothing), the copy keeps one.
    long = collect(range(-1.0, 1.0; length = 10_000))
    for kernel in (nested, statement)
        kernel(long, 0.7)
        @test allocated(kernel, long, 0.7) == 0
    end
    copied(long, 0.7)
    @test allocated(copied, long, 0.7) > 10_000 * sizeof(Float64)
    for kernel in (lse, repeated)
        kernel(long)
        @test allocated(kernel, long) == 0
    end
end

end # module
