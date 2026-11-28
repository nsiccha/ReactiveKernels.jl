module VectorLogTests

using Test, ReactiveKernels, MutatingFunctions
using DifferentiationInterface: AutoEnzyme
import Enzyme
const RK = ReactiveKernels

@kernel exp_sum(x::Vector{Float64}) = begin
    cells = plate(x) do v
        log((exp(v) + exp(-v)) + exp(2.0 * v))
    end
    total::Float64 = sum(cells)
    return total
end

@kernel mixed_sum(x::Vector{Float64}) = begin
    cells = plate(x) do v
        sin(v) + log((exp(v) + exp(-v)) + exp(2.0 * v))
    end
    total::Float64 = sum(cells)
    return total
end

@kernel guarded_sum(x::Vector{Float64}, use_log::Vector{Bool}) = begin
    cells = plate(x, use_log) do v, selected
        selected ? log((exp(v) + exp(-v)) + exp(2.0 * v)) : sqrt(v)
    end
    total::Float64 = sum(cells)
    return total
end

@kernel ragged_logs(rows::Vector{Vector{Float64}}) = begin
    cells = plate(rows) do row
        isempty(row) ? 0.0 : log((exp(first(row)) + exp(-first(row))) + exp(last(row)))
    end
    total::Float64 = sum(cells)
    return total
end

@kernel kept_logs(x::Vector{Float64}) = begin
    cells = plate(x) do v
        log((exp(v) + exp(-v)) + exp(2.0 * v))
    end
    return cells
end

@kernel two_terms(x::Vector{Float64}) = begin
    cells = plate(x) do v
        log(exp(v) + exp(-v))
    end
    total::Float64 = sum(cells)
    return total
end

@kernel nested_log(x::Vector{Float64}) = begin
    cells = plate(x) do v
        log((exp(log(exp(v)) - v) + exp(-abs(v))) + exp(-abs(v)))
    end
    total::Float64 = sum(cells)
    return total
end

@kernel flat_sum(x::Vector{Float64}) = begin
    cells = plate(x) do v
        log(exp(v) + exp(-v) + exp(2.0 * v))
    end
    total::Float64 = sum(cells)
    return total
end

@kernel weighted_sum(x::Vector{Float64}) = begin
    cells = plate(x) do v
        log(0.3 * exp(v) + 0.7 * exp(-v))
    end
    total::Float64 = sum(cells)
    return total
end

@kernel scaled_sum(x::Vector{Float64}) = begin
    cells = plate(x) do v
        log(exp(v) * 2 + 0.5 * (exp(-v) + exp(2.0 * v)))
    end
    total::Float64 = sum(cells)
    return total
end

@kernel signed_weights(x::Vector{Float64}) = begin
    cells = plate(x) do v
        log(1.5 * exp(v) + -0.25 * exp(-v))
    end
    total::Float64 = sum(cells)
    return total
end

score(x) = log((exp(x) + exp(-x)) + exp(2x))
partial(x) = (exp(x) - exp(-x) + 2exp(2x)) / (exp(x) + exp(-x) + exp(2x))
function reverse_gradient(k, x)
    dx = zero(x)
    Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(k), Enzyme.Active, Enzyme.Duplicated(x, dx))
    dx
end

@testset "vector log in a summed native plate" begin
    x0 = [0.2, -1.0, 0.3]
    wide = RK._vector_doubles() >= 2
    backend = AutoEnzyme(mode=Enzyme.Reverse)
    for k in (prepare(exp_sum), prepare_nonallocating(exp_sum, x0))
        source = string(code_expr(k))
        @test occursin("_log_exp_sum", source) == wide
        @test occursin("_plate_simd_cells", source)
        for n in (0, 1, 7, 513)
            x = sin.(collect(1:n))
            @test k(x) ≈ sum(score, x; init=0.0)
            op = prepare_ad(k, backend, x; active=:x)
            value, gradient = ad_value_and_gradient!(op, similar(x), x)
            @test value ≈ sum(score, x; init=0.0)
            @test gradient ≈ partial.(x)
        end
    end
    # Rebuilding on another exemplar length preserves the emitted cell count.
    sites = map((x0, repeat(x0, 128))) do x
        source = string(code_expr(prepare_nonallocating(exp_sum, x)))
        (count("_plate_simd_cells", source), count("_log_exp_sum", source))
    end
    @test first(sites) == last(sites)
    mixed = prepare(mixed_sum)
    @test !occursin("_log_exp_sum", string(code_expr(mixed)))
    @test mixed(x0) ≈ sum(x -> score(x) + sin(x), x0)
    @test !occursin("_log_exp_sum", string(code_expr(prepare(kept_logs))))
    guarded = prepare(guarded_sum)
    @test guarded([-1.0, 4.0], [true, false]) ≈ score(-1.0) + 2.0
    @test_throws DomainError guarded([-1.0], [false])
    ragged = prepare(ragged_logs)
    @test !occursin("_log_exp_sum", string(code_expr(ragged)))
    rows = [Float64[], [0.0], [-0.5, 1.0, 0.5]]
    @test ragged(rows) ≈ log(3.0) + log(exp(-0.5) + 2exp(0.5))
    stable = prepare(two_terms)
    @test !occursin("_log_exp_sum", string(code_expr(stable)))
    @test stable([1000.0, -1000.0]) ≈ 2000.0
    @test reverse_gradient(stable, [1000.0, -1000.0]) ≈ [1.0, -1.0]
    nested = prepare(nested_log)
    @test nested([1000.0]) ≈ 0.0 atol=eps()
    @test only(reverse_gradient(nested, [1000.0])) ≈ 0.0 atol=eps()
end

@testset "a vector log of flat and scaled exponential sums" begin
    wide = RK._vector_doubles() >= 2
    backend = AutoEnzyme(mode=Enzyme.Reverse)
    weighted(x) = log(0.3 * exp(x) + 0.7 * exp(-x))
    weighted_partial(x) = (0.3 * exp(x) - 0.7 * exp(-x)) / (0.3 * exp(x) + 0.7 * exp(-x))
    scaled(x) = log(exp(x) * 2 + 0.5 * (exp(-x) + exp(2x)))
    scaled_partial(x) = (2exp(x) + 0.5 * (2exp(2x) - exp(-x))) / (exp(x) * 2 + 0.5 * (exp(-x) + exp(2x)))
    for (spec, value, slope) in ((flat_sum, score, partial), (weighted_sum, weighted, weighted_partial),
                                 (scaled_sum, scaled, scaled_partial))
        k = prepare(spec)
        source = string(code_expr(k))
        @test occursin("_log_exp_sum", source) == wide
        @test occursin("_plate_simd_cells", source)
        for n in (0, 1, 7, 513)
            x = sin.(collect(1:n))
            @test k(x) ≈ sum(value, x; init=0.0)
            op = prepare_ad(k, backend, x; active=:x)
            total, gradient = ad_value_and_gradient!(op, similar(x), x)
            @test total ≈ sum(value, x; init=0.0)
            @test gradient ≈ slope.(x)
        end
    end
    # A negative factor can make the sum negative: the written logarithm stays.
    signed = prepare(signed_weights)
    @test !occursin("_log_exp_sum", string(code_expr(signed)))
    @test signed([0.0, 1.0]) ≈ log(1.25) + log(1.5 * exp(1.0) - 0.25 * exp(-1.0))
    @test_throws DomainError signed([-2.0])
end

struct LoggedOperand
    id::Int
    trace::Vector{Int}
end
Base.exp(x::LoggedOperand) = (push!(x.trace, x.id); x)
Base.:+(x::LoggedOperand, y::LoggedOperand) = (push!(x.trace, 10x.id + y.id); x)
Base.log(x::LoggedOperand) = (push!(x.trace, 0); -1.0)
struct NegativeExp end
Base.exp(::NegativeExp) = -2.0
mutable_exp = exp

@testset "a vector log keeps the source arithmetic and its dispatch" begin
    ref = GlobalRef
    sum_expr = :($(ref(Base,:log))($(ref(Base,:+))($(ref(Base,:+))(
        $(ref(Base,:exp))(x), $(ref(Base,:exp))(y)), $(ref(Base,:exp))(z))))
    body = Expr(:block, sum_expr)
    transformed = RK._vectorize_cell_logs(body)
    @test (transformed !== body) == (RK._vector_doubles() >= 2)
    kernel = RK.compile(Expr(:function, Expr(:tuple, :x, :y, :z), transformed))
    # A global variable can be rebound between preparation and evaluation.
    mutable_call = Expr(:call, ref(@__MODULE__, :mutable_exp), :x)
    mixed = Expr(:block, Expr(:call, ref(Base, :log),
        Expr(:call, ref(Base, :+), mutable_call, Expr(:call, ref(Base, :exp), :y))))
    @test RK._vectorize_cell_logs(mixed) === mixed

    trace = Int[]
    @test kernel((LoggedOperand(i, trace) for i in 1:3)...) === -1.0
    @test trace == [1, 2, 12, 3, 13, 0]
    @test_throws DomainError kernel(NegativeExp(), NegativeExp(), 0.0)
    @test kernel(0f0, 0f0, 0f0) === log(3f0)
    @test kernel(0f0, 0.0, 0.0) === log(3.0)
    @test kernel(-Inf, -Inf, -Inf) === -Inf
    for x in (nextfloat(0.0), floatmin(), prevfloat(1.0), 1.0, nextfloat(1.0), floatmax())
        reference = Float64(log(BigFloat(x)))
        @test abs(RK._vector_log(x) - reference) <= eps(reference)
        @test derivative_cut(RK._vector_log, (true,), x) == (log(x), inv(x))
    end
    @test kernel(Inf, 0.0, 0.0) === Inf
    @test isnan(kernel(NaN, 0.0, 0.0))

    # One `+` call of three terms stays one call: the operations and their
    # order are the written expression's.
    flat_body = Expr(:block, :($(ref(Base,:log))($(ref(Base,:+))(
        $(ref(Base,:exp))(x), $(ref(Base,:exp))(y), $(ref(Base,:exp))(z)))))
    flat_transformed = RK._vectorize_cell_logs(flat_body)
    @test (flat_transformed !== flat_body) == (RK._vector_doubles() >= 2)
    written, rewritten = map((flat_body, flat_transformed)) do body
        RK.compile(Expr(:function, Expr(:tuple, :x, :y, :z), body))
    end
    traces = map((written, rewritten)) do f
        seen = Int[]
        @test f((LoggedOperand(i, seen) for i in 1:3)...) === -1.0
        seen
    end
    @test first(traces) == last(traces) == [1, 2, 3, 12, 13, 0]
    @test_throws DomainError rewritten(NegativeExp(), NegativeExp(), 0.0)
    @test rewritten(0f0, 0f0, 0f0) === log(3f0)
    @test rewritten(0.3, -0.2, 1.1) ≈ written(0.3, -0.2, 1.1)

    # A literal factor multiplies on its written side. One that is negative,
    # or a negative zero, keeps the written logarithm.
    outcome(f, args...) = try f(args...) catch err; typeof(err) end
    product(c, left) = Expr(:block, Expr(:call, ref(Base, :log), Expr(:call, ref(Base, :+),
        left ? Expr(:call, ref(Base, :*), c, Expr(:call, ref(Base, :exp), :x)) :
               Expr(:call, ref(Base, :*), Expr(:call, ref(Base, :exp), :x), c),
        Expr(:call, ref(Base, :exp), :y))))
    for left in (true, false)
        for c in (-0.25, -0.0, -1)
            body = product(c, left)
            @test RK._vectorize_cell_logs(body) === body
        end
        for c in (0.25, 0.0, 2, 0x02, 0.5f0)
            body = product(c, left)
            transformed = RK._vectorize_cell_logs(body)
            @test (transformed !== body) == (RK._vector_doubles() >= 2)
            written, rewritten = map((body, transformed)) do b
                RK.compile(Expr(:function, Expr(:tuple, :x, :y), b))
            end
            for point in ((0.3, -0.2), (-Inf, -Inf), (Inf, 0.0), (NaN, 0.0), (700.0, 700.0), (-745.0, -800.0))
                a, b = written(point...), rewritten(point...)
                @test isequal(a, b) || a ≈ b
            end
            @test rewritten(0f0, 0f0) === written(0f0, 0f0)
            @test outcome(rewritten, NegativeExp(), -Inf) === outcome(written, NegativeExp(), -Inf)
            @test outcome(rewritten, NegativeExp(), 0.0) === outcome(written, NegativeExp(), 0.0)
        end
    end
end

end # module
