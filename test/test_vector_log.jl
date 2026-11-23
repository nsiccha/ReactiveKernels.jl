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
end

end # module
