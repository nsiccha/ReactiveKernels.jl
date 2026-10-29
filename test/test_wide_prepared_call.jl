using ReactiveKernels, Test

# A prepared call takes one argument per HAVE port, and the data-only prefix
# that `prepare(...; bound)` runs takes one per bound port. Inferring the call
# must not need native stack in proportion to that count: a model's prefix over
# 655 bound columns overflowed Julia's inference stack and ran uninferred
# (snag stack-overflow-i-549143aa).

const _WIDE_COLUMN_TYPES = (Vector{Float64}, Vector{Int}, Vector{String},
    Matrix{Float64}, Vector{Bool}, Vector{Int32}, Vector{Float32})

# Column types cycle irregularly, as a dataset's do, so no tail of the argument
# tuple collapses to a compact `NTuple`.
_wide_column_type(i) = i == 1 ? Vector{Float64} :
    _WIDE_COLUMN_TYPES[1 + mod(7i + div(i, 3), length(_WIDE_COLUMN_TYPES))]

function _wide_column(::Type{T}, n) where {T}
    T === Vector{Float64} && return collect(range(0.0, 1.0; length = n))
    T === Vector{Int} && return collect(1:n)
    T === Vector{String} && return [string(j) for j in 1:n]
    T === Matrix{Float64} && return ones(n, 2)
    T === Vector{Bool} && return isodd.(1:n)
    T === Vector{Int32} && return Int32.(1:n)
    T === Vector{Float32} && return ones(Float32, n)
    error("no column of type $T")
end

# `theta` and `n` data columns: a plate over the first column (so the prepared
# callable is an embedded plate pair) and one data-only value per other column.
function _wide_spec(name, n; typed)
    ports = [Symbol(:x, i) for i in 1:n]
    signature = typed ? Any[:($(ports[i])::$(_wide_column_type(i))) for i in 1:n] :
                        Any[ports...]
    body = Any[:(cells = plate(x1) do v
                     2.0 * v
                 end),
               :(m1 = sum(cells))]
    for i in 2:n
        push!(body, :($(Symbol(:m, i)) = 1.0 * length($(ports[i]))))
    end
    push!(body, :(t1 = theta * m1))
    for i in 2:n
        push!(body, :($(Symbol(:t, i)) = $(Symbol(:t, i - 1)) + theta * $(Symbol(:m, i))))
    end
    push!(body, :(total = $(Symbol(:t, n))), :(return total))
    Core.eval(@__MODULE__, :(@kernel $name(theta::Float64, $(signature...)) =
        $(Expr(:block, body...))))
end

# Inference on a task with a small reserved stack. `Base.return_types` raises
# the `StackOverflowError` that compilation would print and swallow.
function _infer_on_small_stack(f, types; stack = 2^18)
    task = Task(() -> Base.return_types(f, types), stack)
    schedule(task)
    fetch(task)
end

@testset "marker searches over a wide argument list infer in one frame" begin
    n = 600
    T = Tuple{(_wide_column_type(i) for i in 1:n)...}
    @test _infer_on_small_stack(ReactiveKernels._dynamic_tensorized_marker, (T,)) ==
          Any[Nothing]
    # every argument a candidate axis, as for an untyped signature
    candidates = Val{Tuple(1:n)}
    @test _infer_on_small_stack(ReactiveKernels._dynamic_embedded_marker,
                                (T, candidates)) == Any[Vector{Float64}]
    # a record with as many fields
    R = NamedTuple{Tuple(Symbol(:f, i) for i in 1:n),T}
    @test _infer_on_small_stack(ReactiveKernels._dynamic_tensorized_value_marker,
                                (R,)) == Any[Nothing]
    @test _infer_on_small_stack(ReactiveKernels._embedded_axis_marker,
                                (R,)) == Any[Vector{Float64}]
    # the searches still find what they look for
    values = Tuple(_wide_column(_wide_column_type(i), 2) for i in 1:n)
    @test ReactiveKernels._dynamic_tensorized_marker(values) === nothing
    @test ReactiveKernels._dynamic_tensorized_marker(()) === nothing
    @test ReactiveKernels._dynamic_embedded_marker(values, Val(Tuple(2:n))) ===
          values[2]
    @test ReactiveKernels._dynamic_embedded_marker(values, Val(())) === nothing
    @test ReactiveKernels._embedded_axis_marker((; a = 1.0, b = values[1])) ===
          values[1]
end

@testset "a kernel with $n data columns prepares, infers and runs" for n in (200,)
    columns = [_wide_column(_wide_column_type(i), 3 + mod(i, 4)) for i in 1:n]
    expected = 0.7 * (sum(2.0 .* columns[1]) + sum(length, columns[2:end]))
    names = Tuple(Symbol(:x, i) for i in 1:n)
    for typed in (true, false)
        spec = _wide_spec(typed ? :wide_typed : :wide_untyped, n; typed)
        kernel = prepare(spec)
        @test kernel.f isa (typed ? ReactiveKernels._EmbeddedFunctionPair :
                                    ReactiveKernels._DynamicEmbeddedFunctionPair)
        arguments = (0.7, columns...)
        @test _infer_on_small_stack(kernel, Tuple{map(typeof, arguments)...}) ==
              Any[Float64]
        @test kernel(arguments...) ≈ expected
        # the bound prefix takes every column as an argument
        bound = prepare(spec; bound = NamedTuple{names}(Tuple(columns)))
        @test bound(0.7) ≈ expected
    end
end
