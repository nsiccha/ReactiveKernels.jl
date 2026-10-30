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

# The signatures Julia compiles while `f()` runs.
function _compiled_signatures(f)
    path = tempname()
    open(path, "w") do io
        ccall(:jl_dump_compiles, Cvoid, (Ptr{Cvoid},), io.handle)
        try
            f()
        finally
            ccall(:jl_dump_compiles, Cvoid, (Ptr{Cvoid},), C_NULL)
        end
    end
    signatures = [String(last(split(line, '\t'; limit = 2))) for line in readlines(path)]
    rm(path)
    signatures
end

# Whether `signature` is specialized on a selection of about `n` ports: a tuple
# of that many `Value`s, or a signature key of that many `ValueId`s.
function _port_width_specialized(signature, n)
    count("ReactiveKernels.Value{", signature) >= n ÷ 2 && return true
    any(m -> parse(Int, m.captures[1]) >= n ÷ 2,
        eachmatch(r"NTuple\{(\d+), ReactiveKernels\.ValueId\}", signature))
end

@testset "preparation compiles once for every port count" begin
    # Preparing a kernel compiles its own callables for its port types; nothing
    # else may specialize on the selection of its ports. Before, the keyword
    # body of `prepare`, the binding's pairs, the preparation cache's closures
    # and its keys compiled again for every port count: 27 of the 30 s of
    # preparing a kernel with 655 bound ports.
    function bound_columns(n)
        columns = [_wide_column(_wide_column_type(i), 3 + mod(i, 4)) for i in 1:n]
        NamedTuple{Tuple(Symbol(:x, i) for i in 1:n)}(Tuple(columns))
    end
    prepare(_wide_spec(:wide_warmup, 5; typed = true); bound = bound_columns(5))
    n = 120
    spec = _wide_spec(:wide_compiles, n; typed = true)
    bound = bound_columns(n)
    signatures = _compiled_signatures(() -> prepare(spec; bound))
    kernel_code(signature) = occursin("ReactiveKernels.PreparedKernel{", signature) ||
                             occursin("RuntimeGeneratedFunctions", signature)
    # Base's `all`/`any` over one recipe's input tuple compile per arity, a few
    # milliseconds each; this kernel's embedded op reads every column.
    recipe_inputs(signature) = any(f -> startswith(signature, "\"Tuple{typeof(Base.$f), "),
                                   ("all", "_all", "any", "_any"))
    @test isempty(filter(s -> !kernel_code(s) && !recipe_inputs(s) &&
                              _port_width_specialized(s, n), signatures))
    @test prepare(spec; bound)(0.7) ≈
          0.7 * (sum(2.0 .* bound.x1) + sum(length, values(bound)[2:end]))
end

@testset "a kernel with $n data columns prepares, infers and runs" for n in (200,)
    columns = [_wide_column(_wide_column_type(i), 3 + mod(i, 4)) for i in 1:n]
    expected = 0.7 * (sum(2.0 .* columns[1]) + sum(length, columns[2:end]))
    names = Tuple(Symbol(:x, i) for i in 1:n)
    for typed in (true, false)
        spec = _wide_spec(typed ? :wide_typed : :wide_untyped, n; typed)
        kernel = prepare(spec)
        @test kernel.f isa ReactiveKernels._ArrayFunction
        arguments = (0.7, columns...)
        @test _infer_on_small_stack(kernel, Tuple{map(typeof, arguments)...}) ==
              Any[Float64]
        @test kernel(arguments...) ≈ expected
        # the bound prefix takes every column as an argument
        bound = prepare(spec; bound = NamedTuple{names}(Tuple(columns)))
        @test bound(0.7) ≈ expected
    end
end
