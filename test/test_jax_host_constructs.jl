using ReactiveKernels, Test

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

# Ordinary data-only Julia in `jax_source`'s host `prepare_data`: each helper
# reads data alone, so the module computes it with NumPy on the host, and its
# value must equal native Julia's. The kernels scale it by `theta` on the
# device.

@traceable jhc_bounds(x) = (; lo = minimum(x), hi = maximum(x), n = length(x))
@traceable function jhc_span(x)
    (; lo, hi, n) = jhc_bounds(x)
    (hi - lo) * n
end
@traceable jhc_first_positions(x) = sum(Int[findfirst(==(v), x) for v in x])
@traceable jhc_positions_in(x, y) = sum(Int[findfirst(==(v), x) for v in y])
@traceable jhc_as_floats(x) = sum(Float64[length(x) - i for i in eachindex(x)])
@traceable function jhc_grow(x)
    rv = [0.0]
    for v in x
        v > 0 && push!(rv, v)
    end
    both = vcat(x, rv)
    push!(both, 10.0)
    tail = x[2:end]
    append!(tail, rv)
    sorted = sort(x)
    push!(sorted, 4.0)
    alias = rv
    push!(rv, 7.0)
    sum(rv) + 2 * sum(both) + 3 * sum(tail) + 4 * sum(sorted) + 5 * length(x) + 6 * sum(x) +
        7 * length(alias)
end
@traceable function jhc_first_zero(x)
    rv = copy(x)
    rv[findfirst(iszero, rv)] = 5.0
    sum(rv) + 10 * x[findfirst(iszero, x)] + 100 * findfirst(iszero, x)
end
@traceable function jhc_order(x)
    s = issorted(x) ? 1.0 : 0.0
    n = 0
    for i in 1:(length(x) - 1)
        if isless(x[i], x[i + 1])
            n += 1
        end
    end
    s + 10 * n + 100 * (isless(x[1], x[end]) ? 1 : 0)
end
@traceable function jhc_distinct(x)
    seen = Set{Float64}()
    n = 0
    for v in x
        v in seen && continue
        push!(seen, v)
        n += 1
    end
    delete!(seen, 1.0)
    all = Set(x)
    union!(all, [1.5, 2.5])
    n + 100 * length(seen) + 1000 * length(all) + (isempty(seen) ? 10000 : 0) + (2.5 in all ? 0.5 : 0.0)
end
@traceable function jhc_views(x)
    rv = sizehint!(Float64[], 2 * length(x))
    append!(rv, @view x[2:end])
    push!(rv, sum(view(x, 1:2)))
    sum(vcat(rv, x)) + 2 * sum(vcat(0.0, x)) + 3 * length(rv)
end
@traceable jhc_total_length(args::Vararg{Any,N}) where {N} = sum(length, args; init = 0)
@traceable jhc_lengths(x, y) = jhc_total_length(x, y, x) * 1.5
@traceable function jhc_fill_head!(rv, v)
    rv[1] = v
    rv
end
@traceable function jhc_grow!(rv, v)
    push!(rv, v)
    nothing
end
@traceable function jhc_sort_then_fill!(rv)
    sort!(rv)
    jhc_fill_head!(rv, -1.0)
end
@traceable function jhc_mutations(x)
    buf = zeros(length(x))
    jhc_fill_head!(buf, 7.0)
    acc = Float64[]
    jhc_grow!(acc, 2.0)
    jhc_grow!(acc, 3.0)
    grown = [1.0]
    jhc_grow!(grown, 4.0)
    s = copy(x)
    jhc_sort_then_fill!(s)
    sum(buf) + 10 * sum(acc) + 20 * sum(grown) + 100 * s[1] + 1000 * s[end] + sum(x)
end
@traceable function jhc_pairs(x)
    d = Float64[x[i] - x[j] for j in eachindex(x) for i in eachindex(x) if i != j]
    c = 0.5
    kept = filter!(v -> v >= 0.0, d)
    sum(sort!(kept)) + length(kept) + length(d) + sum(filter(v -> v > c, x))
end
@traceable function jhc_branch_list(x)
    if length(x) > 3
        rv = Float64[]
        push!(rv, 1.0)
    else
        rv = x
    end
    acc = x
    for v in x
        if v > 2
            acc = Float64[]
        end
    end
    sum(rv .* 2) + length(rv) + sum(acc .* 3)
end
@traceable jhc_maybe(x) = length(x) > 2 ? (x[1], x[2]) : nothing
@traceable function jhc_union_branch(x)
    m = jhc_maybe(x)
    a, b = if m === nothing
        0.0, 0.0
    else
        m
    end
    a + 2b
end
# An optional pair of lists, asked with `isnothing` and `!== nothing`.
@traceable jhc_maybe_lists(x) =
    length(x) > 2 ? (x[2:end], Int[findfirst(==(v), x) for v in x]) : nothing
@traceable function jhc_union_forms(x)
    m = jhc_maybe_lists(x)
    tail, positions = if isnothing(m)
        x, Int[length(x) - i for i in eachindex(x)]
    else
        m
    end
    present = m !== nothing ? 100.0 : 0.0
    sum(tail) + 2 * sum(positions) + present
end

for f in (:jhc_span, :jhc_first_positions, :jhc_as_floats, :jhc_grow, :jhc_first_zero,
          :jhc_order, :jhc_distinct, :jhc_views, :jhc_mutations, :jhc_pairs, :jhc_branch_list,
          :jhc_union_branch, :jhc_union_forms)
    @eval @kernel $(Symbol(f, :_kernel))(theta::Float64, x::Vector{Float64}) = begin
        s = $f(x)
        total = theta * s
        return total
    end
end
@kernel jhc_two_kernel(theta::Float64, x::Vector{Float64}, y::Vector{Float64}) = begin
    s = jhc_lengths(x, y) + jhc_positions_in(x, y)
    total = theta * s
    return total
end

const JHC_X = [3.0, 0.0, 1.0, -0.0, 2.0, 3.0, NaN, 0.5]

# The module of `spec` emitted from `data`, and, where Python runs, its value
# on each data in `datas` against native Julia's.
function jhc_parity(spec, datas; have = (:theta, :x))
    native = prepare(spec; have, want = :total)
    src = jax_source(spec; have, want = :total, data = NamedTuple{Base.tail(have)}(first(datas)))
    text = string(src)
    # The data-only helper runs in `prepare_data`, not in the JAX function.
    @test occursin("def prepare_data", text)
    # Data lengths are data: a longer dataset emits the same module.
    longer = map(d -> vcat(d, d), first(datas))
    @test string(jax_source(spec; have, want = :total,
                            data = NamedTuple{Base.tail(have)}(longer))) == text
    if rk_jax_available()
        for d in datas
            got = rk_jax_evaluate(src, :kernel, (0.5, d...); prepare = true)
            @test isequal(got.value, native(0.5, d...))
        end
    else
        @test_skip false
    end
    src
end

@testset "jax_source host prepare_data: ordinary data-only Julia" begin
    xs = (JHC_X,)
    @testset "(; a, b) = record" begin
        jhc_parity(jhc_span_kernel, [(JHC_X[1:6],), ([1.0, 5.0],)])
    end
    @testset "typed comprehensions" begin
        jhc_parity(jhc_first_positions_kernel, [(JHC_X[1:6],), ([2.0, 2.0, 1.0],)])
        jhc_parity(jhc_as_floats_kernel, [xs, (Float64[],)])
        src = jhc_parity(jhc_two_kernel, [(JHC_X, [0.0, 3.0]), ([1.0], Float64[])];
                         have = (:theta, :x, :y))
        # `Int[findfirst(...)]` of a value that is absent raises, as Julia's
        # `convert(Int, nothing)` does.
        native = prepare(jhc_two_kernel; have = (:theta, :x, :y), want = :total)
        @test_throws MethodError native(0.5, [1.0], [2.0])
        if rk_jax_available()
            err = try
                rk_jax_evaluate(src, :kernel, (0.5, [1.0], [2.0]); prepare = true)
                nothing
            catch e
                e
            end
            @test err isa ErrorException && occursin("Cannot `convert` an object of type Nothing", err.msg)
        end
    end
    @testset "push! and append! onto any vector" begin
        jhc_parity(jhc_grow_kernel, [xs, ([1.0, -2.0],)])
    end
    @testset "indexing with findfirst's result" begin
        src = jhc_parity(jhc_first_zero_kernel, [(JHC_X[1:6],), ([-0.0, 1.0],)])
        native = prepare(jhc_first_zero_kernel; have = (:theta, :x), want = :total)
        @test_throws ArgumentError native(0.5, [1.0, 2.0])
        if rk_jax_available()
            err = try
                rk_jax_evaluate(src, :kernel, (0.5, [1.0, 2.0]); prepare = true)
                nothing
            catch e
                e
            end
            @test err isa ErrorException && occursin("invalid index: nothing", err.msg)
        end
    end
    @testset "issorted and isless in Julia's order" begin
        jhc_parity(jhc_order_kernel, [([-0.0, 0.0, 1.0, NaN],), ([0.0, -0.0],), ([NaN, 1.0],),
                                      ([1.0, NaN, NaN],), ([2.0, 1.0, 3.0],), ([NaN],)])
    end
    @testset "Set: isequal membership" begin
        jhc_parity(jhc_distinct_kernel, [([1.0, -0.0, 0.0, NaN, NaN, 1.0, 2.0],), ([1.0],)])
    end
    @testset "sizehint!, views and vcat" begin
        jhc_parity(jhc_views_kernel, [xs, ([1.0, 2.0],)])
    end
    @testset "sum(f, tuple; init)" begin
        jhc_parity(jhc_two_kernel, [(JHC_X, JHC_X[1:3])]; have = (:theta, :x, :y))
    end
    @testset "helpers that write their arguments" begin
        jhc_parity(jhc_mutations_kernel, [(JHC_X[1:6],), ([2.0, -1.0, 4.0],)])
    end
    @testset "nested generators, filter and filter!" begin
        jhc_parity(jhc_pairs_kernel, [(JHC_X[1:6],), ([1.0],)])
    end
    @testset "a list on some paths only" begin
        jhc_parity(jhc_branch_list_kernel, [(JHC_X[1:6],), ([1.0, 2.0],), ([1.0, 1.5],)])
    end
    @testset "a Union{Nothing, Tuple} result" begin
        jhc_parity(jhc_union_branch_kernel, [xs, ([1.0],)])
        jhc_parity(jhc_union_forms_kernel, [([3.0, 0.5, 1.0, 2.0],), ([1.0],), (Float64[],)])
    end
end

# What still has no host translation says so.
@traceable function jhc_iterate_set(x)
    s = 0.0
    for v in Set(x)
        s += v
    end
    s
end
@traceable jhc_write_slice(x) = sum(jhc_fill_head!(x[1:2], 1.0))
@traceable jhc_apply_all(f, x) = sum(Float64[f(v) for v in x])
@traceable jhc_capture(x) = jhc_apply_all(v -> v * x[1], x)
@traceable function jhc_shared_grow(x)
    w = zeros(length(x))
    v = w
    push!(v, 1.0)
    length(w) * 1.0
end
for f in (:jhc_iterate_set, :jhc_write_slice, :jhc_capture, :jhc_shared_grow)
    @eval @kernel $(Symbol(f, :_kernel))(theta::Float64, x::Vector{Float64}) = begin
        s = $f(x)
        total = theta * s
        return total
    end
end
@testset "jax_source host prepare_data: refusals name the construct" begin
    for (spec, message) in ((jhc_iterate_set_kernel, "iterating a `Set"),
                            (jhc_write_slice_kernel, "writes its argument"),
                            (jhc_capture_kernel, "anonymous function that reads the locals"),
                            (jhc_shared_grow_kernel, "names too"))
        # The host's reason is part of the module's comment or of the error
        # the JAX functions then raise.
        text = try
            string(jax_source(spec; have = (:theta, :x), want = :total, data = (; x = [1.0, 2.0])))
        catch e
            e isa ReactiveKernels.JAXEmissionError || rethrow()
            sprint(showerror, e)
        end
        @test occursin(message, text)
    end
end

# `isless` and a typed comprehension in the JAX functions too.
@traceable jhc_device_less(theta, a, b) = isless(theta * a, theta * b) ? theta : 2 * theta
@traceable jhc_device_floats(theta, x) = theta * sum(Float64[i for i in 1:3]) + sum(x)
@kernel jhc_device_kernel(theta::Float64, a::Float64, b::Float64, x::Vector{Float64}) = begin
    total = jhc_device_less(theta, a, b) + jhc_device_floats(theta, x)
    return total
end
@testset "jax_source: isless and typed comprehensions in JAX functions" begin
    have = (:theta, :a, :b, :x)
    native = prepare(jhc_device_kernel; have, want = :total)
    src = jax_source(jhc_device_kernel; have, want = :total)
    if rk_jax_available()
        for (a, b) in ((-0.0, 0.0), (0.0, -0.0), (NaN, 1.0), (1.0, NaN), (1.0, 2.0), (2.0, 1.0),
                       (NaN, NaN))
            got = rk_jax_evaluate(src, :kernel, (0.5, a, b, [1.0, 2.0]); grad = true)
            @test got.value == native(0.5, a, b, [1.0, 2.0])
        end
    else
        @test_skip false
    end
end
