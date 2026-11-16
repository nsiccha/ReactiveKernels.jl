using ReactiveKernels, Test, Enzyme

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

@traceable function keyword_sum(xs::AbstractVector{T}; scale::T = one(T),
                              shift = scale / 2, count::Integer = length(xs)) where {T}
    total = zero(scale)
    for i in 1:count
        total += scale * xs[i] + shift
    end
    total
end
@traceable function keyword_required(xs; scale::Real, positive::Bool = false)
    total = 0.0
    for i in eachindex(xs)
        x = xs[i]
        if !positive || x > 0
            total += scale * x
        end
    end
    total
end
@traceable keyword_nested(xs; scale = 2.0) =
    keyword_sum(xs; scale) + keyword_required(xs; scale, positive = true)
keyword_plain(xs, scale) = keyword_nested(xs; scale)
@traceable keyword_lazy(x; scale = error("unused keyword default")) = x * scale
@traceable keyword_rest(xs...; scale = 2.0) = scale * sum(length, xs; init = 0)
@traceable keyword_positional(x = 2.0; scale = x + 1.0) = x * scale
@traceable keyword_positional_rest(x = 2.0, xs...; scale = 3.0) = x * scale + sum(identity, xs; init = 0.0)
@traceable keyword_dispatch(x::Real; scale = 2.0) = x * scale
keyword_dispatch(x::Float64) = x + 10.0
@traceable keyword_redispatch(x::Real = 2; scale = 3.0) = x * scale
@traceable keyword_redispatch(x::Int; scale = 4.0) = x + scale
@traceable keyword_arity(x; scale = 2.0) = x * scale
@traceable keyword_arity(x, y) = x + y
@traceable keyword_callable(x; transform::Function = abs) = transform(x)
@kernel keyword_defaults(x::Float64) = begin
    total = keyword_positional(; scale = x) + keyword_positional() +
        keyword_dispatch(x) + keyword_dispatch(x; scale = 3.0) +
        keyword_redispatch() + keyword_redispatch(; scale = x) +
        keyword_arity(x, 2.0) + keyword_arity(x; scale = 2.0) +
        keyword_callable(x; transform = sin) + keyword_callable(-x) +
        keyword_positional_rest(; scale = x) +
        keyword_positional_rest(x, 1.0; scale = 2.0)
    return total
end

@kernel keyword_host(a::Float64, xs::Vector{Float64}, scale::Float64) = begin
    first = keyword_plain(xs, scale) + keyword_callable(scale; transform = sin)
    second = keyword_sum(xs)
    third = keyword_sum(xs; shift = 3.0, scale = 4.0)
    total = a * (first + second + third)
    return total
end
@kernel keyword_device(scale::Float64, xs::Vector{Float64}) = begin
    first = keyword_sum(xs; scale)
    second = keyword_required(xs; scale, positive = true)
    total = first + second + keyword_lazy(scale; scale = 3.0) + keyword_rest(xs, xs; scale)
    return total
end
@kernel keyword_ragged(scale::Float64, rows::Vector{Vector{Float64}}) = begin
    cells = plate(rows) do row
        keyword_sum(row; scale)
    end
    total = sum(cells)
    return total
end
@traceable function keyword_checked(x; scale = 2.0)
    x > 0 || throw(DomainError(x, "positive input required"))
    scale * x
end
@traceable keyword_checked_nested(x; scale = 3.0) = keyword_checked(x; scale)
@kernel keyword_errors(x::Float64) = begin
    result = keyword_checked_nested(x; scale = 4.0)
    return result
end
@kernel keyword_missing(xs::Vector{Float64}) = begin
    total = keyword_required(xs)
    return total
end
@kernel keyword_bad_type(xs::Vector{Float64}) = begin
    total = keyword_sum(xs; count = 1.5)
    return total
end
@kernel keyword_unknown(xs::Vector{Float64}) = begin
    total = keyword_sum(xs; unknown = 2)
    return total
end

module KeywordDefaultScope
using ReactiveKernels
const default_factor = 2.5
@traceable function optional_sum(xs; indices::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                                offsets::AbstractVector{<:Real} = Float64[],
                                scale = default_factor)
    total = 0.0
    if indices === nothing
        for x in xs
            total += scale * x
        end
    else
        for i in indices
            total += scale * xs[i]
        end
    end
    for x in offsets
        total += x
    end
    total
end
end
@kernel keyword_optional(a::Float64, xs::Vector{Float64}, indices::Vector{Int}) = begin
    first = KeywordDefaultScope.optional_sum(xs)
    second = KeywordDefaultScope.optional_sum(xs; indices, offsets = xs)
    total = a * (first + second)
    return total
end

@testset "@traceable keyword methods retain Julia behavior" begin
    xs = [0.2, -0.3, 1.1]
    @test @inferred(keyword_sum(xs)) ≈ sum(xs) + length(xs) / 2
    @test @inferred(keyword_sum(xs; scale = 2.0)) ≈ 2sum(xs) + length(xs)
    @test @inferred(keyword_sum(xs; shift = 0.0, scale = 3.0)) ≈ 3sum(xs)
    @test @inferred(keyword_required(xs; scale = 2.0, positive = true)) ≈ 2.6
    @test keyword_rest(xs, Float64[]; scale = 4.0) == 12.0
    @test keyword_lazy(2.0; scale = 3.0) == 6.0
    @test keyword_positional() == 6.0
    @test keyword_positional(; scale = 4.0) == 8.0
    @test keyword_positional_rest(; scale = 4.0) == 8.0
    @test_throws UndefKeywordError keyword_required(xs)
    @test_throws TypeError keyword_sum(xs; count = 1.5)
    @test_throws MethodError keyword_sum(xs; unknown = 2)
    normal = prepare(keyword_errors; have = :x, want = :result)
    ignored = prepare(keyword_errors; have = :x, want = :result, on_error = :ignore)
    @test_throws DomainError normal(-2.0)
    @test ignored(-2.0) == -8.0
    # Emission must preserve the same required-name and type obligations as
    # ordinary Julia calls above; it cannot silently reinterpret an invalid call.
    @test_throws JAXEmissionError jax_source(keyword_missing; have = :xs, want = :total)
    @test_throws JAXEmissionError jax_source(keyword_bad_type; have = :xs, want = :total)
    @test_throws JAXEmissionError jax_source(keyword_unknown; have = :xs, want = :total)
end

@testset "jax_source binds keyword helpers on host and device" begin
    lengths = (0, 1, 7, 37)
    host_sources, device_sources = String[], String[]
    native_host = prepare(keyword_host; have = (:a, :xs, :scale), want = :total)
    native_device = prepare(keyword_device; have = (:scale, :xs), want = :total)
    for n in lengths
        xs = [(-1.0)^i * i / 10 for i in 1:n]
        scale = 0.7 + n / 20
        host = jax_source(keyword_host; have = (:a, :xs, :scale), want = :total,
                          data = (; xs, scale))
        device = jax_source(keyword_device; have = (:scale, :xs), want = :total)
        push!(host_sources, string(host))
        push!(device_sources, string(device))
        @test occursin("keyword_sum_np(", string(host))
        @test occursin(r"\bfor\b", string(host))
        @test occursin("fori_loop", string(device))
        if rk_jax_available()
            original = copy(xs)
            h = rk_jax_evaluate(host, :kernel, (1.3, xs, scale); grad = true, prepare = :arguments)
            d = rk_jax_evaluate(device, :kernel, (scale, xs); grad = true)
            @test h.value ≈ native_host(1.3, xs, scale)
            @test d.value ≈ native_device(scale, xs)
            hg = Enzyme.autodiff(Enzyme.Reverse, native_host, Enzyme.Active,
                Enzyme.Active(1.3), Enzyme.Const(xs), Enzyme.Const(scale))[1][1]
            dg = Enzyme.autodiff(Enzyme.Reverse, native_device, Enzyme.Active,
                Enzyme.Active(scale), Enzyme.Const(xs))[1][1]
            @test only(h.grad) ≈ hg
            @test only(d.grad) ≈ dg
            @test xs == original
        end
    end
    # CONTRIBUTING.md: data lengths and runtime keyword values must not
    # replicate loop bodies or specialize the emitted source to one dataset.
    @test all(==(first(host_sources)), host_sources)
    @test all(==(first(device_sources)), device_sources)
    if rk_jax_available()
        defaults = jax_source(keyword_defaults; have = :x, want = :total)
        result = rk_jax_evaluate(defaults, :kernel, (1.7,); grad = true)
        @test result.value ≈ prepare(keyword_defaults)(1.7)
        @test only(result.grad) ≈ 15.0 + cos(1.7)
        for xs in (Float64[], [0.2, -0.5, 1.1])
            indices = collect(eachindex(xs))
            source = jax_source(keyword_optional; have = (:a, :xs, :indices),
                                want = :total, data = (:xs, :indices))
            result = rk_jax_evaluate(source, :kernel, (1.3, xs, indices);
                                     grad = true, prepare = :arguments)
            native = prepare(keyword_optional)
            @test result.value ≈ native(1.3, xs, indices)
            @test only(result.grad) ≈ native(1.0, xs, indices)
        end
        rows = [Float64[], [0.2], [0.1, -0.2, 0.5]]
        source = jax_source(keyword_ragged; have = (:scale, :rows), want = :total, data = (:rows,))
        native = prepare(keyword_ragged; have = (:scale, :rows), want = :total)
        result = rk_jax_evaluate(source, :kernel, (2.0, rows); grad = true, prepare = :arguments)
        @test result.value ≈ native(2.0, rows)
        @test only(result.grad) ≈ sum(row -> sum(row) + length(row) / 2, rows)
    else
        @test_skip false
    end
end
