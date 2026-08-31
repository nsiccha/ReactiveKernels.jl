# `plate`: generate a Stan-parity vectorized log density from a scalar per-element
# `@kernel`. The defining property is NO REPEATED WORK — recipes that depend only
# on the shared (scalar) ports (here `σ = exp(logσ)`) are hoisted and computed
# ONCE above a fused per-element loop; only the batch-dependent recipes run per
# element; the scalar want is summed. No per-observation vector is materialized.
#
# This is a pure planning/codegen concern: ReactiveKernels PLANS and COMPUTES the
# kernel and does no AD, so nothing here differentiates — the claims are value,
# hoisting (verified in the generated AST), and no per-element materialization.
using ReactiveKernels
using ReactiveKernels: code_expr
using Test

_axis_plate_call(f, x, location, scale) = f(x, location, scale)
_axis_plate_allocated(f, x, location, scale) =
    @allocated f(x, location, scale)
const _AXIS_LOG_CALLS = Ref(0)
const _AXIS_PRODUCT_CALLS = Ref(0)
const _AXIS_ATOMIC_CALLS = Ref(0)
_axis_counted_log(value) = (_AXIS_LOG_CALLS[] += 1; log(value))
_axis_counted_product(left, right) =
    (_AXIS_PRODUCT_CALLS[] += 1; left * right)
_axis_counted_sum(values) = (_AXIS_ATOMIC_CALLS[] += 1; sum(values))
_axis_reference(value, center, width) =
    -0.5 * log(2π) - log(width) - 0.5 * ((value - center) / width)^2

struct _AxisBackendArray <: AbstractVector{Float64}
    values::Vector{Float64}
end
Base.size(marker::_AxisBackendArray) = size(marker.values)
Base.getindex(marker::_AxisBackendArray, index::Int) = marker.values[index]
ReactiveKernels._requires_tensorized_marker(::_AxisBackendArray) = true
ReactiveKernels._batched_call(
        pair::ReactiveKernels._ArrayFunctionPair, ops, args,
        ::_AxisBackendArray) = pair.tensorized(ops, args...)

@testset "plate: graph dependencies determine broadcast-axis frequency" begin
    graph = Graph()
    x = value!(graph, :x, Float64)
    location = value!(graph, :location, Float64)
    scale = value!(graph, :scale, Float64)
    log_scale = value!(graph, :log_scale, Float64)
    residual = value!(graph, :residual, Float64)
    logpdf = value!(graph, :logpdf, Float64)

    add!(graph, scale => log_scale, _axis_counted_log)
    add!(graph, (x, location) => residual, (value, center) -> value - center)
    add!(graph, (residual, scale, log_scale) => logpdf,
         (difference, width, log_width) ->
             -0.5 * log(2π) - log_width - 0.5 * (difference / width)^2)

    selected = plan(graph;
        have = (x, location, scale), want = (logpdf,))
    reducing = ReactiveKernels._prepare_batched(
        selected; batched = (:x, :location, :scale))
    collecting = ReactiveKernels._prepare_batched(
        selected; batched = (:x, :location, :scale), reduce = nothing)

    observations = collect(range(-1.0, 1.0; length = 7))
    locations = [-0.4, 0.2, 0.7]
    scales = [0.8, 1.1, 1.7, 2.2]
    location_grid = reshape(locations, 1, :)
    scale_grid = reshape(scales, 1, 1, :)
    reference = _axis_reference.(observations, location_grid, scale_grid)

    _AXIS_LOG_CALLS[] = 0
    @test reducing(observations, location_grid, scale_grid) ≈ sum(reference)
    @test _AXIS_LOG_CALLS[] == length(scales)

    _AXIS_LOG_CALLS[] = 0
    collected = collecting(observations, location_grid, scale_grid)
    @test collected ≈ reference
    @test axes(collected) == axes(reference)
    @test _AXIS_LOG_CALLS[] == length(scales)

    generated = string(code_expr(reducing))
    @test occursin("CartesianIndices", generated)
    @test occursin("_plate_dependency_changed", generated)
    @test !occursin("similar", generated)

    _axis_plate_call(reducing, observations, location_grid, scale_grid)
    _AXIS_LOG_CALLS[] = 0
    @test _axis_plate_allocated(
        reducing, observations, location_grid, scale_grid) == 0
    @test _AXIS_LOG_CALLS[] == length(scales)

    @testset "aligned dimensions zip and fail before execution" begin
        zipped_graph = Graph()
        left = value!(zipped_graph, :left, Float64)
        right = value!(zipped_graph, :right, Float64)
        product = value!(zipped_graph, :product, Float64)
        add!(zipped_graph, (left, right) => product, _axis_counted_product)
        zipped_plan = plan(zipped_graph;
            have = (left, right), want = (product,))
        zipped = ReactiveKernels._prepare_batched(
            zipped_plan; batched = (:left, :right))

        _AXIS_PRODUCT_CALLS[] = 0
        @test zipped([1.0, 2.0], [3.0, 4.0]) == 11.0
        @test _AXIS_PRODUCT_CALLS[] == 2
        @test zipped([1.0, 2.0], [3.0]) == 9.0
        @test _AXIS_PRODUCT_CALLS[] == 4
        _AXIS_PRODUCT_CALLS[] = 0
        @test_throws DimensionMismatch zipped(
            [1.0, 2.0], [3.0, 4.0, 5.0])
        @test _AXIS_PRODUCT_CALLS[] == 0
    end

    @testset "HAVE cuts remain authoritative" begin
        cut = plan(graph;
            have = (x, location, log_scale, scale), want = (logpdf,))
        cut_kernel = ReactiveKernels._prepare_batched(
            cut; batched = (:x, :location, :log_scale, :scale))
        _AXIS_LOG_CALLS[] = 0
        @test cut_kernel(
            observations, location_grid, log.(scale_grid), scale_grid) ≈
            sum(reference)
        @test _AXIS_LOG_CALLS[] == 0
    end

    @testset "Ref keeps an array-valued input atomic" begin
        atomic_graph = Graph()
        value = value!(atomic_graph, :value, Float64)
        offsets = value!(atomic_graph, :offsets, Vector{Float64})
        offset = value!(atomic_graph, :offset, Float64)
        shifted = value!(atomic_graph, :shifted, Float64)
        add!(atomic_graph, offsets => offset, _axis_counted_sum)
        add!(atomic_graph, (value, offset) => shifted, +)
        atomic_plan = plan(atomic_graph;
            have = (value, offsets), want = (shifted,))
        atomic_kernel = ReactiveKernels._prepare_batched(
            atomic_plan; batched = (:value, :offsets), reduce = nothing)

        values = [1.0, 2.0, 3.0]
        shared_offsets = [0.25, 0.75]
        _AXIS_ATOMIC_CALLS[] = 0
        @test atomic_kernel(values, Ref(shared_offsets)) == values .+ 1.0
        @test _AXIS_ATOMIC_CALLS[] == 1
    end


    @testset "a later traced HAVE selects tensorized lowering" begin
        native = (ops, args...) -> :native
        tensorized = (ops, args...) -> :tensorized
        pair = ReactiveKernels._BatchedFunctionPair{
            1,(:x, :scale),:+,typeof(native),typeof(tensorized)}(
                native, tensorized)
        @test pair((), observations, scale_grid) === :native
        @test pair((), observations, _AxisBackendArray(scales)) ===
              :tensorized
    end
end

# One scalar per-observation Gaussian log density, authored once.
@kernel plate_nlogpdf(x::Float64, μ::Float64, logσ::Float64) = begin
    σ::Float64 = exp(logσ)
    z::Float64 = (x - μ) / σ
    ld::Float64 = -0.5 * log(2π) - logσ - 0.5 * z^2
end

const embedded_plate_nlogpdf = plate(plate_nlogpdf;
    have = (:x, :μ, :logσ), want = :ld, batched = :x)

@kernel embedded_plate_model(x::Vector{Float64},
                             μ::Float64,
                             logσ::Float64) = begin
    total::Float64 = embedded_plate_nlogpdf(x, μ, logσ)
end

const embedded_plate_middle = prepare(embedded_plate_model;
    have = (:x, :μ, :logσ), want = :total)

@kernel embedded_plate_outer(x::Vector{Float64},
                             μ::Float64,
                             logσ::Float64) = begin
    total::Float64 = embedded_plate_middle(x, μ, logσ)
end

@kernel embedded_plate_shifted(x::Vector{Float64},
                               μ::Float64,
                               logσ::Float64) = begin
    total::Float64 = embedded_plate_nlogpdf(x, μ, logσ)
    shifted::Float64 = total + μ
end

# Function barrier so `@allocated` measures the kernel, not global-ref boxing.
_plate_call(k, xs, μ, logσ) = k(xs, μ, logσ)

@testset "plate: vectorized log density with loop-invariant hoisting" begin
    k = plate(plate_nlogpdf; have = (:x, :μ, :logσ), want = :ld, batched = (:x,))
    xs = randn(200)
    μ = 0.3
    logσ = log(1.2)

    @testset "value equals the summed per-observation density" begin
        ref = sum(-0.5 * log(2π) - logσ - 0.5 * ((xi - μ) / exp(logσ))^2 for xi in xs)
        @test k(xs, μ, logσ) ≈ ref
    end

    @testset "prepare splices generated plate code into an outer kernel" begin
        embedded = prepare(embedded_plate_model;
            have = (:x, :μ, :logσ), want = :total)
        @test embedded(xs, μ, logσ) ≈ k(xs, μ, logσ)
        generated = string(code_expr(embedded))
        @test occursin("for ", generated)
        @test !occursin("similar", generated)
        @test !any(op -> op isa ReactiveKernels.PreparedKernel, embedded.ops)
        @test length(embedded.ops) == length(embedded_plate_nlogpdf.ops)
        readable = string(ReactiveKernels._readable_expr(
            code_expr(embedded), embedded))
        @test !occursin(r"__ops__\[\d+\]", readable)
        @test occursin("-0.5", readable)
    end

    @testset "two-level composition preserves native and tensorized products" begin
        outer = prepare(embedded_plate_outer;
            have = (:x, :μ, :logσ), want = :total)
        @test outer(xs, μ, logσ) ≈ k(xs, μ, logσ)
        @test embedded_plate_middle.f isa ReactiveKernels._EmbeddedFunctionPair
        @test outer.f isa ReactiveKernels._EmbeddedFunctionPair
        @test occursin("for ", string(code_expr(outer)))
        @test !occursin("for ", string(outer.f.tensorized_ast))
        @test !any(op -> op isa ReactiveKernels.PreparedKernel, outer.ops)
        @test length(outer.ops) == length(embedded_plate_nlogpdf.ops)
    end

    @testset "public raw Plan lowering retains its outer operation table" begin
        raw_plan = plan(embedded_plate_shifted;
            have = (:x, :μ, :logσ), want = :shifted)
        raw_ast = ReactiveKernels.lower(raw_plan)
        raw_ops = Tuple(recipe.op for recipe in raw_plan.recipes)
        @test length(raw_ops) == 2
        @test first(raw_ops) === embedded_plate_nlogpdf
        @test ReactiveKernels.compile(raw_ast)(raw_ops, xs, μ, logσ) ≈
              k(xs, μ, logσ) + μ
    end

    @testset "invariant `exp(logσ)` is hoisted ABOVE the loop (Stan-parity)" begin
        # `__ops__[1]` is the `σ = exp(logσ)` recipe. It must be emitted once,
        # above the loop, and NOT recomputed per element.
        body = code_expr(k).args[2].args
        fori = findfirst(s -> s isa Expr && s.head === :for, body)
        @test fori !== nothing
        pre = body[1:(fori - 1)]
        loopbody = body[fori].args[2].args
        @test any(s -> occursin("__ops__[1]", string(s)), pre)
        @test !any(s -> occursin("__ops__[1]", string(s)), loopbody)
        # The batch-dependent recipes DO run in the loop.
        @test any(s -> occursin("__ops__[2]", string(s)), loopbody)
        @test any(s -> occursin("__ops__[3]", string(s)), loopbody)
    end

    @testset "no per-element materialization (allocation is O(1) in the batch)" begin
        ys10 = randn(10)
        ys1000 = randn(1000)
        _plate_call(k, ys10, μ, logσ)
        _plate_call(k, ys1000, μ, logσ)
        a10 = @allocated _plate_call(k, ys10, μ, logσ)
        a1000 = @allocated _plate_call(k, ys1000, μ, logσ)
        # Nothing scales with N — no per-observation vector is built.
        @test a10 == a1000
    end

    @testset "a want with no batched dependency is rejected" begin
        # `σ` depends only on the shared `logσ` — there is nothing to vectorize.
        @test_throws ArgumentError plate(
            plate_nlogpdf; have = (:x, :μ, :logσ), want = :σ, batched = (:x,))
    end

    @testset "MULTIPLE batched ports — normal_lpdf(vec, vec, scalar)" begin
        # The canonical Stan-parity case: both the observations `x` and the
        # per-observation means `μ` are batched (e.g. a regression mean), while
        # the scale is shared. `exp(logσ)`/`log(σ)` must still be computed ONCE.
        km = plate(plate_nlogpdf; have = (:x, :μ, :logσ), want = :ld,
                   batched = (:x, :μ))
        xs = randn(150)
        μs = randn(150)
        logσ = log(0.9)

        @test km(xs, μs, logσ) ≈
            sum(-0.5 * log(2π) - logσ - 0.5 * ((xs[i] - μs[i]) / exp(logσ))^2
                for i in eachindex(xs))

        body = code_expr(km).args[2].args
        fori = findfirst(s -> s isa Expr && s.head === :for, body)
        pre = body[1:(fori - 1)]
        loopbody = body[fori].args[2].args
        # The shared-scale recipe is hoisted once; both batched ports are indexed.
        @test any(s -> occursin("__ops__[1]", string(s)), pre)
        @test !any(s -> occursin("__ops__[1]", string(s)), loopbody)
    end

    @testset "collect mode (reduce=nothing) — per-observation vector, hoisted" begin
        # For LOO/WAIC: the per-observation densities as a vector, sharing the
        # hoisted invariants; only the output vector is materialized.
        kv = plate(plate_nlogpdf; have = (:x, :μ, :logσ), want = :ld,
                   batched = (:x,), reduce = nothing)
        xs = randn(64)
        μ = 0.3
        logσ = log(1.1)

        got = kv(xs, μ, logσ)
        ref = [-0.5 * log(2π) - logσ - 0.5 * ((xi - μ) / exp(logσ))^2 for xi in xs]
        @test got ≈ ref
        @test length(got) == length(xs)

        # Its sum equals the reducing-mode total (both hoist `exp(logσ)` once).
        ks = plate(plate_nlogpdf; have = (:x, :μ, :logσ), want = :ld, batched = (:x,))
        @test sum(got) ≈ ks(xs, μ, logσ)

        body = code_expr(kv).args[2].args
        fori = findfirst(s -> s isa Expr && s.head === :for, body)
        @test any(s -> occursin("__ops__[1]", string(s)), body[1:(fori - 1)])
        @test !any(s -> occursin("__ops__[1]", string(s)), body[fori].args[2].args)
    end
end
