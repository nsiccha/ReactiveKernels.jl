using ReactiveKernels
using Test

@kernel authored_standard_normal() = begin
    logpdf(z::Float64)::Float64 = -0.5 * log(2π) - 0.5 * z^2
end

@kernel authored_standard_cauchy() = begin
    logpdf(z::Float64)::Float64 = -log(π) - log1p(z^2)
end

@kernel authored_location_scale(standard, location::Float64, scale::Float64) = begin
    log_scale::Float64 = log(scale)
    scale::Float64 = exp(log_scale)
    standardized(x::Float64)::Float64 = (x - location) / scale

    logpdf(x::Float64)::Float64 = begin
        z::Float64 = standardized(x)
        standard.logpdf(z) - log_scale
    end
end


@kernel authored_normal = authored_location_scale(authored_standard_normal)
@kernel authored_cauchy = authored_location_scale(authored_standard_cauchy)

@kernel authored_vector_location(location::Vector{Float64}) = begin
    logpdf(x::Vector{Float64})::Float64 =
        -0.5 * sum(abs2, x .- location)
end

@kernel authored_normal_loglik(x::Vector{Float64}, location, scale) = begin
    pointwise = plate(x, location, scale) do xi, li, si
        authored_normal(li, si).logpdf(xi)
    end
    return sum(pointwise)
end

@kernel authored_normal_both_loglik(x, location, scale, log_scale) = begin
    pointwise = plate(x, location, scale, log_scale) do xi, li, si, log_si
        authored_normal(;
            location = li, scale = si, log_scale = log_si).logpdf(xi)
    end
    return sum(pointwise)
end

@kernel authored_eight_schools_prior(
        θ::Vector{Float64}, μ::Float64, τ::Float64, log_τ::Float64) = begin
    μ_prior::Float64 = authored_normal(0.0, 5.0).logpdf(μ)
    τ_cauchy::Float64 = authored_cauchy(;
        location = 0.0, scale = τ, log_scale = log_τ).logpdf(τ)
    effects_pointwise = plate(θ, μ, τ, log_τ) do θj, μj, τj, log_τj
        authored_normal(;
            location = μj, scale = τj, log_scale = log_τj).logpdf(θj)
    end
    prior::Float64 = μ_prior + τ_cauchy + sum(effects_pointwise)
    return prior
end

@kernel untyped_authored_normal_loglik(x, location, scale) = begin
    pointwise = plate(x, location, scale) do xi, li, si
        authored_normal(li, si).logpdf(xi)
    end
    return sum(pointwise)
end

@kernel authored_cauchy_loglik(x::Vector{Float64}, location, scale) = begin
    pointwise = plate(x, location, scale) do xi, li, si
        authored_cauchy(li, si).logpdf(xi)
    end
    return sum(pointwise)
end

@kernel authored_vector_loglik(
        x::Vector{Vector{Float64}}, location::Vector{Float64}) = begin
    pointwise = plate(x, Ref(location)) do xi, li
        authored_vector_location(li).logpdf(xi)
    end
    return sum(pointwise)
end


@kernel untyped_authored_vector_loglik(x, location) = begin
    pointwise = plate(x, Ref(location)) do xi, li
        authored_vector_location(li).logpdf(xi)
    end
    return sum(pointwise)
end

_authored_plate_normal(x, location, scale) =
    -0.5 * log(2π) - log(scale) - 0.5 * ((x - location) / scale)^2
_authored_plate_cauchy(x, location, scale) =
    -log(π) - log(scale) - log1p(((x - location) / scale)^2)

function _authored_plate_allocated(kernel, a, b, c)
    kernel(a, b, c)
    @allocated kernel(a, b, c)
end

struct _AuthoredPlateBackendArray <: AbstractVector{Float64}
    values::Vector{Float64}
end
Base.size(marker::_AuthoredPlateBackendArray) = size(marker.values)
Base.getindex(marker::_AuthoredPlateBackendArray, index::Int) =
    marker.values[index]
ReactiveKernels._requires_tensorized_marker(
    ::_AuthoredPlateBackendArray) = true
ReactiveKernels._batched_call(
        pair::ReactiveKernels._ArrayFunctionPair, ops, args,
        ::_AuthoredPlateBackendArray) = pair.tensorized(ops, args...)

function _authored_plate_head_count(node, head)
    node isa Expr || return 0
    (node.head === head ? 1 : 0) +
        sum(_authored_plate_head_count(child, head) for child in node.args)
end

@testset "authored plate block: transparent distribution log-likelihood" begin
    xs = [-1.2, -0.1, 0.7, 1.8]
    location = 0.3
    scale = 1.2

    total_plan = plan(authored_normal_loglik)
    pointwise_spec = extract(authored_normal_loglik; want = :pointwise)
    both_spec = extract(authored_normal_loglik;
                        want = (:pointwise, :__return__))
    pointwise_plan = plan(pointwise_spec)
    both_plan = plan(both_spec)

    @test keys(authored_normal_loglik) ==
        (:x, :location, :scale, :pointwise, :__return__)
    @test only(outputs(authored_normal_loglik)).name === :__return__
    @test length(total_plan.recipes) == 2
    @test length(pointwise_plan.recipes) == 1
    @test length(both_plan.recipes) == 2
    @test count(recipe -> recipe.source isa Expr && recipe.source.head === :do,
                authored_normal_loglik.graph.recipes) == 1
    @test count(recipe -> recipe.source == :(sum(pointwise)),
                authored_normal_loglik.graph.recipes) == 1

    plate_recipe = first(total_plan.recipes)
    scalar_plan = plate_body(plate_recipe)
    scalar_names = Symbol[output.name for recipe in scalar_plan.recipes
                          for output in recipe.outputs]
    @test :log_scale in scalar_names
    @test :standardized in scalar_names
    @test Symbol("standard.logpdf") in scalar_names
    @test :logpdf in scalar_names
    @test occursin("standard.logpdf", dot_source(total_plan))

    total = prepare(authored_normal_loglik)
    pointwise = prepare(pointwise_spec)
    both = prepare(both_spec)
    reference = [_authored_plate_normal(x, location, scale) for x in xs]

    untyped_total = prepare(untyped_authored_normal_loglik)
    @test untyped_total.f isa ReactiveKernels._DynamicEmbeddedFunctionPair
    @test typeof(untyped_total.f).parameters[1] == (1, 2, 3)
    @test untyped_total(xs, location, scale) ≈ sum(reference)
    @test _authored_plate_allocated(
        untyped_total, xs, location, scale) == 0
    @test !occursin("for ", string(untyped_total.f.tensorized_ast))

    # Positional KernelSpec application requests the distinguished return.
    @test authored_normal_loglik(xs, location, scale) ≈ sum(reference)
    @test total(xs, location, scale) ≈ sum(reference)
    @test pointwise(xs, location, scale) ≈ reference
    @test both(xs, location, scale) == (pointwise(xs, location, scale),
                                       total(xs, location, scale))

    raw_ast = lower(total_plan)
    raw_ops = Tuple(recipe.op for recipe in total_plan.recipes)
    @test ReactiveKernels.compile(raw_ast)(raw_ops, xs, location, scale) ≈
        sum(reference)

    total_ast = code_expr(total)
    pointwise_ast = code_expr(pointwise)
    both_ast = code_expr(both)
    @test _authored_plate_head_count(total_ast, :for) == 1
    @test _authored_plate_head_count(pointwise_ast, :for) == 1
    @test _authored_plate_head_count(both_ast, :for) == 1
    @test !occursin("similar", string(total_ast))
    @test occursin("similar", string(pointwise_ast))
    @test occursin("similar", string(both_ast))
    @test !occursin("for ", string(total.f.tensorized_ast))
    @test occursin("broadcast", string(total.f.tensorized_ast))
    native_text = string(total_ast)
    @test findfirst("_authored_plate_broadcast", native_text) <
          findfirst("__ops__[", native_text)
    for materializing_ast in (pointwise_ast, both_ast)
        materializing_text = string(materializing_ast)
        @test findfirst("_authored_plate_broadcast", materializing_text) <
              findfirst("similar", materializing_text)
        @test findfirst("_authored_plate_broadcast", materializing_text) <
              findfirst("__ops__[", materializing_text)
    end

    readable = string(ReactiveKernels._readable_expr(total_ast, total))
    @test occursin("standard.logpdf", readable)
    @test !occursin(r"__ops__\[\d+\]", readable)

    locations = [0.1, 0.2, 0.4, 0.5]
    scales = [0.8, 1.0, 1.3, 1.5]
    @test untyped_total(0.25, locations, scale) ≈ sum(
        _authored_plate_normal(0.25, item, scale) for item in locations)

    # A native array in an earlier candidate position must not mask a later
    # backend-traced array (or traced scalar) that requires tensorized lowering.
    native_call = (ops, args...) -> :native
    tensorized_call = (ops, args...) -> :tensorized
    pair = ReactiveKernels._DynamicEmbeddedFunctionPair{
        (1, 2),typeof(native_call),typeof(tensorized_call),Expr}(
            native_call, tensorized_call, Expr(:block))
    typed_pair = ReactiveKernels._EmbeddedFunctionPair{
        1,typeof(native_call),typeof(tensorized_call),Expr}(
            native_call, tensorized_call, Expr(:block))
    backend_locations = _AuthoredPlateBackendArray(locations)
    @test pair((), xs, backend_locations) === :tensorized
    @test pair((), xs, locations) === :native
    @test typed_pair((), xs, backend_locations) === :tensorized
    @test typed_pair((), xs, locations) === :native
    zipped_reference = [
        _authored_plate_normal(xs[i], locations[i], scales[i])
        for i in eachindex(xs)
    ]
    @test total(xs, locations, scales) ≈ sum(zipped_reference)
    @test pointwise(xs, locations, scales) ≈ zipped_reference
    @test _authored_plate_allocated(total, xs, locations, scales) == 0
    @test total(xs, locations, scale) ≈
        sum(_authored_plate_normal(xs[i], locations[i], scale)
            for i in eachindex(xs))

    both_have = prepare(authored_normal_both_loglik)
    @test both_have(xs, location, scale, log(scale)) ≈ sum(reference)
    both_have_scalar = plate_body(first(plan(authored_normal_both_loglik).recipes))
    @test [only(recipe.outputs).name for recipe in both_have_scalar.recipes] ==
          [:standardized, Symbol("standard.logpdf"), :logpdf]
    @test !occursin("KernelObjectSpec", string(code_expr(both_have)))

    θ = [0.25 * index for index in 1:8]
    μ = 1.5
    τ = 2.0
    log_τ = log(τ)
    eight_schools_prior = prepare(authored_eight_schools_prior)
    expected_prior =
        _authored_plate_normal(μ, 0.0, 5.0) +
        _authored_plate_cauchy(τ, 0.0, τ) +
        sum(_authored_plate_normal(value, μ, τ) for value in θ)
    @test eight_schools_prior(θ, μ, τ, log_τ) ≈ expected_prior
    eight_schools_plan = plan(authored_eight_schools_prior)
    @test any(recipe -> recipe.source == 0.0, eight_schools_plan.recipes)
    @test any(recipe -> recipe.source == 5.0, eight_schools_plan.recipes)
    @test all(!(recipe.op isa PreparedKernel) for recipe in eight_schools_plan.recipes)
    @test !occursin("KernelObjectSpec", string(code_expr(eight_schools_prior)))

    # Julia broadcast semantics include singleton expansion.
    @test pointwise(xs, locations[1:1], scale) ≈
        [_authored_plate_normal(x, only(locations[1:1]), scale) for x in xs]

    location_grid = reshape(locations[1:3], 1, :)
    scale_grid = reshape(scales[1:2], 1, 1, :)
    grid_reference = _authored_plate_normal.(xs, location_grid, scale_grid)
    @test pointwise(xs, location_grid, scale_grid) ≈ grid_reference
    @test total(xs, location_grid, scale_grid) ≈ sum(grid_reference)

    @test_throws DimensionMismatch total(xs, locations[1:3], scales)
    @test_throws DimensionMismatch total(xs, [locations; 0.6], scales)
end

@testset "authored plate block: second transparent object endpoint" begin
    xs = [-0.8, 0.2, 1.1]
    location = -0.1
    scale = 0.9
    reference = [_authored_plate_cauchy(x, location, scale) for x in xs]
    @test authored_cauchy_loglik(xs, location, scale) ≈ sum(reference)
    @test prepare(extract(authored_cauchy_loglik; want = :pointwise))(
        xs, location, scale) ≈ reference
end


@testset "authored plate block: Ref marks an array-valued atom" begin
    observations = [[1.0, -0.5], [0.3, 0.8], [-0.4, 1.2]]
    location = [0.2, -0.1]
    reference = [-0.5 * sum(abs2, x .- location) for x in observations]

    pointwise = prepare(extract(authored_vector_loglik; want = :pointwise))
    total = prepare(authored_vector_loglik)
    untyped_total = prepare(untyped_authored_vector_loglik)
    @test pointwise(observations, location) ≈ reference
    @test total(observations, location) ≈ sum(reference)
    @test untyped_total(observations, location) ≈ sum(reference)
    @test typeof(untyped_total.f).parameters[1] == (1,)
    @test _authored_plate_head_count(code_expr(total), :for) == 1
    @test !occursin("similar", string(code_expr(total)))
end
