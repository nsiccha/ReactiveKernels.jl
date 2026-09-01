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

const _AUTHORED_PLATE_SCALE_CALLS = Ref(0)
_authored_plate_counted_log(scale) =
    (_AUTHORED_PLATE_SCALE_CALLS[] += 1; log(scale))

# This helper is a transparent RK graph. Its leaf operation is an ordinary
# opaque Julia callable, which carries the normal RK pure-recipe contract.
@kernel authored_counted_scale(scale) = begin
    log_scale::Float64 = _authored_plate_counted_log(scale)
    return log_scale
end

@kernel authored_axis_loglik(x, location, scale) = begin
    pointwise = plate(x, location, scale) do xi, li, si
        log_scale::Float64 = authored_counted_scale(si)
        residual::Float64 = xi - li
        logpdf::Float64 =
            -0.5 * log(2π) - log_scale - 0.5 * (residual / si)^2
        return logpdf
    end
    return sum(pointwise)
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

@kernel authored_namedtuple_eight_schools_prior(parameters) = begin
    μ::Float64 = parameters.μ
    τ::Float64 = parameters.τ
    θ::AbstractVector{Float64} = parameters.θ
    log_τ::Float64 = log(τ)
    μ_prior::Float64 = authored_normal(0.0, 5.0).logpdf(μ)
    τ_cauchy::Float64 = authored_cauchy(0.0, 5.0).logpdf(τ)
    τ_prior::Float64 = log(2.0) + τ_cauchy
    effects_pointwise = plate(θ, μ, τ, log_τ) do θj, μj, τj, log_τj
        authored_normal(;
            location = μj, scale = τj, log_scale = log_τj).logpdf(θj)
    end
    prior::Float64 = μ_prior + τ_prior + sum(effects_pointwise)
    return prior
end

@kernel authored_namedtuple_observed_loglik(
        parameters, observations, observation_scales) = begin
    μ::Float64 = parameters.μ
    pointwise = plate(μ, observations, observation_scales) do μj, xj, sj
        authored_normal(μj, sj).logpdf(xj)
    end
    return sum(pointwise)
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

@kernel authored_eachcol_sum(matrix, offsets) = begin
    pointwise = plate(eachcol(matrix), offsets) do column, offset
        value::Float64 = sum(column) + offset
        return value
    end
    return sum(pointwise)
end

@kernel authored_eachrow_derived_sum(matrix, offsets) = begin
    pointwise = plate(eachrow(matrix), offsets .+ 1.0) do row, offset
        value::Float64 = sum(row) + offset
        return value
    end
    return sum(pointwise)
end

@kernel authored_ref_derived_sum(x, offsets) = begin
    pointwise = plate(x, Ref(offsets .+ 1.0)) do value, atomic_offsets
        result::Float64 = value + sum(atomic_offsets)
        return result
    end
    return sum(pointwise)
end

_authored_logsumexp(values) = log(sum(exp, values))

@kernel authored_categorical_logit(logits::AbstractVector{Float64}) = begin
    logpdf(observed::Int)::Float64 =
        logits[observed] - _authored_logsumexp(logits)
end

@kernel authored_eachcol_categorical(logits, observed) = begin
    pointwise = plate(eachcol(logits), observed) do column, value
        authored_categorical_logit(column).logpdf(value)
    end
    return sum(pointwise)
end

@kernel authored_vcat_normal(W, b) = begin
    pointwise = plate(vcat(vec(W), b)) do coefficient
        authored_normal(0.0, 1.0).logpdf(coefficient)
    end
    return sum(pointwise)
end

@kernel authored_implicit_typed_plate(x::Vector{Float64}) = begin
    terms = plate(x) do value
        squared::Float64 = value * value
        squared
    end
    total::Float64 = sum(terms)
end

@kernel authored_implicit_untyped_plate(x::Vector{Float64}) = begin
    terms = plate(x) do value
        squared = value * value
        squared
    end
    total::Float64 = sum(terms)
end

@kernel authored_implicit_assignment_plate(x::Vector{Float64}) = begin
    terms = plate(x) do value
        squared::Float64 = value * value
    end
    total::Float64 = sum(terms)
end

_authored_plate_normal(x, location, scale) =
    -0.5 * log(2π) - log(scale) - 0.5 * ((x - location) / scale)^2
_authored_plate_cauchy(x, location, scale) =
    -log(π) - log(scale) - log1p(((x - location) / scale)^2)

function _authored_plate_allocated(kernel, a, b, c)
    kernel(a, b, c)
    @allocated kernel(a, b, c)
end

_authored_plate_steady_allocated(kernel, a, b, c) =
    @allocated kernel(a, b, c)

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

@testset "authored plate block: implicit multi-statement result" begin
    values = [1.0, 2.0, 3.0]
    expected = values .^ 2

    for spec in (authored_implicit_typed_plate,
                 authored_implicit_untyped_plate,
                 authored_implicit_assignment_plate)
        scalar_plan = plate_body(first(plan(spec).recipes))
        @test [value.name for value in scalar_plan.have] == [:value]
        @test [value.name for value in scalar_plan.want] == [:squared]
        @test [only(recipe.outputs).name for recipe in scalar_plan.recipes] ==
              [:squared]

        total = prepare(spec; have = (:x,), want = :total)
        pointwise = prepare(extract(spec; have = (:x,), want = :terms))
        @test total(values) == sum(expected)
        @test pointwise(values) == expected
    end
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

    parameters = (; μ = location, τ = scale, θ = xs)
    constrained_prior = prepare(
        authored_namedtuple_eight_schools_prior;
        have = :parameters, want = :prior)
    constrained_reference =
        _authored_plate_normal(location, 0.0, 5.0) +
        log(2.0) + _authored_plate_cauchy(scale, 0.0, 5.0) +
        sum(_authored_plate_normal(value, location, scale) for value in xs)
    @test constrained_prior.f isa
          ReactiveKernels._DynamicEmbeddedFunctionPair
    @test typeof(constrained_prior.f).parameters[1] == (1,)
    @test constrained_prior(parameters) ≈ constrained_reference

    scalar_parameters = (; μ = location)
    observed_scales = fill(scale, length(xs))
    observed_total = prepare(authored_namedtuple_observed_loglik)
    observed_pointwise = prepare(extract(
        authored_namedtuple_observed_loglik; want = :pointwise))
    @test typeof(observed_total.f).parameters[1] == (1, 2, 3)
    @test observed_total(scalar_parameters, xs, observed_scales) ≈ sum(
        _authored_plate_normal(xs[i], location, observed_scales[i])
        for i in eachindex(xs))
    observed_pointwise_text = string(code_expr(observed_pointwise))
    @test findfirst("Base.broadcastable", observed_pointwise_text) <
          findfirst("similar", observed_pointwise_text)
    @test_throws DimensionMismatch observed_pointwise(
        scalar_parameters, xs, observed_scales[1:3])

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
    @test findfirst("Base.broadcastable", native_text) <
          findfirst("__ops__[", native_text)
    for materializing_ast in (pointwise_ast, both_ast)
        materializing_text = string(materializing_ast)
        @test findfirst("Base.broadcastable", materializing_text) <
              findfirst("similar", materializing_text)
        @test findfirst("Base.broadcastable", materializing_text) <
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
    single_candidate_pair = ReactiveKernels._DynamicEmbeddedFunctionPair{
        (1,),typeof(native_call),typeof(tensorized_call),Expr}(
            native_call, tensorized_call, Expr(:block))
    typed_pair = ReactiveKernels._EmbeddedFunctionPair{
        1,typeof(native_call),typeof(tensorized_call),Expr}(
            native_call, tensorized_call, Expr(:block))
    backend_locations = _AuthoredPlateBackendArray(locations)
    @test pair((), xs, backend_locations) === :tensorized
    @test pair((), xs, locations) === :native
    @test pair((), (; μ = location, τ = scale), locations) === :native
    @test pair((), Dict(:μ => location, :τ => scale), locations) === :native
    @test pair((), (; μ = location, τ = scale), backend_locations) ===
          :tensorized
    @test pair((), (; θ = xs), locations) === :native
    @test pair((), (; θ = backend_locations), locations) === :tensorized
    @test single_candidate_pair((), (; μ = location, τ = scale)) === :native
    @test single_candidate_pair(
        (), (; μ = location, τ = scale, θ = backend_locations)) ===
          :tensorized
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

@testset "authored plate block: graph-derived broadcast scheduling" begin
    observations = collect(range(-1.0, 1.0; length = 7))
    locations = reshape([-0.4, 0.2, 0.7], 1, :)
    scales = reshape([0.8, 1.1, 1.7, 2.2], 1, 1, :)
    reference = _authored_plate_normal.(observations, locations, scales)

    total = prepare(authored_axis_loglik)
    pointwise = prepare(extract(authored_axis_loglik; want = :pointwise))
    both = prepare(extract(
        authored_axis_loglik; want = (:pointwise, :__return__)))

    scalar_plan = plate_body(first(plan(authored_axis_loglik).recipes))
    scalar_names = [only(recipe.outputs).name for recipe in scalar_plan.recipes]
    @test :log_scale in scalar_names
    @test all(!(recipe.op isa PreparedKernel) for recipe in scalar_plan.recipes)

    # Julia's instantiated broadcast dimensions are the logical axes. The
    # transparent dependency graph proves that `log(scale)` depends only on
    # the third one, so it runs once per scale coordinate, not once per cell.
    for (kernel, expected) in (
            (total, sum(reference)),
            (pointwise, reference))
        _AUTHORED_PLATE_SCALE_CALLS[] = 0
        @test kernel(observations, locations, scales) ≈ expected
        @test _AUTHORED_PLATE_SCALE_CALLS[] == length(scales)
    end
    _AUTHORED_PLATE_SCALE_CALLS[] = 0
    both_result = both(observations, locations, scales)
    @test first(both_result) ≈ reference
    @test last(both_result) ≈ sum(reference)
    @test _AUTHORED_PLATE_SCALE_CALLS[] == length(scales)

    generated = string(code_expr(total))
    @test occursin("_plate_dependency_changed", generated)
    @test _authored_plate_head_count(code_expr(total), :for) == 1
    @test !occursin("similar", generated)
    @test !occursin("for ", string(total.f.tensorized_ast))

    total(observations, locations, scales)
    _AUTHORED_PLATE_SCALE_CALLS[] = 0
    @test _authored_plate_steady_allocated(
        total, observations, locations, scales) == 0
    @test _AUTHORED_PLATE_SCALE_CALLS[] == length(scales)

    # Broadcast compatibility is established before any pure recipe executes.
    _AUTHORED_PLATE_SCALE_CALLS[] = 0
    @test_throws DimensionMismatch total(
        observations, vec(locations), scales)
    @test _AUTHORED_PLATE_SCALE_CALLS[] == 0

    # A plate is a pure map/reduction contract. Ordinary opaque leaf callables
    # are pure by default; a recipe explicitly known to be effectful is not a
    # plate execution mode and cannot be selected into the scalar plan.
    @test_throws PlanningError (@kernel begin
        scale
        pointwise = plate(scale) do si
            @recipe (effectful = true) y = si + 1
            return y
        end
        return sum(pointwise)
    end)
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

@testset "authored plate block: derived iterable arguments" begin
    matrix = [1.0 2.0 3.0; 4.0 5.0 6.0]
    column_offsets = [0.25, 0.5, 0.75]
    column_reference = [
        sum(column) + column_offsets[index]
        for (index, column) in enumerate(eachcol(matrix))
    ]

    column_total = prepare(authored_eachcol_sum)
    column_pointwise = prepare(extract(authored_eachcol_sum; want = :pointwise))
    @test column_total(matrix, column_offsets) == sum(column_reference)
    @test column_pointwise(matrix, column_offsets) == column_reference
    @test count(recipe -> recipe.source == :(eachcol(matrix)),
                authored_eachcol_sum.graph.recipes) == 1
    @test length(plan(authored_eachcol_sum).recipes) == 3
    @test _authored_plate_head_count(code_expr(column_total), :for) == 1
    @test !occursin("similar", string(code_expr(column_total)))
    @test !occursin("for ", string(column_total.f.tensorized_ast))

    row_offsets = [0.5, 1.5]
    row_reference = [
        sum(row) + row_offsets[index] + 1.0
        for (index, row) in enumerate(eachrow(matrix))
    ]
    row_total = prepare(authored_eachrow_derived_sum)
    row_pointwise = prepare(extract(authored_eachrow_derived_sum; want = :pointwise))
    @test row_total(matrix, row_offsets) == sum(row_reference)
    @test row_pointwise(matrix, row_offsets) == row_reference
    @test count(recipe -> recipe.source == :(eachrow(matrix)),
                authored_eachrow_derived_sum.graph.recipes) == 1
    @test count(recipe -> recipe.source == :(offsets .+ 1.0),
                authored_eachrow_derived_sum.graph.recipes) == 1
    @test _authored_plate_head_count(code_expr(row_total), :for) == 1
    @test !occursin("for ", string(row_total.f.tensorized_ast))

    values = [-1.0, 0.5, 2.0]
    atomic_offsets = [0.25, 0.75]
    atomic_reference = [
        value + sum(atomic_offsets .+ 1.0) for value in values
    ]
    atomic_total = prepare(authored_ref_derived_sum)
    atomic_pointwise = prepare(extract(authored_ref_derived_sum; want = :pointwise))
    @test atomic_total(values, atomic_offsets) == sum(atomic_reference)
    @test atomic_pointwise(values, atomic_offsets) == atomic_reference
    @test count(recipe -> recipe.source == :(offsets .+ 1.0),
                authored_ref_derived_sum.graph.recipes) == 1
    @test _authored_plate_head_count(code_expr(atomic_total), :for) == 1
    @test !occursin("similar", string(code_expr(atomic_total)))
    @test !occursin("for ", string(atomic_total.f.tensorized_ast))

    logits = [1.0 -0.5 0.25; 0.0 1.25 -0.75; -1.0 0.5 1.5]
    observed = [1, 2, 3]
    categorical_reference = [
        column[observed[index]] - _authored_logsumexp(column)
        for (index, column) in enumerate(eachcol(logits))
    ]
    categorical_total = prepare(authored_eachcol_categorical)
    categorical_pointwise = prepare(extract(
        authored_eachcol_categorical; want = :pointwise))
    @test categorical_total(logits, observed) ≈ sum(categorical_reference)
    @test categorical_pointwise(logits, observed) ≈ categorical_reference
    @test count(recipe -> recipe.source == :(eachcol(logits)),
                authored_eachcol_categorical.graph.recipes) == 1
    @test _authored_plate_head_count(code_expr(categorical_total), :for) == 1
    @test !occursin("similar", string(code_expr(categorical_total)))
    @test !occursin("for ", string(categorical_total.f.tensorized_ast))

    W = [0.25 -0.5; 0.75 1.0]
    b = [-0.25, 0.5]
    coefficients = vcat(vec(W), b)
    normal_reference = sum(
        _authored_plate_normal(value, 0.0, 1.0) for value in coefficients)
    normal_total = prepare(authored_vcat_normal)
    @test normal_total(W, b) ≈ normal_reference
    @test count(recipe -> recipe.source == :(vcat(vec(W), b)),
                authored_vcat_normal.graph.recipes) == 1
    @test _authored_plate_head_count(code_expr(normal_total), :for) == 1
    @test !occursin("similar", string(code_expr(normal_total)))
    @test !occursin("for ", string(normal_total.f.tensorized_ast))
end
