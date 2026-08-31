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

_authored_plate_normal(x, location, scale) =
    -0.5 * log(2π) - log(scale) - 0.5 * ((x - location) / scale)^2
_authored_plate_cauchy(x, location, scale) =
    -log(π) - log(scale) - log1p(((x - location) / scale)^2)

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
    zipped_reference = [
        _authored_plate_normal(xs[i], locations[i], scales[i])
        for i in eachindex(xs)
    ]
    @test total(xs, locations, scales) ≈ sum(zipped_reference)
    @test pointwise(xs, locations, scales) ≈ zipped_reference
    @test total(xs, locations, scale) ≈
        sum(_authored_plate_normal(xs[i], locations[i], scale)
            for i in eachindex(xs))

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
    @test pointwise(observations, location) ≈ reference
    @test total(observations, location) ≈ sum(reference)
    @test _authored_plate_head_count(code_expr(total), :for) == 1
    @test !occursin("similar", string(code_expr(total)))
end
