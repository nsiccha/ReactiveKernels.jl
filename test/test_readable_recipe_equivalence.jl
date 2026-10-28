using Test, ReactiveKernels

@testset "embedded readable recipes preserve graph wiring" begin
    recipe(id, inputs, outputs) = ReactiveKernels.Recipe(id, inputs, outputs,
        identity, 1.0, nothing, false)
    a, b, c = (value(:a, Float64), value(:b, Float64), value(:c, Float64))
    x, y, z = (value(:a, Float64), value(:b, Float64), value(:c, Float64))
    left = [recipe(1, (a,), (b,)), recipe(2, (b,), (c,))]
    renamed = [recipe(1, (x,), (y,)), recipe(2, (y,), (z,))]
    @test ReactiveKernels._same_readable_recipes(left, renamed)

    # A second value with the same readable name is a different graph edge.
    disconnected = [recipe(1, (x,), (y,)),
        recipe(2, (value(:b, Float64),), (z,))]
    @test !ReactiveKernels._same_readable_recipes(left, disconnected)
    @test !ReactiveKernels._same_readable_recipes(left,
        [recipe(1, (x,), (y,)), recipe(2, (y,), (value(:c, Int),))])
end

# A law whose component plate nests an observation plate, composed into the
# cells of an enclosing plate, as the whole-event mixture laws are.
@kernel _weighted_log_sum(scores::Vector{Float64}, weights::Vector{Float64}) = begin
    weighted = plate(scores, weights) do score, weight
        log(weight) + score
    end
    return log(sum(exp.(weighted)))
end

@kernel _nested_mixture(observed::Vector{Bool}, probabilities::Matrix{Float64},
        weights::Vector{Float64}) = begin
    components = plate(1:size(probabilities, 2)) do k
        cells = plate(1:length(observed)) do i
            p = probabilities[i, k]
            observed[i] ? log(p) : log1p(-p)
        end
        sum(cells)
    end
    return _weighted_log_sum(components, weights)
end

@kernel _event_mixtures(events::Vector{Vector{Bool}},
        probabilities::Matrix{Float64}, weights::Vector{Float64}) = begin
    scores = plate(events) do observed
        _nested_mixture(observed, probabilities, weights)
    end
    return sum(scores)
end

@testset "backend products of one preparation share nested cell plans" begin
    RK = ReactiveKernels
    # Composing a nested cell's plate chains gives the composed values fresh
    # identities. Each product composed separately, so a fresh value could take
    # an identity a package image had already given another value of the same
    # program, and preparation refused it (snag readable-recipe-75581e09).
    p = plan(_event_mixtures)
    @test RK._needs_embedded_tensorization(p)
    native, tensorized = RK._sharing_cell_plans() do
        lowered = RK._lowering_plan(p)
        (RK._lower_with_ops(lowered), RK._lower_with_ops(lowered; tensorized = true))
    end
    identities(recipes) = [v.id for r in recipes for v in (r.inputs..., r.outputs...)]
    @test identities(native[3]) == identities(tensorized[3])
    @test RK._same_readable_recipes(tensorized[3], native[3])
    events = [Bool[1, 0, 1], Bool[0, 0, 1], Bool[1, 1, 1]]
    probabilities = [0.3 0.8; 0.6 0.2; 0.5 0.5]
    weights = [0.4, 0.6]
    expected = sum(events) do observed
        scores = [sum(observed[i] ? log(probabilities[i, k]) :
                      log1p(-probabilities[i, k]) for i in eachindex(observed))
                  for k in axes(probabilities, 2)]
        log(sum(weights .* exp.(scores)))
    end
    @test prepare(_event_mixtures)(events, probabilities, weights) ≈ expected
    @test prepare(_event_mixtures)(Vector{Bool}[], probabilities, weights) == 0.0
end
