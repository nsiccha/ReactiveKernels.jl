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
