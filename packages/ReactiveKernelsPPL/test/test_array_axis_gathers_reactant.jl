using Reactant

# The shared measurement checks native/compiled primal and AD parity and
# counts operations and regions in both traces, before optimization.
@testset "Reactant: second-axis gathers retain array structure" begin
    for kind in (:columns, :derived, :subset)
        @testset "$kind" begin
            structures = Dict{String,Int}[]
            for (G, n) in ((3, 6), (5, 17))
                fx = _axis_gather_fixture(kind, G, n)
                u = _av_point(fx.built.layout.total)
                sampler = prepare_sampler(fx.built, fx.bound, u;
                    backend = AutoEnzyme(; mode = Enzyme.Reverse))
                native, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
                @test native ≈ _axis_gather_oracle(fx, u)
                @test gradient ≈ _findiff_grad(w -> _axis_gather_oracle(fx, w), u) rtol = 1e-5 atol = 1e-7
                ops = Base.invokelatest(_semantics_compiled_measure, sampler, u)
                @test get(ops, "primal.stablehlo.gather", 0) > 0
                @test get(ops, "primal.stablehlo.reduce", 0) > 0
                @test get(ops, "ad.stablehlo.gather", 0) > 0
                push!(structures, ops)
            end
            @test structures[1] == structures[2]
        end
    end
end

@testset "Reactant limitation: gather from an adjoint matrix" begin
    # Native values and gradients pass for both orientations above.
    # The standalone backend reproducer isolates the incorrect ancestor
    # indices; preserve the authored adjoint and refuse this exact shape.
    fx = _axis_gather_fixture(:adjoint, 3, 6)
    u = _av_point(fx.built.layout.total)
    kernel = prepare_query(fx.built, fx.bound, :sampler)
    ru = Reactant.to_rarray(u)
    err = try
        Reactant.@compile kernel(ru)
        nothing
    catch e
        e
    end
    if err !== nothing
        @test err isa BoundsError
        @test err.a isa LinearIndices{2}
        @test err.i isa Tuple{AbstractVector{CartesianIndex{2}}}
    end
    @test_broken err === nothing
end
