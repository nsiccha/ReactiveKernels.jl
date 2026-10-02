using Reactant

# Helpers and the independent Julia/Distributions oracle are defined in
# test_array_definition_gathers.jl. Fixed observation scale isolates the
# gathers from the separately recorded default scale-gradient defect.
function _adg_compiled(fx, u)
    kernel = prepare_query(fx.built, fx.bound, :sampler)
    ru = Reactant.to_rarray(u)
    hlo = repr(Reactant.@code_hlo optimize = false kernel(ru))
    operations = Dict{String,Int}()
    for m in eachmatch(r"(?:stablehlo|enzyme|chlo|func|arith)\.[a-z_]+", hlo)
        operations[m.match] = get(operations, m.match, 0) + 1
    end
    oracle = _adg_oracle(fx, u)
    compiled = Reactant.@compile kernel(ru)
    @test Float64(compiled(ru)) ≈ oracle.posterior rtol = 1e-9
    q = prepare_sampler(fx.built, fx.bound, u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    value, grad = sampler_value_and_gradient!(q, similar(u), u)
    cad = compile_ad_value_and_gradient(q.ad, ru)
    rvalue, rgrad = cad(ru)
    @test Float64(rvalue) ≈ value rtol = 1e-9
    @test Array(rgrad) ≈ grad rtol = 1e-8 atol = 1e-9
    @test Array(rgrad) ≈ _adg_findiff(w -> _adg_oracle(fx, w).posterior, u) rtol = 1e-5 atol = 1e-7
    return operations
end

@testset "Reactant: positional definition gathers retain array structure" begin
    @testset "$kind" for kind in (:linear_matrix, :opaque, :opaque_vector, :opaque_rows)
        structures = Dict{String,Int}[]
        for (G, n) in ((2, 6), (5, 17))
            fx = _adg_build(kind, G, n)
            u = [0.3 * sin(1.2i) for i in 1:fx.built.layout.total]
            ops = Base.invokelatest(_adg_compiled, fx, u)
            @test get(ops, "stablehlo.gather", 0) > 0
            @test get(ops, "stablehlo.reduce", 0) > 0
            push!(structures, ops)
        end
        # Every traced operation and control-flow region has a fixed
        # count as both the observation and level axes grow.
        @test structures[1] == structures[2]
    end
end

@testset "Reactant limitation: linear gather from adjoint row" begin
    # Native primal and reverse pass in test_array_definition_gathers.jl.
    # benchmark/repro_reactant_adjoint_linear_gather.jl isolates this exact
    # backend indexing failure. Keep ordinary Julia indexing and exclude
    # only this shape from the compiled parity/structure acceptance above.
    fx = _adg_build(:linear, 2, 6)
    u = [0.3 * sin(1.2i) for i in 1:fx.built.layout.total]
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
        @test err.a isa LinearIndices{1}
        @test err.i isa Tuple{AbstractVector{CartesianIndex{2}}}
    end
    @test_broken err === nothing
end
