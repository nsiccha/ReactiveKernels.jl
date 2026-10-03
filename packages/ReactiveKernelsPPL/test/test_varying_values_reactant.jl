using Reactant

@testset "Reactant: membership and stratified values retain iteration" begin
    for kind in (:membership, :distinct, :stratified, :single, :both)
        traces, recipes = Dict{String,Int}[], Int[]
        for (n, S) in ((7, 2), (19, 4))
            bound, built = _cv_build(kind, n, S)
            u = [0.2 * sin(i) for i in 1:built.layout.total]
            push!(recipes, length(built.spec.graph.recipes))
            push!(traces, Base.invokelatest(_pcr_measure, built, bound, u; structure_ad = true))
        end
        @test recipes[1] == recipes[2]
        @test traces[1] == traces[2]
    end
end

@testset "Reactant limitation: data-sized shared LKJ factor reverse" begin
    # benchmark/repro_reactant_shared_matrix_loop_reverse.jl isolates the
    # retained diagonal reduction and shared matrix without RK/PPL code.
    bound, built = _cv_data_width_build()
    u = [0.2 * sin(i) for i in 1:built.layout.total]
    q = prepare_sampler(built, bound, u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    value, _ = sampler_value_and_gradient!(q, similar(u), u)
    ru = Reactant.to_rarray(u)
    kernel = prepare_query(built, bound, :sampler)
    primal = Reactant.@compile kernel(ru)
    @test Float64(primal(ru)) ≈ value rtol = 1e-9
    err = try
        compile_ad_value_and_gradient(q.ad, ru)
        nothing
    catch e
        e
    end
    if err !== nothing
        @test occursin("does not dominate this use", sprint(showerror, err))
    end
    @test_broken err === nothing
end

@testset "Reactant: ordered grouping axes retain iteration" begin
    for kind in (:unique, :alias, :computed, :provided, :range, :stepped,
            :descending, :cell, :crossed, :labels, :factor)
        traces, recipes = Dict{String,Int}[], Int[]
        for (n, G) in ((7, 3), (19, 5))
            bound, built = kind === :cell ? _gv_plate_build(n, G) :
                kind === :factor ? _gv_factor_build(n, G) :
                kind === :crossed ? _gv_crossed_build(n, G) :
                kind === :labels ? _gv_build(:alias, n, G; labels = true) :
                _gv_build(kind, n, G)
            u = [0.2 * sin(i) for i in 1:built.layout.total]
            push!(recipes, length(built.spec.graph.recipes))
            push!(traces, Base.invokelatest(_pcr_measure, built, bound, u; structure_ad = true))
        end
        @test recipes[1] == recipes[2]
        @test traces[1] == traces[2]
    end
end
