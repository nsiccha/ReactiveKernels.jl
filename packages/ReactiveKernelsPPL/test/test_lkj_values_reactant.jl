using Reactant

@testset "Reactant: invalid live LKJ shape keeps its branch inactive" begin
    bound, built = _lkv_build(2, :(e - 2.3))
    u = [0.15 * sin(i) for i in 1:built.layout.total]
    u[findfirst(==(:e), coordinate_names(built.layout))] = log(1.3)
    Base.invokelatest(_pcr_measure, built, bound, u)
end

@testset "Reactant: bound scalar LKJ normalizers" begin
    for eta in (:e, :(2 * e))
        bound, built = _lkv_build(3, eta; sampled = false)
        u = [0.15 * sin(i) for i in 1:built.layout.total]
        Base.invokelatest(_pcr_measure, built, bound, u)
    end
end

@testset "Reactant: data-sized LKJ values retain iteration" begin
    for uplo in ('L', 'U')
        traces, recipes = Dict{String,Int}[], Int[]
        for K in (2, 4)
            bound, built = _lkv_build(K, :e; uplo, data_width = true)
            u = [0.25 * sin(i) for i in 1:built.layout.total]
            push!(recipes, length(built.spec.graph.recipes))
            push!(traces, Base.invokelatest(_pcr_measure, built, bound, u;
                structure_ad = true))
        end
        @test recipes[1] == recipes[2]
        @test traces[1] == traces[2]
    end
end

# _pcr_measure in test_plate_cells_reactant.jl compares compiled primal and
# reverse mode with native Enzyme and records primal/AD backend operations.
@testset "Reactant: live LKJ eta retains iteration" begin
    for kind in (:factor, :stack), uplo in ('L', 'U')
        structures = Dict{String,Int}[]
        recipes = Int[]
        for (S, n) in ((2, 5), (4, 13))
            bound, built = kind === :factor ? _lkv_build(3, :e; n, uplo) :
                _lkv_stack_build(S, n; uplo)
            u = [0.25 * sin(i) for i in 1:built.layout.total]
            u[findfirst(==(:e), coordinate_names(built.layout))] = 0.0
            push!(recipes, length(built.spec.graph.recipes))
            push!(structures, Base.invokelatest(_pcr_measure, built, bound, u;
                structure_ad = true))
        end
        @test recipes[1] == recipes[2]
        @test structures[1] == structures[2]
    end
end
