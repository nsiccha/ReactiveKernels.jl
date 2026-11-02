using Test, ReactiveKernels, ReactiveKernelsPPL, Enzyme
import Distributions

const _STRUCTURE_FACTOR_REVERSE = ReactiveKernels.DifferentiationInterface.AutoEnzyme(; mode = Enzyme.Reverse)

struct DensityStructureReads <: AbstractVector{Float64}
    values::Vector{Float64}
    reads::Vector{Int}
end
Base.size(v::DensityStructureReads) = size(v.values)
Base.IndexStyle(::Type{DensityStructureReads}) = IndexLinear()
Base.getindex(v::DensityStructureReads, i::Int) = (push!(v.reads, i); v.values[i])

function _check_density_structure(built, plan, selected, u, support, reference)
    @test factor_support(selected; u0 = u) == support
    for event in selected
        @test factor_support(selected.catalogue, event; u0 = u) ⊆ support
    end
    q = prepare_query(built, plan, selected; u0 = u)
    recording = DensityStructureReads(u, Int[])
    @test q(recording) ≈ reference(u)
    @test sort(unique(recording.reads)) == support
    ad = prepare_query_ad(prepare_ad, built, plan, selected,
        _STRUCTURE_FACTOR_REVERSE, u; coordinates = :support)
    for shift in (0.0, 0.1)
        active = u[support] .+ shift
        point = copy(u)
        point[support] = active
        value, gradient = ad_value_and_gradient(ad, u, active)
        @test value ≈ reference(point)
        h = 1e-5
        expected = map(support) do index
            plus, minus = copy(point), copy(point)
            plus[index] += h
            minus[index] -= h
            (reference(plus) - reference(minus)) / (2h)
        end
        @test gradient ≈ expected rtol = 1e-6 atol = 1e-8
    end
end

@testset "independent multivariate draws retain local whole-event evaluation" begin
    for orientation in (:rows, :cols), J in (3, 103)
        dims = orientation === :rows ? :(1:J, 1:2) : :(1:2, 1:J)
        slices = orientation === :rows ? :eachrow : :eachcol
        model = RKPPLModel(quote
            M[$(dims.args...)] .~ Normal.(0.3, 1.7)
            $slices(B[$(dims.args...)]) .~ MvNormalCholesky.($slices(M), Ref(F))
        end, @__MODULE__)
        F = [1.2 0.0; 0.3 0.8]
        plan = bind_data(model(; J, F))
        built = build_kernel(plan)
        u = [0.2sin(i) for i in 1:built.layout.total]
        catalogue = density_factors(built, plan)
        @test length(select_factors(catalogue; site = :B)) == J
        selected = select_factors(catalogue; site = :B, entries = 2)
        support = sort!(reduce(vcat, map((:M, :B)) do name
            e = only(e for e in built.layout.entries if e.name === name)
            orientation === :rows ? e.offset .+ [1, J + 1] : e.offset .+ [2, 3]
        end))
        e = only(e for e in built.layout.entries if e.name === :B)
        owned = orientation === :rows ? e.offset .+ [1, J + 1] : e.offset .+ [2, 3]
        @test factor_coordinates(selected) == owned
        reference = point -> begin
            nt = constrain(built.layout, point)
            means = orientation === :rows ? nt.M[2, :] : nt.M[:, 2]
            values = orientation === :rows ? nt.B[2, :] : nt.B[:, 2]
            Distributions.logpdf(Distributions.MvNormal(means, F * F'), values)
        end
        _check_density_structure(built, plan, selected, u, support, reference)
    end
end

@testset "independent coupled transforms retain local whole-event evaluation" begin
    model = RKPPLModel(quote
        unrelated ~ Normal(0.0, 1.0)
        eachrow(P[1:J, 1:3]) .~ Dirichlet([1.0, 2.0, 3.0])
    end, @__MODULE__)
    for J in (3, 103)
        plan = bind_data(model(; J))
        built = build_kernel(plan)
        u = [0.2sin(i) for i in 1:built.layout.total]
        catalogue = density_factors(built, plan)
        @test length(select_factors(catalogue; kind = :prior, site = :P)) == J
        @test length(select_factors(catalogue; kind = :log_jacobian, site = :P)) == J
        selected = select_factors(catalogue; site = :P, entries = 2)
        e = only(e for e in built.layout.entries if e.name === :P)
        support = e.offset .+ [1, J + 1]
        @test factor_coordinates(selected) == support
        @test all(event -> factor_coordinates(catalogue, event) == support, selected)
        reference = point -> begin
            values = simplex_constrain(point[support])
            Distributions.logpdf(Distributions.Dirichlet([1.0, 2.0, 3.0]), values) +
                simplex_logjac(point[support])
        end
        _check_density_structure(built, plan, selected, u, support, reference)
    end
end

@rkppl _density_named_child(m) = begin
    b ~ Normal(m, 1.7)
    return b
end

@testset "factor identities preserve authored submodel paths" begin
    model = @rkppl begin
        m ~ Normal(0.3, 1.2)
        child ~ _density_named_child(m)
        var"child.b" ~ Normal(0.0, 2.0)
    end
    plan = bind_data(model())
    built = build_kernel(plan)
    u = [0.2sin(i) for i in 1:built.layout.total]
    catalogue = density_factors(built, plan)
    @test Set(f.site for f in catalogue) == Set([:m, Symbol("child.b"), Symbol("var\"child.b\"")])
    selected = select_factors(catalogue; site = Symbol("child.b"))
    names = coordinate_names(built.layout)
    support = sort!([findfirst(==(:m), names), findfirst(==(Symbol("child.b")), names)])
    reference = point -> begin
        nt = constrain(built.layout, point)
        Distributions.logpdf(Distributions.Normal(nt.m, 1.7), nt.child.b)
    end
    _check_density_structure(built, plan, selected, u, support, reference)
end
