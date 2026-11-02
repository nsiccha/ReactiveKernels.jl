using Test, ReactiveKernels, ReactiveKernelsPPL, Enzyme

module DensityCellLaw
using ReactiveKernelsPPL
struct Joint{M,S}
    mean::M
    scale::S
end
@inline ReactiveKernelsPPL.sampling_logdensity(d::Joint, x) =
    sum(-0.5log(2pi) - log(d.scale) - 0.5*((x[i] - d.mean[i])/d.scale)^2
        for i in eachindex(x); init = 0.0)
ReactiveKernelsPPL.sampling_geometry(::Type{Joint}, args, shape) =
    ParameterGeometry((:(length($(args[1]))),), :(length($(args[1])));
        support = :real,
        constrain = (u, shape, args...) -> collect(u),
        unconstrain = (x, shape, args...) -> collect(x),
        logjac = (u, shape, args...) -> 0.0)
end

@testset "one authored joint factor per ragged cell" begin
    model = @rkppl begin
        scale ~ Exponential(1.4)
        @plate for i in eachindex(means)
            z[i] ~ DensityCellLaw.Joint(means[i], scale)
        end
    end
    emitted, reads = Int[], Int[]
    nodes(x) = x isa Expr ? 1 + sum(nodes, x.args; init = 0) : 1
    for J in (3, 103)
        means = [[0.2sin(i+j) for j in 1:mod1(i, 3)] for i in 1:J]
        plan = bind_data(model(; means))
        built = build_kernel(plan)
        u = [0.2sin(i) for i in 1:built.layout.total]
        catalogue = density_factors(built, plan)
        @test length(select_factors(catalogue; kind = :prior, site = :z)) == J
        selected = select_factors(catalogue; kind = :prior, site = :z, entries = 2)
        names = coordinate_names(built.layout)
        support = findall(n -> n in (:scale, Symbol("z.2.1"), Symbol("z.2.2")), names)
        @test factor_coordinates(selected) == findall(n -> n in
            (Symbol("z.2.1"), Symbol("z.2.2")), names)
        reference = point -> begin
            nt = constrain(built.layout, point)
            sum(-0.5log(2pi) - log(nt.scale) - 0.5*((nt.z[2][i]-means[2][i])/nt.scale)^2
                for i in eachindex(means[2]))
        end
        _check_density_structure(built, plan, selected, u, support, reference)
        with_jacobian = select_factors(catalogue; site = :z, entries = 2)
        _check_density_structure(built, plan, with_jacobian, u, support, reference)
        other_jacobian = select_factors(catalogue; kind = :log_jacobian, site = :z, entries = 3)
        different_cells = select_factors(catalogue, vcat(collect(selected), collect(other_jacobian)))
        combined_support = sort!(unique!(vcat(support, factor_support(other_jacobian; u0 = u))))
        _check_density_structure(built, plan, different_cells, u, combined_support, reference)
        q = prepare_query(built, plan, selected; u0 = u)
        push!(emitted, nodes(code_expr(q)))
        recording = DensityStructureReads(u, Int[])
        q(recording)
        push!(reads, length(recording.reads))
        @test prepare_query(built, plan, select_factors(catalogue))(u) ≈
            prepare_query(built, plan, :sampler)(u)
    end
    @test emitted[1] == emitted[2]
    @test reads[1] == reads[2]
end

@testset "factor selections are checked against the bound program" begin
    model = @rkppl begin
        z[1:J] .~ Normal.(means, 1.7)
    end
    means = [0.2, -0.1, 0.4]
    plan = bind_data(model(; J = 3, means))
    built = build_kernel(plan)
    catalogue = density_factors(built, plan)
    selected = select_factors(catalogue; site = :z, entries = 2)
    equivalent = bind_data(model(; J = 3, means = copy(means)))
    @test prepare_query(build_kernel(equivalent), equivalent, selected)(zeros(3)) ≈
        prepare_query(built, plan, selected)(zeros(3))
    changed = bind_data(model(; J = 3, means = means .+ 1))
    @test_throws ContractValidationError prepare_query(build_kernel(changed), changed, selected)
    malformed = FactorSelection(catalogue, [DensityFactor(:prior, :z, 4)])
    @test_throws ArgumentError prepare_query(built, plan, malformed)
    # Inventory support belongs to its original binding snapshot. Preparing
    # values against changed data must still reject a stale selection.
    support = factor_support(catalogue, only(selected))
    owned = factor_coordinates(selected)
    plan.columns[:means][2] += 1
    @test_throws ContractValidationError prepare_query(built, plan, selected)
    @test factor_support(catalogue, only(selected)) == support
    @test factor_coordinates(selected) == owned
end
