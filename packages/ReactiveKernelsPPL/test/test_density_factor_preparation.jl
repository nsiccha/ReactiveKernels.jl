using Test, ReactiveKernels, ReactiveKernelsPPL, Enzyme

const _FACTOR_PREPARATION_REVERSE = ReactiveKernels.DifferentiationInterface.AutoEnzyme(
    ; mode = Enzyme.Reverse)

struct FactorPreparationReads <: AbstractVector{Float64}
    data::Vector{Float64}
    reads::Vector{Int}
end
Base.size(x::FactorPreparationReads) = size(x.data)
Base.IndexStyle(::Type{FactorPreparationReads}) = IndexLinear()
Base.getindex(x::FactorPreparationReads, i::Int) = (push!(x.reads, i); x.data[i])

module FactorPreparationAudit
using ReactiveKernels
include(joinpath(pkgdir(ReactiveKernels), "test", "type_stability_audit.jl"))
end

@testset "batched authored factors preserve values, support and binding checks" begin
    model = @rkppl begin
        tau ~ Exponential(1.4)
        z[1:2n] .~ Normal.(means, tau)
        b[1:n] .~ Uniform.(-2.0, 3.0)
    end
    bodies = Int[]
    nodes(x) = x isa Expr ? 1 + sum(nodes, x.args; init = 0) : 1
    for n in (3, 27)
        means = fill(0.4, 2n)
        plan = bind_data(model(; n, means))
        built = build_kernel(plan)
        u = 0.1 .* sin.(1:built.layout.total)
        original = copy(u)
        c = density_factors(built, plan)
        selections = [select_factors(c, vcat(
            collect(select_factors(c; site = :z, entries = (2i-1):2i)),
            collect(select_factors(c; site = :b, entries = i)))) for i in 1:n]
        preparation_bases = Vector{Float64}[]
        preparation_actives = Vector{Float64}[]
        # A compact query's operator takes its selection's arguments (coordinate
        # slots, cell memberships, row plan) after `(base, active)`.
        function record_prepare(kernel, backend, base, active, selection...; kwargs...)
            push!(preparation_bases, base)
            push!(preparation_actives, active)
            prepare_ad(kernel, backend, base, active, selection...; kwargs...)
        end
        queries = prepare_query_ad(record_prepare, built, plan, selections,
            _FACTOR_PREPARATION_REVERSE, u; coordinates = :support)
        @test length(queries) == n
        @test length(preparation_bases) == n
        @test all(base -> base === first(preparation_bases), preparation_bases)
        @test first(preparation_bases) !== u
        @test length(Set(objectid.(preparation_actives))) == n
        push!(bodies, nodes(code_expr(first(queries).prepared.kernel)))
        if n == 3
            audit = FactorPreparationAudit.TypeStabilityAudit
            q = first(queries)
            @test isempty(audit.ad_call_findings(q.prepared, u, u[q.coordinates], q.context...;
                recurse = audit.rk_or_generated(ReactiveKernelsPPL)))
            full = only(prepare_query_ad(prepare_ad, built, plan, selections[1:1],
                _FACTOR_PREPARATION_REVERSE, u))
            full_value, full_gradient = ad_value_and_gradient(full, u)
            compact_value, compact_gradient = ad_value_and_gradient(q, u, u[q.coordinates])
            @test full_value ≈ compact_value
            @test full_gradient[q.coordinates] ≈ compact_gradient
            @test all(iszero, full_gradient[setdiff(eachindex(u), q.coordinates)])
        end
        names = coordinate_names(built.layout)
        at(name) = only(findall(==(name), names))
        for i in (1, n)
            q = queries[i]
            zindices = [at(Symbol("z.", 2i-1)), at(Symbol("z.", 2i))]
            bindex, tauindex = at(Symbol("b.", i)), at(:tau)
            @test q.coordinates == sort([tauindex; zindices; bindex])
            @test q.coordinates == factor_support(selections[i]; u0 = u)
            for shift in (0.0, 0.13)
                point = u .+ shift
                active = point[q.coordinates]
                tau, b = exp(point[tauindex]), point[bindex]
                residual = point[zindices] .- 0.4
                expected = -log(2pi) - 2log(tau) - sum(abs2, residual)/(2tau^2) -
                    log1p(exp(-b)) - log1p(exp(b))
                oracle = zeros(length(point))
                oracle[tauindex] = -2 + sum(abs2, residual)/tau^2
                oracle[zindices] = -residual/tau^2
                oracle[bindex] = 1 - 2/(1 + exp(-b))
                value, gradient = ad_value_and_gradient(q, point, active)
                @test value ≈ expected
                @test gradient ≈ oracle[q.coordinates]
                @test active == point[q.coordinates]
                @test point == u .+ shift
            end
        end
        @test u == original
        @test first(preparation_bases) == original
        # Cached event inventory reads a local footprint, not a copy of all u.
        event = only(select_factors(c; kind = :prior, site = :z, entries = 2))
        readpoint = FactorPreparationReads(u, Int[])
        @test factor_support(c, event; u0 = readpoint) == factor_support(c, event; u0 = u)
        @test length(readpoint.reads) < length(u)
        @test factor_support(c, event) == factor_support(c, event; u0 = u)
        @test factor_support(c, event; u0 = Float32.(u)) == factor_support(c, event; u0 = u)
        @test_throws ContractValidationError factor_support(c, event; u0 = u[2:end])
        empty_selection = select_factors(c, DensityFactor[])
        empty_ad = only(prepare_query_ad(prepare_ad, built, plan, [empty_selection],
            _FACTOR_PREPARATION_REVERSE, u; coordinates = :support))
        @test isempty(empty_ad.coordinates)
        value, gradient = ad_value_and_gradient(empty_ad, u, Float64[])
        @test iszero(value)
        @test isempty(gradient)
        @test isempty(prepare_query_ad(prepare_ad, built, plan, FactorSelection[],
            _FACTOR_PREPARATION_REVERSE, u; coordinates = :support))
        second_catalogue = density_factors(built, plan)
        mixed = [first(selections), select_factors(second_catalogue; site = :z, entries = 1)]
        @test_throws ArgumentError prepare_query_ad(prepare_ad, built, plan, mixed,
            _FACTOR_PREPARATION_REVERSE, u; coordinates = :support)
        malformed = FactorSelection(c, [DensityFactor(:prior, :z, 2n + 1)])
        @test_throws ArgumentError prepare_query_ad(prepare_ad, built, plan, [malformed],
            _FACTOR_PREPARATION_REVERSE, u; coordinates = :support)
        # A later mutation still invalidates both preparation spellings.
        support = factor_support(c, event; u0 = u)
        plan.columns[:means][1] += 1
        @test_throws ContractValidationError prepare_query_ad(prepare_ad, built, plan,
            selections, _FACTOR_PREPARATION_REVERSE, u; coordinates = :support)
        @test_throws ContractValidationError prepare_query_ad(prepare_ad, built, plan,
            first(selections), _FACTOR_PREPARATION_REVERSE, u; coordinates = :support)
        @test factor_support(c, event; u0 = u) == support
    end
    @test bodies[1] == bodies[2]
end

@testset "selections of one structure share one prepared program" begin
    # Snag prepare-query-ad-96b98277: each selection copied the catalogue
    # graph, and the copies stayed alive in the binding's data store and in
    # every query, so retained memory grew by a model graph per selection.
    model = @rkppl begin
        sd_r ~ Exponential(1.0)
        r[1:J] .~ Normal.(0.0, sd_r)
        sigma ~ Exponential(1.0)
        @plate for i in eachindex(y)
            y[i] ~ Normal(r[g[i]] * x[i], sigma)
        end
    end
    stored_graphs = Int[]
    for J in (8, 64)
        n = 4J
        plan = model(; J, x = sin.(1:n), g = repeat(1:J; inner = 4)) | (; y = 0.3 .* cos.(1:n))
        built = build_kernel(plan)
        u = 0.1 .* sin.(1:built.layout.total)
        c = density_factors(built, plan)
        selections = [select_factors(c; kind = :prior, site = :r, entries = j) for j in 1:J]
        data = getfield(plan, :data_values)
        stored = length(data.graphs)
        queries = prepare_query_ad(prepare_ad, built, plan, selections,
            _FACTOR_PREPARATION_REVERSE, u; coordinates = :support)
        # The binding gains the row-plan graph, the query graph and the site's
        # footprint kernel, whatever the number of selections.
        push!(stored_graphs, length(data.graphs) - stored)
        # One query graph; the shared program plans its selections itself, so
        # compact queries build no separate planner graph.
        @test length(getfield(c, :structures)) == 1
        @test length(unique(objectid(q.prepared.kernel.plan.graph) for q in queries)) == 1
        # Snag per-unit-partiti-9d46c6de: the queries of one structure call one
        # prepared program with their own selection, instead of a program
        # rebound per selection (a kernel, plan and constants table each).
        @test length(unique(objectid(q.prepared.kernel.plan.recipes) for q in queries)) == 1
        @test all(q -> !isempty(q.context), queries)
        # The catalogue reads the binding's own arrays; it keeps no copy of them.
        @test getfield(c, :binding).x === plan.columns[:x]
        names = coordinate_names(built.layout)
        at(name) = only(findall(==(name), names))
        for j in (1, J)
            q = queries[j]
            sd, rj = at(:sd_r), at(Symbol("r.", j))
            @test q.coordinates == sort([sd, rj])
            for shift in (0.0, 0.21)
                point = u .+ shift
                s, r = exp(point[sd]), point[rj]
                value, gradient = ad_value_and_gradient(q, point, point[q.coordinates])
                @test value ≈ -0.5log(2pi) - log(s) - r^2 / (2s^2)
                oracle = zeros(length(point))
                oracle[sd] = -1 + r^2 / s^2
                oracle[rj] = -r / s^2
                @test gradient ≈ oracle[q.coordinates]
            end
        end
        # The uncompacted spelling shares its program in the same way.
        full = prepare_query_ad(prepare_ad, built, plan, selections[1:2],
            _FACTOR_PREPARATION_REVERSE, u)
        @test full[1].prepared.kernel.plan.graph === full[2].prepared.kernel.plan.graph
        @test ad_value_and_gradient(full[2], u)[1] ≈
            ad_value_and_gradient(queries[2], u, u[queries[2].coordinates])[1]
    end
    @test stored_graphs[1] == stored_graphs[2] <= 3
end

@testset "a selection of many sites prepares a concretely typed program" begin
    # Snag slices-of-one-bo-c43eb9ee: a slice's prior on ODL v2 selects dozens
    # of sites. Their totals were added by one `+`, which Julia does not infer
    # past 32 operands, so the prepared program dispatched it at run time.
    sites = 40
    ast = Expr(:block, (:($(Symbol(:a_, k)) ~ Normal(0.0, 1.0)) for k in 1:sites)...,
        :(y .~ Normal.(a_1, 1.0)))
    data = (; y = [0.1, -0.2])
    plan = bind_data(lower_rkppl(ast, data; conditioned = (:y,), mod = @__MODULE__), data)
    built = build_kernel(plan)
    @test built.layout.total == sites
    u = 0.1 .* sin.(1:sites)
    selection = select_factors(density_factors(built, plan); kind = :prior)
    q = only(prepare_query_ad(prepare_ad, built, plan, [selection],
        _FACTOR_PREPARATION_REVERSE, u; coordinates = :support))
    @test q.coordinates == collect(1:sites)
    audit = FactorPreparationAudit.TypeStabilityAudit
    @test isempty(audit.ad_call_findings(q.prepared, u, u[q.coordinates], q.context...;
        recurse = audit.rk_or_generated(ReactiveKernelsPPL)))
    value, gradient = ad_value_and_gradient(q, u, u[q.coordinates])
    @test value ≈ -sites * 0.5log(2pi) - sum(abs2, u) / 2
    @test gradient ≈ -u
end
