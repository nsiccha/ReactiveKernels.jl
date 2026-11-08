using Test, ReactiveKernels, ReactiveKernelsPPL, Enzyme

const _FACTOR_REVERSE = ReactiveKernels.DifferentiationInterface.AutoEnzyme(; mode = Enzyme.Reverse)
_factor_normal(x, m, s) = -0.5log(2pi) - log(s) - 0.5*((x-m)/s)^2
_factor_nodes(x) = x isa Expr ? 1 + sum(_factor_nodes, x.args; init = 0) : 1

struct FactorReadVector <: AbstractVector{Float64}
    values::Vector{Float64}
    reads::Vector{Int}
end

@testset "joint laws and coupled transforms stay atomic" begin
    model = @rkppl begin
        location ~ Normal(0.3, 1.7)
        means = [location, -location]
        joint[1:2] ~ MvNormalCholesky(means, F)
        concentration = [exp(location), 1.5, 2.0]
        simplex ~ Dirichlet(concentration)
    end
    plan = bind_data(model(; F = [1.2 0.0; 0.3 0.8]))
    built = build_kernel(plan)
    u = collect(unconstrain(built.layout, (; location = 0.2,
        joint = [0.4, -0.1], simplex = [0.2, 0.3, 0.5])))
    c = density_factors(built, plan)
    @test length(select_factors(c; kind = :prior, site = :joint)) == 1
    @test length(select_factors(c; kind = :prior, site = :simplex)) == 1
    @test length(select_factors(c; kind = :log_jacobian, site = :simplex)) == 1
    @test_throws ArgumentError select_factors(c; site = :joint, entries = 2)
    joint = select_factors(c; kind = :prior, site = :joint)
    joint_support = factor_support(joint; u0 = u)
    @test length(joint_support) == 3
    location = only(e.offset for e in built.layout.entries if e.name === :location)
    @test factor_coordinates(joint) == setdiff(joint_support, [location])
    simplex = select_factors(c; site = :simplex)
    e = only(e for e in built.layout.entries if e.name === :simplex)
    @test factor_coordinates(simplex) == collect(e.offset:e.offset + e.size - 1)
    # Its prior reads its own block and the concentration's hyperparent; its
    # Jacobian reads the block alone (snag factor-support-r-2b8c2968).
    block = collect(e.offset:e.offset + e.size - 1)
    @test factor_support(select_factors(c; kind = :prior, site = :simplex);
        u0 = u) == sort([location; block])
    @test factor_support(select_factors(c; kind = :log_jacobian, site = :simplex);
        u0 = u) == block
    @test factor_support(c, only(joint); u0 = u) == joint_support
    selected = select_factors(c)
    q = prepare_query_ad(prepare_ad, built, plan, selected, _FACTOR_REVERSE, u)
    sampler = prepare_query_ad(prepare_ad, built, plan, :sampler, _FACTOR_REVERSE, u)
    for point in (u, u .+ 0.12)
        value, gradient = ad_value_and_gradient(q, point)
        target, target_gradient = ad_value_and_gradient(sampler, point)
        @test value ≈ target
        @test gradient ≈ target_gradient
        h = 1e-5
        value_query = prepare_query(built, plan, selected; u0 = point)
        finite_difference = map(eachindex(point)) do i
            plus, minus = copy(point), copy(point)
            plus[i] += h
            minus[i] -= h
            (value_query(plus) - value_query(minus)) / (2h)
        end
        @test gradient ≈ finite_difference rtol = 1e-6 atol = 1e-8
    end
end
Base.size(v::FactorReadVector) = size(v.values)
Base.IndexStyle(::Type{FactorReadVector}) = IndexLinear()
Base.getindex(v::FactorReadVector, i::Int) = (push!(v.reads, i); v.values[i])

@testset "authored density factors, support and native Reverse" begin
    model = RKPPLModel(quote
        center ~ Normal(0.3, 1.7)
        scale ~ Exponential(1.4)
        z[1:J] .~ Normal.(center, scale)
        bounded[1:J] .~ Uniform.(-2.0, 3.0)
        y .~ Normal.(z[group], 0.9)
    end, @__MODULE__)
    emitted, reads = Int[], Int[]
    for J in (3, 301)
        data = (; group = [1, 2], y = [0.2, -0.1])
        plan = model(; J, group = data.group) | (; data.y)
        built = build_kernel(plan)
        u = collect(unconstrain(built.layout, (; center = 0.2, scale = 0.8,
            z = fill(0.4, J), bounded = fill(0.6, J))))
        c = density_factors(built, plan)
        @test all(f -> f.kind in (:prior, :log_jacobian), c)
        @test count(f -> f.kind === :prior && f.site === :z, c) == J
        @test count(f -> f.kind === :log_jacobian && f.site === :bounded, c) == J
        local_prior = select_factors(c; kind = :prior, site = :z, entries = 2)
        support = factor_support(local_prior; u0 = u)
        expected_support = sort([only(e.offset for e in built.layout.entries if e.name === n)
            for n in (:center, :scale, :z)] .+ [0, 0, 1])
        @test support == expected_support
        @test factor_coordinates(local_prior) == [expected_support[end]]
        q = prepare_query(built, plan, local_prior; u0 = u)
        recording = FactorReadVector(u, Int[])
        @test q(recording) ≈ _factor_normal(0.4, 0.2, 0.8)
        @test sort(unique(recording.reads)) == support
        push!(emitted, _factor_nodes(code_expr(q)))
        push!(reads, length(recording.reads))
        compact = prepare_query_ad(prepare_ad, built, plan, local_prior,
            _FACTOR_REVERSE, u; coordinates = :support)
        @test compact.coordinates == support
        for delta in (0.0, 0.13)
            point = u .+ delta
            values = constrain(built.layout, point)
            expected = _factor_normal(values.z[2], values.center, values.scale)
            value, gradient = ad_value_and_gradient(compact, point, point[support])
            @test value ≈ expected
            residual = values.z[2] - values.center
            analytic = [residual / values.scale^2,
                -1 + (residual / values.scale)^2, -residual / values.scale^2]
            @test gradient ≈ analytic
        end
        # The world-age barrier passes the query by reference: it copies
        # nothing per call beyond the boxed result (an immutable query was
        # copied whole with its prepared operator, about 9 KB per call).
        active = u[support]
        direct() = ReactiveKernels.ad_value_and_gradient(compact.prepared, u, active,
            compact.context...)
        wrapped() = ad_value_and_gradient(compact, u, active)
        direct(); wrapped()
        @test (@allocated wrapped()) - (@allocated direct()) < 256
        all_prior = prepare_query(built, plan, select_factors(c; kind = :prior); u0 = u)
        all_jac = prepare_query(built, plan, select_factors(c; kind = :log_jacobian); u0 = u)
        @test all_prior(u) ≈ prepare_query(built, plan, :prior)(u)
        @test all_jac(u) ≈ prepare_query(built, plan, :log_jacobian)(u)
        if J == 3
            factors_ad = prepare_query_ad(prepare_ad, built, plan, select_factors(c),
                _FACTOR_REVERSE, u)
            likelihood_ad = prepare_query_ad(prepare_ad, built, plan, :likelihood,
                _FACTOR_REVERSE, u)
            sampler_ad = prepare_query_ad(prepare_ad, built, plan, :sampler,
                _FACTOR_REVERSE, u)
            for point in (u, u .+ 0.17)
                fv, fg = ad_value_and_gradient(factors_ad, point)
                lv, lg = ad_value_and_gradient(likelihood_ad, point)
                sv, sg = ad_value_and_gradient(sampler_ad, point)
                @test fv + lv ≈ sv
                @test fg + lg ≈ sg
            end
        end
        empty = select_factors(c, DensityFactor[])
        @test prepare_query(built, plan, empty)(u) == 0.0
        @test isempty(factor_support(empty; u0 = u))
        @test length(select_factors(c, [first(c), first(c)])) == 1
        @test_throws ArgumentError select_factors(c; site = :z, entries = J + 1)
    end
    @test emitted[1] == emitted[2]
    @test reads[1] == reads[2]
end

@testset "fresh ownership excludes retained parent and child events" begin
    model = @rkppl begin
        m ~ Normal(0.0, 4.0)
        z[1:nb] .~ Normal.(m, 2.0)
        y .~ Normal.(z[block], 1.0)
    end
    for nb in (2, 3)
        plan = model(; nb, block = [nb]) | (; y = [1.4])
        built = build_kernel(plan)
        c = density_factors(built, plan)
        names = coordinate_names(built.layout)
        m = only(findall(==(:m), names))
        z = only(findall(==(Symbol("z.$nb")), names))
        fresh = select_factors(c, [f for f in c if z in factor_coordinates(c, f)])
        @test collect(fresh) == [DensityFactor(:prior, :z, nb)]
        @test factor_coordinates(fresh) == [z]
        @test factor_support(fresh) == sort([m, z])
        parent = select_factors(c, [f for f in c if m in factor_coordinates(c, f)])
        @test collect(parent) == [DensityFactor(:prior, :m, 1)]
        @test isempty(factor_coordinates(select_factors(c, DensityFactor[])))
        @test_throws ArgumentError factor_coordinates(c, DensityFactor(:prior, :z, nb + 1))
        u = [0.2sin(i) for i in 1:built.layout.total]
        ad = prepare_query_ad(prepare_ad, built, plan, fresh, _FACTOR_REVERSE, u;
            coordinates = :support)
        for point in (u, u .+ 0.1)
            value, gradient = ad_value_and_gradient(ad, point, point[ad.coordinates])
            @test value ≈ _factor_normal(point[z], point[m], 2.0)
            @test gradient ≈ [(point[z] - point[m]) / 4, (point[m] - point[z]) / 4]
        end
    end
    derived_model = @rkppl begin
        z ~ Normal(0.0, 1.0)
        squared = z^2
        squared ~ Normal(0.3, 1.7)
    end
    plan = bind_data(derived_model(;))
    built = build_kernel(plan)
    c = density_factors(built, plan)
    derived = select_factors(c; site = :squared)
    @test isempty(factor_coordinates(derived))
    @test factor_support(derived) == [1]
end

@testset "Jacobian support includes authored bound dependencies" begin
    model = @rkppl begin
        center ~ Normal(0.0, 1.3)
        width ~ Exponential(1.7)
        upper = center + width
        bounded ~ Uniform(center, upper)
        unrelated[1:50] .~ Normal.(1.0, 2.0)
    end
    plan = bind_data(model())
    built = build_kernel(plan)
    u = collect(unconstrain(built.layout, (; center = 0.2, width = 1.1,
        bounded = 0.6, unrelated = zeros(50))))
    catalogue = density_factors(built, plan)
    jac = select_factors(catalogue; kind = :log_jacobian, site = :bounded)
    support = factor_support(jac; u0 = u)
    @test support == sort([only(e.offset for e in built.layout.entries if e.name === n)
        for n in (:center, :width, :bounded)])
    q = prepare_query(built, plan, jac; u0 = u)
    recording = FactorReadVector(u, Int[])
    @test q(recording) ≈ q(u)
    @test sort(unique(recording.reads)) == support
    ad = prepare_query_ad(prepare_ad, built, plan, jac, _FACTOR_REVERSE, u;
        coordinates = :support)
    for shift in (0.0, 0.2)
        active = u[support] .+ shift
        point = copy(u)
        point[support] = active
        value, gradient = ad_value_and_gradient(ad, u, active)
        @test value ≈ prepare_query(built, plan, jac; u0 = u)(point)
        h = 1e-5
        for (k, coordinate) in enumerate(support)
            plus, minus = copy(point), copy(point)
            plus[coordinate] += h
            minus[coordinate] -= h
            q = prepare_query(built, plan, jac; u0 = u)
            @test gradient[k] ≈ (q(plus) - q(minus)) / (2h) atol = 1e-8 rtol = 1e-6
        end
    end
end
