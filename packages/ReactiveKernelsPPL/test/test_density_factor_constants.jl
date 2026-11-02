using Test, ReactiveKernels, ReactiveKernelsPPL, Enzyme

const _CONSTANT_FACTOR_REVERSE =
    ReactiveKernels.DifferentiationInterface.AutoEnzyme(; mode = Enzyme.Reverse)

struct ConstantFactorReads <: AbstractVector{Float64}
    values::Vector{Float64}
    reads::Vector{Int}
end
Base.size(v::ConstantFactorReads) = size(v.values)
Base.IndexStyle(::Type{ConstantFactorReads}) = IndexLinear()
Base.getindex(v::ConstantFactorReads, i::Int) = (push!(v.reads, i); v.values[i])

@testset "selected flat events retain ownership without coordinate reads" begin
    model = @rkppl begin
        z[1:nb] .~ Flat()
        y .~ Normal.(z[block], 1)
    end
    emitted = Int[]
    nodes(x) = x isa Expr ? 1 + sum(nodes, x.args; init = 0) : 1
    for nb in (0, 1, 2, 301)
        block = nb <= 1 ? Int[] : [nb]
        plan = model(; nb, block) | (; y = fill(1.4, length(block)))
        built = build_kernel(plan)
        c = density_factors(built, plan)
        u = fill(0.2, nb)
        @test length(c) == nb
        selections = nb == 0 ? (Int[],) : (Int[], [nb], collect(1:nb))
        for entries in selections
            selected = select_factors(c; site = :z, entries)
            @test factor_coordinates(selected) == entries
            @test isempty(factor_support(selected; u0 = u))
            for event in selected
                @test factor_coordinates(c, event) == [event.entry]
                @test isempty(factor_support(c, event; u0 = u))
            end
            q = prepare_query(built, plan, selected; u0 = u)
            recording = ConstantFactorReads(fill(NaN, nb), Int[])
            @test q(recording) === 0.0
            @test isempty(recording.reads)
            full = prepare_query_ad(prepare_ad, built, plan, selected,
                _CONSTANT_FACTOR_REVERSE, u)
            compact = prepare_query_ad(prepare_ad, built, plan, selected,
                _CONSTANT_FACTOR_REVERSE, u; coordinates = :support)
            @test isempty(compact.coordinates)
            for point in (u, u .+ 0.7)
                value, gradient = ad_value_and_gradient(full, point)
                @test value === 0.0
                @test gradient == zeros(nb)
                value, gradient = ad_value_and_gradient(compact, point, Float64[])
                @test value === 0.0
                @test isempty(gradient)
            end
            entries == [nb] && push!(emitted, nodes(code_expr(q)))
        end
        if !isempty(block)
            fresh = select_factors(c; site = :z, entries = nb)
            likelihood = prepare_query_ad(prepare_ad, built, plan, :likelihood,
                _CONSTANT_FACTOR_REVERSE, u)
            value, gradient = ad_value_and_gradient(likelihood, u)
            @test factor_coordinates(fresh) == [nb]
            @test value ≈ -0.5log(2pi) - 0.5(1.4 - u[nb])^2
            @test gradient ≈ [zeros(nb - 1); 1.4 - u[nb]]
        end
    end
    @test all(==(first(emitted)), emitted)
end

@testset "mixed constant and active factor selections" begin
    model = @rkppl begin
        z[1:nb] .~ Flat()
        w[1:nb] .~ Normal.(0, 2)
    end
    plan = bind_data(model(; nb = 3))
    built = build_kernel(plan)
    c = density_factors(built, plan)
    selected = select_factors(c, [DensityFactor(:prior, :z, 2),
        DensityFactor(:prior, :w, 3), DensityFactor(:prior, :z, 2)])
    zi = only(factor_coordinates(c, DensityFactor(:prior, :z, 2)))
    wi = only(factor_coordinates(c, DensityFactor(:prior, :w, 3)))
    u = [0.2sin(i) for i in 1:built.layout.total]
    @test length(selected) == 2
    @test factor_coordinates(selected) == sort([zi, wi])
    @test factor_support(selected; u0 = u) == [wi]
    q = prepare_query(built, plan, selected; u0 = u)
    compact = prepare_query_ad(prepare_ad, built, plan, selected,
        _CONSTANT_FACTOR_REVERSE, u; coordinates = :support)
    for delta in (0.0, 0.4)
        active = u[[wi]] .+ delta
        point = copy(u)
        point[wi] = only(active)
        recording = ConstantFactorReads(point, Int[])
        expected = -0.5log(2pi) - log(2.0) - point[wi]^2 / 8
        @test q(recording) ≈ expected
        @test unique(recording.reads) == [wi]
        value, gradient = ad_value_and_gradient(compact, u, active)
        @test value ≈ expected
        @test gradient ≈ [-point[wi] / 4]
    end
end
