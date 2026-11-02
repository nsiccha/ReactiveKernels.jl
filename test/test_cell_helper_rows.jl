module HelperRowTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test

weighted(z, w, indices, scale) = [z[i] + scale * w[i] for i in indices]
@kernel helper_rows(z::AbstractVector{Float64}, w::AbstractVector{Float64},
                    indices::AbstractVector{Int}, subject::AbstractVector{Int},
                    scale::Float64) = begin
    field = weighted(z, w, indices, scale)
    density = plate(field) do f
        f^2
    end
    total = sum(density)
end

@testset "helper gathers preserve partition order and derivatives" begin
    z, w, scale = [0.2, 0.7, -0.3, 0.4], [0.6, -0.2, 0.1, 0.8], 0.9
    for indices in ([4, 2, 4, 1], 1:4, 4:-1:1)
        subject = [1, 2, 1, 2]
        spec = plate_partition(helper_rows, :density; by=:subject, index=:label,
                               locality=:error)
        k = prepare(spec; have=(:z, :w, :indices, :subject, :scale, :label),
            want=:partition_total, bound=(; indices, subject))
        @test isempty(ReactiveKernels._code_findings(k,
            (typeof(z), typeof(w), typeof(scale), Int);
            recurse=ReactiveKernels._rk_or_generated(@__MODULE__)))
        ad = prepare_ad(k, AutoEnzyme(; mode=Enzyme.Reverse), z, w, scale, 1; active=:z)
        for label in 1:2
            selected = indices[findall(==(label), subject)]
            expected = sum((z[i] + scale*w[i])^2 for i in selected; init=0.0)
            @test k(z, w, scale, label) ≈ expected
            value, gradient = ad_value_and_gradient(ad, z, w, scale, label)
            @test value ≈ expected
            @test gradient ≈ [2count(==(i), selected)*(z[i]+scale*w[i]) for i in eachindex(z)]
        end
    end
end

# Indexing and iteration need not agree for a custom vector. Recovering a
# comprehension as an indexed gather without proving that order is incorrect.
struct ReversedIteration <: AbstractVector{Int}
    values::Vector{Int}
end
Base.size(x::ReversedIteration) = size(x.values)
Base.getindex(x::ReversedIteration, i::Int) = x.values[i]
Base.iterate(x::ReversedIteration, state=length(x)) =
    state == 0 ? nothing : (x.values[state], state-1)
@testset "unknown helper iterator semantics retain the ordinary path" begin
    indices = ReversedIteration([1, 2, 3, 4])
    subject = [1, 2, 1, 2]
    spec = plate_partition(helper_rows, :density; by=:subject, index=:label,
                           locality=:error)
    @test_throws PartitionLocalityError prepare(spec;
        have=(:z, :w, :indices, :subject, :scale, :label), want=:partition_total,
        bound=(; indices, subject))
end

@kernel selected_helper_rows(z::AbstractVector{Float64}, w::AbstractVector{Float64},
                            indices::Vector{Int}, selections::Vector{Vector{Int}},
                            scale::Float64) = begin
    field = weighted(z, w, indices, scale)
    density = plate(selections) do selected
        sum(field[selected])
    end
end

@testset "empty helper row windows retain values and ordinary derivatives" begin
    z, w, scale = [0.2, 0.7, -0.3, 0.4], [0.6, -0.2, 0.1, 0.8], 0.9
    indices, selections = [4, 2, 4, 1], [Int[], [1, 3], [2, 4]]
    spec = plate_cell(selected_helper_rows, :density; index=:cell, locality=:error)
    k = prepare(spec; have=(:z, :w, :indices, :selections, :scale, :cell),
                want=:density_cell, bound=(; indices, selections))
    ad = prepare_ad(k, AutoEnzyme(; mode=Enzyme.Reverse), z, w, scale, 1; active=:z)
    for cell in eachindex(selections)
        chosen = indices[selections[cell]]
        expected = sum(z[i] + scale*w[i] for i in chosen; init=0.0)
        @test k(z, w, scale, cell) ≈ expected
        value, gradient = ad_value_and_gradient(ad, z, w, scale, cell)
        @test value ≈ expected
        @test gradient ≈ [count(==(i), chosen) for i in eachindex(z)]
    end
end

dispatched(z, indices) = [z[i] for i in indices]
dispatched(z::Vector{Float64}, indices) = [z[i] + sum(z) for i in indices]
@kernel dispatched_rows(z::AbstractVector{Float64}, indices::Vector{Int}) = begin
    field = dispatched(z, indices)
    density = plate(field) do f
        f^2
    end
end
@testset "abstract helper arguments do not select a concrete dispatch fallback" begin
    spec = plate_cell(dispatched_rows, :density; index=:cell, locality=:error)
    @test_throws PartitionLocalityError prepare(spec; have=(:z, :indices, :cell),
        want=:density_cell, bound=(indices=[4, 2, 1, 3],))
end

trajectory(x, scale) = [sin(x), scale*x^2]
trajectories(z, indices, scale) = [trajectory(z[i], scale) for i in indices]
@kernel trajectory_rows(z::AbstractVector{Float64}, indices::Vector{Int},
                        scale::Float64) = begin
    paths = trajectories(z, indices, scale)
    density = plate(paths) do path
        sum(path)
    end
end
@testset "ordinary helper calls returning one array per selected row" begin
    z, scale, indices = [0.2, 0.7, -0.3, 0.4], 0.9, [4, 2, 4, 1]
    spec = plate_cell(trajectory_rows, :density; index=:cell, locality=:error)
    k = prepare(spec; have=(:z, :indices, :scale, :cell), want=:density_cell,
                bound=(; indices))
    ad = prepare_ad(k, AutoEnzyme(; mode=Enzyme.Reverse), z, scale, 1; active=:z)
    for cell in eachindex(indices)
        i = indices[cell]
        value, gradient = ad_value_and_gradient(ad, z, scale, cell)
        @test k(z, scale, cell) ≈ sin(z[i]) + scale*z[i]^2
        @test value ≈ sin(z[i]) + scale*z[i]^2
        @test gradient ≈ [j==i ? cos(z[i])+2scale*z[i] : 0.0 for j in eachindex(z)]
    end
end
end
