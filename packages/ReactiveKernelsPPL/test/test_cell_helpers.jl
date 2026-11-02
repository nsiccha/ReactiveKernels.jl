module OrdinaryHelperCellTests
using ReactiveKernels, ReactiveKernelsPPL, Enzyme, Test
using DifferentiationInterface: AutoEnzyme, SecondOrder

plain(z, indices) = [z[i] for i in indices]
wrapped(z, indices) = plain(z, indices)
rowwise(z, indices, scale) = [scale * sin(z[i]) + z[i]^2 for i in indices]
crossrow(z, indices) = [z[i] + sum(z) for i in indices]

const plain_model = @rkppl begin
    beta ~ Normal(0, 1)
    z[1:n] .~ Normal.(0, 1)
    field = plain(z, indices)
    @plate for i in eachindex(x)
        y[i] ~ Poisson(exp(beta * x[i] + field[i]))
    end
end
const wrapped_model = @rkppl begin
    beta ~ Normal(0, 1)
    z[1:n] .~ Normal.(0, 1)
    field = wrapped(z, indices)
    @plate for i in eachindex(x)
        y[i] ~ Poisson(exp(beta * x[i] + field[i]))
    end
end
const rowwise_model = @rkppl begin
    beta ~ Normal(0, 1)
    z[1:n] .~ Normal.(0, 1)
    field = rowwise(z, indices, beta)
    @plate for i in eachindex(x)
        y[i] ~ Poisson(exp(beta * x[i] + field[i]))
    end
end
const crossrow_model = @rkppl begin
    z[1:n] .~ Normal.(0, 1)
    field = crossrow(z, indices)
    @plate for i in eachindex(x)
        y[i] ~ Poisson(exp(field[i]))
    end
end

struct RecordingPoint <: AbstractVector{Float64}
    values::Vector{Float64}
    reads::Vector{Int}
end
Base.size(p::RecordingPoint) = size(p.values)
Base.IndexStyle(::Type{RecordingPoint}) = IndexLinear()
Base.getindex(p::RecordingPoint, i::Int) = (push!(p.reads, i); p.values[i])
nodes(x::Expr) = 1 + sum(nodes, x.args; init = 0)
nodes(x) = 0

@testset "ordinary helper rows in strict values and factors" begin
    for model in (plain_model, wrapped_model, rowwise_model)
        structures = Int[]
        for n in (4, 80)
            indices, x = [n, 2, 1, 2], [0.4, -0.2, 0.8, 0.1]
            bound = model(; n, indices, x) | (; y=ones(Int, length(indices)))
            built = build_kernel(bound)
            names = coordinate_names(built.layout)
            beta = findfirst(==(:beta), names)
            z = [findfirst(==(Symbol("z.$i")), names) for i in 1:n]
            u = collect(range(-0.2, 0.3; length=length(names)))
            q = prepare_cell_query(built, bound, :y; partitioned=(:z=>collect(1:n),),
                                   locality=:error, u0=u)
            factors = prepare_cell_factors(built, bound, :y; locality=:error)
            outer = prepare_cell_factor_query(factors, :outer)
            @test isempty(cell_reductions(factors))
            push!(structures, nodes(code_expr(outer.kernel)))
            for c in eachindex(indices)
                zi = u[z[indices[c]]]
                field = model === rowwise_model ? u[beta] * sin(zi) + zi^2 : zi
                eta = u[beta] * x[c] + field
                expected = eta - exp(eta)
                @test q(u, c) ≈ expected
                @test outer(u, Float64[], c) ≈ expected
                p = RecordingPoint(u, Int[])
                @test outer(p, Float64[], c) ≈ expected
                @test sort!(unique!(p.reads)) == sort([beta, z[indices[c]]])
                support = Base.invokelatest(partition_footprint_kernel(outer.kernel),
                                           u, Float64[], c).unconstrained
                @test support == sort([beta, z[indices[c]]])
            end
            old = q(u, 1)
            u[z[3]] += 0.7
            @test q(u, 1) == old
            u[z[n]] += 0.2
            @test q(u, 1) != old
            u[beta] += 0.1
            @test q(u, 1) ≈ outer(u, Float64[], 1)
        end
        @test structures[1] == structures[2]
    end
end

@testset "ordinary helper native compact derivatives" begin
    n, indices, x = 4, [4, 2, 1, 2], [0.4, -0.2, 0.8, 0.1]
    for model in (plain_model, rowwise_model)
        bound = model(; n, indices, x) | (; y=ones(Int, 4))
        built = build_kernel(bound)
        names = coordinate_names(built.layout)
        beta = findfirst(==(:beta), names)
        z = [findfirst(==(Symbol("z.$i")), names) for i in 1:n]
        u = collect(range(-0.2, 0.3; length=length(names)))
        selection = sort([beta, z[4]])
        ad = prepare_compact_cell_ad(prepare_ad, built, bound, :y, u, selection,
                                    AutoEnzyme(; mode=Enzyme.Reverse); locality=:error)
        factors = prepare_cell_factors(built, bound, :y; locality=:error)
        factor_ad = prepare_cell_factor_ad(prepare_ad, factors, :outer, u, selection,
                                           AutoEnzyme(; mode=Enzyme.Reverse))
        for cell in (1, 2, 4)
            coordinate = z[indices[cell]]
            selection = sort([beta, coordinate])
            value, gradient = ad_value_and_gradient(ad, u, u[selection], selection, cell)
            factor_value, (sum_gradient, factor_gradient) = ad_value_and_gradient(
                factor_ad, u, Float64[], u[selection], selection, cell)
            zi, b = u[coordinate], u[beta]
            field = model === rowwise_model ? b * sin(zi) + zi^2 : zi
            eta = b * x[cell] + field
            d_beta = model === rowwise_model ? x[cell] + sin(zi) : x[cell]
            d_z = model === rowwise_model ? b * cos(zi) + 2zi : 1.0
            @test value ≈ eta - exp(eta)
            oracle = Dict(beta => (1-exp(eta))*d_beta, coordinate => (1-exp(eta))*d_z)
            @test gradient ≈ [oracle[i] for i in selection]
            @test factor_value ≈ value
            @test isempty(sum_gradient)
            @test factor_gradient ≈ gradient
        end
    end
end

@testset "ordinary helper compact Hessian and third-order contraction" begin
    n, indices, x = 4, [4, 2, 1, 2], [0.4, -0.2, 0.8, 0.1]
    bound = plain_model(; n, indices, x) | (; y=ones(Int, 4))
    built = build_kernel(bound)
    names = coordinate_names(built.layout)
    beta = findfirst(==(:beta), names)
    z = [findfirst(==(Symbol("z.$i")), names) for i in 1:n]
    u = collect(range(-0.2, 0.3; length=length(names)))
    selection(cell) = sort([beta, z[indices[cell]], z[3]])
    units = ntuple(k -> Float64[j == k for j in 1:3], 3)
    v = [0.4, -0.7, 1.1]
    second = SecondOrder(AutoEnzyme(; mode=Enzyme.Reverse),
                         AutoEnzyme(; mode=Enzyme.Forward))
    third = SecondOrder(second, AutoEnzyme(; mode=Enzyme.Forward))
    h = prepare_compact_cell_ad(prepare_ad_hvp, built, bound, :y, u, selection(1),
                               second, units; locality=:error)
    t = prepare_compact_cell_ad(prepare_ad_third, built, bound, :y, u, selection(1),
                               third, (v,); locality=:error)
    for cell in (1, 2)
        slots = selection(cell)
        coordinate = z[indices[cell]]
        rate = exp(u[beta] * x[cell] + u[coordinate])
        d = [j == beta ? x[cell] : j == coordinate ? 1.0 : 0.0 for j in slots]
        gradient = (1-rate) .* d
        hessian = -rate .* (d * d')
        contraction = -rate * sum(d .* v)^2 .* d
        gh, columns = ad_gradient_and_hvp(h, units, u, u[slots], slots, cell)
        gt, (hv,), (tv,) = ad_gradient_hvp_and_third(t, (v,), u, u[slots], slots, cell)
        @test gh ≈ gradient
        @test reduce(hcat, columns) ≈ hessian
        @test gt ≈ gradient
        @test hv ≈ hessian * v
        @test tv ≈ contraction
    end
end

@testset "cross-row helper keeps strict locality checks" begin
    bound = crossrow_model(; n=4, indices=[4, 2, 1, 3], x=zeros(4)) | (; y=ones(Int, 4))
    built = build_kernel(bound)
    # Every output depends on every z through sum(z), so selecting one cell
    # cannot satisfy the selected-coordinate locality contract.
    @test_throws PartitionLocalityError prepare_cell_query(built, bound, :y;
        partitioned=(:z=>collect(1:4),), locality=:error, u0=zeros(4))
end
end
