module CellJointFootprintTests
using Test, ReactiveKernels, ReactiveKernelsPPL
using DifferentiationInterface: AutoEnzyme
import Enzyme
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

# A joint cell latent `field[i] ~ Law(n[i])` packs one block of coordinates per
# index. An observation that reads selected entries of its own block reads
# those coordinates only: the block is a view of the packed vector, so a strict
# cell's footprint is the selected positions, and an empty selection reads
# nothing. A transform that couples the block reads all of it.

@kernel identity_coordinates(u, shape, n) = begin
    return u
end
@kernel zero_jacobian(u, shape, n) = begin
    return 0.0
end
# Cumulative sums: value k reads coordinates 1:k; unit triangular Jacobian.
@kernel cumulative_coordinates(u, shape, n) = begin
    value = cumsum(u)
    return value
end
@kernel cumulative_inverse(value, shape, n) = begin
    u = value .- vcat(zeros(min(length(value), 1)), value[1:end-1])
    return u
end
# Elementwise: value k reads coordinate k, computed for the whole block.
@kernel exp_coordinates(u, shape, n) = begin
    value = exp.(u)
    return value
end
@kernel exp_inverse(value, shape, n) = begin
    u = log.(value)
    return u
end
@kernel exp_jacobian(u, shape, n) = begin
    return sum(u)
end
# A wrong constrained shape: one entry short of the declared one.
@kernel short_coordinates(u, shape, n) = begin
    value = u[2:end]
    return value
end
@kernel block_density(value, n) = begin
    terms = plate(1:n) do j
        return normal(0.0, 1.0).logpdf(value[j])
    end
    return sum(terms)
end

for (Law, constrain, unconstrain, jacobian) in (
        (:IdentityLaw, identity_coordinates, identity_coordinates, zero_jacobian),
        (:CumulativeLaw, cumulative_coordinates, cumulative_inverse, zero_jacobian),
        (:ExpLaw, exp_coordinates, exp_inverse, exp_jacobian),
        (:ShortLaw, short_coordinates, identity_coordinates, zero_jacobian))
    @eval begin
        struct $Law{N}
            n::N
        end
        ReactiveKernelsPPL.sampling_geometry(::Type{$Law}, args, shape) =
            ParameterGeometry((args[1],), args[1]; support = :real,
                constrain = $constrain, unconstrain = $unconstrain, logjac = $jacobian)
        ReactiveKernelsPPL.sampling_distribution_graph(::Type{$Law}, args::Tuple) =
            block_density
    end
end

values_of(law, block) = law === :IdentityLaw ? block :
    law === :CumulativeLaw ? cumsum(block) : law === :ExpLaw ? exp.(block) :
    block[2:end]

# Records every packed position a call reads.
struct ReadVector <: AbstractVector{Float64}
    data::Vector{Float64}
    reads::Vector{Int}
end
Base.size(x::ReadVector) = size(x.data)
Base.IndexStyle(::Type{ReadVector}) = IndexLinear()
Base.getindex(x::ReadVector, i::Int) = (push!(x.reads, i); x.data[i])

function fixture(law, n, picks)
    data = (; n, picks, y = [fill(0.1, length(p)) for p in picks])
    ast = quote
        @plate for i in eachindex(n)
            field[i] ~ $(GlobalRef(@__MODULE__, law))(n[i])
            mu[i] = field[i][picks[i]]
            y[i] .~ Normal.(mu[i], 0.2)
        end
    end
    bound = bind_data(lower_rkppl(ast, data; mod = @__MODULE__, conditioned = (:y,)), data)
    built = build_kernel(bound)
    u = [0.1 + 0.07i for i in 1:built.layout.total]
    (; data, bound, built, u, offsets = cumsum([0; n]))
end

function observation(law, fx, v, i)
    block = v[fx.offsets[i] + 1:fx.offsets[i + 1]]
    sum((-log(2pi) / 2 - log(0.2) - ((0.1 - values_of(law, block)[j]) / 0.2)^2 / 2
         for j in fx.data.picks[i]); init = 0.0)
end

function central(f, v)
    map(eachindex(v)) do j
        plus, minus = copy(v), copy(v)
        plus[j] += 1e-6
        minus[j] -= 1e-6
        (f(plus) - f(minus)) / 2e-6
    end
end

const SELECTIONS = (([3, 2], [[1], Int[]]),
                    ([3, 2, 0], [[3, 1], [2], Int[]]),
                    ([4, 1], [[2, 2, 4], [1]]))

@testset "joint cell latent observations read their selected coordinates" begin
    for law in (:IdentityLaw, :CumulativeLaw, :ExpLaw), (n, picks) in SELECTIONS
        fx = fixture(law, n, picks)
        saved = deepcopy((fx.data, fx.u))
        cell = prepare_cell_query(fx.built, fx.bound, :y; locality = :error)
        footprint = partition_footprint_kernel(cell.kernel)
        ad = prepare_ad(cell.kernel, AutoEnzyme(; mode = Enzyme.Reverse), fx.u, 1;
            active = :unconstrained)
        for i in eachindex(n)
            block = collect(fx.offsets[i] + 1:fx.offsets[i + 1])
            expected = observation(law, fx, fx.u, i)
            @test Base.invokelatest(cell, fx.u, i) ≈ expected
            tracked = ReadVector(copy(fx.u), Int[])
            @test Base.invokelatest(cell, tracked, i) ≈ expected
            read = Base.invokelatest(footprint, fx.u, i).unconstrained
            if law === :IdentityLaw
                # Exact: the selected coordinates, repeated ones once, and
                # nothing for an empty selection or an empty block.
                @test read == sort!(unique(fx.offsets[i] .+ picks[i]))
                @test sort!(unique(tracked.reads)) == read
            else
                # The transform runs on the whole block: its own block only.
                @test read == block
                @test issubset(tracked.reads, read)
            end
            value, gradient = ad_value_and_gradient(ad, fx.u, i)
            @test value ≈ expected
            @test gradient ≈ central(v -> observation(law, fx, v, i), fx.u) atol = 1e-8 rtol = 1e-6
            @test issubset(findall(!iszero, gradient), read)
        end
        # The joint prior stays one factor per index over its whole block.
        catalogue = density_factors(fx.built, fx.bound)
        for i in eachindex(n)
            selected = select_factors(catalogue; kind = :prior, site = :field, entries = i)
            @test factor_coordinates(selected) == collect(fx.offsets[i] + 1:fx.offsets[i + 1])
        end
        @test (fx.data, fx.u) == saved
    end
end

astsize(x::Expr) = 1 + sum(astsize, x.args; init = 0)
astsize(x) = 0
function bytes(f, args...)
    Base.invokelatest(f, args...)
    minimum(@allocated(Base.invokelatest(f, args...)) for _ in 1:3)
end

@testset "joint cell latent observation cost follows the selection" begin
    # Neither the number of indices nor the length of a block changes the
    # emitted programs, or what a cell and its footprint allocate.
    sizes, cell_bytes, footprint_bytes = Int[], Int[], Int[]
    for (groups, len) in ((3, 3), (600, 3), (3, 3000))
        fx = fixture(:IdentityLaw, fill(len, groups), fill([2], groups))
        cell = prepare_cell_query(fx.built, fx.bound, :y; locality = :error)
        footprint = partition_footprint_kernel(cell.kernel)
        @test Base.invokelatest(footprint, fx.u, groups).unconstrained == [len * (groups - 1) + 2]
        @test Base.invokelatest(cell, fx.u, groups) ≈ observation(:IdentityLaw, fx, fx.u, groups)
        push!(sizes, astsize(code_expr(cell.kernel)) + astsize(code_expr(footprint)))
        push!(cell_bytes, bytes(cell, fx.u, 1))
        push!(footprint_bytes, bytes(footprint, fx.u, 1))
    end
    @test allequal(sizes)
    @test maximum(cell_bytes) <= minimum(cell_bytes) + 512
    @test maximum(footprint_bytes) <= minimum(footprint_bytes) + 1024
end

@testset "joint cell latent observations keep the shape check and the sampler" begin
    fx = fixture(:ShortLaw, [3, 2], [[1], Int[]])
    cell = prepare_cell_query(fx.built, fx.bound, :y; locality = :error)
    for i in 1:2
        @test_throws ArgumentError Base.invokelatest(cell, fx.u, i)
    end
    for law in (:IdentityLaw, :CumulativeLaw), (n, picks) in SELECTIONS
        fx = fixture(law, n, picks)
        function joint(v)
            total = 0.0
            for i in eachindex(n)
                w = values_of(law, v[fx.offsets[i] + 1:fx.offsets[i + 1]])
                total += sum(-log(2pi) / 2 .- w .^ 2 ./ 2; init = 0.0) +
                    observation(law, fx, v, i)
            end
            total
        end
        @test Base.invokelatest(prepare_query(fx.built, fx.bound, :sampler), fx.u) ≈ joint(fx.u)
        sampler = prepare_sampler(fx.built, fx.bound, fx.u;
            backend = AutoEnzyme(; mode = Enzyme.Reverse))
        value, gradient = sampler_value_and_gradient!(sampler, similar(fx.u), fx.u)
        @test value ≈ joint(fx.u)
        @test gradient ≈ central(joint, fx.u) atol = 1e-8 rtol = 1e-6
    end
end
end
