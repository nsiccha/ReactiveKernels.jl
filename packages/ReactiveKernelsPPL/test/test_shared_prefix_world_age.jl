module SharedPrefixWorldAgeTests
using ReactiveKernels, ReactiveKernelsPPL, Distributions, Enzyme, Test
using DifferentiationInterface: AutoEnzyme

struct RecordingPoint <: AbstractVector{Float64}
    values::Vector{Float64}
    reads::Vector{Int}
end
Base.size(point::RecordingPoint) = size(point.values)
Base.IndexStyle(::Type{RecordingPoint}) = IndexLinear()
function Base.getindex(point::RecordingPoint, i::Int)
    push!(point.reads, i)
    point.values[i]
end

function exercise(kind, n)
    mod = Module(gensym(:FreshSharedPrefix))
    Core.eval(mod, :(using ReactiveKernelsPPL, Distributions))
    Core.eval(mod, quote
        row_shift(x) = 2 .* x
        shared_values(a, x) = a .+ sum(x)
    end)
    model = quote
        a[1:2] .~ Normal.(0, 1)
        z[levels(subject)] .~ Normal.(0, 1)
        values = shared_values(a, row_shift(x))
        eta = z[subject]
        mu = values[subject] .+ eta
        y .~ Normal.(mu, 1)
    end
    data = (; subject=repeat(1:2; inner=n), x=collect(1:2n) ./ (2n),
        y=sin.(1:2n))
    original = deepcopy(data)
    bound = bind_data(lower_rkppl(model, data; mod, conditioned=(:y,)), data)
    built = build_kernel(bound)
    names = coordinate_names(built.layout)
    shared = findall(name -> startswith(string(name), "a."), names)
    locals = findall(name -> startswith(string(name), "z."), names)
    u = zeros(built.layout.total)
    u[shared] = [0.2, -0.4]
    u[locals] = [-0.1, 0.3]

    # Stay in this function's entry world after creating the helper methods.
    # Each public constructor must cover its own preparation-time calls;
    # wrapping this exercise in invokelatest would conceal that boundary.
    q = if kind === :cell
        prepare_cell_query(built, bound, :y;
            partitioned=:z, u0=u, locality=:error)
    elseif kind === :partition
        prepare_partition(built, bound, :y; by=:subject,
            partitioned=:z, u0=u, locality=:error)
    else
        prepare_partition_sampler(built, bound, :y, u; by=:subject,
            partitioned=:z, locality=:error, compact=false,
            backend=AutoEnzyme(; mode=Enzyme.Reverse))
    end
    @test q.prefix !== nothing
    if kind !== :sampler
        @test !any(e -> e.status === :whole, ReactiveKernels.partition_locality(q))
    end
    initial = copy(u)
    for change in (0.0, 0.2), label in (1, 2, 1)
        point = copy(u)
        point[shared] .+= change
        point[locals] .+= change
        rows = kind === :cell ? [1 + (label - 1) * n] : findall(==(label), data.subject)
        residual = data.y[rows] .- (point[shared[label]] + point[locals[label]] + sum(2 .* data.x))
        expected = sum(logpdf.(Normal(), residual))
        index = kind === :cell ? only(rows) : label
        if kind === :sampler
            gradient = zero(point)
            value, returned = partition_value_and_gradient!(q, gradient, point, index)
            oracle = zero(point)
            oracle[shared[label]] = sum(residual)
            oracle[locals[label]] = sum(residual)
            @test value ≈ expected
            @test returned === gradient
            @test gradient ≈ oracle
            @test partition_footprint(q, label; shared=false) == [locals[label]]
        else
            recorded = RecordingPoint(point, Int[])
            @test q(recorded, index) ≈ expected
            @test intersect(unique(recorded.reads), locals) == [locals[label]]
            @test q(point, index) ≈ expected
        end
    end
    @test data == original
    @test u == initial
end

@testset "shared-prefix preparation uses freshly defined helpers" begin
    for kind in (:cell, :partition, :sampler), n in (2, 7)
        @testset "$kind, $n rows per subject" begin
            exercise(kind, n)
        end
    end
end
end
