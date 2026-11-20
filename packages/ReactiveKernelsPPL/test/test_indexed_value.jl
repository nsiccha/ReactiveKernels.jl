using DifferentiationInterface, Enzyme, ReactiveKernels, ReactiveKernelsPPL, Test

const _INDEXED_MODEL = @rkppl begin
    a ~ Normal(0, 1)
    z[levels(group)] .~ Normal.(0, 1)
    grouped = z[group]
    @plate for i in eachindex(rows)
        coefficients[i] = a .+ grouped[rows[i]]
    end
    @plate for i in eachindex(group)
        offset[i] = a + z[group[i]]
    end
end

function _indexed_fixture(n)
    rows = [[1, 2], Int[], [2], [2, 1, 2]]
    model = _INDEXED_MODEL(; group=collect(1:n), rows)
    built = build_kernel(model)
    names = coordinate_names(built.layout)
    selected = [findfirst(==(:a), names), findfirst(==(Symbol("z.2")), names)]
    base = zeros(length(names))
    base[findfirst(==(Symbol("z.1")), names)] = 1.5
    (; model, built, base, selected, rows)
end

function _indexed_queries(f, name, seed, coordinates=f.selected)
    target = indexed_value(name)
    q = prepare_query(f.built, f.model, target; f.base, coordinates, locality=:error)
    pb = prepare_query_ad(prepare_ad_pullback, f.built, f.model, target,
        AutoEnzyme(; mode=Enzyme.Reverse), seed, f.base[coordinates], 1;
        f.base, coordinates, locality=:error)
    return q, pb
end

@testset "indexed values over a frozen packed point" begin
    f = _indexed_fixture(4)
    q, pb = _indexed_queries(f, :coefficients, zeros(2))
    f.base .= 99
    f.selected .= 1
    for (a, z) in ((0.25, -0.75), (2.0, 3.0)), i in eachindex(f.rows)
        x = [a, z]
        expected = [a + (j == 1 ? 1.5 : z) for j in f.rows[i]]
        seed = Float64.(eachindex(expected))
        dx = [sum(seed), sum(seed[f.rows[i] .== 2])]
        @test q(x, i) == expected
        @test ad_pullback(pb, seed, x, i) == dx
        @test ad_value_and_pullback(pb, seed, x, i) == (expected, dx)
    end
    @test_throws ContractValidationError q([0.0], 1)
    @test_throws ContractValidationError ad_pullback(pb, ones(2), zeros(3), 1)
    @test_throws BoundsError q(zeros(2), 5)
    @test_throws DimensionMismatch ad_pullback(pb, ones(2), zeros(2), 3)
end

@testset "scalar indexed values and fixed coordinate sets" begin
    f = _indexed_fixture(4)
    for coords in (Int[], f.selected[1:1], f.selected)
        q, pb = _indexed_queries(f, :offset, 1.0, coords)
        x = fill(2.0, length(coords))
        @test q(x, 1) == (isempty(x) ? 1.5 : 3.5)
        @test q(x, 2) == 2.0 * length(x)
        @test ad_pullback(pb, 0.5, x, 2) == fill(0.5, length(x))
    end
    for bad in ([0], [f.built.layout.total + 1], [2, 1], [1, 1])
        @test_throws ContractValidationError prepare_query(f.built, f.model,
            indexed_value(:offset); f.base, coordinates=bad)
    end
    @test_throws ContractValidationError prepare_query(f.built, f.model,
        indexed_value(:offset); base=[0.0], coordinates=Int[])
end

# Both preparations evaluate the same selected row over two coordinates.
# A dense shadow of the inactive positions would grow by ≥ 8 bytes apiece.
_indexed_expr_nodes(x) = x isa Expr ? 1 + sum(_indexed_expr_nodes, x.args; init=0) : 0

function _indexed_size(n)
    f = _indexed_fixture(n)
    q, pb = _indexed_queries(f, :coefficients, zeros(2))
    x, seed = [2.0, 3.0], [1.0, 2.0]
    @test q(x, 1) == [3.5, 5.0]
    @test ad_pullback(pb, seed, x, 1) == [3.0, 2.0]
    bytes() = @allocated ad_pullback(pb, seed, x, 1)
    bytes()
    return (; bytes=bytes(), nodes=_indexed_expr_nodes(code_expr(q.kernel)))
end

@testset "indexed pullback storage across inactive extents" begin
    small = _indexed_size(8)
    large = _indexed_size(2048)
    @test large.bytes < small.bytes + 8 * (2048 - 8)
    # A 256-fold increase in binding size must not replicate the row body.
    @test large.nodes < 2 * small.nodes
end
