using Test, ReactiveKernels, ReactiveKernelsPPL
using CategoricalArrays: categorical, compress, levels!, ordered!

@testset "a density selection reads categorical data across pools" begin
    model = @rkppl begin
        theta ~ Normal(0.0, length(labels))
        y .~ Normal.(theta, 1.0)
    end
    entries = ["a", "b", "a", "c"]
    declared = ["c", "b", "a", "unused"]
    column() = categorical(entries; levels = declared, ordered = true)
    binding(labels; extra...) = model(; labels, extra...) | (; y = [0.2, 0.4])
    original, independent = column(), column()
    plan = binding(original)
    built = build_kernel(plan)
    selected = select_factors(density_factors(built, plan))
    u = [0.3]
    score(labels) = prepare_query(built, binding(labels), selected; u0 = u)(u)
    reference = -log(4sqrt(2pi)) - (only(u) / 4)^2 / 2
    @test score(independent) ≈ reference
    # Comparing values can populate a pool's internal comparison cache.
    @test isequal(first(original), first(independent))
    @test score(original) ≈ reference
    @test score(independent) ≈ reference

    for changed in (categorical(reverse(entries); levels = declared, ordered = true),
            levels!(column(), reverse(declared)), levels!(column(), declared[1:3]),
            ordered!(column(), false), compress(column()), reshape(column(), 2, 2))
        @test_throws ContractValidationError score(changed)
    end

    function partition(labels)
        obs = binding(labels; unit = [2, 1])
        jax_source(build_kernel(obs), obs, :y; by = :unit, global_factors = selected,
            unit_factors = Dict(), expected_layout = built.layout)
    end
    same, rebuilt = partition(original), partition(independent)
    @test rebuilt.source.text == same.source.text
    @test rebuilt.partition_data == same.partition_data
    @test rebuilt.partition_data.labels == [1, 2]
    @test_throws "density selection belongs to another program or data binding" partition(
        ordered!(column(), false))

    original[3] = "b"
    @test_throws "density selection belongs to another program or data binding" prepare_query(
        built, plan, selected; u0 = u)
end

@testset "categorical fingerprints retain unobserved levels and missing entries" begin
    fingerprint = ReactiveKernelsPPL._binding_fingerprint
    for entries in (String[], Union{String,Missing}[missing, missing],
            Union{String,Missing}["b", missing])
        make() = categorical(entries; levels = ["a", "b"])
        a, b = make(), make()
        @test fingerprint(a) == fingerprint(b)
        @test fingerprint(view(a, :)) == fingerprint(view(b, :))
        levels!(b, ["b", "a"])
        @test fingerprint(a) != fingerprint(b)
        @test fingerprint(view(a, :)) != fingerprint(view(b, :))
    end
    many = categorical(fill!(Vector{Union{String,Missing}}(undef, 4096), "a"))
    original = fingerprint(many)
    many[29] = missing
    @test fingerprint(many) != original
    a, b = (categorical(["a", "b", "a"]) for _ in 1:2)
    # Individual categorical values can also occur outside a pooled array.
    @test fingerprint((a[2], a[1])) == fingerprint((b[2], b[1]))
    ordered!(b, true)
    @test fingerprint((a[2], a[1])) != fingerprint((b[2], b[1]))
end
