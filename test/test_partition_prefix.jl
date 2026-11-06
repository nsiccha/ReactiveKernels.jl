# A structural shared/local cut must preserve full derivatives, including
# cotangents through cached shared intermediates. Finite differences are an
# independent oracle; this is not a numerical sparsity test.
module PartitionPrefixTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test
const _PREFIX_RK = ReactiveKernels

@kernel _prefix_packed_model(u::Vector{Float64}, group::Vector{Int},
                     y::Vector{Float64}, subject::Vector{Int}) = begin
    theta = view(u, 1:3)
    transformed = exp.(theta)
    offset::Float64 = sum(transformed)
    z = view(u, 4:9)
    mu = offset .+ z[group]
    density = plate(mu, y) do m, obs
        -(obs - m)^2 / 2
    end
    total = sum(density)
end

@testset "shared cut preserves multi-output ownership" begin
    graph = Graph()
    x = value!(graph, :x, Float64)
    theta = value!(graph, :theta, Float64)
    a = value!(graph, :a, Float64)
    b = value!(graph, :b, Float64)
    c = value!(graph, :c, Float64)
    result = value!(graph, :result, Float64)
    add!(graph, x => (a, b), x -> (x + 1, x + 2))
    add!(graph, theta => (b, c), theta -> (theta + 10, theta + 20))
    add!(graph, (a, b, c) => result, (a, b, c) -> a + b + c)
    original = prepare(graph; have=(x, theta), want=result)
    parts = _PREFIX_RK._partition_shared_parts(original.plan, (); varying=(:x,))
    @test only(parts.frontier).name === :c
    shared = prepare(parts.prefix)
    rest = prepare(parts.residual)
    for x0 in (1.0, 2.0), theta0 in (3.0, 4.0)
        @test rest(x0, theta0, shared(theta0)) == original(x0, theta0)
    end
end

# The same model reading its packed blocks at bound positions, as
# ReactiveKernelsPPL reads declared arrays (snag `rkppl-sized-vect-69604d32`):
# its bound endpoints keep the range view, read exactly as a literal one.
@kernel _prefix_bound_range_model(u::Vector{Float64}, group::Vector{Int},
                     y::Vector{Float64}, subject::Vector{Int}, lo::Int, hi::Int,
                     zlo::Int, zhi::Int) = begin
    theta = view(u, lo:hi)
    transformed = exp.(theta)
    offset::Float64 = sum(transformed)
    z = view(u, zlo:zhi)
    mu = offset .+ z[group]
    density = plate(mu, y) do m, obs
        -(obs - m)^2 / 2
    end
    total = sum(density)
end

u = [0.1, 0.2, -0.3, -0.2, 0.3, 0.1, 0.6, -0.4, 0.2]
group = [1, 2, 1, 3, 4, 3, 5, 6, 5]
subject = repeat(1:3; inner=3)
y = sin.(eachindex(group))
coordinate_labels = Union{Missing,Int}[missing, missing, missing, 1, 1, 2, 2, 3, 3]
spec = plate_partition(_prefix_packed_model, :density; by=:subject, index=:label,
                       partitioned=(:u => coordinate_labels,))
kernel = prepare(spec; have=(:u, :group, :y, :subject, :label), want=:partition_total,
                 bound=(; group, y, subject))
split = _PREFIX_RK._partition_shared_parts(kernel.plan, (:u => coordinate_labels,); varying=(:label,))
@test getproperty.(split.frontier, :name) == [:offset]
@test _PREFIX_RK._partition_prefix_positions(split.prefix, :u) == [1, 2, 3]
@testset "bound range endpoints keep the prefix's exact read set" begin
    bound_spec = plate_partition(_prefix_bound_range_model, :density; by=:subject,
        index=:label, partitioned=(:u => coordinate_labels,))
    bound_kernel = prepare(bound_spec;
        have=(:u, :group, :y, :subject, :lo, :hi, :zlo, :zhi, :label),
        want=:partition_total, bound=(; group, y, subject, lo=1, hi=3, zlo=4, zhi=9))
    bound_split = _PREFIX_RK._partition_shared_parts(bound_kernel.plan,
        (:u => coordinate_labels,); varying=(:label,))
    @test _PREFIX_RK._partition_prefix_positions(bound_split.prefix, :u) == [1, 2, 3]
    for label in 1:3
        @test bound_kernel(u, label) == kernel(u, label)
    end
end
prefix = prepare(split.prefix; exemplars=(u,))
cached = prefix(u)
residual = prepare(split.residual; exemplars=(u, 1, cached))
backend = AutoEnzyme(; mode=Enzyme.Reverse)
full_ad = prepare_ad(kernel, backend, u, 1; active=:u)
residual_ad = prepare_ad(residual, backend, u, 1, cached; active=(:u, :offset))
prefix_ad = prepare_ad_pullback(prefix, backend, 1.0, u; active=:u)
function staged(point, label, cached_value)
    value, (direct, seed) = ad_value_and_gradient(residual_ad, point, label, cached_value)
    propagated = ad_pullback(prefix_ad, seed, point)
    value, direct + propagated
end

@testset "staged shared prefix preserves primal and full gradient" begin
    for label in 1:3, (change, fresh_prefix) in ((0.0, false), (0.15, false), (-0.1, true))
        point = copy(u)
        point[4:end] .+= change
        fresh_prefix && (point[1:3] .+= 0.25)
        boundary = fresh_prefix ? prefix(point) : cached
        value, gradient = staged(point, label, boundary)
        full_value, full_gradient = ad_value_and_gradient(full_ad, point, label)
        @test value ≈ full_value
        @test gradient ≈ full_gradient
        for j in eachindex(point)
            direction = Float64.(eachindex(point) .== j)
            finite = (kernel(point + 1e-6direction, label) - kernel(point - 1e-6direction, label)) / 2e-6
            @test gradient[j] ≈ finite atol=1e-6
        end
        @test all(!iszero, gradient[1:3])
        expected = sort!([1, 2, 3, 2label+2, 2label+3])
        @test findall(!iszero, gradient) == expected
    end
end

@kernel _prefix_reads(u::Vector{Float64}, indices::Vector{Int}, chosen::Int, label::Float64) = begin
    a = u[1]
    block = view(u, 2:3)
    picked = u[indices]
    dynamic = u[chosen]
    opaque = sin(sum(u))
    total = a + sum(block) + sum(picked) + dynamic + opaque + label
end

@testset "shared reads use structural positions, not current values" begin
    point = [0.2, 0.1, -0.3, 0.4, 0.7]
    labels = Union{Missing,Int}[missing, missing, missing, 1, 2]
    original = prepare(_prefix_reads; have=(:u, :indices, :chosen, :label), want=:total,
        bound=(indices=[3, 1, 3],))
    parts = _PREFIX_RK._partition_shared_parts(original.plan, (:u => labels,);
        varying=(:chosen, :label))
    @test Set(v.name for v in parts.frontier) == Set((:a, :block, :picked))
    prefix = prepare(parts.prefix)
    boundary = prefix(point)
    residual = prepare(parts.residual; exemplars=(point, 4, 1.0, boundary...))
    for chosen in (1, 4, 5), shift in (0.0, 0.25)
        local_point = copy(point)
        local_point[4:5] .+= shift
        @test residual(local_point, chosen, 2.0, boundary...) ≈ original(local_point, chosen, 2.0)
        shared_point = point .+ shift
        @test residual(shared_point, chosen, 3.0, prefix(shared_point)...) ≈
              original(shared_point, chosen, 3.0)
    end
    @test_throws BoundsError residual(point, 0, 1.0, boundary...)
    @test _PREFIX_RK._partition_shared_parts(original.plan, (:u => [1, 1, 1, 2, 2],);
        varying=(:chosen, :label)) === nothing
    # Declaring the whole port shared also admits an opaque whole read.
    all_shared = _PREFIX_RK._partition_shared_parts(original.plan, (:u => fill(missing, 5),);
        varying=(:chosen, :label))
    @test :opaque in (v.name for v in all_shared.frontier)
    @test _PREFIX_RK._partition_prefix_positions(all_shared.prefix, :u) === :whole
end

@kernel _prefix_branch_reads(u::Vector{Float64}) = begin
    value = u[1] > 0 ? exp(u[2]) : exp(u[3])
end
@testset "prefix active reads never specialize to an exemplar's branch" begin
    kernel = prepare(_prefix_branch_reads; have=:u, want=:value,
        exemplars=([1.0, 0.2, 0.3],))
    @test _PREFIX_RK._partition_prefix_positions(kernel.plan, :u) === :whole
end

@testset "pullbacks include lowered producers absent from the source graph" begin
    graph = Graph()
    x = value!(graph, :x, Float64)
    a = value!(graph, :a, Float64)
    add!(graph, x => a, sin)
    original = ReactiveKernels.plan(graph; have=(x,), want=(a,))
    # Outlining introduces values and recipes in the selected plan without
    # registering those producers as alternate source-graph recipes.
    b = value!(graph, :b, Float64)
    lowered = ReactiveKernels.Recipe(-1, (x, a), (b,), (x, a) -> x * a,
        1.0, nothing, false, ReactiveKernels._NO_KERNEL_SOURCE)
    cut = _PREFIX_RK._partial_subplan(original, ReactiveKernels.Value[x], ReactiveKernels.Value[a, b],
        [original.recipes; lowered])
    kernel = prepare(cut; exemplars=(0.4,))
    pullback = prepare_ad_pullback(kernel, backend, (1.0, 1.0), 0.4; active=:x)
    for point in (-0.3, 0.4), seed in ((1.0, 0.0), (0.0, 1.0), (1.3, -0.7))
        value, derivative = ad_value_and_pullback(pullback, seed, point)
        @test value == (sin(point), point * sin(point))
        @test derivative ≈ seed[1] * cos(point) +
                           seed[2] * (sin(point) + point * cos(point))
    end
end

struct ReadOnlyPrefixVector{V<:AbstractVector{Float64}} <: AbstractVector{Float64}
    values::V
end
Base.size(v::ReadOnlyPrefixVector) = size(v.values)
Base.getindex(v::ReadOnlyPrefixVector, i::Int) = v.values[i]
Base.IndexStyle(::Type{<:ReadOnlyPrefixVector}) = IndexLinear()

@kernel _readonly_prefix_view(u::Vector{Float64}) = begin
    readonly = ReadOnlyPrefixVector(u)
    selected = view(readonly, 2:4)
end

@testset "pullback reads an array output with a read-only parent" begin
    point = [0.1, -0.4, 0.3, 0.7, 0.2]
    kernel = prepare(_readonly_prefix_view; have=:u, want=:selected)
    for mode in (Enzyme.Reverse, Enzyme.Forward)
        ad = AutoEnzyme(; mode)
        prepared = prepare_ad_pullback(kernel, ad, ones(3), point; active=:u)
        for shift in (0.0, 0.2), seed in ([1.0, -0.5, 0.3], [-0.2, 0.4, 0.7])
            current = point .+ shift
            before = copy(current)
            expected = [0.0; seed; 0.0]
            value, derivative = ad_value_and_pullback(prepared, seed, current)
            @test value == current[2:4]
            @test derivative ≈ expected
            @test ad_pullback(prepared, seed, current) ≈ expected
            @test ad_pullback(kernel, ad, seed, current; active=:u) ≈ expected
            destination = fill(NaN, length(current))
            value, returned = ad_value_and_pullback!(prepared, destination, seed, current)
            @test returned === destination
            @test value == current[2:4]
            @test destination ≈ expected
            @test current == before
        end
    end
end

end # module
