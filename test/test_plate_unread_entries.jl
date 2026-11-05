using ReactiveKernels, DifferentiationInterface, Test
import Enzyme

# A plate whose cell returns a tuple that other plates read only by constant
# entries (`c[k]`) computes, in each query, only the entries that query reads.
# `derived` counts its calls; the scale reads `s`, the derived entry `d`.
module PlateUnreadEntries
using ReactiveKernels

const CALLS = Ref(0)
derived(m) = (CALLS[] += 1; m^2)
# A closure `map` over a captured constant beside an active argument: native
# Enzyme Reverse cannot prove its activity statically.
derived_map(m, ys) = (CALLS[] += 1; sum(map(yi -> yi * m, ys)))

@kernel shared(q::Vector{Float64}, x::Vector{Float64}, y::Vector{Float64}) = begin
    a::Float64 = sum(view(q, 1:1))
    cell = plate(x, a) do xi, ai
        m = exp(ai * xi)
        s = 1 + m
        d = derived(m)
        (m, s, d)
    end
    loc = plate(cell) do c
        c[1]
    end
    scale = plate(cell) do c
        c[2]
    end
    extra = plate(cell) do c
        c[3]
    end
    pointwise = plate(loc, scale, y) do l, s, yi
        -0.5 * ((yi - l) / s)^2 - log(s)
    end
    total::Float64 = sum(pointwise)
    return total
end

@kernel shared_map(q::Vector{Float64}, x::Vector{Float64}, y::Vector{Float64}) = begin
    a::Float64 = sum(view(q, 1:1))
    cell = plate(x, a, Ref(y)) do xi, ai, ys
        m = exp(ai * xi)
        d = derived_map(m, ys)
        (m, d)
    end
    loc = plate(cell) do c
        c[1]
    end
    extra = plate(cell) do c
        c[2]
    end
    pointwise = plate(loc, y) do l, yi
        -0.5 * (yi - l)^2
    end
    total::Float64 = sum(pointwise)
    return total
end

# Readers that do not read the tuple by constant entries keep every entry.
@kernel whole(q::Vector{Float64}, x::Vector{Float64}) = begin
    a::Float64 = sum(view(q, 1:1))
    cell = plate(x, a) do xi, ai
        m = exp(ai * xi)
        d = derived(m)
        (m, d)
    end
    loc = plate(cell) do c
        first(c)
    end
    total::Float64 = sum(loc)
    return total
end

@kernel atomic(q::Vector{Float64}, x::Vector{Float64}) = begin
    a::Float64 = sum(view(q, 1:1))
    cell = plate(x, a) do xi, ai
        m = exp(ai * xi)
        d = derived(m)
        (m, d)
    end
    loc = plate(x, Ref(cell)) do xi, cs
        xi * cs[1][1]
    end
    total::Float64 = sum(loc)
    return total
end

# Both selected observation plates read the projections as whole arrays, so
# fusion cannot absorb those projections into their consumers. Retrying their
# composition must terminate even though the tuple producer remains live.
@kernel partitioned(q::Vector{Float64}, x::Vector{Float64}, key::Vector{Int}) = begin
    a = sum(q)
    cell = plate(x, a) do xi, ai
        m = exp(ai * xi)
        s = 1 + m
        d = derived(m)
        (m, s, d)
    end
    loc = plate(cell) do c
        c[1]
    end
    scale = plate(cell) do c
        c[2]
    end
    first = plate(x, Ref(loc), Ref(scale)) do xi, ls, ss
        sum(ls) + sum(ss) + xi
    end
    second = plate(x, Ref(loc), Ref(scale)) do xi, ls, ss
        sum(ls) * sum(ss) + xi
    end
    total = sum(first) + sum(second)
    return total
end
end

@testset "tuple-entry pruning reaches a fixed point with shared projections" begin
    RK = ReactiveKernels
    p = plan(PlateUnreadEntries.shared; want = (:loc, :scale))
    lowered = RK._fuse_authored_plate_chains(p; cells_only = true)
    @test lowered !== p
    @test RK._fuse_authored_plate_chains(lowered; cells_only = true) === lowered
    @test RK._fuse_authored_plate_chains(lowered) === lowered
    # Both projections keep the tuple producer live. Repeated pruning must
    # retain their values and leave the unread derived entry unevaluated.
    q, x = [0.3], [-1.0, 0.5, 2.0]
    PlateUnreadEntries.CALLS[] = 0
    loc, scale = prepare(lowered)(q, x, zeros(length(x)))
    @test loc == exp.(only(q) .* x)
    @test scale == 1 .+ loc
    @test PlateUnreadEntries.CALLS[] == 0
end

@testset "partition preparation terminates with shared tuple projections" begin
    P = PlateUnreadEntries
    partition = plate_partition(P.partitioned, (:first, :second); by = :key,
                                locality = :error)
    for n in (0, 3, 20)
        q, x = [0.3], collect(range(-1.0, 1.0; length = n))
        key = [isodd(i) ? 1 : 2 for i in 1:n]
        kernel = prepare(partition; bound = (; x, key))
        m = exp.(only(q) .* x)
        s = 1 .+ m
        @test_throws ArgumentError kernel(q, 3)
        for label in unique(key)
            indices = findall(==(label), key)
            expected = length(indices) * (sum(m) + sum(s) + sum(m) * sum(s)) +
                       2sum(x[indices])
            P.CALLS[] = 0
            @test kernel(q, label) ≈ expected
            @test P.CALLS[] == 0
        end
    end
end

@testset "a plate computes only the tuple entries a query reads" begin
    P = PlateUnreadEntries
    q = [0.3]
    for n in (0, 5, 40)
        x = collect(range(-1.0, 1.0; length = n))
        y = collect(range(0.5, 2.0; length = n))
        m = exp.(only(q) .* x)
        s = 1 .+ m
        expected = sum(-0.5 .* ((y .- m) ./ s) .^ 2 .- log.(s))
        kernel = prepare(P.shared)
        P.CALLS[] = 0
        @test kernel(q, x, y) ≈ expected rtol = 1e-14
        @test P.CALLS[] == 0
        # A query of the derived entry computes it, once per index.
        P.CALLS[] = 0
        @test prepare(P.shared; want = :extra)(q, x, y) == m .^ 2
        @test P.CALLS[] == n
        P.CALLS[] = 0
        total, extra = prepare(P.shared; want = (:total, :extra))(q, x, y)
        @test total ≈ expected rtol = 1e-14
        @test extra == m .^ 2
        @test P.CALLS[] == n
        # Bound data and a cut at the location keep the same entries.
        P.CALLS[] = 0
        @test prepare(P.shared; bound = (; x, y))(q) ≈ expected rtol = 1e-14
        @test prepare(P.shared; want = :loc)(q, x, y) == m
        @test P.CALLS[] == 0

        # Ordinary Enzyme Reverse differentiates only the read entries.
        backend = AutoEnzyme(; mode = Enzyme.Reverse)
        ad = prepare_ad(kernel, backend, q, x, y; active = :q)
        gradient = zeros(1)
        P.CALLS[] = 0
        value, _ = ad_value_and_gradient!(ad, gradient, q, x, y)
        @test P.CALLS[] == 0
        @test value ≈ expected rtol = 1e-14
        h = cbrt(eps())
        fd = (kernel(q .+ h, x, y) - kernel(q .- h, x, y)) / (2h)
        @test only(gradient) ≈ fd rtol = 1e-6 atol = 1e-8

        map_kernel = prepare(P.shared_map)
        map_ad = prepare_ad(map_kernel, backend, q, x, y; active = :q)
        P.CALLS[] = 0
        value, _ = ad_value_and_gradient!(map_ad, gradient, q, x, y)
        @test P.CALLS[] == 0
        @test value ≈ sum(-0.5 .* (y .- m) .^ 2) rtol = 1e-14
        @test only(gradient) ≈ sum((y .- m) .* m .* x) rtol = 1e-12 atol = 1e-12
        P.CALLS[] = 0
        @test prepare(P.shared_map; want = :extra)(q, x, y) ≈
            [sum(y .* mi) for mi in m] rtol = 1e-14
        @test P.CALLS[] == n
    end

    # A reader that does not read constant entries, or reads the tuples whole
    # through `Ref`, keeps every entry: the cell runs as authored.
    q, x = [0.3], collect(range(-1.0, 1.0; length = 7))
    for k in (PlateUnreadEntries.whole, PlateUnreadEntries.atomic)
        P = PlateUnreadEntries
        P.CALLS[] = 0
        prepare(k)(q, x)
        @test P.CALLS[] == length(x)
    end
end
