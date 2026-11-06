using ReactiveKernels, DifferentiationInterface, Test
import Enzyme

# A cell lowering computes a plate once, at the union of the rows its readers ask
# for. A tuple plate whose readers read different entries is split by which of
# them each statement serves, so a row runs only the statements something reads
# there, and each statement still runs once per row (snag
# `absent-odl-vecto-d98e48bf`).
module PlateEntrySplit
using ReactiveKernels

const SHAREDWORK, EXCLUSIVE, MIXED, OWN, FIRST_ENTRY, SECOND_ENTRY = ntuple(_ -> Ref(0), 6)
const COUNTERS = (; sharedwork = SHAREDWORK, exclusive = EXCLUSIVE, mixed = MIXED, own = OWN,
                  first_entry = FIRST_ENTRY, second_entry = SECOND_ENTRY)
runs(counter, x) = (counter[] += 1; x)
reset!() = foreach(counter -> counter[] = 0, COUNTERS)
counted() = Dict(name => counter[] for (name, counter) in pairs(COUNTERS) if counter[] > 0)

# `h` (label 1 only) reads `total`, `m` (label 1 only) reads `mixed`, and `s`
# (labels 1 and 2) reads `own`. `sharedwork` serves `total` and `mixed` but not
# `own`, and only `exclusive` reads the field at coordinates 3:4.
@kernel entries(u::AbstractVector{Float64}, w::AbstractVector{Float64},
                hpath::AbstractVector{Int}, spath::AbstractVector{Int}, mpath::AbstractVector{Int},
                yh::AbstractVector{Float64}, ys::AbstractVector{Float64}, ym::AbstractVector{Float64},
                hkey::AbstractVector{Int}, skey::AbstractVector{Int}, mkey::AbstractVector{Int}) = begin
    field = exp.(u[3:4])
    cells = plate(eachindex(w)) do i
        sharedwork = runs(SHAREDWORK, u[i] * w[i])
        exclusive = runs(EXCLUSIVE, w[i] * sum(field))
        total = exclusive + sharedwork
        mixed = runs(MIXED, sharedwork * 3.0)
        own = runs(OWN, u[i] * 2.0)
        (total, own, mixed)
    end
    totals = plate(eachindex(w), cells) do i, c
        c[1]
    end
    owns = plate(eachindex(w), cells) do i, c
        c[2]
    end
    mixeds = plate(eachindex(w), cells) do i, c
        c[3]
    end
    th = totals[hpath]
    so = owns[spath]
    mi = mixeds[mpath]
    h = plate(yh, th) do y, t
        density::Float64 = -(y - t)^2 / 2
        density
    end
    s = plate(ys, so) do y, o
        density::Float64 = -(y - o)^2 / 2
        density
    end
    m = plate(ym, mi) do y, x
        density::Float64 = -(y - x)^2 / 2
        density
    end
end

# One reader reads two entries, another one of those two.
@kernel two_entries(u::AbstractVector{Float64}, w::AbstractVector{Float64},
                    apath::AbstractVector{Int}, bpath::AbstractVector{Int},
                    ya::AbstractVector{Float64}, yb::AbstractVector{Float64},
                    akey::AbstractVector{Int}, bkey::AbstractVector{Int}) = begin
    field = exp.(u[3:4])
    cells = plate(eachindex(w)) do i
        first_entry = runs(FIRST_ENTRY, w[i] * sum(field))
        second_entry = runs(SECOND_ENTRY, u[i] * 2.0)
        (first_entry, second_entry)
    end
    both = plate(eachindex(w), cells) do i, c
        one = c[1]
        two = c[2]
        one + two
    end
    seconds = plate(eachindex(w), cells) do i, c
        c[2]
    end
    ta = both[apath]
    tb = seconds[bpath]
    a = plate(ya, ta) do y, t
        density::Float64 = -(y - t)^2 / 2
        density
    end
    b = plate(yb, tb) do y, t
        density::Float64 = -(y - t)^2 / 2
        density
    end
end

# The cell reads captures through `vec` of an index array, which is not bound and
# which both entries use: the captures stay read at rows after the split.
@kernel through_vec(u::AbstractVector{Float64}, index::AbstractVector{Int},
                    w::AbstractVector{Float64},
                    apath::AbstractVector{Int}, bpath::AbstractVector{Int},
                    ya::AbstractVector{Float64}, yb::AbstractVector{Float64},
                    akey::AbstractVector{Int}, bkey::AbstractVector{Int}) = begin
    left = plate(eachindex(w)) do i
        exp(u[i])
    end
    right = plate(eachindex(w)) do i
        exp(u[i + 2])
    end
    cells = plate(eachindex(w)) do i
        linear = vec(index)
        a = left[linear[i]]
        b = right[linear[i]]
        (a, b)
    end
    lefts = plate(eachindex(w), cells) do i, c
        c[1]
    end
    rights = plate(eachindex(w), cells) do i, c
        c[2]
    end
    ta = lefts[apath]
    tb = rights[bpath]
    a = plate(ya, ta) do y, t
        density::Float64 = -(y - t)^2 / 2
        density
    end
    b = plate(yb, tb) do y, t
        density::Float64 = -(y - t)^2 / 2
        density
    end
end

# Every statement serves both readers: nothing to split.
@kernel all_shared(u::AbstractVector{Float64}, w::AbstractVector{Float64},
                   ya::AbstractVector{Float64}, yb::AbstractVector{Float64}) = begin
    cells = plate(w, u) do wi, ui
        m = wi * ui
        (m, m)
    end
    firsts = plate(cells) do c
        c[1]
    end
    seconds = plate(cells) do c
        c[2]
    end
    a = plate(ya, firsts) do y, t
        -(y - t)^2 / 2
    end
    b = plate(yb, seconds) do y, t
        -(y - t)^2 / 2
    end
end
end

function _entry_split_gradient_check(q, u, label, expected, extra...)
    @test q(u, extra..., label) ≈ expected(u, label)
    ad = prepare_ad(q, AutoEnzyme(; mode = Enzyme.Reverse), u, extra..., label; active = :u)
    value, gradient = ad_value_and_gradient(ad, u, extra..., label)
    @test value ≈ expected(u, label)
    for j in eachindex(u)
        plus, minus = copy(u), copy(u)
        plus[j] += 1e-6
        minus[j] -= 1e-6
        @test gradient[j] ≈ (expected(plus, label) - expected(minus, label)) / 2e-6 atol=2e-8 rtol=2e-6
    end
end

@testset "a tuple plate's readers run only the statements they read" begin
    P = PlateEntrySplit
    @testset "statements run once, only at rows that read them" begin
        bound = (; w = [0.5, 0.7], hpath = [1], spath = [1, 2], mpath = [1], yh = [0.3],
                 ys = [0.1, 0.2], ym = [0.4], hkey = [1], skey = [1, 2], mkey = [1])
        function expected(u, label)
            value = -(bound.ys[label] - 2u[label])^2 / 2
            label == 1 || return value
            shared = u[1] * bound.w[1]
            value - (bound.yh[1] - (bound.w[1] * sum(exp.(u[3:4])) + shared))^2 / 2 -
                (bound.ym[1] - shared * 3.0)^2 / 2
        end
        spec = plate_partition(P.entries, (:h, :s, :m); by = (h = :hkey, s = :skey, m = :mkey),
                               locality = :error, partitioned = (:u => [1, 2, 1, 1],))
        q = prepare(spec; want = :partition_total, bound)
        u = [0.2, -0.1, 0.3, 0.4]
        P.reset!()
        @test q(u, 1) ≈ expected(u, 1)
        @test P.counted() == Dict(:sharedwork => 1, :exclusive => 1, :mixed => 1, :own => 1)
        # Label 2 reads only `own`, at its own row. `sharedwork` serves two
        # readers, neither of which has a cell here.
        P.reset!()
        @test q(u, 2) ≈ expected(u, 2)
        @test P.counted() == Dict(:own => 1)
        @test partition_footprint(q, u, 1).u == [1, 3, 4]
        @test partition_footprint(q, u, 2).u == [2]
        @test partition_separability(q, u, 2).u == [2]
        _entry_split_gradient_check(q, u, 1, expected)
        _entry_split_gradient_check(q, u, 2, expected)
    end

    @testset "with a plan port" begin
        bound = (; w = [0.5, 0.7], hpath = [1], spath = [1, 2], mpath = [1], yh = [0.3],
                 ys = [0.1, 0.2], ym = [0.4], hkey = [1], skey = [1, 2], mkey = [1])
        function expected(u, label)
            value = -(bound.ys[label] - 2u[label])^2 / 2
            label == 1 || return value
            shared = u[1] * bound.w[1]
            value - (bound.yh[1] - (bound.w[1] * sum(exp.(u[3:4])) + shared))^2 / 2 -
                (bound.ym[1] - shared * 3.0)^2 / 2
        end
        spec = plate_partition(P.entries, (:h, :s, :m); by = (h = :hkey, s = :skey, m = :mkey),
                               locality = :error, partitioned = (:u => [1, 2, 1, 1],),
                               plan = :plan)
        u = [0.2, -0.1, 0.3, 0.4]
        q = prepare(spec; want = :partition_total, bound, exemplars = (u, 1, nothing))
        for label in (1, 2)
            rows = partition_plan(q, u, label, nothing)
            P.reset!()
            @test q(u, label, rows) ≈ expected(u, label)
            @test P.counted() == (label == 1 ?
                Dict(:sharedwork => 1, :exclusive => 1, :mixed => 1, :own => 1) : Dict(:own => 1))
            @test partition_footprint(q, u, label, nothing).u == (label == 1 ? [1, 3, 4] : [2])
        end
    end

    @testset "a reader of several entries runs each of them" begin
        bound = (; w = [0.5, 0.7], apath = [1], bpath = [1, 2], ya = [0.3], yb = [0.1, 0.2],
                 akey = [1], bkey = [1, 2])
        function expected(u, label)
            value = -(bound.yb[label] - 2u[label])^2 / 2
            label == 1 || return value
            value - (bound.ya[1] - (bound.w[1] * sum(exp.(u[3:4])) + 2u[1]))^2 / 2
        end
        spec = plate_partition(P.two_entries, (:a, :b); by = (a = :akey, b = :bkey),
                               locality = :error, partitioned = (:u => [1, 2, 1, 1],))
        q = prepare(spec; want = :partition_total, bound)
        u = [0.2, -0.1, 0.3, 0.4]
        P.reset!()
        @test q(u, 1) ≈ expected(u, 1)
        @test P.counted() == Dict(:first_entry => 1, :second_entry => 1)
        P.reset!()
        @test q(u, 2) ≈ expected(u, 2)
        @test P.counted() == Dict(:second_entry => 1)
        @test partition_footprint(q, u, 2).u == [2]
        _entry_split_gradient_check(q, u, 1, expected)
        _entry_split_gradient_check(q, u, 2, expected)
    end

    @testset "a capture read through `vec` stays read at rows" begin
        bound = (; w = [0.5, 0.7], apath = [1], bpath = [2], ya = [0.3], yb = [0.1],
                 akey = [1], bkey = [2])
        index = [1, 2]
        expected(u, label) = label == 1 ? -(bound.ya[1] - exp(u[1]))^2 / 2 :
            -(bound.yb[1] - exp(u[4]))^2 / 2
        spec = plate_partition(P.through_vec, (:a, :b); by = (a = :akey, b = :bkey),
                               locality = :error, partitioned = (:u => [1, 2, 1, 2],))
        q = prepare(spec; want = :partition_total, bound)
        u = [0.2, -0.1, 0.3, 0.4]
        for label in (1, 2)
            @test partition_footprint(q, u, index, label).u == (label == 1 ? [1] : [4])
            _entry_split_gradient_check(q, u, label, expected, index)
        end
    end

    # The split follows the cell's statements, not the data: the emitted plates
    # and loops are the same at every size, and each statement runs once per row.
    @testset "the emitted structure does not grow with the rows" begin
        structure = map((2, 7)) do n
            bound = (; w = fill(0.5, n), hpath = [1], spath = collect(1:n), mpath = [1],
                     yh = [0.3], ys = fill(0.1, n), ym = [0.4], hkey = [1],
                     skey = collect(1:n), mkey = [1])
            labels = vcat(collect(1:n), 1, 1)
            spec = plate_partition(P.entries, (:h, :s, :m); by = (h = :hkey, s = :skey, m = :mkey),
                                   locality = :error, partitioned = (:u => labels,))
            q = prepare(spec; want = :partition_total, bound)
            u = fill(0.2, n + 2)
            P.reset!()
            q(u, 2)
            @test P.counted() == Dict(:own => 1)
            P.reset!()
            q(u, 1)
            @test P.counted() == Dict(:sharedwork => 1, :exclusive => 1, :mixed => 1, :own => 1)
            [(e.kind, e.depth) for e in recipe_inventory(q) if e.kind !== :ordinary]
        end
        @test structure[1] == structure[2]
    end

    @testset "the split is a fixed point, and only when statements differ" begin
        RK = ReactiveKernels
        split = RK._cell_split_tuple_plates
        p = RK._lowering_plan(plan(P.entries; want = (:h, :s, :m)))
        once = split(p)
        @test once !== p
        @test split(once) === once
        all_shared = RK._lowering_plan(plan(P.all_shared; want = (:a, :b)))
        @test split(all_shared) === all_shared
    end
end
