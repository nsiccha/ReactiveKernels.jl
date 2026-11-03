module GatherMapInterchangeTests

# An elementwise call over an expanding data-indexed gather, f.(x[k]), runs f
# once per element of x where partial evaluation caches k (snag
# data-only-subexp-c79363b3, user decision 1j3rv6f; src/gather_maps.jl).

using Test
using ReactiveKernels
using Enzyme
using Random
using ReactiveKernels: plate, _rk_gather_cover

const CALLS = Ref(0)
counted_exp(x) = (CALLS[] += 1; exp(x))
counted_log(x) = (CALLS[] += 1; log(x))

# Bruno's brm7 dose-amount shape: amounts at the event operations, scaled by
# the bioavailability of the dose each operation belongs to (position 1: none).
@kernel amounts(log_F, op_amount, op_dose_pos, op_rows, dose_rows) = begin
    totals = plate(eachindex(op_rows)) do i
        amount = op_amount[op_rows[i]] .*
            counted_exp.(vcat(0.0, log_F[dose_rows[i]])[op_dose_pos[op_rows[i]] .+ 1])
        sum(amount)
    end
    return totals
end

# The same through a nested kernel's computed argument (RK's own recipe for it).
@kernel consume(amount) = begin
    total = sum(amount)
    return total
end
@kernel amounts_argument(log_F, op_amount, op_dose_pos, op_rows, dose_rows) = begin
    totals = plate(eachindex(op_rows)) do i
        consume(op_amount[op_rows[i]] .*
            counted_exp.(vcat(0.0, log_F[dose_rows[i]])[op_dose_pos[op_rows[i]] .+ 1]))
    end
    return totals
end

@kernel amounts_exp(log_F, op_amount, op_dose_pos, op_rows, dose_rows) = begin
    totals = plate(eachindex(op_rows)) do i
        amount = op_amount[op_rows[i]] .*
            exp.(vcat(0.0, log_F[dose_rows[i]])[op_dose_pos[op_rows[i]] .+ 1])
        sum(amount)
    end
    return totals
end

# Without the `vcat` (which allocates at every position whatever the gather
# does), so a position reader's allocation shows the interchanged arm's own.
@kernel doses(log_F, op_amount, op_dose_pos, op_rows, dose_rows) = begin
    totals = plate(eachindex(op_rows)) do i
        amount = op_amount[op_rows[i]] .* exp.(log_F[dose_rows[i]][op_dose_pos[op_rows[i]]])
        sum(amount)
    end
    return totals
end

# A gather whose index may leave elements of its source unread.
@kernel logs(z, picks, groups) = begin
    totals = plate(groups) do rows
        sum(counted_log.(z[picks[rows]]))
    end
    return totals
end

# A plain Julia reference of `amounts`.
reference(log_F, d) = [sum(d.op_amount[d.op_rows[i]] .*
                           exp.(vcat(0.0, log_F[d.dose_rows[i]])[d.op_dose_pos[d.op_rows[i]] .+ 1]))
                       for i in eachindex(d.op_rows)]

function design(rng; nsub = 4, nops = 11, ndose = 3)
    op_rows = [collect((s - 1) * nops + 1:s * nops) for s in 1:nsub]
    dose_rows = [collect((s - 1) * ndose + 1:s * ndose) for s in 1:nsub]
    op_amount = zeros(nsub * nops)
    op_dose_pos = zeros(Int, nsub * nops)
    for s in 1:nsub
        doses = sort(randperm(rng, nops)[1:ndose])
        for (k, op) in enumerate(doses)
            op_amount[(s - 1) * nops + op] = 10.0 * k
            op_dose_pos[(s - 1) * nops + op] = k
        end
    end
    (; op_amount, op_dose_pos, op_rows, dose_rows, ndose = nsub * ndose)
end
args(d) = (d.op_amount, d.op_dose_pos, d.op_rows, d.dose_rows)
data(d) = (; op_amount = d.op_amount, op_dose_pos = d.op_dose_pos,
           op_rows = d.op_rows, dose_rows = d.dose_rows)
calls(f) = (CALLS[] = 0; f(); CALLS[])
bytes(f, xs...) = (f(xs...); minimum(@allocated(f(xs...)) for _ in 1:3))

@testset "gather map interchange" begin
@testset "cover of an index" begin
    @test _rk_gather_cover([1, 2, 1, 3, 2]) == 3
    @test _rk_gather_cover([1, 3, 1, 3]) == 0          # leaves 2 unread
    @test _rk_gather_cover([1, 2, 3]) == 0             # no repeat
    @test _rk_gather_cover([2, 1]) == 0
    @test _rk_gather_cover([0, 1, 1]) == 0
    @test _rk_gather_cover(Int[]) == 0
    @test _rk_gather_cover([true, true, false]) == 0   # logical indexing
    @test _rk_gather_cover([1 2; 2 1]) == 2
    @test _rk_gather_cover(Int32[1, 1, 2]) == 2
    @test _rk_gather_cover([1.0, 1.0]) == 0
end

@testset "dose amounts: f once per source element, values bit for bit" begin
    rng = Xoshiro(3)
    d = design(rng)
    cells, nops, sources = length(d.op_rows), length(first(d.op_rows)), 4  # 3 doses + exp(0)
    npos = 5
    log_F = 0.3 .* randn(rng, d.ndose, npos)
    for spec in (amounts, amounts_argument)
        scalar = prepare(spec)
        # Unbound scalar preparation caches nothing: the authored evaluation,
        # the reference the interchanged values must equal bit for bit.
        @test calls(() -> scalar(log_F[:, 1], args(d)...)) == cells * nops
        expected = reduce(hcat, [scalar(log_F[:, k], args(d)...) for k in 1:npos])
        @test expected ≈ reduce(hcat, [reference(log_F[:, k], d) for k in 1:npos])
        bound = prepare(spec; bound = data(d))
        @test calls(() -> bound(log_F[:, 1])) == cells * sources
        @test bound(log_F[:, 1]) == expected[:, 1]
        readers = (vectorize(scalar; batched = :log_F),
                   vectorize(scalar; batched = :log_F, reuse = true),
                   vectorize(scalar; batched = :log_F,
                             schedule = NativeScheduling(workers = 2, chunk_size = 2)))
        for reader in readers
            reader(log_F, args(d)...)
            @test calls(() -> reader(log_F, args(d)...)) == cells * sources * npos
            @test reader(log_F, args(d)...) == expected
        end
        bound_reader = vectorize(bound; batched = :log_F, reuse = true)
        @test calls(() -> bound_reader(log_F)) == cells * sources * npos
        @test bound_reader(log_F) == expected
    end
    # A position reader recomputes each cover from the call's data.
    reader = vectorize(amounts; batched = :log_F, reuse = true)
    changed = design(Xoshiro(4); nsub = 3, nops = 6, ndose = 2)
    F2 = randn(Xoshiro(5), changed.ndose, npos)
    scalar = prepare(amounts)
    authored(F, d) = reduce(hcat, [scalar(F[:, k], args(d)...) for k in axes(F, 2)])
    @test reader(F2, args(changed)...) == authored(F2, changed)
    @test calls(() -> reader(F2, args(changed)...)) == 3 * 3 * npos
    # Empty and ragged cells.
    ragged = (; op_amount = [5.0, 0.0, 7.0, 0.0], op_dose_pos = [1, 0, 1, 0],
              op_rows = [[1, 2], Int[], [3, 4, 4]], dose_rows = [[1], Int[], [2]])
    G = randn(Xoshiro(6), 2, npos)
    @test reader(G, args(ragged)...) == authored(G, ragged)
    @test size(reader(G, args((; ragged..., op_rows = Vector{Int}[],
                                dose_rows = Vector{Int}[]))...)) == (0, npos)
    # A dose position past the subject's doses throws as the authored gather does.
    bad = (; ragged..., op_dose_pos = [3, 0, 1, 0])
    @test_throws BoundsError reader(G, args(bad)...)
    @test_throws BoundsError prepare(amounts; bound = data(bad))(G[:, 1])
    # A covering index that reads past a shorter source throws there too.
    short = (; op_amount = [5.0, 0.0, 7.0, 1.0, 2.0], op_dose_pos = [1, 0, 2, 1, 0],
             op_rows = [[1, 2], [3, 4, 5, 5]], dose_rows = [[1], [2]])
    @test _rk_gather_cover(short.op_dose_pos[short.op_rows[2]] .+ 1) == 3
    @test_throws BoundsError scalar(G[:, 1], args(short)...)
    @test_throws BoundsError reader(G, args(short)...)
    @test_throws BoundsError prepare(amounts; bound = data(short))(G[:, 1])
end

@testset "an index that leaves source elements unread keeps the authored form" begin
    z = [0.5, -1.0, 2.0, 3.0]          # z[2] < 0 is never read
    picks = [1, 3, 1, 4, 3, 1]
    groups = [[1, 2, 3], [4, 5, 6], Int[]]
    expected = [sum(log.(z[picks[rows]])) for rows in groups]
    bound = prepare(logs; bound = (; picks, groups))
    @test bound(z) == expected
    @test calls(() -> bound(z)) == 6
    Z = hcat(z, z .+ 1.5)
    reader = vectorize(logs; batched = :z, reuse = true)
    @test reader(Z, picks, groups) == hcat(expected, [sum(log.(Z[picks[rows], 2])) for rows in groups])
    # Covering cells interchange, the others do not, in the same plate.
    w = [0.5, 2.0, 3.0]
    mixed = [[1, 2, 1, 2], [3, 3], [1, 2, 3]]
    flat = reduce(vcat, mixed)
    cells = [collect(1:4), collect(5:6), collect(7:9)]
    b2 = prepare(logs; bound = (; picks = flat, groups = cells))
    @test b2(w) == [sum(log.(w[m])) for m in mixed]
    # Cell 1 covers 1:2 (2 calls); cell 2 covers nothing from 1 (2 calls); cell 3 has no repeat (3).
    @test calls(() -> b2(w)) == 2 + 2 + 3
end

@testset "reverse derivatives agree with the authored form" begin
    rng = Xoshiro(7)
    d = design(rng)
    q = 0.2 .* randn(rng, d.ndose)
    loss(f, x) = sum(f(x))
    bound = prepare(amounts_exp; bound = data(d))
    scalar = prepare(amounts_exp)
    @test bound(q) == scalar(q, args(d)...)
    gradient = Enzyme.gradient(Enzyme.Reverse, loss, Const(bound), q)[2]
    authored = Enzyme.gradient(Enzyme.Reverse, (x, a...) -> sum(scalar(x, a...)), q,
                               map(Const, args(d))...)[1]
    @test gradient ≈ authored rtol = 8eps()
    @test any(!iszero, gradient)
end

@testset "position readers keep their buffers" begin
    rng = Xoshiro(8)
    d = design(rng; nsub = 16, nops = 64, ndose = 4)
    d = (; d..., op_dose_pos = max.(d.op_dose_pos, 1))   # every operation reads a dose
    @test all(i -> _rk_gather_cover(d.op_dose_pos[d.op_rows[i]]) == 4, eachindex(d.op_rows))
    scalar = prepare(doses)
    reader = vectorize(doses; batched = :log_F, reuse = true)
    small = randn(rng, d.ndose, 2)
    large = randn(rng, d.ndose, 16)
    @test reader(large, args(d)...) ==
        reduce(hcat, [scalar(large[:, k], args(d)...) for k in axes(large, 2)])
    @test bytes(reader, large, args(d)...) <= bytes(reader, small, args(d)...) + 2048
end
end

end # module
