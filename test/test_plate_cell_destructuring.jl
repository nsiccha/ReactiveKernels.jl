using ReactiveKernels, DifferentiationInterface, Enzyme, Test

# A destructuring `(a, b, …) = f(x…)` of an opaque call inside an authored
# plate cell, or inside a kernel the cell calls, binds each element as Julia
# does. The cell plans it as one recipe with several outputs and lowers it as
# one recipe computing the whole value plus one element read per name. Before
# that, lowering failed with `ArgumentError: Tuple contains 3 elements, must
# contain exactly 1 element` (snag `destructure-an-o-88407817`).

_cd_design(t, d) = (t + d, 2t, t * d)
_cd_mixed(t, d) = (round(Int, t), t * d, [t, d])
_cd_bounds(t, d) = (lo = t - d, hi = t + d)
_cd_vector(t, d) = [t, d, t * d]
_cd_longer(t, d) = (t, d, t + d)
_cd_shorter(t, d) = (t, d)
_cd_pair(q) = (q[1], q[2])
_cd_scaled(a, k) = (exp(k) * a, k^2)
struct _CDIterable
    a::Float64
    b::Float64
end
Base.iterate(x::_CDIterable, state = 1) =
    state > 2 ? nothing : (state == 1 ? x.a : x.b, state + 1)

@kernel _cd_cell(ts, ds) = begin
    vals = plate(ts, ds) do t, d
        (a, b, c) = _cd_design(t, d)
        c * c + a
    end
    total = sum(vals)
    return total
end

@kernel _cd_entry(t, d) = begin
    (a, b, c) = _cd_design(t, d)
    out = a * b + c
    return out
end
@kernel _cd_called(ts, ds) = begin
    vals = plate(ts, ds) do t, d
        _cd_entry(t, d)
    end
    return vals
end

@kernel _cd_elements(ts, ds) = begin
    vals = plate(ts, ds) do t, d
        (n, p, v) = _cd_mixed(t, d)
        (lo, hi) = _cd_bounds(t, d)
        (x, y, z) = _cd_vector(t, d)
        (i, j) = _CDIterable(t, d)
        (e, f) = _cd_longer(t, d)
        n + p + sum(v) + (hi - lo) + x * y * z + (i - j) + e * f
    end
    return vals
end

@kernel _cd_short(ts, ds) = begin
    vals = plate(ts, ds) do t, d
        (a, b, c) = _cd_shorter(t, d)
        a * b * c
    end
    return vals
end

@kernel _cd_shared(ts, q) = begin
    vals = plate(ts, Ref(q)) do t, qq
        (a, b) = _cd_pair(qq)
        -0.5 * (a * t + b)^2
    end
    total = sum(vals)
    return total
end

@kernel _cd_loop(ts, ds) = begin
    @plate for i in eachindex(ts, ds)
        (a, b, c) = _cd_design(ts[i], ds[i])
        z[i] = ds[i] > 0.5 ? a * b : c
    end
    return z
end

@kernel _cd_density(ts, ds, k::Float64) = begin
    lp = plate(ts, ds, k) do t, d, kk
        (a, b, c) = _cd_design(t, d)
        (s, k2) = _cd_scaled(a, kk)
        -0.5 * (s - c)^2 - b * k2
    end
    total = sum(lp)
    return total
end

const _CD_TS = [0.5, 1.0, 2.0]
const _CD_DS = [1.0, 0.0, 3.0]
_cd_reference(f) = [f(t, d) for (t, d) in zip(_CD_TS, _CD_DS)]

@testset "plate cells destructure an opaque call's result" begin
    ts, ds = _CD_TS, _CD_DS
    cell = prepare(_cd_cell)
    @test @inferred(cell(ts, ds)) ≈ sum(_cd_reference((t, d) -> (t * d)^2 + t + d))
    @test cell(Float64[], Float64[]) == 0.0
    @test prepare(_cd_cell; want = (:vals, :total))(ts, ds)[1] ==
          _cd_reference((t, d) -> (t * d)^2 + t + d)
    @test @inferred(prepare(_cd_called)(ts, ds)) ==
          _cd_reference((t, d) -> (t + d) * 2t + t * d)

    # Every element follows `Base.indexed_iterate`: heterogeneous and named
    # tuples, arrays, a plain iterable, and a longer value whose extra element
    # is dropped. A shorter value throws `BoundsError`, as in Julia.
    @test prepare(_cd_elements)(ts, ds) == _cd_reference((t, d) ->
        round(Int, t) + t * d + t + d + 2d + t * d * t * d + (t - d) + t * d)
    @test_throws BoundsError prepare(_cd_short)(ts, ds)

    q = [2.0, 0.25]
    @test @inferred(prepare(_cd_shared)(ts, q)) ≈ sum(-0.5 .* (2.0 .* ts .+ 0.25) .^ 2)
    @test prepare(_cd_loop)(ts, ds) ==
          _cd_reference((t, d) -> d > 0.5 ? (t + d) * 2t : t * d)
    @test prepare(_cd_loop; bound = (; ds))(ts) == prepare(_cd_loop)(ts, ds)
    @test prepare(_cd_cell; bound = (; ds))(ts) ≈ cell(ts, ds)

    # The cell lowers single-output recipes. Outside a plate the destructuring
    # stays the one multi-output recipe that planning chose.
    plate_recipe = only(r for r in plan(_cd_cell).recipes
                        if r.op isa ReactiveKernels._AuthoredPlateOp)
    body = ReactiveKernels.plate_body(plate_recipe)
    @test all(r -> length(r.outputs) == 1, body.recipes)
    @test count(r -> r.op isa ReactiveKernels._KernelDestructureElement,
                body.recipes) == 3
    @test count(r -> length(r.outputs) == 3, kernel_graph(_cd_entry).recipes) == 1

    batch = vectorize(_cd_cell; batched = :ts, want = :total)
    @test batch(hcat(ts, 2 .* ts), ds) ≈ [cell(ts, ds), cell(2 .* ts, ds)]
end

@testset "plate-cell destructuring under native Enzyme Reverse" begin
    ts, ds, k = _CD_TS, _CD_DS, 0.3
    value(k) = sum(-0.5 * (exp(k) * (t + d) - t * d)^2 - 2t * k^2
                   for (t, d) in zip(ts, ds))
    slope(k) = sum(-(exp(k) * (t + d) - t * d) * exp(k) * (t + d) - 4t * k
                   for (t, d) in zip(ts, ds))
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    prepared = prepare_ad(prepare(_cd_density), backend, ts, ds, k; active = :k)
    v, g = ad_value_and_gradient(prepared, ts, ds, k)
    @test v ≈ value(k)
    @test g ≈ slope(k)
    bound = prepare_ad(prepare(_cd_density; bound = (; ts, ds)), backend, k;
                       active = :k)
    v, g = ad_value_and_gradient(bound, k)
    @test v ≈ value(k)
    @test g ≈ slope(k)
    q = [0.7, -0.2]
    shared = prepare_ad(prepare(_cd_shared), backend, ts, q; active = :q)
    v, g = ad_value_and_gradient(shared, ts, q)
    residual = q[1] .* ts .+ q[2]
    @test v ≈ sum(-0.5 .* residual .^ 2)
    @test g ≈ [-sum(residual .* ts), -sum(residual)]
end

# Outside a plate a destructuring stays one recipe with several outputs, and
# `bound=` splits it around its data-only nodes like any other recipe. That
# split used to fail with `ArgumentError: Tuple contains 2 elements, must
# contain exactly 1 element` (snag `prepare-cell-fac-469bcafb`).
const _CD_DATA_RUNS = Ref(0)
const _CD_CALL_RUNS = Ref(0)
const _CD_EXP_RUNS = Ref(0)
_cd_data(x) = (_CD_DATA_RUNS[] += 1; exp.(x))
_cd_counted(u, a) = (_CD_CALL_RUNS[] += 1; (sum(u) * a, a^2, a))
_cd_exp(t) = (_CD_EXP_RUNS[] += 1; exp(t))
_cd_rows(rows) = rows
_cd_read(m, v) = (sum(m), sum(v))
_cd_scaled_total(u, a) = (sum(u) * a, a^2)

@kernel _cd_top(x, a) = begin
    (s, q) = _cd_counted(_cd_data(x), a)
    out = s + q
    return out
end
@kernel _cd_top_short(x, a) = begin
    (s, q, w, z) = _cd_counted(_cd_data(x), a)
    out = s + q + w + z
    return out
end
@kernel _cd_top_density(x, a) = begin
    (s, q) = _cd_scaled_total(exp.(x), a)
    out = s + q
    return out
end
# The call reads `v` beside `exp.(v[rows])`, so the gather stays in the root.
@kernel _cd_top_gathered(v, rows) = begin
    (s, q) = _cd_read(_cd_exp.(v[_cd_rows(rows)]), v)
    out = s * q
    return out
end
# A live condition keeps the destructured branch lazy and unsplit.
@kernel _cd_top_branch(x, a) = begin
    (s, q) = a > 0 ? _cd_scaled_total(exp.(x), a) : (0.0, 1.0)
    out = s + q
    return out
end

@testset "a top-level destructuring under bound= caches its data-only nodes" begin
    x = [0.1, 0.2, 0.3]
    reference(x, a) = sum(exp.(x)) * a + a^2
    prepared = prepare(_cd_top; bound = (; x))
    _CD_DATA_RUNS[] = _CD_CALL_RUNS[] = 0
    for a in (0.7, -0.4, 1.1)
        @test prepared(a) ≈ reference(x, a)
        @test prepared(a) ≈ prepare(_cd_top)(x, a)
    end
    # 3 + 3 calls of the bound kernel and 3 of the unbound one, which is the
    # only kernel that computes the data-only node again.
    @test _CD_DATA_RUNS[] == 3
    @test _CD_CALL_RUNS[] == 9

    # Another binding, and no data at all, read their own values.
    @test prepare(_cd_top; bound = (; x = 2 .* x))(0.7) ≈ reference(2 .* x, 0.7)
    @test prepare(_cd_top; bound = (; x = Float64[]))(0.7) ≈ 0.7^2

    # A missing element is Julia's BoundsError, bound or not.
    @test_throws BoundsError prepare(_cd_top_short)(x, 0.7)
    @test_throws BoundsError prepare(_cd_top_short; bound = (; x))(0.7)

    # The gather-map interchange also applies at a root with several outputs.
    v, rows = [0.3, -0.2, 0.5], [1, 2, 1, 2, 2]
    gathered = prepare(_cd_top_gathered; bound = (; rows))
    _CD_EXP_RUNS[] = 0
    @test gathered(v) ≈ sum(exp.(v[rows])) * sum(v)
    @test _CD_EXP_RUNS[] == 2

    branch = prepare(_cd_top_branch; bound = (; x))
    @test branch(0.7) ≈ sum(exp.(x)) * 0.7 + 0.7^2
    @test branch(-0.7) == 1.0

    # Splitting introduces no run-time dispatch into a concretely typed program.
    strict = ReactiveKernels._with_type_stability(:error, :all) do
        prepare(_cd_top_density; bound = (; x))
    end
    @test @inferred(strict(0.7)) ≈ reference(x, 0.7)
end

@testset "a split top-level destructuring under native Enzyme Reverse" begin
    x = [0.1, 0.2, 0.3]
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    bound = prepare_ad(prepare(_cd_top_density; bound = (; x)), backend, 0.7; active = :a)
    v, g = ad_value_and_gradient(bound, 0.7)
    @test v ≈ sum(exp.(x)) * 0.7 + 0.7^2
    @test g ≈ sum(exp.(x)) + 2 * 0.7
end
