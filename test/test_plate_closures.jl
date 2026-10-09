using ReactiveKernels, DifferentiationInterface, Enzyme, Test

# A `plate(...) do` cell or `scan(...) do` step reads enclosing names the way a
# Julia closure does: a name it reads without passing it is the whole value,
# never zipped. Closures are the preferred spelling for reading a whole value;
# an explicit `Ref(name)` operand stays supported and lowers to the same program
# (todo `1f3ku0g`, snag `rk-plate-body-ca-091ef037`).

# Program text with every generated local renamed by first appearance, so two
# spellings compare equal exactly when they lower to the same program.
function _closure_program(k)
    names = Dict{String,String}()
    replace(string(readable_code(k)),
            r"var\"[^\"]*\"" => m -> get!(names, m, "v$(length(names) + 1)"))
end

# Each closure kernel below is paired with its `Ref` spelling: the captured
# names passed as trailing `Ref` operands, in the order the body first reads
# them (a generator's iterator before its term), under the same formal names.

@kernel _closure_indexed(x, d) = begin
    cells = plate(eachindex(d), d) do s, dd
        x[s] + dd
    end
    return cells
end
@kernel _closure_indexed_ref(x, d) = begin
    cells = plate(eachindex(d), d, Ref(x)) do s, dd, x
        x[s] + dd
    end
    return cells
end

@kernel _closure_first(x, d) = begin
    cells = plate(d) do dd
        x[1] + dd
    end
    return cells
end

@kernel _closure_length(x, d) = begin
    cells = plate(d) do dd
        length(x) + dd
    end
    return cells
end

# The plate zips `x` and also reads it whole.
@kernel _closure_normalized(x) = begin
    shares = plate(x) do xi
        xi / sum(x)
    end
    return shares
end

@kernel _closure_ragged(xs, d) = begin
    cells = plate(eachindex(d), d) do s, dd
        sum(xs[s]; init = 0.0) * dd + length(xs)
    end
    return cells
end
@kernel _closure_ragged_ref(xs, d) = begin
    cells = plate(eachindex(d), d, Ref(xs)) do s, dd, xs
        sum(xs[s]; init = 0.0) * dd + length(xs)
    end
    return cells
end

# A recipe-computed local, read before its own statement.
@kernel _closure_derived(x, d) = begin
    cells = plate(eachindex(d), d) do s, dd
        shifted[s] * dd
    end
    shifted = x .+ 1.0
    return cells
end
@kernel _closure_derived_ref(x, d) = begin
    cells = plate(eachindex(d), d, Ref(shifted)) do s, dd, shifted
        shifted[s] * dd
    end
    shifted = x .+ 1.0
    return cells
end

# Captures read inside a generator and inside a lazy arm.
@kernel _closure_generator_branch(x, d) = begin
    cells = plate(d) do dd
        dd > 15 ? sum(x[j] * dd for j in eachindex(x); init = 0.0) : length(x) * dd
    end
    return cells
end
@kernel _closure_generator_branch_ref(x, d) = begin
    cells = plate(d, Ref(x)) do dd, x
        dd > 15 ? sum(x[j] * dd for j in eachindex(x); init = 0.0) : length(x) * dd
    end
    return cells
end

# Nested plates: the inner cell captures an outer-cell local and a kernel port;
# the outer cell captures the port because its inner plate reads it.
@kernel _closure_nested(groups, w) = begin
    totals = plate(groups) do observations
        scaled = observations .* 2.0
        inner = plate(eachindex(observations)) do i
            scaled[i] + w[i]
        end
        sum(inner)
    end
    return totals
end
@kernel _closure_nested_ref(groups, w) = begin
    totals = plate(groups, Ref(w)) do observations, w
        scaled = observations .* 2.0
        inner = plate(eachindex(observations), Ref(scaled), Ref(w)) do i, scaled, w
            scaled[i] + w[i]
        end
        sum(inner)
    end
    return totals
end

@kernel _closure_scan(q::Vector{Float64}, series::Vector{Float64}) = begin
    decay::Float64 = q[1]
    trajectory = scan(series; init = 0.0) do carry, x
        next = decay * carry + q[2] * x
        (next, next)
    end
    total::Float64 = sum(trajectory)
end
@kernel _closure_scan_ref(q::Vector{Float64}, series::Vector{Float64}) = begin
    decay::Float64 = q[1]
    trajectory = scan(series, Ref(decay), Ref(q); init = 0.0) do carry, x, decay, q
        next = decay * carry + q[2] * x
        (next, next)
    end
    total::Float64 = sum(trajectory)
end

# A scan in a plate cell captures the cell's local and a kernel port.
@kernel _closure_scan_in_plate(groups, w::Vector{Float64}) = begin
    totals = plate(groups) do g
        rate = w[1] * length(g)
        trajectory = scan(g; init = 0.0) do carry, x
            next = carry * w[2] + x * rate
            (next, next)
        end
        sum(trajectory; init = 0.0)
    end
    total::Float64 = sum(totals)
end
@kernel _closure_scan_in_plate_ref(groups, w::Vector{Float64}) = begin
    totals = plate(groups, Ref(w)) do g, w
        rate = w[1] * length(g)
        trajectory = scan(g, Ref(w), Ref(rate); init = 0.0) do carry, x, w, rate
            next = carry * w[2] + x * rate
            (next, next)
        end
        sum(trajectory; init = 0.0)
    end
    total::Float64 = sum(totals)
end

# A plate in a scan step captures the step's element and a kernel port.
@kernel _closure_plate_in_scan(xs, w) = begin
    out = scan(xs; init = 0.0) do carry, x
        cells = plate(eachindex(w)) do j
            w[j] * x
        end
        next = carry + sum(cells)
        (next, next)
    end
    return out
end
@kernel _closure_plate_in_scan_ref(xs, w) = begin
    out = scan(xs, Ref(w); init = 0.0) do carry, x, w
        cells = plate(eachindex(w), Ref(w), Ref(x)) do j, w, x
            w[j] * x
        end
        next = carry + sum(cells)
        (next, next)
    end
    return out
end

# A scan in a scan step captures the outer step's element and a kernel port.
@kernel _closure_scan_in_scan(groups, w::Float64) = begin
    out = scan(groups; init = 0.0) do carry, g
        inner = scan(g; init = 0.0) do c, x
            next = c + x * w + length(g)
            (next, next)
        end
        total = carry + sum(inner; init = 0.0)
        (total, total)
    end
    return out
end
@kernel _closure_scan_in_scan_ref(groups, w::Float64) = begin
    out = scan(groups, Ref(w); init = 0.0) do carry, g, w
        inner = scan(g, Ref(w), Ref(g); init = 0.0) do c, x, w, g
            next = c + x * w + length(g)
            (next, next)
        end
        total = carry + sum(inner; init = 0.0)
        (total, total)
    end
    return out
end

# A scan feeding one reducing plate streams the plate's cells inside the carry
# loop; the captured scale is a shared operand there, as `Ref(scale)` is.
@kernel _closure_scan_plate_sum(xs::Vector{Float64}, scale) = begin
    cumulative = scan(xs; init = 0.0) do carry, x
        next = carry + x
        (next, next)
    end
    pointwise = plate(cumulative) do value
        -0.5 * (value / scale)^2
    end
    total::Float64 = sum(pointwise)
    return total
end
@kernel _closure_scan_plate_sum_ref(xs::Vector{Float64}, scale) = begin
    cumulative = scan(xs; init = 0.0) do carry, x
        next = carry + x
        (next, next)
    end
    pointwise = plate(cumulative, Ref(scale)) do value, scale
        -0.5 * (value / scale)^2
    end
    total::Float64 = sum(pointwise)
    return total
end

# A fold written as its own statement of a plain scan step runs in the scan's
# strip region; the captured coefficients and decay are shared operands there.
@kernel _closure_scan_strip(steps, pa, decay) = begin
    out = scan(steps; init = 0.0) do carry, k
        x = 1 / (k + 0.5)
        s = evalpoly(x, pa)
        next = muladd(decay, carry, s)
        (next, next)
    end
    return out
end
@kernel _closure_scan_strip_ref(steps, pa, decay) = begin
    out = scan(steps, Ref(pa), Ref(decay); init = 0.0) do carry, k, pa, decay
        x = 1 / (k + 0.5)
        s = evalpoly(x, pa)
        next = muladd(decay, carry, s)
        (next, next)
    end
    return out
end

# The dose-superposition `get` cell (dose-outer native lowering) and an
# `evalpoly` cell over shared coefficients (coefficient-outer lowering).
@kernel _closure_superposition(observations, shifts, units, weights) = begin
    response = plate(observations) do t
        sum(weights[j] * get(units, t - shifts[j], 0.0) for j in eachindex(shifts);
            init = 0.0)
    end
    return response
end
@kernel _closure_superposition_ref(observations, shifts, units, weights) = begin
    response = plate(observations, Ref(shifts), Ref(weights), Ref(units)) do t, shifts, weights, units
        sum(weights[j] * get(units, t - shifts[j], 0.0) for j in eachindex(shifts);
            init = 0.0)
    end
    return response
end
@kernel _closure_evalpoly(xs, c) = begin
    values = plate(xs) do x
        evalpoly(x, c)
    end
    return values
end
@kernel _closure_evalpoly_ref(xs, c) = begin
    values = plate(xs, Ref(c)) do x, c
        evalpoly(x, c)
    end
    return values
end

# A reader plate over a bound subject domain: a data-only index chain beside a
# live parameter vector read by subject (`reactivekernels-use` §4a).
@kernel _closure_reader(live, kinds_by_subject, read_idx, subjects) = begin
    out = plate(subjects) do s
        kinds = kinds_by_subject[s]
        read_positions = findall(isone, kinds)
        observation_operations = read_positions[read_idx[s]]
        sum(live[observation_operations]; init = 0.0) * live[s]
    end
    total::Float64 = sum(out)
end
@kernel _closure_reader_ref(live, kinds_by_subject, read_idx, subjects) = begin
    out = plate(subjects, Ref(kinds_by_subject), Ref(read_idx), Ref(live)) do s, kinds_by_subject, read_idx, live
        kinds = kinds_by_subject[s]
        read_positions = findall(isone, kinds)
        observation_operations = read_positions[read_idx[s]]
        sum(live[observation_operations]; init = 0.0) * live[s]
    end
    total::Float64 = sum(out)
end

@testset "plate and scan closures read whole enclosing values" begin
    x, d = [1.0, 2.0, 3.0], [10.0, 20.0, 30.0]
    @test prepare(_closure_first)(x, d) == [11.0, 21.0, 31.0]
    @test prepare(_closure_length)(x, d) == [13.0, 23.0, 33.0]
    @test prepare(_closure_indexed)(x, d) == [11.0, 22.0, 33.0]
    # A capture of another length than the plate axis is not broadcast.
    @test prepare(_closure_first)([1.0, 2.0], d) == [11.0, 21.0, 31.0]
    @test prepare(_closure_normalized)(x) == x ./ sum(x)

    xs = [[1.0, 2.0], Float64[], [3.0]]
    @test prepare(_closure_ragged)(xs, d) ==
          [sum(xs[s]; init = 0.0) * d[s] + 3 for s in eachindex(d)]
    @test prepare(_closure_derived)(x, d) == (x .+ 1.0) .* d
    @test prepare(_closure_generator_branch)(x, d) ==
          [dd > 15 ? sum(x) * dd : 3dd for dd in d]

    groups, w = [[1.0, 2.0], [4.0]], [0.5, 0.25]
    @test prepare(_closure_nested)(groups, w) ==
          [sum(2.0 .* g .+ w[eachindex(g)]) for g in groups]

    q, series = [0.6, 1.5], [0.5, -1.0, 2.0, 0.25]
    @test prepare(_closure_scan)(q, series) ≈
          sum(accumulate((c, s) -> q[1] * c + q[2] * s, series; init = 0.0))
    sgroups = [[1.0, 2.0, 3.0], Float64[], [0.5]]
    expected_scan = sum(sgroups) do g
        sum(accumulate((c, s) -> c * w[2] + s * w[1] * length(g), g; init = 0.0);
            init = 0.0)
    end
    @test prepare(_closure_scan_in_plate)(sgroups, w) ≈ expected_scan
    @test prepare(_closure_plate_in_scan)([1.0, 2.0], w) ≈ cumsum([1.0, 2.0] .* sum(w))
    @test prepare(_closure_scan_in_scan)(sgroups, 0.5) ≈ cumsum(map(sgroups) do g
        sum(accumulate((c, x) -> c + 0.5x + length(g), g; init = 0.0); init = 0.0)
    end)
    @test prepare(_closure_scan_plate_sum)(series, 1.5) ≈
          sum(v -> -0.5 * (v / 1.5)^2, cumsum(series))
    # The streamed plate stores no scan vector, closure or not.
    @test !occursin("similar", _closure_program(prepare(_closure_scan_plate_sum)))

    # Live and bound captures give the same values.
    @test prepare(_closure_indexed; bound = (; x))(d) == [11.0, 22.0, 33.0]
    @test prepare(_closure_ragged; bound = (; xs))(d) == prepare(_closure_ragged)(xs, d)
end

@testset "closures lower exactly as their Ref spelling" begin
    x, d = [1.0, 2.0, 3.0], [10.0, 20.0, 30.0]
    xs = [[1.0, 2.0], Float64[], [3.0]]
    groups, w = [[1.0, 2.0], [4.0]], [0.5, 0.25]
    q, series = [0.6, 1.5], [0.5, -1.0, 2.0, 0.25]
    sgroups = [[1.0, 2.0, 3.0], Float64[], [0.5]]
    shifts, units, weights = [0, 2, 5], collect(1.0:8.0), [0.5, 1.0, 0.25]
    coefficients = [1.0, -0.5, 0.25, 0.125]
    cases = (
        (_closure_indexed, _closure_indexed_ref, (x, d)),
        (_closure_ragged, _closure_ragged_ref, (xs, d)),
        (_closure_derived, _closure_derived_ref, (x, d)),
        (_closure_generator_branch, _closure_generator_branch_ref, (x, d)),
        (_closure_nested, _closure_nested_ref, (groups, w)),
        (_closure_scan, _closure_scan_ref, (q, series)),
        (_closure_scan_in_plate, _closure_scan_in_plate_ref, (sgroups, w)),
        (_closure_plate_in_scan, _closure_plate_in_scan_ref, ([1.0, 2.0], w)),
        (_closure_scan_in_scan, _closure_scan_in_scan_ref, (sgroups, 0.5)),
        (_closure_scan_plate_sum, _closure_scan_plate_sum_ref, (series, 1.5)),
        (_closure_scan_strip, _closure_scan_strip_ref,
         (0:299, coefficients, 0.75)),
        (_closure_superposition, _closure_superposition_ref,
         (1:8, shifts, units, weights)),
        (_closure_evalpoly, _closure_evalpoly_ref,
         (collect(range(-1.0, 1.0; length = 70)), coefficients)),
    )
    # Equal programs include the native loop-order lowerings: the superposition
    # cell's dose-outer loop and the evalpoly cell's coefficient-outer chunks.
    for (closure, reffed, args) in cases
        k, r = prepare(closure), prepare(reffed)
        @test _closure_program(k) == _closure_program(r)
        @test k(args...) == r(args...)
    end
end

@testset "bound= caching treats a capture like a Ref operand" begin
    caches(p) = sort([only(r.outputs).name for r in p.recipes
                      if r.op isa ReactiveKernels._BoundConstant &&
                         startswith(String(only(r.outputs).name), "bound_plate_")])
    kinds = [[1, 2, 1, 1, 3], [2, 1, 1, 3, 1, 1], [3, 3]]
    read_idx = [[1, 3], [2, 4], Int[]]
    data = (; kinds_by_subject = kinds, read_idx, subjects = 1:3)
    k = prepare(_closure_reader; bound = data)
    r = prepare(_closure_reader_ref; bound = data)
    @test !isempty(caches(k.plan))
    @test caches(k.plan) == caches(r.plan)
    @test _closure_program(k) == _closure_program(r)
    for live in (collect(1.0:6.0), [0.5, -1.0, 2.0, 3.0, 0.0, 7.0])
        @test k(live) == r(live) == prepare(_closure_reader)(live, kinds, read_idx, 1:3)
    end

    # A bound capture lowers as the bound `Ref` operand.
    x, d = [1.0, 2.0, 3.0], [10.0, 20.0, 30.0]
    @test _closure_program(prepare(_closure_indexed; bound = (; x))) ==
          _closure_program(prepare(_closure_indexed_ref; bound = (; x)))
end

@testset "native Reverse differentiates a capture as its Ref spelling" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    function value_gradient(spec, active, args...; bound = (;))
        kernel = isempty(bound) ? prepare(spec) : prepare(spec; bound)
        prepared = prepare_ad(kernel, backend, args...; active)
        ad_value_and_gradient(prepared, args...)
    end

    q, series = [0.6, 1.5], [0.5, -1.0, 2.0, 0.25]
    @test value_gradient(_closure_scan, :q, q, series) ==
          value_gradient(_closure_scan_ref, :q, q, series)

    w, sgroups = [0.5, 0.25], [[1.0, 2.0, 3.0], Float64[], [0.5]]
    closure = value_gradient(_closure_scan_in_plate, :w, sgroups, w)
    @test closure == value_gradient(_closure_scan_in_plate_ref, :w, sgroups, w)
    reference(w) = sum(sgroups) do g
        sum(accumulate((c, s) -> c * w[2] + s * w[1] * length(g), g; init = 0.0);
            init = 0.0)
    end
    fd = map(eachindex(w)) do i
        e = zeros(length(w)); e[i] = 1e-6
        (reference(w .+ e) - reference(w .- e)) / 2e-6
    end
    @test last(closure) ≈ fd rtol = 1e-6

    kinds = [[1, 2, 1, 1, 3], [2, 1, 1, 3, 1, 1], [3, 3]]
    read_idx = [[1, 3], [2, 4], Int[]]
    bound = (; kinds_by_subject = kinds, read_idx, subjects = 1:3)
    live = [0.5, -1.0, 2.0, 3.0, 0.0, 7.0]
    @test value_gradient(_closure_reader, :live, live; bound) ==
          value_gradient(_closure_reader_ref, :live, live; bound)
end
