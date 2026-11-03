using ReactiveKernels, DifferentiationInterface, Enzyme, Test

# `plate_partition`: the cells of one or more plates whose key equals a label
# (one subject's observations) as ONE compiled pass over those cells, every
# value computed only at the rows the partition reads, or refused by name
# (decision
# `ReactiveKernels:snag.partition-graph-6306f756/decisions/2026-10-30T11-52-01-648-08uci2h`,
# option 1, increment 2; the test ladder of todo
# `ReactiveKernels:refactor.plate-partition/todos/2026-10-30T12-52-53-505-10j0aa7`).

_pp_central(f, x; h = 1e-6) = (f(x + h) - f(x - h)) / (2h)

function _pp_window_sum(x, rows, indices, sorted)
    shape = (Base.OneTo(isempty(rows) ? 0 : maximum(rows)),)
    window = ReactiveKernels._CellWindow(x, rows, shape, sorted)
    sum(window[indices])
end

@testset "partition row lookup preserves window indexing" begin
    # Contiguous, gapped, empty and extreme rows retain the window's checked
    # membership semantics. Non-partition windows retain first-match order.
    for rows in (Int[], [7], [7, 8, 9], [7, 9, 12],
                 [typemin(Int), typemin(Int) + 1, typemin(Int) + 2],
                 [typemax(Int) - 2, typemax(Int) - 1, typemax(Int)],
                 [typemin(Int), typemax(Int)])
        probes = isempty(rows) ? BigInt[0] :
            unique([BigInt.(rows); BigInt(first(rows)) - 1; BigInt(last(rows)) + 1])
        for row in probes
            expected = findfirst(==(row), rows)
            if expected === nothing
                @test_throws BoundsError ReactiveKernels._cell_row_position(rows, row, row, true)
            else
                @test ReactiveKernels._cell_row_position(rows, row, row, true) === expected
                @test ReactiveKernels._cell_row_position(rows, Int(row), row, true) === expected
            end
        end
    end
    @test_throws BoundsError ReactiveKernels._cell_row_position([7, 9, 12], 8, 8, true)
    for row in (9, 7, 8)
        @test ReactiveKernels._cell_row_position([9, 7, 9, 8], row, row, false) ==
            findfirst(==(row), [9, 7, 9, 8])
    end
    @test_throws BoundsError ReactiveKernels._cell_row_position([9, 7, 9, 8], 6, 6, false)

    # Exercise uncached gathers with repeats and reversed order, both with
    # contiguous windows and the bisection fallback. Native Reverse must
    # accumulate each repeated read at the same local position as Julia.
    for n in (0, 1, 7, 37), stride in (1, 2)
        rows = collect(11:stride:(11 + stride * (n - 1)))
        indices = [reverse(rows); rows]
        x = [0.2 + 0.1 * i for i in 1:n]
        before = copy(rows)
        expected = [count(==(row), indices) for row in rows]
        @test _pp_window_sum(x, rows, indices, true) ≈ sum(x .* expected)
        gradient = zeros(n)
        Enzyme.autodiff(Enzyme.Reverse, _pp_window_sum, Enzyme.Active,
            Enzyme.Duplicated(x, gradient), Enzyme.Const(rows),
            Enzyme.Const(indices), Enzyme.Const(true))
        @test gradient ≈ expected
        @test rows == before
    end
end

# Rung 1: rows with a subject key, a per-row predictor and a group effect
# gathered by a group nested in the subject, with the parameters unpacked
# from a packed vector as RKPPL unpacks them (a scalar, a 10 x 2 block, a
# scalar).
@kernel _pp_subjects(u::AbstractVector{Float64}, g::AbstractVector{Int},
                     w::AbstractVector{Float64}, subject::AbstractVector{Int}) = begin
    a::Float64 = u[1]
    zflat::AbstractVector{Float64} = view(u, 2:21)
    Z::AbstractMatrix{Float64} = reshape(zflat, 10, 2)
    tau::Float64 = u[22]
    R = Z * [1.0 0.5; 0.0 1.0]
    log_k = a .+ w .* tau .+ R[g, 1]
    log_v = R[g, 2]
    dens = plate(log_k, log_v, w) do lk, lv, wi
        -(wi - exp(lk))^2 / 2 - lv^2 / 2
    end
    total = sum(dens)
    return total
end

# `per_subject[s]` rows for subject `s`, in two groups each.
function _pp_subject_data(per_subject)
    subject = Int[s for (s, n) in enumerate(per_subject) for _ in 1:n]
    g = Int[2(s - 1) + 1 + i % 2 for (s, n) in enumerate(per_subject) for i in 1:n]
    w = [sin(i / 3) for i in eachindex(subject)]
    # No exact zero among the effects: a zero effect makes a derivative vanish
    # where the coordinate is still read.
    u = [0.3; [0.1 * ((k * 7) % 5 - 1.5) for k in 1:20]; 0.7]
    (; u, g, w, subject)
end
const _PP_HAVE = (:u, :g, :w, :subject)

@testset "plate_partition: a key holds one entry per cell" begin
    d = _pp_subject_data((3, 4, 5))
    spec = plate_partition(_pp_subjects, :dens; by = :subject, index = :label)
    k = prepare(spec; have = (_PP_HAVE..., :label), want = :partition_total)
    @test k(d.u, d.g, d.w, d.subject, 1) isa Float64
    n = length(d.subject)
    # A short key would leave cells in no partition, whatever label is asked.
    for key in (d.subject[1:(n - 1)], [d.subject; 1])
        message = "the key :subject has $(length(key)) entries, but its plate has $n cells"
        @test_throws message k(d.u, d.g, d.w, key, 1)
        @test_throws message prepare(spec; have = (_PP_HAVE..., :label), want = :partition_total,
            bound = (; subject = key))(d.u, d.g, d.w, 1)
    end
    # With the plate's data bound too, the plan knows its domain, so the check
    # runs when each label is planned, before any evaluation.
    planned = plate_partition(_pp_subjects, :dens; by = :subject, index = :label, plan = :rows)
    planner = partition_plan_kernel(planned; have = (_PP_HAVE..., :label, :rows),
        want = :partition_total, bound = (; d.g, d.w, subject = d.subject[1:(n - 1)]))
    @test_throws "has $(n - 1) entries, but its plate has $n cells" planner(d.u, 1, nothing)
end

@testset "plate_partition: one subject's cells as one pass" begin
    spec = plate_partition(_pp_subjects, :dens; by = :subject, index = :label)
    @test spec isa KernelSpec
    k = prepare(spec; have = (_PP_HAVE..., :label), want = :partition_total)
    full = prepare(_pp_subjects; have = _PP_HAVE, want = :dens)
    cell = prepare(plate_cell(_pp_subjects, :dens; index = :i);
                   have = (_PP_HAVE..., :i), want = :dens_cell)
    for sizes in ((3, 4, 5, 2, 6), (40, 1, 70, 12, 9))
        d = _pp_subject_data(sizes)
        dens = full(d.u, d.g, d.w, d.subject)
        for s in 1:5
            rows = findall(==(s), d.subject)
            value = k(d.u, d.g, d.w, d.subject, s)
            @test value ≈ sum(dens[rows])
            @test value ≈ sum(cell(d.u, d.g, d.w, d.subject, i) for i in rows)
        end
        @test sum(k(d.u, d.g, d.w, d.subject, s) for s in 1:5) ≈ sum(dens)
    end
    # Bound keys: the label's positions are a dictionary read; the cost of a
    # partition does not grow with the other partitions' rows.
    d = _pp_subject_data((10, 10, 10, 10, 10))
    bound = prepare(spec; have = (_PP_HAVE..., :label), want = :partition_total,
                    bound = (; d.g, d.w, d.subject))
    @test [bound(d.u, s) for s in 1:5] ≈ [k(d.u, d.g, d.w, d.subject, s) for s in 1:5]
    allocated(sizes) = let d = _pp_subject_data(sizes),
            b = prepare(spec; have = (_PP_HAVE..., :label), want = :partition_total,
                        bound = (; d.g, d.w, d.subject))
        b(d.u, 1)
        @allocated b(d.u, 1)
    end
    @test allocated((10, 10, 10, 10, 10)) == allocated((10, 1000, 1000, 1000, 1000))
    # Every value is at the partition's rows, a scalar, or an input.
    report = partition_locality(bound)
    status(name) = only(e.status for e in report if e.value === name)
    @test status(:zflat) === :rows && status(:Z) === :rows && status(:R) === :rows
    @test status(:log_k) === :rows && status(:log_v) === :rows
    @test status(:a) === :scalar && status(:tau) === :scalar
    @test all(e.status !== :whole for e in report)
    # A label no cell has is refused by name.
    @test_throws ArgumentError bound(d.u, 6)
    @test occursin("no cell has the key 6", sprint(showerror,
        try bound(d.u, 6) catch e; e end))
    # Native Reverse through the partition: the gradient of subject 2's
    # density is nonzero on `a`, its two groups' rows of both columns of
    # `Z`, and `tau`.
    ad = prepare_ad(bound, AutoEnzyme(; mode = Enzyme.Reverse), d.u, 2; active = :u)
    gradient = similar(d.u)
    value, _ = ad_value_and_gradient!(ad, gradient, d.u, 2)
    @test value ≈ bound(d.u, 2)
    for j in eachindex(d.u)
        @test gradient[j] ≈ _pp_central(h -> bound(d.u .+ h .* (eachindex(d.u) .== j), 2),
                                        0.0) atol = 1e-6
    end
    groups = unique(d.g[d.subject .== 2])
    @test findall(!iszero, gradient) == sort!([1; 1 .+ groups; 11 .+ groups; 22])
    # The ordinary (table) call of the operation agrees with the lowered one.
    plan = ReactiveKernels.plan(spec; have = (_PP_HAVE..., :label), want = :partition_total)
    partition = only(r for r in plan.recipes if r.op isa ReactiveKernels._AuthoredPlateCellOp)
    @test partition.op.rows
end

# Rung 2: two plates with different row sets and keys. An observation plate
# over rows reads a completed covariate (`vcat(observed, drawn)[order]`, the
# shape of a completion) and a group block through a product; a totals plate
# over groups reads several rows through a shared index array.
@kernel _pp_two_plates(u::AbstractVector{Float64}, g::AbstractVector{Int},
                       observed::AbstractVector{Float64}, order::AbstractVector{Int},
                       y::AbstractVector{Float64}, rows::AbstractVector{Vector{Int}},
                       totals::AbstractVector{Float64}, subject::AbstractVector{Int},
                       group_subject::AbstractVector{Int}) = begin
    a::Float64 = u[1]
    drawn::AbstractVector{Float64} = view(u, 2:6)
    zflat::AbstractVector{Float64} = view(u, 7:26)
    Z::AbstractMatrix{Float64} = reshape(zflat, 10, 2)
    sigma::Float64 = u[27]
    completed = vcat(observed, drawn)[order]
    R = Z * [1.0 0.5; 0.0 1.0]
    log_w = a .+ 0.1 .* completed .+ R[g, 1]
    dens = plate(log_w, y) do lw, yi
        -(yi - exp(lw))^2 / 2
    end
    group_totals = plate(rows, totals) do r, t
        -(t - sum(exp.(log_w[r])))^2 / 2 - log(sigma)
    end
    total = sum(dens) + sum(group_totals)
    return total
end
function _pp_two_plate_data(per_subject)
    subject = Int[s for (s, n) in enumerate(per_subject) for _ in 1:n]
    n = length(subject)
    g = Int[2(s - 1) + 1 + i % 2 for (s, n) in enumerate(per_subject) for i in 1:n]
    # Five drawn (missing) entries at the rows `order` selects past the observed.
    observed = [cos(i / 2) for i in 1:n-5]
    order = [i <= n - 5 ? i : (n - 5) + (i - (n - 5)) for i in 1:n]
    y = [0.8 + sin(i / 5) for i in 1:n]
    groups = unique(g)
    rows = [findall(==(gi), g) for gi in groups]
    totals = [1.5 + 0.1 * k for k in eachindex(groups)]
    group_subject = [subject[first(r)] for r in rows]
    u = [0.2; [0.05 * k for k in 1:5]; [0.1 * ((k * 7) % 5 - 2) for k in 1:20]; 0.9]
    (; u, g, observed, order, y, rows, totals, subject, group_subject)
end
const _PP_TWO_HAVE = (:u, :g, :observed, :order, :y, :rows, :totals, :subject, :group_subject)

@testset "plate_partition: two plates with their own keys, in one pass" begin
    spec = plate_partition(_pp_two_plates, (:dens, :group_totals);
        by = (dens = :subject, group_totals = :group_subject), index = :label)
    have = (_PP_TWO_HAVE..., :label)
    k = prepare(spec; have, want = :partition_total)
    full = prepare(_pp_two_plates; have = _PP_TWO_HAVE, want = (:dens, :group_totals))
    d = _pp_two_plate_data((4, 6, 3, 5, 7))
    args(d) = (d.u, d.g, d.observed, d.order, d.y, d.rows, d.totals, d.subject, d.group_subject)
    dens, group_totals = full(args(d)...)
    for s in 1:5
        expected = sum(dens[d.subject .== s]) + sum(group_totals[d.group_subject .== s])
        @test k(args(d)..., s) ≈ expected
    end
    bound = prepare(spec; have, want = :partition_total,
        bound = (; d.g, d.observed, d.order, d.y, d.rows, d.totals, d.subject, d.group_subject))
    @test [bound(d.u, s) for s in 1:5] ≈ [sum(dens[d.subject .== s]) +
        sum(group_totals[d.group_subject .== s]) for s in 1:5]
    # One entry per cell: both partition cells read `completed` and `log_w` at rows.
    report = partition_locality(bound)
    @test all(e.status !== :whole for e in report)
    statuses(name) = unique(e.status for e in report if e.value === name)
    @test statuses(:completed) == [:rows] && statuses(:log_w) == [:rows]
    @test length(unique(e.cell for e in report)) == 2
    allocated(sizes) = let d = _pp_two_plate_data(sizes),
            b = prepare(spec; have, want = :partition_total,
                bound = (; d.g, d.observed, d.order, d.y, d.rows, d.totals, d.subject,
                           d.group_subject))
        b(d.u, 1)
        @allocated b(d.u, 1)
    end
    @test allocated((8, 8, 8, 8, 8)) == allocated((8, 800, 800, 800, 800))
    # A label one plate has no cell of: that plate contributes nothing.
    e = _pp_two_plate_data((5, 5, 5, 5, 5))
    only_rows = (; e..., group_subject = [s == 1 ? 1 : 2 for s in e.group_subject])
    b = prepare(spec; have, want = :partition_total, bound = (; only_rows.g, only_rows.observed,
        only_rows.order, only_rows.y, only_rows.rows, only_rows.totals, only_rows.subject,
        only_rows.group_subject))
    dens, group_totals = full(args(e)...)
    @test b(e.u, 3) ≈ sum(dens[e.subject .== 3])
    @test b(e.u, 2) ≈ sum(dens[e.subject .== 2]) + sum(group_totals[only_rows.group_subject .== 2])
    # Native Reverse: subject 1's gradient reads `a`, its drawn entries, its
    # groups' rows of `Z` and `sigma` only.
    ad = prepare_ad(bound, AutoEnzyme(; mode = Enzyme.Reverse), d.u, 1; active = :u)
    gradient = similar(d.u)
    value, _ = ad_value_and_gradient!(ad, gradient, d.u, 1)
    @test value ≈ bound(d.u, 1)
    for j in eachindex(d.u)
        @test gradient[j] ≈ _pp_central(h -> bound(d.u .+ h .* (eachindex(d.u) .== j), 1),
                                        0.0) atol = 1e-6
    end
    @test all(iszero, gradient[6 .+ (3:10)])   # other subjects' rows of column 1 of Z
end

# A value an ordinary function computes, read at the partition's rows.
_pp_opaque_field(x, s) = s .* cumsum(x)
@kernel _pp_opaque(x::AbstractVector{Float64}, s::Float64, w::AbstractVector{Float64},
                   key::AbstractVector{Int}) = begin
    field::Vector{Float64} = _pp_opaque_field(x, s)
    eta = field .+ w
    dens = plate(eta, w) do ei, wi
        -(wi - ei)^2 / 2
    end
    total = sum(dens)
    return total
end

@testset "plate_partition: the spellings of `by`" begin
    # The docstring promises `plate => key` pairs, and a vector or tuple of them
    # used to be read by position (snag `prepare-partitio-1865f53f`).
    keys_by_plate = Dict(:dens => :subject, :group_totals => :group_subject)
    names = (:dens, :group_totals)
    for by in ((dens = :subject, group_totals = :group_subject), keys_by_plate,
               pairs((dens = :subject, group_totals = :group_subject)),
               [:dens => :subject, :group_totals => :group_subject],
               (:dens => :subject, :group_totals => :group_subject),
               (name => keys_by_plate[name] for name in names),
               ["dens" => "subject", "group_totals" => "group_subject"],
               [:dens => :subject, :group_totals => :group_subject, :dens => :subject],
               # Loosely typed containers hold the same names.
               Dict{Symbol,Any}(:dens => :subject, :group_totals => :group_subject),
               Pair{Any,Any}[:dens => :subject, "group_totals" => :group_subject],
               Any[:dens => :subject, :group_totals => "group_subject"])
        @test ReactiveKernels._partition_by_map(by, names) == keys_by_plate
    end
    @test ReactiveKernels._partition_by_map(:dens => :subject, (:dens,)) == Dict(:dens => :subject)
    @test ReactiveKernels._partition_by_map(:subject, names) ==
        Dict(:dens => :subject, :group_totals => :subject)

    # End to end, against the plates' own cells.
    spec = plate_partition(_pp_two_plates, names;
        by = [:dens => :subject, :group_totals => :group_subject], index = :label)
    k = prepare(spec; have = (_PP_TWO_HAVE..., :label), want = :partition_total)
    full = prepare(_pp_two_plates; have = _PP_TWO_HAVE, want = names)
    d = _pp_two_plate_data((4, 6, 3, 5, 7))
    args = (d.u, d.g, d.observed, d.order, d.y, d.rows, d.totals, d.subject, d.group_subject)
    dens, group_totals = full(args...)
    @test [k(args..., s) for s in 1:5] ≈ [sum(dens[d.subject .== s]) +
        sum(group_totals[d.group_subject .== s]) for s in 1:5]

    # A `by` that selects no key is refused naming the accepted spellings (the
    # reported error named none, and a pair or `nothing` raised a `MethodError`).
    function refusal(by)
        try
            plate_partition(_pp_subjects, :dens; by)
            nothing
        catch err
            err
        end
    end
    for by in ([:subject], "subject", 1, nothing, [:dens => 1])
        err = refusal(by)
        @test err isa ArgumentError
        @test occursin("`by` must be", err.msg)
        @test occursin("`plate => key` pair", err.msg)
    end
    for (by, named) in ([:other => :subject] => ":other", (; other = :subject) => ":other",
                        Pair{Symbol,Symbol}[] => "nothing")
        err = refusal(by)
        @test err isa ArgumentError
        @test occursin("no key for plate :dens; it names $named", err.msg)
        @test occursin("`plate => key` pair", err.msg)
    end
    # Two keys partition one plate's cells in two different ways, so the choice
    # cannot be left to the entries' order.
    err = refusal([:dens => :subject, :dens => :g])
    @test err isa ArgumentError
    @test occursin("plate :dens two keys, :subject and :g", err.msg)
end

@testset "plate_partition: refusals and arguments" begin
    @test_throws ArgumentError plate_partition(_pp_subjects, :total; by = :subject)
    @test_throws ArgumentError plate_partition(_pp_subjects, :dens; by = (; other = :subject))
    @test_throws ArgumentError plate_partition(_pp_subjects, :dens; by = :subject, locality = :loud)
    @test_throws ArgumentError plate_partition(_pp_subjects, :dens; by = :subject,
                                               partitioned = (:u,))
    # A value a cell reads at rows but would compute whole is refused by name,
    # the partition's default.
    err = try
        prepare(plate_partition(_pp_opaque, :dens; by = :key, index = :label);
                have = (:x, :s, :w, :key, :label), want = :partition_total)
        nothing
    catch e
        e
    end
    @test err isa PartitionLocalityError
    @test [v.value for v in err.violations] == [:field]
    warned = @test_logs (:warn, r"computes a value whole") prepare(
        plate_partition(_pp_opaque, :dens; by = :key, index = :label, locality = :warn);
        have = (:x, :s, :w, :key, :label), want = :partition_total)
    x, w, key = [1.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4], [1, 2, 1, 2]
    dens = prepare(_pp_opaque; have = (:x, :s, :w, :key), want = :dens)(x, 0.5, w, key)
    @test warned(x, 0.5, w, key, 1) ≈ dens[1] + dens[3]
    @test warned(x, 0.5, w, key, 2) ≈ dens[2] + dens[4]
    # A multidimensional plate domain is not partitioned (not built yet).
    @kernel _pp_matrix(M::Matrix{Float64}, key::Vector{Int}) = begin
        cells = plate(M) do m
            m^2
        end
        total = sum(cells)
        return total
    end
    spec = plate_partition(_pp_matrix, :cells; by = :key, index = :label)
    k = prepare(spec; have = (:M, :key, :label), want = :partition_total)
    @test_throws ArgumentError k([1.0 2.0; 3.0 4.0], [1, 1, 2, 2], 1)
end

@testset "plate_partition: footprint and separability" begin
    spec = plate_partition(_pp_subjects, :dens; by = :subject, index = :label)
    d = _pp_subject_data((4, 3, 5, 2, 6))
    bound = prepare(spec; have = (_PP_HAVE..., :label), want = :partition_total,
                    bound = (; d.g, d.w, d.subject))
    # The footprint of `u` is the support of the partition's gradient: `a`,
    # the subject's groups' rows of both columns of `Z`, and `tau`.
    for s in 1:5
        footprint = partition_footprint(bound, d.u, s)
        groups = unique(d.g[d.subject .== s])
        @test footprint.u == sort!([1; 1 .+ groups; 11 .+ groups; 22])
        @test footprint.label === :whole
    end
    ad = prepare_ad(bound, AutoEnzyme(; mode = Enzyme.Reverse), d.u, 3; active = :u)
    gradient = similar(d.u)
    ad_value_and_gradient!(ad, gradient, d.u, 3)
    @test findall(!iszero, gradient) == partition_footprint(bound, d.u, 3).u
    # A footprint returns the reads, not the wanted value: its body keeps the
    # row checks but not the density plate that only the value needs, which
    # the value kernel's body does compute (snag frozen-atlas-fir-2d4269a6).
    value_code = sprint(show, code_expr(bound))
    footprint_code = sprint(show, code_expr(partition_footprint_kernel(bound)))
    @test occursin("_plate_total_value", value_code)
    @test !occursin("_plate_total_value", footprint_code)
    @test occursin("_partition_check_rows", footprint_code)
    # Unbound, the key and row-wise ports are read at the partition's rows.
    k = prepare(spec; have = (_PP_HAVE..., :label), want = :partition_total)
    footprint = partition_footprint(k, d.u, d.g, d.w, d.subject, 2)
    @test footprint.w == findall(==(2), d.subject) == footprint.g
    @test footprint.u == partition_footprint(bound, d.u, 2).u
    # A one-cell kernel's footprint is that cell's.
    cell = prepare(plate_cell(_pp_subjects, :dens; index = :i);
                   have = (_PP_HAVE..., :i), want = :dens_cell, bound = (; d.g, d.w, d.subject))
    @test partition_footprint(cell, d.u, 6).u == sort!([1, 1 + d.g[6], 11 + d.g[6], 22])
    # Separability: `Z`'s rows belong to the group's subject; `a` and `tau`
    # are shared. Reading another subject's group refuses by name.
    labels = Any[missing; [1 + (r - 1) ÷ 2 for r in 1:10]; [1 + (r - 1) ÷ 2 for r in 1:10]; missing]
    rule = plate_partition(_pp_subjects, :dens; by = :subject, index = :label,
                           partitioned = (:u => labels,))
    separable = prepare(rule; have = (_PP_HAVE..., :label), want = :partition_total,
                        bound = (; d.g, d.w, d.subject))
    for s in 1:5
        @test partition_separability(separable, d.u, s) == partition_footprint(separable, d.u, s)
    end
    mixed = (; d..., g = [s == 2 ? 5 : g for (s, g) in zip(d.subject, d.g)])   # subject 2 reads subject 3's group
    coupled = prepare(rule; have = (_PP_HAVE..., :label), want = :partition_total,
                      bound = (; mixed.g, mixed.w, mixed.subject))
    err = try
        partition_separability(coupled, d.u, 2)
        nothing
    catch e
        e
    end
    @test err isa PartitionSeparabilityError
    @test err.port === :u && err.label == 2
    @test err.positions == [6, 16] && all(==(3), err.labels)
    @test occursin("belong to other partitions", sprint(showerror, err))
    @test partition_separability(coupled, d.u, 1) == partition_footprint(coupled, d.u, 1)
    @test_throws ArgumentError plate_partition(_pp_subjects, :dens; by = :subject,
                                               partitioned = (:nope => labels,))
end

# A pushed plate reading rows of shared matrices, `tau[k, :]` (a class's
# scales) and `Z[m, :]` (a muni's effects): captures gathered at the
# partition's rows, so a subject's partition reads its own classes' and munis'
# rows only.
@kernel _pp_rows(u::AbstractVector{Float64}, muni::AbstractVector{Int},
                 klass::AbstractVector{Int}, w::AbstractVector{Float64},
                 subject::AbstractVector{Int}) = begin
    tflat::AbstractVector{Float64} = view(u, 1:4)
    tau::AbstractMatrix{Float64} = reshape(tflat, 2, 2)
    zflat::AbstractVector{Float64} = view(u, 5:16)
    Z::AbstractMatrix{Float64} = reshape(zflat, 6, 2)
    eff = plate(muni, klass) do m, k
        sum(tau[k, :] .* Z[m, :])
    end
    dens = plate(eff, w) do e, wi
        -(wi - e)^2 / 2
    end
    total = sum(dens)
    return total
end
# Two munis per subject; subject 1 is class 1, the others class 2.
function _pp_rows_data(per_subject)
    subject = Int[s for (s, n) in enumerate(per_subject) for _ in 1:n]
    muni = Int[2(s - 1) + 1 + i % 2 for (s, n) in enumerate(per_subject) for i in 1:n]
    klass = Int[s == 1 ? 1 : 2 for s in subject]
    w = [cos(i / 2) for i in eachindex(subject)]
    u = [0.1 * ((k * 7) % 5 - 1.5) + 0.05 * k for k in 1:16]
    (; u, muni, klass, w, subject)
end
const _PP_ROWS_HAVE = (:u, :muni, :klass, :w, :subject)

# A lane of one row is extruded over the partition's rows.
@kernel _pp_extruded(x::AbstractVector{Float64}, c::AbstractVector{Float64},
                     subject::AbstractVector{Int}) = begin
    dens = plate(x, c) do xi, ci
        -(xi - ci)^2 / 2
    end
    total = sum(dens)
    return total
end

# A literal range of a matrix is a linear read, not rows: refused by name.
@kernel _pp_matrix_range(M::AbstractMatrix{Float64}, g::AbstractVector{Int},
                         w::AbstractVector{Float64}, subject::AbstractVector{Int}) = begin
    v = M[2:3]
    eta = v[g]
    dens = plate(eta, w) do ei, wi
        -(wi - ei)^2 / 2
    end
    total = sum(dens)
    return total
end

# A computed value captured as `field[q]` through a lane of no known type.
@kernel _pp_untyped(v::AbstractVector{Float64}, q, w::AbstractVector{Float64},
                    subject::AbstractVector{Int}) = begin
    field = exp.(v)
    dens = plate(w, q) do wi, qi
        -(wi - field[qi])^2 / 2
    end
    total = sum(dens)
    return total
end

# A product of unknown shape (`c` undeclared), gathered by one partitioned
# plate and read as a lane by the other: pushed for the gather, then dropped
# for the lane.
@kernel _pp_dropped(u::AbstractVector{Float64}, c, g::AbstractVector{Int},
                    w::AbstractVector{Float64}, subject::AbstractVector{Int},
                    group_subject::AbstractVector{Int}) = begin
    zflat::AbstractVector{Float64} = view(u, 1:20)
    Z::AbstractMatrix{Float64} = reshape(zflat, 10, 2)
    R = Z * c
    eta = R[g]
    dens = plate(eta, w) do ei, wi
        -(wi - ei)^2 / 2
    end
    pen = plate(R) do ri
        -ri^2 / 2
    end
    total = sum(dens) + sum(pen)
    return total
end

@testset "plate_partition: rows of shared matrices, extruded lanes, named refusals" begin
    # Rows of `tau` and `Z` gathered at the partition's rows.
    spec = plate_partition(_pp_rows, :dens; by = :subject, index = :label)
    d = _pp_rows_data((3, 4, 5))
    have = (_PP_ROWS_HAVE..., :label)
    bound = prepare(spec; have, want = :partition_total, bound = (; d.muni, d.klass, d.w, d.subject))
    full = prepare(_pp_rows; have = _PP_ROWS_HAVE, want = :dens)
    dens = full(d.u, d.muni, d.klass, d.w, d.subject)
    report = partition_locality(bound)
    @test all(e.status !== :whole for e in report)
    statuses(name) = unique(e.status for e in report if e.value === name)
    @test statuses(:tau) == [:rows] && statuses(:Z) == [:rows]
    # Static Reverse is required; the upstream repair for constant window
    # rows is recorded in Enzyme snag `native-reverse-r-285478e2`.
    ad = prepare_ad(bound, AutoEnzyme(; mode = Enzyme.Reverse),
                    d.u, 1; active = :u)
    gradient = similar(d.u)
    for s in 1:3
        @test bound(d.u, s) ≈ sum(dens[d.subject .== s])
        classes = unique(d.klass[d.subject .== s])
        munis = unique(d.muni[d.subject .== s])
        expected = sort!(vcat([[k, k + 2] for k in classes]..., [[4 + m, 10 + m] for m in munis]...))
        @test partition_footprint(bound, d.u, s).u == expected
        ad_value_and_gradient!(ad, gradient, d.u, s)
        @test findall(!iszero, gradient) == expected
    end
    # The cost of a partition does not grow with the other subjects' rows.
    allocated(sizes) = let dd = _pp_rows_data(sizes),
            k = prepare(spec; have, want = :partition_total,
                        bound = (; dd.muni, dd.klass, dd.w, dd.subject))
        k(dd.u, 1)
        @allocated k(dd.u, 1)
    end
    @test allocated((5, 5, 5)) == allocated((5, 500, 500))

    # An extruded lane: every row of the partition reads its one row.
    x = [0.1, 0.4, 0.9, 0.3, 0.7]
    subject = [1, 1, 2, 2, 2]
    extruded = prepare(plate_partition(_pp_extruded, :dens; by = :subject, index = :label);
                       have = (:x, :c, :subject, :label), want = :partition_total)
    @test extruded(x, [0.4], subject, 2) ≈ sum(-(x[3:5] .- 0.4) .^ 2 ./ 2)
    footprint = partition_footprint(extruded, x, [0.4], subject, 2)
    @test footprint.x == [3, 4, 5] && footprint.c == [1]

    # A literal range of a matrix reads linear positions, not rows: refused.
    M = [1.0 3.0; 2.0 4.0]
    g = [1, 2, 2, 1, 2]
    w = [0.5, 0.1, 0.9, 0.2, 0.4]
    range_spec = plate_partition(_pp_matrix_range, :dens; by = :subject, index = :label)
    range_have = (:M, :g, :w, :subject, :label)
    err = try
        prepare(range_spec; have = range_have, want = :partition_total)
        nothing
    catch e
        e
    end
    @test err isa PartitionLocalityError
    @test any(v -> v.value === :v && v.reason === :operands, err.violations)
    ignored = prepare(plate_partition(_pp_matrix_range, :dens; by = :subject, index = :label,
                                      locality = :ignore);
                      have = range_have, want = :partition_total)
    v = M[2:3]
    @test ignored(M, g, w, subject, 2) ≈ sum(-(w[3:5] .- v[g[3:5]]) .^ 2 ./ 2)

    # A capture through a lane of no known type is read whole, and named.
    vdata = [0.2, 0.5, 0.1]
    q = [1, 3, 2, 2, 1]
    untyped_spec = plate_partition(_pp_untyped, :dens; by = :subject, index = :label)
    untyped_have = (:v, :q, :w, :subject, :label)
    err = try
        prepare(untyped_spec; have = untyped_have, want = :partition_total)
        nothing
    catch e
        e
    end
    @test err isa PartitionLocalityError
    @test any(e -> e.value === :field && e.reason === :untyped_index, err.violations)
    @test occursin("not provably an integer position", sprint(showerror, err))
    typed = prepare(untyped_spec; have = untyped_have, want = :partition_total,
                    bound = (; q, w, subject))
    @test all(e.status !== :whole for e in partition_locality(typed))
    @test typed(vdata, 2) ≈ sum(-(w[3:5] .- exp.(vdata)[q[3:5]]) .^ 2 ./ 2)

    # A value pushed for a gather and dropped for a lane is named, not silent.
    dd = _pp_subject_data((4, 3, 5, 2, 6))
    group_subject = [1 + (r - 1) ÷ 2 for r in 1:10]
    dropped_spec = plate_partition(_pp_dropped, (:dens, :pen);
                                   by = (dens = :subject, pen = :group_subject), index = :label)
    dropped_have = (:u, :c, :g, :w, :subject, :group_subject, :label)
    err = try
        prepare(dropped_spec; have = dropped_have, want = :partition_total,
                bound = (; dd.g, dd.w, dd.subject, group_subject))
        nothing
    catch e
        e
    end
    @test err isa PartitionLocalityError
    @test any(e -> e.value === :R && e.reason === :unshaped, err.violations)
    c = [1.0, 0.5]
    uR = [0.1 * ((k * 7) % 5 - 1.5) + 0.02 * k for k in 1:20]
    silent = prepare(plate_partition(_pp_dropped, (:dens, :pen);
                                     by = (dens = :subject, pen = :group_subject), index = :label,
                                     locality = :ignore);
                     have = dropped_have, want = :partition_total,
                     bound = (; dd.g, dd.w, dd.subject, group_subject))
    R = reshape(uR, 10, 2) * c
    @test silent(uR, c, 3) ≈ sum(-(dd.w[dd.subject .== 3] .- R[dd.g[dd.subject .== 3]]) .^ 2 ./ 2) +
        sum(-R[group_subject .== 3] .^ 2 ./ 2)
end

@testset "plate_partition: a plan per label" begin
    # With a plan port the body reads its integer bookkeeping (rows, unions,
    # positions, checks) from the port, computed once per label by
    # `partition_plan`; values, footprints and gradients are those of the
    # partition without a plan.
    d = _pp_subject_data((4, 3, 5, 2, 6))
    plain = prepare(plate_partition(_pp_subjects, :dens; by = :subject, index = :label);
                    have = (_PP_HAVE..., :label), want = :partition_total,
                    bound = (; d.g, d.w, d.subject))
    spec = plate_partition(_pp_subjects, :dens; by = :subject, index = :label, plan = :plan)
    # An untyped plan port is typed by evaluating the plan on the exemplars.
    planned = prepare(spec; have = (_PP_HAVE..., :label, :plan), want = :partition_total,
                      bound = (; d.g, d.w, d.subject), exemplars = (d.u, 1, nothing))
    plans = Dict(s => partition_plan(planned, d.u, s, nothing) for s in 1:5)
    # No plan field is a type, nor a scalar or an array computed from `u`
    # (`u`'s values appear in no field).
    @test all(values(plans)) do p
        all(x -> !(x isa Type) && !(x isa AbstractFloat && x in d.u), p)
    end
    @test all(p -> p isa NamedTuple && !isempty(p), values(plans))
    for s in 1:5
        @test planned(d.u, s, plans[s]) ≈ plain(d.u, s)
        @test partition_footprint(planned, d.u, s, nothing).u == partition_footprint(plain, d.u, s).u
        before = deepcopy(plans[s])
        for scale in (2.0, -1.0, 1.0)
            @test planned(scale .* d.u, s, plans[s]) ≈ plain(scale .* d.u, s)
            @test all(isequal(plans[s][i], before[i]) for i in eachindex(before)
                      if before[i] isa AbstractArray)
        end
    end
    @test_throws ArgumentError partition_plan(planned, d.u, 6, nothing)
    @test_throws ArgumentError planned(d.u, 2, plans[1])
    @test_throws ArgumentError planned(d.u, 6, plans[1])
    # The planned body computes no rows of its own, and reads no data column
    # at rows: `w`'s rows are label-constant, computed once per label.
    @test !occursin("_cell_canonical_rows", string(planned.ast))
    @test occursin("Base.broadcastable(w)", string(plain.ast))
    @test !occursin("Base.broadcastable(w)", string(planned.ast))
    @test !occursin("_cell_take_positions", string(planned.ast))
    @test occursin("_cell_canonical_rows", string(partition_plan_kernel(planned).ast))
    # A plan per label costs no bytes that grow with the other subjects, and
    # fewer than the body without one.
    planned(d.u, 1, plans[1])
    @test (@allocated planned(d.u, 1, plans[1])) < (@allocated plain(d.u, 1))
    # Plan fields must differentiate under ordinary Reverse. Enzyme's
    # upstream repair is on dogfood from 1280e6ee; registered 0.13.216 has
    # the known activity-analysis defect `plan-const-int-r-4671f4a7`.
    ad = prepare_ad(planned, AutoEnzyme(; mode = Enzyme.Reverse),
                    d.u, 3, plans[3]; active = :u)
    gradient = similar(d.u)
    value, _ = ad_value_and_gradient!(ad, gradient, d.u, 3, plans[3])
    @test value ≈ plain(d.u, 3)
    @test findall(!iszero, gradient) == partition_footprint(plain, d.u, 3).u
    reference = similar(d.u)
    ad_value_and_gradient!(prepare_ad(plain, AutoEnzyme(; mode = Enzyme.Reverse), d.u, 3; active = :u),
                           reference, d.u, 3)
    @test gradient ≈ reference
    # The differentiated call receives the plan behind a reference, not as a
    # by-value aggregate of every field (snag frozen-atlas-fir-2d4269a6);
    # callers still pass the named tuple, and every label's plan is boxed alike.
    _, contexts = ReactiveKernels._ad_prepared_arguments(ad, (d.u, 3, plans[3]), (;))
    @test any(c -> DifferentiationInterface.unwrap(c) isa ReactiveKernels._PartitionPlanRef, contexts)
    @test !any(c -> DifferentiationInterface.unwrap(c) isa typeof(plans[3]), contexts)
    for s in (1, 5)
        expected = similar(d.u)
        ad_value_and_gradient!(prepare_ad(plain, AutoEnzyme(; mode = Enzyme.Reverse), d.u, s; active = :u),
                               expected, d.u, s)
        @test last(ad_value_and_gradient!(ad, similar(d.u), d.u, s, plans[s])) ≈ expected
    end
    # Every prepared operator pairs the boxed plan of its preparation with its
    # evaluation, not only the gradient.
    direction = cos.(eachindex(d.u))
    pullback = prepare_ad_pullback(planned, AutoEnzyme(; mode = Enzyme.Reverse), 1.0,
                                   d.u, 3, plans[3]; active = :u)
    @test ad_pullback(pullback, 1.0, d.u, 3, plans[3]) ≈ reference
    pushforward = prepare_ad_pushforward(planned, AutoEnzyme(; mode = Enzyme.Forward), (direction,),
                                         d.u, 3, plans[3]; active = :u)
    @test only(ad_pushforward(pushforward, (direction,), d.u, 3, plans[3])) ≈ sum(reference .* direction)
    second_order = SecondOrder(AutoEnzyme(; mode = Enzyme.Reverse), AutoEnzyme(; mode = Enzyme.Forward))
    hvp = prepare_ad_hvp(planned, second_order, (direction,), d.u, 3, plans[3]; active = :u)
    plain_hvp = prepare_ad_hvp(plain, second_order, (direction,), d.u, 3; active = :u)
    @test only(ad_hvp(hvp, (direction,), d.u, 3, plans[3])) ≈ only(ad_hvp(plain_hvp, (direction,), d.u, 3))
    @test_throws ArgumentError plate_partition(_pp_subjects, :dens; by = :subject, index = :label,
                                               plan = :label)
    # Index bookkeeping as against data, nested (a ragged row structure).
    @test ReactiveKernels._plan_bookkeeping([[1, 2], [3]])
    @test ReactiveKernels._plan_bookkeeping(Dict(1 => 2:3))
    @test ReactiveKernels._plan_bookkeeping((1, [2, 3]))
    @test !ReactiveKernels._plan_bookkeeping([1.0, 2.0])
    @test !ReactiveKernels._plan_bookkeeping([[1.0]])
    # This plan's row bookkeeping depends only on bound data and the packed
    # layout. Final loop axes no longer pull active values into preparation,
    # so the plan port can also be typed without exemplar values.
    automatic = prepare(spec; have = (_PP_HAVE..., :label, :plan),
                        want = :partition_total, bound = (; d.g, d.w, d.subject))
    # The plan kernel of the spec types the port: the same plans, the same values.
    planner = partition_plan_kernel(spec; have = (_PP_HAVE..., :label, :plan),
                                    want = :partition_total, bound = (; d.g, d.w, d.subject))
    typed_plans = Dict(s => planner(d.u, s, nothing) for s in 1:5)
    T = typeof(typed_plans[1])
    @test all(p -> typeof(p) === T, values(typed_plans))
    @test all(s -> automatic(d.u, s, typed_plans[s]) ≈ plain(d.u, s), 1:5)
    typed = prepare(plate_partition(_pp_subjects, :dens; by = :subject, index = :label,
                                    plan = :plan => T);
                    have = (_PP_HAVE..., :label, :plan), want = :partition_total,
                    bound = (; d.g, d.w, d.subject))
    @test all(s -> typed(d.u, s, typed_plans[s]) ≈ plain(d.u, s), 1:5)
    @test !occursin("_cell_canonical_rows", string(typed.ast))
end

@testset "partition plans share one type across empty plate portions" begin
    d = _pp_two_plate_data((5, 5, 5, 5, 5))
    # Label 1 has observations only, label 5 has gathered group totals only,
    # and label 2 has both. Both inferred and explicit plan ports must accept
    # these different combinations of empty portions.
    subject = [s == 5 ? 6 : s for s in d.subject]
    group_subject = [s == 1 ? 2 : s for s in d.group_subject]
    bound = (; d.g, d.observed, d.order, d.y, d.rows, d.totals, subject, group_subject)
    by = (dens = :subject, group_totals = :group_subject)
    spec = plate_partition(_pp_two_plates, (:dens, :group_totals);
        by, index = :label, plan = :plan)
    have = (_PP_TWO_HAVE..., :label, :plan)
    planner = partition_plan_kernel(spec; have, want = :partition_total, bound)
    labels = (1, 2, 5, 6)
    plans = Dict(s => planner(d.u, s, nothing) for s in labels)
    T = typeof(plans[2])
    @test isconcretetype(T)
    @test all(p -> typeof(p) === T, values(plans))
    typed = prepare(plate_partition(_pp_two_plates, (:dens, :group_totals);
        by, index = :label, plan = :plan => T);
        have, want = :partition_total, bound)
    full = prepare(_pp_two_plates; have = _PP_TWO_HAVE, want = (:dens, :group_totals))
    for s in labels
        automatic = prepare(spec; have, want = :partition_total, bound,
            exemplars = (d.u, s, nothing))
        for scale in (1.0, 1.2)
            u = scale .* d.u
            dens, totals = full(u, d.g, d.observed, d.order, d.y, d.rows,
                d.totals, subject, group_subject)
            expected = sum(dens[subject .== s]) + sum(totals[group_subject .== s])
            @test typed(u, s, plans[s]) ≈ expected
            @test automatic(u, s, plans[s]) ≈ expected
        end
    end
end

@testset "partition plans restart scheduled loops on every evaluation" begin
    # The previous-coordinate cache is evaluation state. Preparing a shape
    # may run the loop, but its final state must not become a plan field.
    body = quote
        rows = collect(1:label)
        previous = nothing
        first_coordinate = true
        total = zero(x)
        for i in rows
            if first_coordinate || i != previous[1]
                total = total + x * i
            end
            previous = CartesianIndex(i)
            first_coordinate = false
        end
        return total
    end
    plan, main, needed, _ = ReactiveKernels._partition_plan_split(body, [:x], Any[])
    planner = ReactiveKernels.compile(Expr(:function, Expr(:tuple, :x, :label),
        Expr(:block, plan..., Expr(:return, Expr(:tuple, needed...)))))
    reads = [Expr(:(=), name, :(p[$i])) for (i, name) in enumerate(needed)]
    planned = ReactiveKernels.compile(Expr(:function, Expr(:tuple, :x, :label, :p),
        Expr(:block, reads..., main...)))
    plans = [planner(1.0, label) for label in (0, 1, 3)]
    @test all(p -> typeof(p) === typeof(first(plans)), plans)
    for (label, p) in zip((0, 1, 3), plans), x in (2.0, -3.0, 4.0)
        @test planned(x, label, p) == x * label * (label + 1) / 2
    end
end

@testset "partition plans preserve writes and scratch ownership" begin
    # Execute the compiler split, reusing one plan with changing inputs.
    # The final sum observes indexed/property writes, aliases, mutating
    # calls whose destination is not first, and ordinary reassignments.
    for work in (
            :(begin a = zeros(2); a[1] = u; y = sum(a) end),
            :(begin a = zeros(2); a[1] += u; y = sum(a) end),
            :(begin a = zeros(2); b = view(a, :); b[1] = u; y = sum(a) end),
            :(begin t = (zeros(2), 1); (a, b) = t; a[1] = u; y = sum(t[1]) + b end),
            :(begin a = zeros(2); b = reshape(a, 1, 2); b .+= u; y = sum(a) end),
            :(begin a = zeros(2); broadcast!(identity, a, u); y = sum(a) end),
            :(begin a = zeros(2); fill!(a, 1.0); y = sum(a) * u end),
            :(begin a = Ref(0.0); a.x = u; y = a[] end),
            :(begin a = 0.0; a = u; y = a + 1 end),
            :(begin a = 0.0; a += u; y = a + 1 end),
            :(begin a = 0.0; a = 2.0; y = a * u end),
            :(begin a = Float64[]; push!(a, u); dims = axes(a); y = sum(a) + length(dims[1]) end))
        body = Expr(:block, :(saved = [1, 2]), work.args..., :(return y + sum(saved)))
        split = ReactiveKernels._partition_plan_split(body, [:u], Any[])
        @test split !== nothing
        plan, main, needed, _ = split
        planner = ReactiveKernels.compile(Expr(:function, Expr(:tuple, :u),
            Expr(:block, plan..., Expr(:return, Expr(:tuple, needed...)))))
        reads = [Expr(:(=), name, :(p[$i])) for (i, name) in enumerate(needed)]
        planned = ReactiveKernels.compile(Expr(:function, Expr(:tuple, :u, :p),
            Expr(:block, reads..., main...)))
        plain = ReactiveKernels.compile(Expr(:function, Expr(:tuple, :u), body))
        p = planner(1.0)
        before = deepcopy(p)
        for u in (2.0, -3.0, 4.0)
            @test planned(u, p) == plain(u)
            @test isequal(p, before)
        end
    end
end

@kernel _pp_buffered_rows(u::AbstractVector{Float64}, rows::AbstractVector{Int},
                          subject::AbstractVector{Int}) = begin
    transformed = plate(u) do x
        x^2 + x
    end
    selected = transformed[rows]
    dens = plate(selected) do x
        -x^2
    end
    total = sum(dens)
    return total
end

@kernel _pp_shape_only_prediction(u::AbstractVector{Float64}, x::AbstractVector{Float64},
                                  subject::AbstractVector{Int}) = begin
    a::Float64 = log(u[1])
    prediction = a .* x
    dens = plate(prediction) do value
        value^2
    end
    total = sum(dens)
    return total
end

@testset "partition planning does not evaluate predictions for loop axes" begin
    x = [0.2, 0.4, 0.7, 0.9]
    subject = [1, 2, 1, 2]
    have = (:u, :x, :subject, :label, :plan)
    spec = plate_partition(_pp_shape_only_prediction, :dens;
        by = :subject, index = :label, plan = :plan)
    planner = partition_plan_kernel(spec; have, want = :partition_total,
        bound = (; x, subject))
    # The exemplar gives the needed shapes. Its log is outside the value
    # domain, but the partition's row bookkeeping does not read that value.
    p = planner([-1.0], 1, nothing)
    typed = plate_partition(_pp_shape_only_prediction, :dens;
        by = :subject, index = :label, plan = :plan => typeof(p))
    k = prepare(typed; have, want = :partition_total, bound = (; x, subject))
    for value in (2.0, 4.0)
        u = [value]
        expected = log(value)^2 * sum(abs2, x[subject .== 1])
        @test k(u, 1, p) ≈ expected
        ad = prepare_ad(k, AutoEnzyme(; mode = Enzyme.Reverse), u, 1, p; active = :u)
        gradient = similar(u)
        actual, _ = ad_value_and_gradient!(ad, gradient, u, 1, p)
        @test actual ≈ expected
        @test only(gradient) ≈ 2log(value) / value * sum(abs2, x[subject .== 1])
    end
    @test_throws DomainError k([-1.0], 1, p)
end

@testset "partition plans separate fresh buffers from allocation inputs" begin
    # The allocator's marker and shape cannot alias the storage it returns.
    # A write through a view must still keep that storage per call, while
    # its read-only marker and row map can stay in the reusable plan.
    for n in (0, 1, 5)
        body = quote
            rows = collect(1:$n)
            marker = map(Float64, rows)
            dims = axes(marker)
            scratch = $(GlobalRef(ReactiveKernels, :_plate_similar_output))(marker, Float64, dims)
            alias = view(scratch, :)
            for i in eachindex(rows)
                alias[i] = u * rows[i]
            end
            return sum(scratch) + sum(rows)
        end
        plan, main, needed, _ = ReactiveKernels._partition_plan_split(body, [:u], Any[])
        @test :rows in needed
        @test :marker in needed
        @test :scratch ∉ needed && :alias ∉ needed
        planner = ReactiveKernels.compile(Expr(:function, Expr(:tuple, :u),
            Expr(:block, plan..., Expr(:return, Expr(:tuple, needed...)))))
        reads = [Expr(:(=), name, :(p[$i])) for (i, name) in enumerate(needed)]
        planned = ReactiveKernels.compile(Expr(:function, Expr(:tuple, :u, :p),
            Expr(:block, reads..., main...)))
        p = planner(1.0)
        before = deepcopy(p)
        for u in (2.0, -3.0, 4.0)
            @test planned(u, p) == (u + 1) * n * (n + 1) / 2
            @test isequal(p, before)
        end
    end

    # Public lowering: an intermediate authored plate allocates a buffer,
    # then a downstream gather reads it. Row bookkeeping belongs in the
    # plan, but the intermediate values must follow every changed input.
    loop_counts = Int[]
    for n in (6, 30)
        u = collect(range(0.1, 0.8; length = n))
        rows = repeat(collect(1:n), 2)
        subject = repeat([1, 2], n)
        spec = plate_partition(_pp_buffered_rows, :dens;
            by = :subject, index = :label, plan = :plan)
        k = prepare(spec; have = (:u, :rows, :subject, :label, :plan),
            want = :partition_total, bound = (; rows, subject),
            exemplars = (u, 1, nothing))
        ast = string(k.ast)
        @test !occursin("_cell_canonical_rows", ast)
        @test !occursin("_cell_window_positions", ast)
        push!(loop_counts, length(findall("for ", ast)))
        p = partition_plan(k, u, 1, nothing)
        before = deepcopy(p)
        for scale in (2.0, -1.0, 1.0)
            x = scale .* u
            selected = rows[subject .== 1]
            @test k(x, 1, p) ≈ -sum((x[selected].^2 .+ x[selected]).^2)
            @test isequal(p, before)
        end
        ad = prepare_ad(k, AutoEnzyme(; mode = Enzyme.Reverse), u, 1, p; active = :u)
        gradient = similar(u)
        value, _ = ad_value_and_gradient!(ad, gradient, u, 1, p)
        expected = [-count(==(i), rows[subject .== 1]) * 2 * (u[i]^2 + u[i]) *
                    (2 * u[i] + 1) for i in eachindex(u)]
        @test value ≈ k(u, 1, p)
        @test gradient ≈ expected
    end
    @test first(loop_counts) > 0
    @test first(loop_counts) == last(loop_counts)
end

# The grouped field BRM emits for `hsgp(x; by = g)` (snag
# `run-brm-s-full-g-2dc0667a`): one group's rows as one pass. The field
# `vec(sum(PHI .* scaled_z[g, :]; dims = 2))` with `scaled_z = z .*
# transpose(weights)` is computed at the group's rows; `z` is read at the
# group's row only, so the footprint and the gradient's support are one row of
# `z` beside the two scales.
@kernel _pp_field(u::AbstractVector{Float64}, PHI::Matrix{Float64},
                  omega2::Vector{Float64}, g::Vector{Int}, y::Vector{Float64}) = begin
    sigma::Float64 = u[1]
    rho::Float64 = u[2]
    zflat::AbstractVector{Float64} = view(u, 3:14)
    z::AbstractMatrix{Float64} = reshape(zflat, 4, 3)
    weights = plate(omega2) do w2
        sigma * (rho * sqrt(2pi))^0.5 * exp(-0.25 * rho^2 * w2)
    end
    scaled_z = z .* transpose(weights)
    rows = scaled_z[g, :]
    value = vec(sum(PHI .* rows; dims = 2))
    dens = plate(value, y) do v, yi
        -(yi - v)^2 / 2
    end
    total = sum(dens)
    return total
end
# Group 1 owns the first `first` rows; the other three groups share the rest, so
# growing `n` grows the other groups only (the cost comparison below).
function _pp_field_data(n; B = 3, first = 3)
    u = [0.8; 1.3; [0.1 * ((k * 7) % 5 - 2) + 0.05 for k in 1:(4 * B)]]
    PHI = [sin(0.3 * i * b) for i in 1:n, b in 1:B]
    omega2 = [(b * pi / 2)^2 for b in 1:B]
    g = [i <= first ? 1 : 2 + mod(i - first - 1, 3) for i in 1:n]
    y = [cos(i / 3) for i in 1:n]
    (; u, PHI, omega2, g, y)
end

@testset "plate_partition: a grouped field's rows as one pass" begin
    spec = plate_partition(_pp_field, :dens; by = :g, index = :label)
    have = (:u, :PHI, :omega2, :g, :y, :label)
    d = _pp_field_data(12)
    full = prepare(_pp_field; have = (:u, :PHI, :omega2, :g, :y), want = :dens)
    dens = full(d.u, d.PHI, d.omega2, d.g, d.y)
    bound = prepare(spec; have, want = :partition_total,
                    bound = (; d.PHI, d.omega2, d.g, d.y))
    for s in 1:4
        @test bound(d.u, s) ≈ sum(dens[d.g .== s])
    end
    report = partition_locality(bound)
    @test all(e.status !== :whole for e in report)
    status(name) = only(e.status for e in report if e.value === name)
    @test status(:value) === :rows && status(:rows) === :rows
    @test status(:scaled_z) === :rows && status(:z) === :rows && status(:zflat) === :rows
    @test status(:weights) === :shared && status(:PHI) === :data
    # The footprint of `u`: the scales and row `s` of `z` (`reshape(view(u, 3:14), 4, 3)`).
    for s in 1:4
        @test partition_footprint(bound, d.u, s).u == [1, 2, 2 + s, 6 + s, 10 + s]
    end
    # Native Reverse through the partition: exact, with the footprint as its
    # support. Enzyme's type analysis divides by a known-zero induction value
    # (`FnTypeInfo::knownIntegralValues`, the `SDiv` fold) on a partition whose
    # cells broadcast a data matrix read at rows against an active matrix
    # window and read the result by column — `PHI .* Z` with `Z` a reshape of
    # the packed vector trips it too, with no new form involved. That is an
    # Enzyme gap, fixed in Enzyme rather than worked around here (Enzyme snag
    # `enzyme-type-anal-b401ef8f`; user principle
    # `enzyme-fixed-in-enzyme-everywhere`), so the gradient is pinned broken:
    # an "Unexpected Pass" signals the repair. The one-cell gradient of the same
    # field passes (`test_plate_cell.jl`).
    result = try
        ad = prepare_ad(bound, AutoEnzyme(; mode = Enzyme.Reverse), d.u, 2; active = :u)
        gradient = similar(d.u)
        value, _ = ad_value_and_gradient!(ad, gradient, d.u, 2)
        (value, gradient)
    catch e
        e
    end
    if result isa Exception
        @test_broken !(result isa Exception)
        @test result isa DivideError
    else
        value, gradient = result
        @test value ≈ bound(d.u, 2)
        for j in eachindex(d.u)
            @test gradient[j] ≈ _pp_central(h -> bound(d.u .+ h .* (eachindex(d.u) .== j), 2),
                                            0.0) atol = 1e-6
        end
        @test findall(!iszero, gradient) == partition_footprint(bound, d.u, 2).u
    end
    # The cost of group 1's pass (three rows) does not grow with the other groups' rows.
    allocated(n) = let e = _pp_field_data(n),
            b = prepare(spec; have, want = :partition_total,
                        bound = (; e.PHI, e.omega2, e.g, e.y))
        b(e.u, 1)
        @allocated b(e.u, 1)
    end
    @test allocated(12) == allocated(4_000)
end
