using ReactiveKernels, DifferentiationInterface, Enzyme, Test

# `plate_partition`: the cells of one or more plates whose key equals a label
# (one subject's observations) as ONE compiled pass over those cells, every
# value computed only at the rows the partition reads, or refused by name
# (decision
# `ReactiveKernels:snag.partition-graph-6306f756/decisions/2026-10-30T11-52-01-648-08uci2h`,
# option 1, increment 2; the test ladder of todo
# `ReactiveKernels:refactor.plate-partition/todos/2026-10-30T12-52-53-505-10j0aa7`).

_pp_central(f, x; h = 1e-6) = (f(x + h) - f(x - h)) / (2h)

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
    u = [0.3; [0.1 * ((k * 7) % 5 - 2) for k in 1:20]; 0.7]
    (; u, g, w, subject)
end
const _PP_HAVE = (:u, :g, :w, :subject)

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
