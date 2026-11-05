module PartitionLabelsTests
using Test, ReactiveKernels, ReactiveKernelsPPL, Distributions, Enzyme
using DifferentiationInterface: AutoEnzyme

# A partition's key given as its labels: a per-cell label that no statement of
# the model reads and that need not be in the binding at all, as for a model
# bound through BRM, which binds only the columns a statement reads (snag
# `prepare-partitio-7e89ede1`). The observations read a row-level value through
# a row index, and the label of a cell is its row's sensor.
const MODEL = quote
    a ~ Normal(0, 1)
    s[1:3] .~ Normal.(0, 1)
    row_mean = a .+ s[row_sensor]
    lower .~ interval_censored.(Normal.(row_mean[interval_row], 1.0), upper)
    limit .~ censored.(Normal.(row_mean[censored_row], 1.0), limit, Inf)
end
const DATA = (; row_sensor = [1, 1, 2, 2, 3, 3], interval_row = [1, 2, 3, 5, 6],
    censored_row = [2, 4, 4, 6], lower = [0.0, 0.1, -0.2, 0.3, 0.2],
    upper = [0.5, 0.6, 0.3, 0.8, 0.7], limit = [-1.0, -0.5, -0.7, -1.2])
const NAMES = (:lower, :limit)
const LOWER_UNIT = DATA.row_sensor[DATA.interval_row]
const LIMIT_UNIT = DATA.row_sensor[DATA.censored_row]
const BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

function build(data)
    bound = bind_data(lower_rkppl(MODEL, data; conditioned = NAMES, mod = @__MODULE__), data)
    bound, build_kernel(bound)
end
const BOUND, BUILT = build(DATA)
# The same model with its keys bound as data columns: the equivalent column form.
const COLUMN_BOUND, COLUMN_BUILT =
    build(merge(DATA, (; lower_unit = LOWER_UNIT, limit_unit = LIMIT_UNIT)))
const COLUMN_BY = (; lower = :lower_unit, limit = :limit_unit)
const LABEL_BY = (; lower = LOWER_UNIT, limit = LIMIT_UNIT)

# The log density of the cells of one sensor, written out from the model above.
function density(u, label)
    mean(row) = u[1] + u[1 + DATA.row_sensor[row]]
    total = 0.0
    for (i, row) in enumerate(DATA.interval_row)
        LOWER_UNIT[i] == label || continue
        d = Normal(mean(row), 1.0)
        total += log(cdf(d, DATA.upper[i]) - cdf(d, DATA.lower[i]))
    end
    for (i, row) in enumerate(DATA.censored_row)
        LIMIT_UNIT[i] == label || continue
        total += logcdf(Normal(mean(row), 1.0), DATA.limit[i])
    end
    total
end
central(f, u, j; h = 1e-6) = (f(setindex!(copy(u), u[j] + h, j)) - f(setindex!(copy(u), u[j] - h, j))) / (2h)

@testset "partition keys given as labels" begin
    u = [0.2, -0.3, 0.1, 0.4]
    @test :lower_unit ∉ keys(BOUND.columns)

    @testset "the labels partition the observations" begin
        query = prepare_partition(BUILT, BOUND, NAMES; by = LABEL_BY, locality = :error)
        columns = prepare_partition(COLUMN_BUILT, COLUMN_BOUND, NAMES; by = COLUMN_BY, locality = :error)
        @test sort!(collect(partition_labels(query))) == [1, 2, 3]
        for label in 1:3
            @test query(u, label) ≈ density(u, label)
            @test query(u, label) == columns(u, label)
        end
        @test_throws ArgumentError query(u, 4)
        # Each observation alone, keyed by its own labels.
        lower = prepare_partition(BUILT, BOUND, :lower; by = (; lower = LOWER_UNIT))
        limit = prepare_partition(BUILT, BOUND, :limit; by = (; limit = LIMIT_UNIT))
        for label in 1:3
            @test lower(u, label) + limit(u, label) ≈ density(u, label)
        end
    end

    @testset "every spelling, and labels beside a column" begin
        expected = [density(u, label) for label in 1:3]
        for by in (Dict(:lower => LOWER_UNIT, :limit => LIMIT_UNIT),
                   pairs(LABEL_BY),
                   [:lower => LOWER_UNIT, :limit => LIMIT_UNIT],
                   (:lower => LOWER_UNIT, :limit => LIMIT_UNIT))
            query = prepare_partition(BUILT, BOUND, NAMES; by)
            @test [query(u, label) for label in 1:3] ≈ expected
        end
        query = prepare_partition(BUILT, BOUND, :lower; by = :lower => LOWER_UNIT)
        @test partition_labels(query) == unique(LOWER_UNIT)
        mixed = prepare_partition(COLUMN_BUILT, COLUMN_BOUND, NAMES;
            by = (; lower = LOWER_UNIT, limit = :limit_unit))
        @test [mixed(u, label) for label in 1:3] ≈ expected
        # A label is any value, as in a data column.
        named = ["s1", "s1", "s2", "s3", "s3"]
        strings = prepare_partition(BUILT, BOUND, :lower; by = (; lower = named))
        @test partition_labels(strings) == ["s1", "s2", "s3"]
        @test strings(u, "s3") ≈ prepare_partition(BUILT, BOUND, :lower; by = (; lower = LOWER_UNIT))(u, 3)
    end

    @testset "the labels stay out of the caller's build and binding" begin
        ports = copy(BUILT.spec.port_order)
        columns = copy(BOUND.columns)
        labels = copy(LOWER_UNIT)
        query = prepare_partition(BUILT, BOUND, :lower; by = (; lower = labels))
        before = query(u, 3)
        @test BUILT.spec.port_order == ports
        @test BOUND.columns == columns
        @test_throws ContractValidationError prepare_partition(BUILT, BOUND, :lower; by = :lower_unit)
        # The query owns its labels: a later edit of the caller's vector moves nothing.
        labels .= 1
        @test query(u, 3) == before
        @test partition_labels(query) == unique(LOWER_UNIT)
    end

    @testset "gradients over each label's footprint" begin
        for compact in (true, false)
            sampler = prepare_partition_sampler(BUILT, BOUND, NAMES, u; by = LABEL_BY,
                backend = BACKEND, compact)
            for label in 1:3
                footprint = partition_footprint(sampler, label)
                @test footprint == [1, 1 + label]
                g = zeros(compact ? length(footprint) : length(u))
                value, g = partition_value_and_gradient!(sampler, g, u, label)
                @test value ≈ density(u, label)
                full = compact ? setindex!(zeros(length(u)), g, footprint) : g
                for j in eachindex(u)
                    @test full[j] ≈ central(x -> density(x, label), u, j) atol = 1e-6
                end
            end
        end
    end

    @testset "`partitioned` gives each level's label" begin
        rule = (:s => [1, 2, 3],)
        sampler = prepare_partition_sampler(BUILT, BOUND, NAMES, u; by = LABEL_BY,
            backend = BACKEND, partitioned = rule)
        query = prepare_partition(BUILT, BOUND, NAMES; by = LABEL_BY, partitioned = rule, u0 = u)
        for label in 1:3
            @test query(u, label) ≈ density(u, label)
            footprint = partition_footprint(sampler, label)
            value, _ = partition_value_and_gradient!(sampler, zeros(length(footprint)), u, label)
            @test value ≈ density(u, label)
        end
        # A label foreign to the coordinate it reads is refused by name, as for a column.
        @test_throws ReactiveKernels.PartitionSeparabilityError prepare_partition_sampler(
            BUILT, BOUND, NAMES, u; by = LABEL_BY, backend = BACKEND, partitioned = (:s => [2, 3, 1],))
    end

    @testset "a key holds one entry per cell" begin
        # A short key would leave cells in no partition: the labels would no
        # longer sum to the likelihood, with nothing to say so.
        for labels in (LOWER_UNIT[1:3], [LOWER_UNIT; 1])
            @test_throws "has $(length(labels)) entries, but its plate has 5 cells" prepare_partition(
                BUILT, BOUND, :lower; by = (; lower = labels))
            @test_throws "has $(length(labels)) entries, but its plate has 5 cells" prepare_partition_sampler(
                BUILT, BOUND, :lower, u; by = (; lower = labels), backend = BACKEND)
        end
        short = merge(DATA, (; lower_unit = LOWER_UNIT[1:3]))
        short_bound, short_built = build(short)
        @test_throws "the key :lower_unit has 3 entries, but its plate has 5 cells" prepare_partition(
            short_built, short_bound, :lower; by = :lower_unit)
    end

    @testset "spellings that give no labels per observation are refused" begin
        # A bare vector names no observation, and a long one must not fill the message.
        err = try
            prepare_partition(BUILT, BOUND, :lower; by = collect(1:100_000))
        catch e
            e
        end
        @test err isa ContractValidationError
        @test occursin("`by` must be", err.message)
        @test occursin("a vector of labels", err.message)
        @test length(err.message) < 1000
        # A repeated entry that agrees names one key, and two that differ are refused.
        repeated = prepare_partition(BUILT, BOUND, :lower;
            by = [:lower => LOWER_UNIT, :lower => copy(LOWER_UNIT)])
        @test partition_labels(repeated) == unique(LOWER_UNIT)
        @test_throws "two different vectors of labels" prepare_partition(BUILT, BOUND, :lower;
            by = [:lower => LOWER_UNIT, :lower => reverse(LOWER_UNIT)])
        @test_throws "two keys" prepare_partition(COLUMN_BUILT, COLUMN_BOUND, :lower;
            by = [:lower => LOWER_UNIT, :lower => :lower_unit])
        @test_throws ContractValidationError prepare_partition(BUILT, BOUND, NAMES;
            by = (; lower = LOWER_UNIT))
        @test_throws "`by` must be" prepare_partition(BUILT, BOUND, :lower;
            by = [:lower => LOWER_UNIT, :limit => 1])
    end
end

@testset "whole compressed transforms use constrained-axis labels" begin
    model = quote
        a ~ Normal(0, 1)
        L ~ LKJCholesky(K, 2.0)
        M ~ LKJCholesky(K, 2.0)
        y .~ Normal.(a .+ L[end, 1], 1.0)
        z .~ Normal.(a .+ M[end, 1], 1.0)
    end
    for K in (2, 4)
        data = (; K, y = [0.2, -0.4, 0.1], z = [-0.3, 0.5])
        bound = bind_data(lower_rkppl(model, data; conditioned = (:y, :z),
            mod = @__MODULE__), data)
        built = build_kernel(bound)
        metadata = only(filter(p -> p.path == (:L,), parameter_metadata(built)))
        @test metadata.shape == (K, K)
        @test metadata.ncoordinates == K * (K - 1) ÷ 2
        @test metadata.ncoordinates != K
        u = collect(range(-0.3, 0.4; length = built.layout.total))
        original = copy(u)
        by = (; y = fill(:left, length(data.y)), z = fill(:right, length(data.z)))
        rule = (:L => fill(:left, K), :M => fill(:right, K))
        positions(name) = let e = only(filter(e -> e.name == name, built.layout.entries))
            collect(e.offset:(e.offset + e.size - 1))
        end
        local_positions = Dict(:left => positions(:L), :right => positions(:M))
        function reference(point, label)
            p = constrain(built.layout, point)
            observations, factor = label === :left ? (data.y, p.L) : (data.z, p.M)
            sum(logpdf(Normal(p.a + factor[end, 1], 1.0), y) for y in observations)
        end
        for (observation, label) in ((:y, :left), (:z, :right))
            query = prepare_partition(built, bound, observation; by, partitioned = rule, u0 = u)
            for point in (u, u .+ 0.15)
                fp = ReactiveKernels.partition_footprint(query.kernel, point, label, nothing).unconstrained
                @test fp == sort!([positions(:a); local_positions[label]])
                @test query(point, label) ≈ reference(point, label)
            end
        end
        @test u == original
        # The public declaration counts constrained first-axis entries, not
        # packed coordinates, even when the latter are fewer or more.
        @test_throws "has $K levels" prepare_partition(built, bound, :y;
            by, partitioned = (:L => fill(:left, metadata.ncoordinates),), u0 = u)
        # Atomicity does not waive separability: assigning all of L to the
        # other unit would let :left read a foreign local block.
        @test_throws ReactiveKernels.PartitionSeparabilityError prepare_partition(
            built, bound, :y; by,
            partitioned = (:L => fill(:right, K), :M => fill(:right, K)), u0 = u)
    end
end
end
