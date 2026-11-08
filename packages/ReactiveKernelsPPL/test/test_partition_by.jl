module PartitionBySpellingTests
using Test, ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme

# Every spelling of a partition's `by` selects the same cells. The documentation
# promises `observation => column` pairs, and a vector or tuple of them used to
# be read by position (snag `prepare-partitio-1865f53f`).

const DATA = (; lower_unit = [1, 1, 2, 3, 3, 3], limit_unit = [2, 3, 3, 1],
    lower = [0.1, -0.2, 0.3, 0.4, -0.1, 0.2], limit = [1.0, 0.8, -0.3, 0.2])
const NAMES = (:lower, :limit)

# Two observations with different row counts, each keyed by its own column.
function build()
    bound = bind_data(lower_rkppl(quote
            a ~ Normal(0, 1)
            z[1:3] .~ Normal.(0, 1)
            lower_mu = a .+ z[lower_unit]
            limit_mu = a .+ z[limit_unit]
            lower .~ Normal.(lower_mu, 1.0)
            limit .~ Normal.(limit_mu, 1.0)
        end, DATA; conditioned = NAMES, mod = @__MODULE__), DATA)
    bound, build_kernel(bound)
end

# The label's cells of both observations, written out from the model above.
function cells(u, label)
    rows(y, unit) = [(y[i], u[1] + u[1 + label]) for i in eachindex(y) if unit[i] == label]
    [rows(DATA.lower, DATA.lower_unit); rows(DATA.limit, DATA.limit_unit)]
end
density(u, label) = sum(-(log(2pi) + (y - mu)^2) / 2 for (y, mu) in cells(u, label))
function gradient_of_density(u, label)
    gradient = zeros(length(u))
    for (y, mu) in cells(u, label)
        gradient[1] += y - mu
        gradient[1 + label] += y - mu
    end
    gradient
end

@testset "partition keys: every spelling of `by`" begin
    bound, built = build()
    u = [0.2, -0.3, 0.1, 0.4]
    @testset "$(typeof(by))" for by in (
            (; lower = :lower_unit, limit = :limit_unit),
            Dict(:lower => :lower_unit, :limit => :limit_unit),
            pairs((; lower = :lower_unit, limit = :limit_unit)),
            [:lower => :lower_unit, :limit => :limit_unit],
            (:lower => :lower_unit, :limit => :limit_unit),
            (name => Symbol(name, :_unit) for name in NAMES),
            ["lower" => "lower_unit", "limit" => "limit_unit"],
            # A repeated entry that agrees names one key.
            [:lower => :lower_unit, :limit => :limit_unit, :lower => :lower_unit],
            # Loosely typed containers hold the same names.
            Dict{Symbol,Any}(:lower => :lower_unit, :limit => :limit_unit),
            Pair{Any,Any}[:lower => :lower_unit, "limit" => :limit_unit],
            Any[:lower => :lower_unit, :limit => "limit_unit"])
        query = prepare_partition(built, bound, NAMES; by, locality = :error)
        @test sort!(collect(partition_labels(query))) == [1, 2, 3]
        @test [query(u, label) for label in 1:3] ≈ [density(u, label) for label in 1:3]
    end

    @testset "one observation by a single pair" begin
        query = prepare_partition(built, bound, :limit; by = :limit => :limit_unit,
                                  locality = :error)
        @test sort!(collect(partition_labels(query))) == [1, 2, 3]
        @test query(u, 2) ≈ -(log(2pi) + (DATA.limit[1] - u[1] - u[3])^2) / 2
    end

    @testset "the sampler reads the same spellings" begin
        sampler = prepare_partition_sampler(built, bound, NAMES, u;
            by = [:lower => :lower_unit, :limit => :limit_unit],
            backend = AutoEnzyme(; mode = Enzyme.Reverse), compact = false)
        gradient = similar(u)
        for label in 1:3
            value, _ = partition_value_and_gradient!(sampler, gradient, u, label)
            @test value ≈ density(u, label)
            @test gradient ≈ gradient_of_density(u, label)
        end
    end
end

# Naming a query's observations in another order prepares the same program and
# type: they are evaluated in the program's order whatever the caller lists.
# BRM's NLME partition named one ODL program's observations in two orders for
# two bindings that differ only in which plates have rows, so one package image
# served only one of them (snag `derived-kernels-b8900b11`). Which key column an
# observation reads never changed the type; the order did.
@testset "the observations' order does not change the program" begin
    bound, built = build()
    u = [0.2, -0.3, 0.1, 0.4]
    by = (; lower = :lower_unit, limit = :limit_unit)
    queries = [prepare_partition(built, bound, names; by, locality = :error)
               for names in (NAMES, reverse(NAMES))]
    @test typeof(queries[1]) === typeof(queries[2])
    @test queries[1].observations == queries[2].observations == NAMES
    @test [queries[1](u, label) for label in 1:3] == [queries[2](u, label) for label in 1:3]
    @test [queries[1](u, label) for label in 1:3] ≈ [density(u, label) for label in 1:3]
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    for compact in (true, false), partitioned in ((), (:z => [1, 2, 3],))
        samplers = [prepare_partition_sampler(built, bound, names, u; by, backend, compact,
                        partitioned, locality = :error) for names in (NAMES, reverse(NAMES))]
        @test typeof(samplers[1]) === typeof(samplers[2])
        for label in 1:3
            g = [zeros(compact ? length(partition_footprint(s, label)) : length(u))
                 for s in samplers]
            values = [first(partition_value_and_gradient!(s, gi, u, label))
                      for (s, gi) in zip(samplers, g)]
            @test values[1] == values[2]
            @test g[1] == g[2]
            @test values[1] ≈ density(u, label)
        end
    end
    # A cell query of two observations of one row count, in either order.
    data = (; a_obs = [0.1, -0.2, 0.3], b_obs = [1.0, 0.8, -0.3])
    two = bind_data(lower_rkppl(quote
            a ~ Normal(0, 1)
            a_obs .~ Normal.(a, 1.0)
            b_obs .~ Normal.(2a, 1.0)
        end, data; conditioned = (:a_obs, :b_obs), mod = @__MODULE__), data)
    built_two = build_kernel(two)
    cells = [prepare_cell_query(built_two, two, names) for names in
             ((:a_obs, :b_obs), (:b_obs, :a_obs))]
    @test typeof(cells[1]) === typeof(cells[2])
    @test [cells[1]([0.3], i) for i in 1:3] == [cells[2]([0.3], i) for i in 1:3]
    @test cells[1]([0.3], 2) ≈ -(log(2pi) + (data.a_obs[2] - 0.3)^2) / 2 -
                                (log(2pi) + (data.b_obs[2] - 0.6)^2) / 2
    # A name the program does not observe is still refused by name.
    err = try
        prepare_partition(built, bound, (:nope, :lower); by = (; nope = :lower_unit,
            lower = :lower_unit), locality = :error)
        nothing
    catch e
        e
    end
    @test err isa ContractValidationError
end

@testset "partition keys: refused by name" begin
    bound, built = build()
    function refusal(by; names = :lower)
        try
            prepare_partition(built, bound, names; by, locality = :error)
            nothing
        catch err
            err
        end
    end
    # A `by` that selects no column is refused naming the accepted spellings
    # (the reported error named none, and a pair or `nothing` raised a
    # `MethodError` from `keys`).
    for by in ([:lower_unit], "lower_unit", 1, nothing, [:lower => 1], [1 => :lower_unit])
        err = refusal(by)
        @test err isa ContractValidationError
        @test occursin("`by` must be", err.message)
        @test occursin("`observation => key` pair", err.message)
    end
    for (by, named) in ([:other => :lower_unit] => ":other", (; other = :lower_unit) => ":other",
                        Pair{Symbol,Symbol}[] => "nothing", () => "nothing")
        err = refusal(by)
        @test err isa ContractValidationError
        @test occursin("no key for observation :lower; it names $named", err.message)
        @test occursin("`observation => key` pair", err.message)
    end
    # One entry short of a key for every observation is the same refusal.
    err = refusal([:lower => :lower_unit]; names = NAMES)
    @test err isa ContractValidationError
    @test occursin("no key for observation :limit; it names :lower", err.message)

    # Two columns partition one observation's cells in two different ways, so
    # the choice cannot be left to the entries' order.
    err = refusal([:lower => :lower_unit, :lower => :limit_unit])
    @test err isa ContractValidationError
    @test occursin("observation :lower two keys, :lower_unit and :limit_unit", err.message)

    # A key must still be a supplied data column, in the new spellings too.
    err = refusal([:lower => :nope])
    @test err isa ContractValidationError
    @test occursin("is not a supplied data column", err.message)
end
end
