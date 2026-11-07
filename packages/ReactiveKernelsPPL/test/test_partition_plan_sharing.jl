module PartitionPlanSharingTests
using ReactiveKernels, ReactiveKernelsPPL, Distributions, Enzyme, Test
using DifferentiationInterface: AutoEnzyme

# Snag per-unit-partiti-9d46c6de: a partition query or sampler keeps one plan
# per label, and a shared-prefix sampler two (its residual's and its local
# kernel's). Each held its label's rows and gathered data once per statement
# that reads them: on a one-unit-per-label toy, 82 arrays with 11 distinct
# contents for one label. Equal arrays of the plans of one preparation are now
# one object.
sharedtrend(a, w) = a .* w
const MODEL = quote
    a ~ Normal(0, 1)
    sigma ~ Exponential(1)
    z[levels(group)] .~ Normal.(0, 1)
    trend = sharedtrend(a, w)
    mu = trend .+ z[group] .* x
    y .~ Normal.(mu, sigma)
end

# Every array of plain bits a plan holds per label, through views and tuples.
function plan_arrays!(out, x)
    if x isa Array
        isbitstype(eltype(x)) && push!(out, x)
    elseif x isa SubArray
        plan_arrays!(out, parent(x)); plan_arrays!(out, x.indices)
    elseif x isa Base.Broadcast.Extruded
        plan_arrays!(out, x.x)
    elseif x isa Union{Tuple,NamedTuple}
        foreach(v -> plan_arrays!(out, v), x)
    end
    out
end
# Equal contents through tuples, views and extrusions (`Extruded` has no `==`).
same_contents(a::Union{Tuple,NamedTuple}, b::Union{Tuple,NamedTuple}) =
    typeof(a) === typeof(b) && all(map(same_contents, Tuple(a), Tuple(b)))
same_contents(a::AbstractArray, b::AbstractArray) = typeof(a) === typeof(b) && isequal(a, b)
same_contents(a::Base.Broadcast.Extruded, b::Base.Broadcast.Extruded) =
    same_contents(a.x, b.x) && a.keeps == b.keeps && a.defaults == b.defaults
same_contents(a, b) = isequal(a, b)
distinct_contents(arrays) = length(unique(a -> (typeof(a), size(a), vec(a)), arrays))
distinct_objects(arrays) = length(unique(objectid, arrays))

@testset "per-label plans share equal arrays" begin
    units = 12
    rows = 4
    n = units * rows
    data = (; group = repeat(1:units; inner = rows), x = sin.(1:n), w = cos.(1:n),
        y = 0.2 .* sin.(3 .* (1:n)))
    bound = bind_data(lower_rkppl(MODEL, data; conditioned = (:y,), mod = @__MODULE__), data)
    built = build_kernel(bound)
    u = collect(range(-0.3, 0.4; length = built.layout.total))
    declaration = (:z => collect(1:units),)
    query = prepare_partition(built, bound, :y; by = :group, partitioned = declaration)
    sampler = prepare_partition_sampler(built, bound, :y, u; by = :group,
        partitioned = declaration, backend = AutoEnzyme(; mode = Enzyme.Reverse))
    @test sampler.prefix !== nothing
    for label in 1:units
        # A sampler's residual and local plans of one label: equal arrays are
        # one object, within a plan and across the two.
        arrays = plan_arrays!(Any[], (sampler.plans[label],
            sampler.prefix.local_state.plans[label]))
        @test length(arrays) > distinct_contents(arrays)
        @test distinct_objects(arrays) == distinct_contents(arrays)
        query_arrays = plan_arrays!(Any[], query.plans[label])
        @test distinct_objects(query_arrays) == distinct_contents(query_arrays)
    end
    # Interning keeps the plan's type and values, and calls are unchanged.
    plan = first(values(query.plans))
    pool = ReactiveKernels._PartitionPlanPool()
    interned = ReactiveKernels._intern_partition_plan(pool, plan)
    @test typeof(interned) === typeof(plan)
    @test same_contents(interned, plan)
    # The label check compares a label's rows with its table's by identity.
    @test interned.partition_labels === plan.partition_labels
    for label in (1, units), shift in (0.0, 0.17)
        point = u .+ shift
        fp = partition_footprint(sampler, label)
        g = zeros(length(fp))
        value, _ = partition_value_and_gradient!(sampler, g, point, label)
        @test value ≈ query(point, label)
        for (slot, position) in enumerate(fp)
            plus, minus = copy(point), copy(point)
            plus[position] += 1e-6; minus[position] -= 1e-6
            @test g[slot] ≈ (query(plus, label) - query(minus, label)) / 2e-6 rtol = 1e-6 atol = 1e-6
        end
    end
end
end
