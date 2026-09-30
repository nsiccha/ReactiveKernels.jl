using ReactiveKernels, Test

module SharedPreparationFixture
using ReactiveKernels
const calls = zeros(Int, 3)
basis(data) = (calls[1] += 1; data .^ 2)
weights(q, scale) = (calls[2] += 1; q .* scale)
normalizer(w, b) = (calls[3] += 1; sum(w) + sum(b))

@kernel explicit(samples, data, q, scale) = begin
    b = basis(data)
    w = weights(q, scale)
    offset = normalizer(w, b)
    values = plate(samples, Ref(w), Ref(offset)) do x, shared_w, shared_offset
        x * sum(shared_w) - shared_offset
    end
    return values
end

function opaque(samples, data, q, scale)
    out = similar(samples)
    for i in eachindex(samples)
        b = basis(data)
        w = weights(q, scale)
        out[i] = samples[i] * sum(w) - normalizer(w, b)
    end
    return out
end
end

@testset "explicit shared dependencies and bound preparation" begin
    C = SharedPreparationFixture
    data, q, scale = [2., 3.], [.5, -.2], 1.7
    original_data, original_q = copy(data), copy(q)
    for n in (3, 30)
        samples = collect(1.:n)
        fill!(C.calls, 0)
        reference = C.opaque(samples, data, q, scale)
        @test C.calls == [n, n, n]
        fill!(C.calls, 0)
        live = prepare(C.explicit; bound=(; data))
        @test C.calls == [1, 0, 0]
        @test live(samples, q, scale) == reference
        @test C.calls == [1, 1, 1]
        @test live(samples, 2q, scale) == C.opaque(samples, data, 2q, scale)
        fill!(C.calls, 0)
        @test live(samples, q, 2scale) == C.opaque(samples, data, q, 2scale)
        @test C.calls == [n, n+1, n+1]

        fill!(C.calls, 0)
        fixed = prepare(C.explicit; bound=(; data, q, scale))
        @test C.calls == [1, 1, 1]
        @test fixed(samples) == reference
        @test fixed(samples .+ 1) == (samples .+ 1) .* sum(q .* scale) .-
            (sum(q .* scale) + sum(data .^ 2))
        @test C.calls == [1, 1, 1]
        fill!(C.calls, 0)
        rebound = prepare(C.explicit; bound=(; data=2data, q, scale))
        @test C.calls == [1, 1, 1]
        @test rebound(samples) == samples .* sum(q .* scale) .-
            (sum(q .* scale) + sum((2data) .^ 2))

        fill!(C.calls, 0)
        weights_only = prepare(C.explicit; want=:w, bound=(; data))
        @test C.calls == [0, 0, 0]
        @test weights_only(samples, q, scale) == q .* scale
        @test C.calls == [0, 1, 0]
    end
    @test data == original_data && q == original_q
end

# A prepared child spliced into the outer graph, the shape of the ShinyRK
# simulation graph (three prepared children behind a bound schedule).
module CachedBoundFixture
using ReactiveKernels
const calls = Ref(0)
basis(data) = (calls[] += 1; data .^ 2)
const child = prepare(@kernel scaled_child(w, q) = begin
    r = w .* q
    return r
end)
@kernel nested(data, q) = begin
    b = basis(data)
    r = child(b, q)
    total = sum(r)
    return total
end
# A plate cell whose lazy branch reads a bound plate argument: the partial
# evaluator splits the plate per binding, so its residual is value-dependent.
@kernel masked(x, mask) = begin
    values = plate(x, mask) do xi, m
        m > 0 ? 2xi : -xi
    end
    return values
end
end

@testset "cached bound preparation reuses the plan and compiled residual" begin
    C = SharedPreparationFixture
    data, q, scale = [2., 3.], [.5, -.2], 1.7
    samples = collect(1.:5)
    cache = PreparationCache()

    # The residual (weights, normalizer, the plate) is value-independent here:
    # a second binding reruns only the prefix and shares the compiled callable.
    fill!(C.calls, 0)
    k1 = prepare!(cache, C.explicit; bound = (; data))
    @test C.calls == [1, 0, 0]
    @test k1(samples, q, scale) == C.opaque(samples, data, q, scale)
    fill!(C.calls, 0)
    k2 = prepare!(cache, C.explicit; bound = (; data = 2data))
    @test C.calls == [1, 0, 0]
    @test k2.f === k1.f
    @test typeof(k2) === typeof(k1)
    @test k2(samples, q, scale) == C.opaque(samples, 2data, q, scale)
    @test k1(samples, q, scale) == C.opaque(samples, data, q, scale)
    @test k2(samples, q, scale) == prepare(C.explicit; bound = (; data = 2data))(samples, q, scale)
    @test length(cache) == 1

    # Another bound port set is its own entry, again shared across bindings.
    fill!(C.calls, 0)
    k3 = prepare!(cache, C.explicit; bound = (; data, q, scale))
    @test C.calls == [1, 1, 1]
    k4 = prepare!(cache, C.explicit; bound = (; data = 2data, q = 3q, scale))
    @test C.calls == [2, 2, 2]
    @test k4.f === k3.f
    @test k3(samples) == C.opaque(samples, data, q, scale)
    @test k4(samples) == C.opaque(samples, 2data, 3q, scale)
    @test length(cache) == 2

    # A binding of differently TYPED data on the same ports reuses the entry
    # (no re-planning, same prefix kernel) and, since the inner-plate pass
    # leaves this plate alone, the same compiled residual.
    fill!(C.calls, 0)
    k5 = prepare!(cache, C.explicit; bound = (; data = [2, 3]))
    @test C.calls == [1, 0, 0]
    @test k5.f === k1.f
    @test k5(samples, q, scale) == C.opaque(samples, [2, 3], q, scale)
    @test k1(samples, q, scale) == C.opaque(samples, data, q, scale)
    @test length(cache) == 2

    # A binding that hoists nothing the residual reads returns one kernel.
    w1 = prepare!(cache, C.explicit; want = :w, bound = (; data))
    w2 = prepare!(cache, C.explicit; want = :w, bound = (; data = 2data))
    @test w1 === w2
    @test w1(samples, q, scale) == q .* scale

    # The graph-level form.
    g = Graph()
    x = value!(g, :x, Float64)
    d = value!(g, :d, Vector{Float64})
    s = value!(g, :s, Float64)
    r = value!(g, :r, Float64)
    add!(g; inputs = (d,), outputs = (s,), op = sum)
    add!(g; inputs = (x, s), outputs = (r,), op = *)
    gk1 = prepare!(cache, g; have = (x, d), want = (r,), bound = (d => [1.0, 2.0],))
    gk2 = prepare!(cache, g; have = (x, d), want = (r,), bound = (d => [5.0, 5.0],))
    @test gk1(2.0) == 6.0 && gk2(2.0) == 20.0
    @test gk2.f === gk1.f
    # Bound-port validation is the same as `prepare`'s and caches nothing.
    @test_throws ArgumentError prepare!(cache, g; have = (x, d), want = (r,), bound = (s => 1.0,))
    @test_throws ArgumentError prepare!(cache, g; have = (x, d), want = (r,),
                                        bound = (d => [1.0], d => [2.0]))
end

@testset "cached bound preparation with a spliced prepared child" begin
    F = CachedBoundFixture
    cache = PreparationCache()
    data, q = [1.0, 2.0, 3.0], [2.0, 0.5, 1.0]
    F.calls[] = 0
    k1 = prepare!(cache, F.nested; bound = (; data))
    @test F.calls[] == 1
    @test k1(q) == sum((data .^ 2) .* q)
    k2 = prepare!(cache, F.nested; bound = (; data = 2data))
    @test F.calls[] == 2
    @test k2.f === k1.f
    @test k2(q) == sum(((2data) .^ 2) .* q)
    @test k1(q) == sum((data .^ 2) .* q)
    # The readable residual reports the new binding's constant.
    @test any(recipe -> recipe.op isa ReactiveKernels._BoundConstant &&
                        recipe.op.value == (2data) .^ 2, k2.plan.recipes)
end

module CachedBoundPlanFixture
using ReactiveKernels
struct LatticePlan
    shifts::Vector{Int}
end
struct ExactPlan
    rows::Matrix{Int}
end
count_of(plan::LatticePlan) = length(plan.shifts)
count_of(plan::ExactPlan) = size(plan.rows, 1)
offsets(plan::LatticePlan) = plan.shifts
offsets(plan::ExactPlan) = vec(sum(plan.rows; dims = 2))
const child = prepare(@kernel weighted_child(o, amounts) = begin
    r = o .* amounts
    return r
end)
@kernel dosing(plan, amounts) = begin
    n = count_of(plan)
    o = offsets(plan)
    weights = if n == 0
        Float64[]
    else
        child(o, amounts)
    end
    total = sum(weights; init = 0.0)
    return total
end
end

@testset "cached bound preparation across bound value types and a lazy arm" begin
    F = CachedBoundPlanFixture
    cache = PreparationCache()
    plans = (F.LatticePlan([1, 3, 4]), F.ExactPlan([1 2; 3 4]), F.LatticePlan(Int[]), F.ExactPlan([0 1; 1 1; 2 2]))
    amounts = ([1.0, 2.0, 3.0], [0.5, 0.25], Float64[], [1.0, 1.0, 1.0])
    kernels = [prepare!(cache, F.dosing; bound = (; plan)) for plan in plans]
    for (plan, a, k) in zip(plans, amounts, kernels)
        @test k(a) == prepare(F.dosing; bound = (; plan))(a)
        @test k(a) == sum(F.offsets(plan) .* a; init = 0.0)
    end
    @test all(k -> k.f === first(kernels).f, kernels)
    @test length(cache) == 1
end

# The ShinyRK simulation graph with its superposition plate authored inline:
# a bound schedule fixes the plate's domain and unit response, the dose
# amounts stay live. A plate with a bound-only cell value, which the
# inner-plate pass specializes only on a domain longer than one lane.
module InlinePlateFixture
using ReactiveKernels
struct LatticePlan
    nobs::Int
    shifts::Vector{Int}
    lags::Vector{Float64}
end
row_index(observation, plan::LatticePlan, i) = observation - plan.shifts[i]
dose_slots(plan::LatticePlan) = eachindex(plan.shifts)
observation_domain(plan::LatticePlan) = 1:plan.nobs
@kernel simulation(kernel, plan, amounts::Vector{Float64}) = begin
    units::Vector{Float64} = exp.(-kernel.k .* plan.lags)
    weights::Vector{Float64} = amounts .* kernel.F
    observations = observation_domain(plan)
    concentration::Vector{Float64} = plate(observations, Ref(plan), Ref(units), Ref(weights)) do o, p, u, w
        sum((w[i] * get(u, row_index(o, p, i), 0.0) for i in dose_slots(p)); init = 0.0)
    end
    total = sum(concentration)
    return total
end
schedule(n, doses; offset = 0) =
    LatticePlan(n, collect(offset .+ 3 .* (1:doses)), collect(0.1 .* (0:(n - 1))))
reference(kernel, plan, amounts) = sum(
    sum((amounts[i] * kernel.F * get(exp.(-kernel.k .* plan.lags), o - plan.shifts[i], 0.0)
         for i in eachindex(plan.shifts)); init = 0.0) for o in 1:plan.nobs)
@kernel scaled(x::Vector{Float64}, data) = begin
    values = plate(x, data) do xi, d
        c = exp(d)
        xi * c
    end
    return values
end
end

# An inline plate the inner-plate pass leaves alone rebinds to new bound array
# lengths through the one compiled residual, as the same plate in a prepared
# child does; the pass still decides every binding, so a binding it rewrites is
# specialized and the next binding it declines reuses the template (snag
# inline-plate-reb-7387f072: templates keyed on bound array sizes re-lowered
# the ShinyRK simulation graph at every new schedule length and kept one
# template per length).
@testset "bound preparation shares an inline plate's residual across lengths" begin
    F = InlinePlateFixture
    kernel = (; k = 0.3, F = 0.8)
    fresh(spec, bound) = prepare(plan(spec);
        bound = ReactiveKernels._kernel_bound_pairs(spec, bound))
    kernels = map([(40, 3, 0), (42, 3, 0), (40, 3, 1), (57, 5, 0), (8, 1, 2)]) do (n, doses, offset)
        schedule = F.schedule(n, doses; offset)
        k = prepare(F.simulation; bound = (; kernel, plan = schedule))
        amounts = collect(1.0:doses)
        @test k(amounts) == fresh(F.simulation, (; kernel, plan = schedule))(amounts)
        @test k(amounts) ≈ F.reference(kernel, schedule, amounts)
        k
    end
    # One compiled residual (the template's lowering) for every length.
    @test all(k -> k.f === first(kernels).f && k.ast === first(kernels).ast, kernels)
    entries = collect(values(F.simulation.graph.preparations.cache.bound))
    @test length(entries) == 1
    @test only(entries).template isa ReactiveKernels._BoundTemplate
    # The same through a caller-owned cache, and across a kernel's type.
    cache = PreparationCache()
    k1 = prepare!(cache, F.simulation; bound = (; kernel, plan = F.schedule(30, 2)))
    k2 = prepare!(cache, F.simulation; bound = (; kernel = (; k = 1, F = 2), plan = F.schedule(31, 2)))
    @test k2.ast === k1.ast
    @test k2([1.0, 2.0]) ≈ F.reference((; k = 1, F = 2), F.schedule(31, 2), [1.0, 2.0])

    # Specialized (a domain longer than one lane) and unspecialized (one lane)
    # bindings interleaved: each follows the pass's verdict for its own values.
    declined = []
    for (x, data) in (([1.0, 2.0, 3.0], [0.1, 0.2, 0.3]), ([1.0, 2.0, 3.0, 4.0], [0.5]),
                      ([1.0, 2.0], [0.0, 1.0]), ([2.0, 5.0], [0.25]))
        k = prepare(F.scaled; bound = (; data))
        @test k(x) == x .* exp.(data)
        @test k(x) == fresh(F.scaled, (; data))(x)
        length(data) == 1 ? push!(declined, k) :
            @test any(r -> r.op isa ReactiveKernels._AuthoredPlateOp &&
                           length(r.inputs) > 2, k.plan.recipes)
    end
    @test declined[2].ast === declined[1].ast
end

module FirstRebindingFixture
using ReactiveKernels
square(data) = data .^ 2
@kernel warm(data, q) = begin
    b = square(data)
    total = sum(b .* q)
    return total
end
@kernel fresh(data, q) = begin
    b = square(data)
    scaled = 2 .* b
    total = sum(scaled .* q)
    return total
end
end

# Rebinding code is not specialized per kernel type, so once any graph has
# rebound in a process, the first rebinding of another graph compiles nothing
# (snag plain-prepare-wi-4cd01ccf: 290 ms and 40 MB once per process for the
# ShinyRK simulation graph). The cache keeps no binding's data.
@testset "cached rebinding compiles nothing and retains no bound data" begin
    F = FirstRebindingFixture
    cache = PreparationCache()
    q = [1.0, 2.0, 3.0]
    for data in ([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
        @test prepare!(cache, F.warm; bound = (; data))(q) == sum(data .^ 2 .* q)
    end
    k1 = prepare!(cache, F.fresh; bound = (; data = [1.0, 2.0, 3.0]))
    @test k1(q) == 72.0
    data2 = [2.0, 0.0, 1.0]
    # Read both counters before any arithmetic on them: anything compiled
    # between the reads (even `first` on the counter tuple) is counted.
    Base.cumulative_compile_timing(true)
    before = Base.cumulative_compile_time_ns()
    k2 = prepare!(cache, F.fresh; bound = (; data = data2))
    after = Base.cumulative_compile_time_ns()
    Base.cumulative_compile_timing(false)
    compile_ns = first(after) - first(before)
    @test compile_ns == 0
    @test k2.f === k1.f
    @test k2(q) == 14.0
    @test k1(q) == 72.0

    # Neither a template nor an entry keeps the values of the binding it was
    # built from: a dropped kernel's hoisted data is collectable while the
    # cache lives on.
    function first_binding_hoisted(cache)
        kernel = prepare!(cache, F.fresh; have = (:data, :q), want = :scaled,
                          bound = (; data = [4.0, 5.0]))
        constant = only(op for op in kernel.ops if op isa ReactiveKernels._BoundConstant)
        WeakRef(constant.value)
    end
    hoisted = first_binding_hoisted(cache)
    @test prepare!(cache, F.fresh; have = (:data, :q), want = :scaled,
                   bound = (; data = [1.0, 1.0]))(q) == [2.0, 2.0]
    GC.gc(); GC.gc()
    @test hoisted.value === nothing
end

# The testset above warms the rebinding path on another graph first. The first
# rebinding in a FRESH process must compile nothing too: the path is in the
# package image, not compiled by whichever rebinding comes first (snag
# first-rebinding-64d82bfc: 41 ms and 0.55 MB once per process on the ShinyRK
# simulation graph, for a residual with or without an authored plate).
@testset "the first rebinding in a fresh process compiles nothing" begin
    fixture = raw"""
        using ReactiveKernels
        square(data) = data .^ 2
        @kernel plain(data, q) = begin
            b = square(data)
            total = sum(b .* q)
            return total
        end
        # An inline plate over a bound domain whose cells read a live value:
        # the inner-plate pass runs on every binding and leaves it alone.
        @kernel plated(data, q) = begin
            b = square(data)
            observations::UnitRange{Int} = 1:length(b)
            values::Vector{Float64} = plate(observations, Ref(b), Ref(q)) do t, shared, w
                shared[t] * w[t]
            end
            total = sum(values)
            return total
        end
        spec = ARGS[1] == "plated" ? plated : plain
        function first_rebinding(spec)
            q = [1.0, 2.0, 3.0]
            k1 = prepare(spec; bound = (; data = [1.0, 2.0, 3.0]))
            # Read both counters before any arithmetic on them.
            Base.cumulative_compile_timing(true)
            before = Base.cumulative_compile_time_ns()
            k2 = prepare(spec; bound = (; data = [3.0, 2.0, 1.0]))
            after = Base.cumulative_compile_time_ns()
            Base.cumulative_compile_timing(false)
            (first(after) - first(before), k2.f === k1.f, k1(q), k2(q))
        end
        compile_ns, shared, value1, value2 = first_rebinding(spec)
        entry = only(values(ReactiveKernels._graph_preparations(spec.graph).bound))
        print(join((compile_ns, shared, value1, value2, entry.plates), ","))
        """
    # Package-image native code is not used under coverage or without images.
    options = Base.JLOptions()
    if options.code_coverage != 0 || options.use_pkgimages == 0
        @test_skip "package-image native code is not in use"
    else
        project = Base.active_project()
        for (name, plates) in (("plain", false), ("plated", true))
            out = read(`$(Base.julia_cmd()) --startup-file=no --project=$project -e $fixture $name`,
                       String)
            compile_ns, shared, value1, value2, has_plates = split(out, ",")
            @test parse(Int, compile_ns) == 0
            @test shared == "true"
            @test parse(Float64, value1) == 1.0 + 8.0 + 27.0
            @test parse(Float64, value2) == 9.0 + 8.0 + 3.0
            @test has_plates == string(plates)
        end
    end
end

module PlainBoundFixture
using ReactiveKernels
const calls = Ref(0)
basis(data) = (calls[] += 1; data .^ 2)
@kernel spec(samples, data, q) = begin
    b = basis(data)
    w = q .* 2
    values = plate(samples, Ref(b), Ref(w)) do x, shared_b, shared_w
        x * sum(shared_b) + sum(shared_w)
    end
    return values
end
end

# Plain `prepare(…; bound)` reuses the value-independent work of earlier
# bindings through a cache the graph holds; no caller-owned cache is needed
# (snag plain-prepare-wi-4cd01ccf).
@testset "plain bound preparation reuses the graph's earlier bindings" begin
    B = PlainBoundFixture
    data, q = [2., 3.], [.5, -.2]
    samples = collect(1.:5)
    expected(data) = samples .* sum(data .^ 2) .+ sum(q .* 2)
    fresh(bound) = prepare(plan(B.spec);
        bound = ReactiveKernels._kernel_bound_pairs(B.spec, bound))
    B.calls[] = 0
    k1 = prepare(B.spec; bound = (; data))
    k2 = prepare(B.spec; bound = (; data = 2data))
    @test B.calls[] == 2
    @test k2.f === k1.f
    @test typeof(k2) === typeof(k1)
    @test k1(samples, q) == fresh((; data))(samples, q) == expected(data)
    @test k2(samples, q) == fresh((; data = 2data))(samples, q) == expected(2data)
    @test k1(samples, q) == expected(data)
    @test length(B.spec.graph.preparations.cache) == 1

    # Another boundary of the same graph is its own entry; a residual that
    # reads no bound value is one kernel for every binding.
    w1 = prepare(B.spec; want = :w, bound = (; data))
    @test w1 === prepare(B.spec; want = :w, bound = (; data = 2data))
    @test w1(samples, q) == q .* 2
    @test length(B.spec.graph.preparations.cache) == 2

    # A spliced prepared child and a lazy arm over differently typed bindings.
    F = CachedBoundFixture
    nested = [prepare(F.nested; bound = (; data = d)) for d in ([1.0, 2.0], [3.0, 4.0])]
    @test nested[2].f === nested[1].f
    @test nested[2]([2.0, 0.5]) == 9.0 * 2.0 + 16.0 * 0.5
    P = CachedBoundPlanFixture
    for plan_data in (P.LatticePlan([1, 3, 4]), P.ExactPlan([1 2; 3 4]), P.LatticePlan(Int[]))
        amounts = fill(0.5, P.count_of(plan_data))
        @test prepare(P.dosing; bound = (; plan = plan_data))(amounts) ==
            sum(P.offsets(plan_data) .* amounts; init = 0.0)
    end

    # A value-dependent residual still specializes per binding.
    x = [1.0, 2.0, 3.0, 4.0]
    for mask in ([1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        @test prepare(F.masked; bound = (; mask))(x) ==
            [m > 0 ? 2xi : -xi for (xi, m) in zip(x, mask)]
    end

    # The graph form; a mutation of the graph starts afresh.
    g = Graph()
    xv = value!(g, :x, Float64)
    d = value!(g, :d, Vector{Float64})
    s = value!(g, :s, Float64)
    r = value!(g, :r, Float64)
    add!(g; inputs = (d,), outputs = (s,), op = sum)
    add!(g; inputs = (xv, s), outputs = (r,), op = *)
    gk1 = prepare(g; have = (xv, d), want = (r,), bound = (d => [1.0, 2.0],))
    gk2 = prepare(g; have = (xv, d), want = (r,), bound = (d => [5.0, 5.0],))
    @test gk1(2.0) == 6.0 && gk2(2.0) == 20.0
    @test gk2.f === gk1.f
    old = g.preparations
    r2 = value!(g, :r2, Float64)
    add!(g; inputs = (r,), outputs = (r2,), op = -)
    @test prepare(g; have = (xv, d), want = (r2,), bound = (d => [1.0, 1.0],))(3.0) == -6.0
    @test g.preparations !== old && g.preparations.version == g.version
    @test length(g.preparations.cache) == 1

    # A pass that is not a singleton (a closure) is not retained.
    h = Graph()
    hx = value!(h, :x, Float64)
    hd = value!(h, :d, Vector{Float64})
    hs = value!(h, :s, Float64)
    hr = value!(h, :r, Float64)
    add!(h; inputs = (hd,), outputs = (hs,), op = sum)
    add!(h; inputs = (hx, hs), outputs = (hr,), op = *)
    offset = Ref(0)
    closure_pass = ast -> (offset[] += 1; ast)
    @test prepare(h; have = (hx, hd), want = (hr,), passes = (closure_pass,),
                  bound = (hd => [1.0, 2.0],))(2.0) == 6.0
    @test offset[] == 1
    @test h.preparations === nothing
    # An empty binding is the ordinary unbound preparation.
    @test prepare(h; have = (hx, hs), want = (hr,), bound = ())(2.0, 3.0) == 6.0
    @test h.preparations === nothing

    # Rebinding through the graph keeps no binding's data either.
    function plain_first_binding_hoisted()
        kernel = prepare(FirstRebindingFixture.fresh; have = (:data, :q),
                         want = :scaled, bound = (; data = [6.0, 7.0]))
        constant = only(op for op in kernel.ops if op isa ReactiveKernels._BoundConstant)
        WeakRef(constant.value)
    end
    hoisted = plain_first_binding_hoisted()
    @test prepare(FirstRebindingFixture.fresh; have = (:data, :q), want = :scaled,
                  bound = (; data = [1.0, 2.0]))([0.0, 0.0]) == [2.0, 8.0]
    GC.gc(); GC.gc()
    @test hoisted.value === nothing
end

@testset "cached bound preparation specializes a value-dependent residual per binding" begin
    F = CachedBoundFixture
    cache = PreparationCache()
    x = [1.0, 2.0, 3.0, 4.0]
    for mask in ([1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0])
        cached = prepare!(cache, F.masked; bound = (; mask))
        expected = [m > 0 ? 2xi : -xi for (xi, m) in zip(x, mask)]
        @test cached(x) == expected
        @test cached(x) == prepare(F.masked; bound = (; mask))(x)
    end
    @test length(cache) == 1
end
