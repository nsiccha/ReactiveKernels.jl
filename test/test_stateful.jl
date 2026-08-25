using ReactiveKernels
using Test

function _compiled_state_allocations(state, source, wanted)
    get!(state, wanted)
    touch!(state, source)
    get!(state, wanted)
    touch_allocations = @allocated touch!(state, source)
    recompute_allocations = @allocated get!(state, wanted)
    cached_allocations = @allocated get!(state, wanted)
    (; touch_allocations, recompute_allocations, cached_allocations)
end

@testset "compiled reactive KernelSpec boundary" begin
    calls = Ref(0)
    spec = @kernel begin
        source::Vector{Float64}
        total::Float64 = begin
            calls[] += 1
            sum(source)
        end
        return (total,)
    end

    program = prepare_reactive(spec)
    state = program([1.0, 2.0])
    source = statevalue(program, spec.source)
    total = statevalue(program, spec.total)
    @test get!(state, total) == 3.0
    @test calls[] == 1
    mutate!(state, source) do value
        value[1] = 4.0
    end
    @test get!(state, total) == 6.0
    @test calls[] == 2

    by_symbol = prepare_reactive(spec; have = :source, want = :total)
    @test get!(by_symbol([3.0]), statevalue(by_symbol, spec.total)) == 3.0
    @test_throws KeyError prepare_reactive(spec; have = :missing)
end

@testset "compiled reactive state" begin
    graph = Graph()
    position = value!(graph, :position, Vector{Float64})
    momentum = value!(graph, :momentum, Vector{Float64})
    potential = value!(graph, :potential, Float64)
    gradient = value!(graph, :gradient, Vector{Float64})
    velocity = value!(graph, :velocity, Vector{Float64})
    hamiltonian = value!(graph, :hamiltonian, Float64)

    calls = (geometry = Ref(0), momentum = Ref(0), hamiltonian = Ref(0))
    add!(
        graph,
        position => (potential, gradient),
        q -> begin
            calls.geometry[] += 1
            (sum(abs2, q) / 2, copy(q))
        end,
    )
    add!(graph, momentum => velocity, p -> begin
        calls.momentum[] += 1
        copy(p)
    end)
    add!(graph, (potential, momentum) => hamiltonian, (u, p) -> begin
        calls.hamiltonian[] += 1
        u + sum(abs2, p) / 2
    end)

    program = prepare_reactive(
        graph;
        have = (position, momentum),
        want = (potential, gradient, velocity, hamiltonian),
    )
    state = program([1.0, 2.0], [0.5, -0.25])
    q = statevalue(program, position)
    p = statevalue(program, momentum)
    u = statevalue(program, potential)
    grad = statevalue(program, gradient)
    vel = statevalue(program, velocity)
    ham = statevalue(program, hamiltonian)

    @test inputs(program) == (position, momentum)
    @test outputs(program) == (potential, gradient, velocity, hamiltonian)
    @test map(ref -> ref[], calls) == (geometry = 0, momentum = 0, hamiltonian = 0)
    @test get!(state, u) == 2.5
    @test get!(state, grad) == [1.0, 2.0]
    @test get!(state, vel) == [0.5, -0.25]
    @test get!(state, ham) == 2.65625
    @test @inferred(get!(state, ham)) == 2.65625
    initial_calls = map(ref -> ref[], calls)

    # Momentum mutation retains position geometry but invalidates momentum and
    # Hamiltonian descendants.
    mutate!(state, p) do momentum_value
        momentum_value .*= 2
    end
    @test get!(state, grad) == [1.0, 2.0]
    @test calls.geometry[] == initial_calls.geometry
    @test get!(state, vel) == [1.0, -0.5]
    @test get!(state, ham) == 3.125
    @test calls.momentum[] == initial_calls.momentum + 1
    @test calls.hamiltonian[] == initial_calls.hamiltonian + 1

    # Position mutation invalidates its geometry and the Hamiltonian while the
    # momentum-only velocity stays valid.
    set!(state, q, [2.0, 0.0])
    @test get!(state, vel) == [1.0, -0.5]
    @test calls.momentum[] == initial_calls.momentum + 1
    @test get!(state, ham) == 2.625
    @test get!(state, grad) == [2.0, 0.0]
    @test calls.geometry[] == initial_calls.geometry + 1

    @test occursin("__valid__", string(code_expr(program, ham)))
    receipt = _compiled_state_allocations(state, p, ham)
    @test receipt.touch_allocations == 0
    @test receipt.cached_allocations == 0
    # Scalar recipe orchestration adds no allocation; the selected operations
    # in this path allocate nothing either.
    @test receipt.recompute_allocations == 0

    copied = copy(state)
    mutate!(copied, q) do position_value
        position_value .+= 1
    end
    @test get!(copied, q) != get!(state, q)
    copyto!(copied, state)
    @test get!(copied, q) == get!(state, q)
    @test get!(copied, ham) == get!(state, ham)

    @test_throws ArgumentError touch!(state, grad)
    @test_throws ArgumentError set!(state, grad, zeros(2))
    derived_callback_called = Ref(false)
    @test_throws ArgumentError mutate!(state, grad) do gradient_value
        derived_callback_called[] = true
        gradient_value .= 0
    end
    @test !derived_callback_called[]
    @test get!(state, grad) == [2.0, 0.0]
    @test_throws ArgumentError freeze!(state, q)
    @test all(!, state.frozen)
    @test_throws ArgumentError checkpoint(state, (q,))
end

@testset "compiled reactive fresh and partial state copies" begin
    graph = Graph()
    source = value!(graph, :source, Vector{Float64})
    doubled = value!(graph, :doubled, Vector{Float64})
    total = value!(graph, :total, Float64)
    add!(graph, source => doubled, x -> 2 .* x)
    add!(graph, source => total, sum)
    program = prepare_reactive(
        graph; have = (source,), want = (doubled, total),
    )
    source_state = statevalue(program, source)
    doubled_state = statevalue(program, doubled)
    total_state = statevalue(program, total)

    fresh = program([1.0, 2.0])
    fresh_copy = copy(fresh)
    @test get!(fresh_copy, doubled_state) == [2.0, 4.0]
    mutate!(fresh_copy, source_state) do x
        x[1] = 10.0
    end
    @test get!(fresh_copy, source_state) == [10.0, 2.0]
    @test get!(fresh, source_state) == [1.0, 2.0]

    partial = program([3.0, 4.0])
    @test get!(partial, doubled_state) == [6.0, 8.0]
    partial_copy = copy(partial)
    @test get!(partial_copy, doubled_state) == [6.0, 8.0]
    @test get!(partial_copy, total_state) == 7.0

    destination = program([0.0])
    copyto!(destination, partial)
    @test get!(destination, doubled_state) == [6.0, 8.0]
    @test get!(destination, total_state) == 7.0
    mutate!(destination, source_state) do x
        resize!(x, 3)
        x .= (1.0, 2.0, 3.0)
    end
    get!(destination, doubled_state)
    copyto!(destination, partial)
    @test get!(destination, doubled_state) == [6.0, 8.0]

    # Handles carry an identity token in addition to their literal slot, so a
    # same-graph program and a forged out-of-range handle fail uniformly before
    # any tuple or validity-mask indexing.
    other_program = prepare_reactive(
        graph; have = (source,), want = (doubled, total),
    )
    foreign_same_graph = statevalue(other_program, total)
    forged = ReactiveValue{99,Float64}(program.token, graph, total)
    @test_throws ArgumentError get!(partial, foreign_same_graph)
    @test_throws ArgumentError checkpoint(partial, (foreign_same_graph,))
    @test_throws ArgumentError code_expr(program, foreign_same_graph)
    @test_throws ArgumentError checkpoint(partial, (forged,))
    @test_throws ArgumentError code_expr(program, forged)

    foreign_graph = Graph()
    foreign_source = value!(foreign_graph, :source, Vector{Float64})
    foreign_total = value!(foreign_graph, :total, Float64)
    add!(foreign_graph, foreign_source => foreign_total, sum)
    foreign_program = prepare_reactive(
        foreign_graph; have = (foreign_source,), want = (foreign_total,),
    )
    foreign_handle = statevalue(foreign_program, foreign_total)
    @test_throws ArgumentError checkpoint(partial, (foreign_handle,))
    @test_throws ArgumentError code_expr(program, foreign_handle)

    stale = program([1.0])
    stale_handle = statevalue(program, total)
    value!(graph, :late_graph_change, Float64)
    @test_throws ArgumentError get!(stale, stale_handle)
    @test_throws ArgumentError copy(stale)
    @test_throws ArgumentError code_expr(program, stale_handle)
end

@testset "compiled reactive state preserves planner/lowering contracts" begin
    graph = Graph()
    position = value!(graph, :position, Float64)
    supplied_potential = value!(graph, :supplied_potential, Float64)
    gradient = value!(graph, :gradient, Float64)
    hamiltonian = value!(graph, :hamiltonian, Float64)
    oracle_calls = Ref(0)
    add!(graph, position => (supplied_potential, gradient), q -> begin
        oracle_calls[] += 1
        (100q, 2q)
    end)
    add!(graph, (supplied_potential, gradient) => hamiltonian, +)

    program = prepare_reactive(
        graph;
        have = (position, supplied_potential),
        want = (gradient, hamiltonian),
    )
    state = program(3.0, 7.0)
    q = statevalue(program, position)
    potential = statevalue(program, supplied_potential)
    grad = statevalue(program, gradient)
    ham = statevalue(program, hamiltonian)

    # The selected multi-output oracle must run for its missing gradient while
    # its collateral potential output is discarded at the authoritative HAVE
    # boundary, exactly as in straight-line lowering.
    @test get!(state, ham) == 13.0
    @test get!(state, potential) == 7.0
    @test oracle_calls[] == 1
    set!(state, potential, 11.0)
    @test get!(state, ham) == 17.0
    @test oracle_calls[] == 1
    set!(state, q, 4.0)
    @test get!(state, grad) == 8.0
    @test get!(state, potential) == 11.0
    @test get!(state, ham) == 19.0
    @test oracle_calls[] == 2

    # A mutation which throws may have partially changed its object. The
    # transaction still invalidates descendants before propagating the error.
    array_graph = Graph()
    source = value!(array_graph, :source, Vector{Float64})
    total = value!(array_graph, :total, Float64)
    total_calls = Ref(0)
    add!(array_graph, source => total, x -> (total_calls[] += 1; sum(x)))
    array_program = prepare_reactive(
        array_graph; have = (source,), want = (total,),
    )
    array_state = array_program([1.0, 2.0])
    source_state = statevalue(array_program, source)
    total_state = statevalue(array_program, total)
    @test get!(array_state, total_state) == 3.0
    @test_throws ErrorException mutate!(array_state, source_state) do x
        x[1] = 10.0
        error("after mutation")
    end
    @test get!(array_state, total_state) == 12.0
    @test total_calls[] == 2

    # Effectful recipes with the same structural key remain distinct and are
    # never admitted into the prepared pure state program.
    effect_graph = Graph()
    x = value!(effect_graph, :x, Float64)
    pure = value!(effect_graph, :pure, Float64)
    effect = value!(effect_graph, :effect, Float64)
    effects = Ref(0)
    add!(effect_graph, x => effect,
         v -> (effects[] += 1; v + 10);
         cse_key = :same, effectful = true)
    add!(effect_graph, x => pure, v -> v + 1; cse_key = :same)
    @test canon_id(effect_graph, pure.id) != canon_id(effect_graph, effect.id)
    pure_program = prepare_reactive(
        effect_graph; have = (x,), want = (pure,),
    )
    @test get!(pure_program(1.0), statevalue(pure_program, pure)) == 2.0
    @test effects[] == 0
    @test_throws PlanningError prepare_reactive(
        effect_graph; have = (x,), want = (effect,),
    )
end

@testset "compiled reactive freeze and checkpoint cut points" begin
    graph = Graph()
    raw = value!(graph, :raw, Float64)
    location = value!(graph, :location, Float64)
    centered = value!(graph, :centered, Float64)
    calls = (location = Ref(0), centered = Ref(0))
    add!(graph, raw => location, x -> (calls.location[] += 1; x / 2))
    add!(graph, (raw, location) => centered,
         (x, loc) -> (calls.centered[] += 1; x - loc))

    program = prepare_reactive(
        graph; have = (raw,), want = (location, centered),
    )
    state = program(10.0)
    raw_state = statevalue(program, raw)
    location_state = statevalue(program, location)
    centered_state = statevalue(program, centered)
    freeze!(state, location_state)
    saved = checkpoint(state, (location_state,))

    set!(state, raw_state, 20.0)
    @test get!(state, location_state) == 5.0
    @test get!(state, centered_state) == 15.0
    @test calls.location[] == 1

    unfreeze!(state, location_state)
    @test get!(state, centered_state) == 10.0
    @test calls.location[] == 2

    replay = program(40.0; frozen = saved)
    @test get!(replay, location_state) == 5.0
    @test get!(replay, centered_state) == 35.0
end

@testset "copy_group! grouped HAVE-boundary copy" begin
    # Two-particle graph: sources a_pos, b_pos, r_pos, t_pos (Vector) and a
    # scalar source s_flag; derived a_cost, b_cost, r_cost, delta = a_cost-b_cost.
    calls = (a = Ref(0), b = Ref(0), r = Ref(0), delta = Ref(0))
    graph = Graph()
    a_pos = value!(graph, :a_pos, Vector{Float64})
    b_pos = value!(graph, :b_pos, Vector{Float64})
    r_pos = value!(graph, :r_pos, Vector{Float64})
    t_pos = value!(graph, :t_pos, Vector{Float64})
    a_cost = value!(graph, :a_cost, Float64)
    b_cost = value!(graph, :b_cost, Float64)
    r_cost = value!(graph, :r_cost, Float64)
    delta = value!(graph, :delta, Float64)
    add!(graph, a_pos => a_cost, x -> (calls.a[] += 1; sum(abs2, x)))
    add!(graph, b_pos => b_cost, x -> (calls.b[] += 1; sum(abs2, x)))
    add!(graph, r_pos => r_cost, x -> (calls.r[] += 1; sum(abs2, x)))
    add!(graph, (a_cost, b_cost) => delta,
         (a, b) -> (calls.delta[] += 1; a - b))

    program = prepare_reactive(graph;
        have = (a_pos, b_pos, r_pos, t_pos),
        want = (a_cost, b_cost, r_cost, delta))
    A = [1.0, 0.0]; B = [0.0, 2.0]; R = [3.0, 4.0]
    state = program(copy(A), copy(B), copy(R), zeros(2))
    ha = statevalue(program, a_pos); hb = statevalue(program, b_pos)
    hr = statevalue(program, r_pos); ht = statevalue(program, t_pos)
    hac = statevalue(program, a_cost); hbc = statevalue(program, b_cost)
    hd = statevalue(program, delta)

    @test get!(state, hd) == 1.0 - 4.0
    @test (calls.a[], calls.b[], calls.delta[]) == (1, 1, 1)

    # multi-subscriber / two-particle: touching b invalidates only {b_cost, delta}
    set!(state, hb, [1.0, 1.0])
    @test get!(state, hd) == 1.0 - 2.0
    @test calls.a[] == 1              # a_cost NOT recomputed
    @test (calls.b[], calls.delta[]) == (2, 2)

    # copy_group! array group a_pos <- r_pos: in-place, invalidates a_cost + delta
    copy_group!(state, (ha,), (hr,))
    @test get!(state, hac) == sum(abs2, R)
    @test get!(state, hd) == sum(abs2, R) - 2.0
    @test calls.a[] == 2

    # 0-alloc for a homogeneous array group, measured behind a function barrier
    # so global-scope boxing does not contaminate the allocation count.
    _cg_alloc(st, dh, sh) = (copy_group!(st, dh, sh); @allocated copy_group!(st, dh, sh))
    @test _cg_alloc(state, (ha,), (hr,)) == 0

    # swap via temp: a <-> b (multi-slot group in one call also exercised)
    a_before = get!(state, hac); b_before = get!(state, hbc)
    copy_group!(state, (ht,), (ha,))
    copy_group!(state, (ha,), (hb,))
    copy_group!(state, (hb,), (ht,))
    @test get!(state, hac) == b_before
    @test get!(state, hbc) == a_before

    # copy / detached copy: mutating the original must not touch the copy
    snapshot = copy(state)
    hd_val = get!(snapshot, hd)
    set!(state, ha, [9.0, 9.0])
    @test get!(state, hac) == sum(abs2, [9.0, 9.0])
    @test get!(snapshot, hd) == hd_val          # copy independent
    @test snapshot.program === state.program     # program shared (no cloned deps)

    # destinations must be HAVE sources
    @test_throws ArgumentError copy_group!(state, (hac,), (hr,))
    @test_throws DimensionMismatch copy_group!(state, (ha, hb), (hr,))
end
