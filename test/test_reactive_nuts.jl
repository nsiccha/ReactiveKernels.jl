using ReactiveKernels
using LinearAlgebra
using Random
using DifferentiationInterface
import Enzyme
using Test

# Increments 1-2 of the flat compiled-reactive ca9 NUTS port. Proves the flattened
# phase-point group is a genuine compiled reactive object (active-endpoint
# selection, energy error `dham`, and `diverged` are reactive nodes over ONE wide
# ReactiveProgram spanning init/fwd/bwd — not hand-refreshed), that each endpoint's
# potential+gradient and kinetic work is an owned single-output bundle reused in
# place per CompiledReactiveState slot (near-zero reactive overhead, sound under
# copy/copyto!), and that the DI+Enzyme scalar-potential boundary is the sampled
# gradient path with no shared caller buffer.

const _NUTS_ENZYME_BACKEND = AutoEnzyme(;
    mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
    function_annotation = Enzyme.Const,
)

# Standard normal potential: U(q) = 0.5||q||^2, grad U(q) = q. The sampler boundary
# is in-place: fill the (slot-owned) gradient buffer, return the scalar potential.
_std_pot(q) = 0.5 * sum(abs2, q)
_std_pot_grad!(gradient, q) = (copyto!(gradient, q); 0.5 * sum(abs2, q))
_det_pos(D) = [sin(1.0i) for i in 1:D]
_det_mom(D) = [cos(0.7i) for i in 1:D]

@testset "flat compiled-reactive NUTS group — reactive dham selection" begin
    D = 6
    metric = Matrix{Float64}(I, D, D)
    q0 = _det_pos(D)
    m0 = _det_mom(D)
    group = ReactiveKernels.reactive_nuts_group(
        _std_pot_grad!, metric, q0, m0; gofwd = true)

    # All three endpoints start equal (init == fwd == bwd), so every hamiltonian
    # agrees and dham == 0 (gofwd=true selects fwd == init).
    @test group.init_ham ≈ group.fwd_ham ≈ group.bwd_ham
    @test group.active_ham ≈ group.fwd_ham
    @test group.dham ≈ 0.0 atol = 1e-12
    @test group.diverged == false

    # Reactive selection: mutate the ACTIVE endpoint (fwd) momentum. `dham` is a
    # compiled reactive node, so it invalidates and recomputes with no manual
    # refresh: dham == init_ham - fwd_ham.
    group.fwd_mom = 2.0 .* m0
    dham_fwd = group.dham
    @test dham_fwd ≈ group.init_ham - group.fwd_ham
    @test !(dham_fwd ≈ 0.0)

    # Flip gofwd: active endpoint reselects to bwd with no graph rebuild.
    group.gofwd = false
    dham_bwd = group.dham
    @test group.active_ham ≈ group.bwd_ham
    @test dham_bwd ≈ group.init_ham - group.bwd_ham
    @test dham_bwd ≈ 0.0 atol = 1e-12   # bwd still untouched, equals init

    # Mutating the now-INACTIVE endpoint (fwd) invalidates dham (spurious) but it
    # recomputes to the same value — still selecting bwd. Value-correct.
    group.fwd_mom = 5.0 .* m0
    @test group.dham ≈ dham_bwd

    # Reactive divergence: drive the active endpoint's hamiltonian huge so
    # dham < min_dham; `diverged` flips reactively.
    group.gofwd = true
    group.fwd_mom = 1.0e3 .* m0
    @test group.dham < -1000.0
    @test group.diverged == true
    group.fwd_mom = m0
    @test group.diverged == false
end

@testset "flat NUTS group — parity with per-endpoint Hamiltonian work" begin
    D = 8
    metric = (A = [1.0 / (i + j) for i in 1:D, j in 1:D]; A'A + D * I)  # dense SPD
    q0 = _det_pos(D)
    m_fwd = _det_mom(D)
    m_bwd = [cos(0.3i) for i in 1:D]
    group = ReactiveKernels.reactive_nuts_group(_std_pot_grad!, metric, q0, q0)
    group.fwd_mom = m_fwd
    group.bwd_mom = m_bwd

    chol = cholesky(metric)
    manual_ham(q, m) = begin
        pot = _std_pot(q)
        v = chol \ m
        pot + 0.5 * (logdet(chol) + dot(m, v))
    end
    @test group.fwd_ham ≈ manual_ham(q0, m_fwd)
    @test group.bwd_ham ≈ manual_ham(q0, m_bwd)
    @test group.fwd_dham_dmom ≈ chol \ m_fwd
    @test group.fwd_dpot_dpos ≈ q0
end

@testset "flat NUTS group — owned bundles reused in place, per-slot" begin
    D = 6
    metric = Matrix{Float64}(I, D, D)
    q0 = _det_pos(D)
    m0 = _det_mom(D)
    group = ReactiveKernels.reactive_nuts_group(_std_pot_grad!, metric, q0, m0)

    # Bundle slots are reused in place: the exposed gradient/velocity arrays keep
    # a STABLE identity across an invalidate→recompute, and their values update.
    grad_buffer = group.fwd_dpot_dpos
    vel_buffer = group.fwd_dham_dmom
    group.fwd_pos = 2.0 .* q0
    group.fwd_mom = 3.0 .* m0
    @test group.fwd_dpot_dpos === grad_buffer     # same buffer, filled in place
    @test group.fwd_dham_dmom === vel_buffer
    @test grad_buffer ≈ 2.0 .* q0                 # grad U(q) = q
    @test vel_buffer ≈ metric \ (3.0 .* m0)

    # Endpoint slots are distinct buffers (no cross-endpoint aliasing).
    @test group.init_dpot_dpos !== group.fwd_dpot_dpos !== group.bwd_dpot_dpos
    @test group.init_dham_dmom !== group.fwd_dham_dmom !== group.bwd_dham_dmom
end

@testset "flat NUTS group — copy / copyto! isolation (buffers not aliased)" begin
    D = 5
    metric = Matrix{Float64}(I, D, D)
    q0 = _det_pos(D)
    m0 = _det_mom(D)
    source = ReactiveKernels.reactive_nuts_group(_std_pot_grad!, metric, q0, m0)
    source.fwd_mom = 2.0 .* m0
    source_dham = source.dham
    source_grad = source.fwd_dpot_dpos

    # copy: an independent compiled state sharing the (immutable) handles, with
    # DEEP-COPIED owned bundle buffers (per-slot ownership).
    clone = copy(source)
    @test clone.handles === source.handles
    @test clone.dham ≈ source_dham
    @test clone.fwd_dpot_dpos !== source_grad     # buffers deep-copied, not aliased
    # Mutating the clone must not disturb the source's reactive state or buffers.
    clone.fwd_pos = 4.0 .* q0
    clone.fwd_mom = 10.0 .* m0
    @test clone.dham ≈ clone.init_ham - clone.fwd_ham
    @test !(clone.dham ≈ source_dham)
    @test source.dham ≈ source_dham               # source unchanged
    @test source.fwd_dpot_dpos ≈ q0               # source gradient (fwd_pos == q0)

    # copyto!: restore a destination's HAVE sources from another state of the SAME
    # program (the proposal-swap pattern). Requires identical handles, so the
    # destination is a copy of the source's compiled state.
    destination = copy(source)
    destination.bwd_mom = 3.0 .* m0
    destination.fwd_mom = 9.0 .* m0
    @test !(destination.dham ≈ source_dham)
    copyto!(destination, source)
    @test destination.dham ≈ source_dham
    @test destination.fwd_mom ≈ source.fwd_mom
    @test destination.fwd_dpot_dpos !== source.fwd_dpot_dpos   # still per-instance
    source.fwd_mom = 7.0 .* m0
    @test !(source.dham ≈ destination.dham)       # independence after copyto!
end

@testset "flat NUTS group — hidden bundle copy isolation regression" begin
    # The exposed gradient/velocity arrays are pure projections of HIDDEN owned
    # bundle slots (_ValueGradient/_Kinetic). copy(state) must deep-copy those
    # hidden bundles too, or a clone's in-place recompute would corrupt the
    # source's hidden caches. Materialize both endpoints, copy, recompute the
    # clone, then invalidate/recompute the source, and prove bidirectional
    # isolation of both the exposed arrays and the hidden bundle buffers.
    D = 5
    metric = Matrix{Float64}(I, D, D)
    q0 = _det_pos(D)
    m0 = _det_mom(D)
    source = ReactiveKernels.reactive_nuts_group(_std_pot_grad!, metric, q0, m0)

    # Materialize hidden bundles + exposed projections on the source.
    src_grad = source.fwd_dpot_dpos
    src_vel = source.fwd_dham_dmom
    src_grad_snapshot = copy(src_grad)
    src_vel_snapshot = copy(src_vel)
    src_ham = source.fwd_ham

    clone = copy(source)

    # Recompute the CLONE with different pos/mom; its buffers must be distinct
    # objects and its values must diverge from the source.
    clone.fwd_pos = 3.0 .* q0
    clone.fwd_mom = 4.0 .* m0
    @test clone.fwd_dpot_dpos !== src_grad          # distinct exposed arrays
    @test clone.fwd_dham_dmom !== src_vel
    @test clone.fwd_dpot_dpos ≈ 3.0 .* q0
    @test clone.fwd_dham_dmom ≈ metric \ (4.0 .* m0)
    # Source's hidden caches are UNTOUCHED by the clone's recompute.
    @test source.fwd_dpot_dpos === src_grad         # same source buffer identity
    @test source.fwd_dham_dmom === src_vel
    @test src_grad == src_grad_snapshot             # values not overwritten
    @test src_vel == src_vel_snapshot
    @test source.fwd_ham ≈ src_ham

    # Now recompute the SOURCE; the clone must remain isolated in the other
    # direction.
    clone_grad_snapshot = copy(clone.fwd_dpot_dpos)
    clone_vel_snapshot = copy(clone.fwd_dham_dmom)
    source.fwd_pos = 9.0 .* q0
    @test source.fwd_dpot_dpos ≈ 9.0 .* q0
    @test clone.fwd_dpot_dpos == clone_grad_snapshot   # clone untouched
    @test clone.fwd_dham_dmom == clone_vel_snapshot
end

@testset "flat NUTS group — warmed invalidate→recompute is allocation-free" begin
    D = 6
    metric = Matrix{Float64}(I, D, D)
    q0 = _det_pos(D)
    m0 = _det_mom(D)
    group = ReactiveKernels.reactive_nuts_group(_std_pot_grad!, metric, q0, m0)
    state = group.state
    handles = group.handles

    # A leapfrog-shaped cycle on the active endpoint: half-kick mom (in place),
    # drift pos (in place), re-read gradient/velocity/ham. With owned in-place
    # bundles and the analytic 0-alloc boundary, the reactive overhead is 0 bytes.
    function leapfrog_cycle!(state, handles, stepsize, n)
        for _ in 1:n
            gradient = ReactiveKernels.get!(state, handles.fwd_dpot_dpos)
            ReactiveKernels.mutate!(state, handles.fwd_mom) do mom
                @. mom -= 0.5 * stepsize * gradient
                mom
            end
            velocity = ReactiveKernels.get!(state, handles.fwd_dham_dmom)
            ReactiveKernels.mutate!(state, handles.fwd_pos) do pos
                @. pos += stepsize * velocity
                pos
            end
            ReactiveKernels.get!(state, handles.fwd_ham)
        end
    end
    leapfrog_cycle!(state, handles, 0.01, 1)          # warm the steady path
    alloc = @allocated leapfrog_cycle!(state, handles, 0.01, 1000)
    println("REACTIVE_NUTS_LEAPFROG_CYCLE_ALLOC_BYTES\t", alloc)
    @test alloc == 0

    # Scalar-selection engine invalidation is likewise 0 bytes.
    select_alloc(g) = (ReactiveKernels.set!(g.state, g.handles.gofwd, true);
                       g.active_ham;
                       @allocated (ReactiveKernels.set!(g.state, g.handles.gofwd, false);
                                   ReactiveKernels.get!(g.state, g.handles.active_ham)))
    @test select_alloc(group) == 0
end

@testset "flat NUTS group — DI+Enzyme owned-buffer gradient is the sampled path" begin
    D = 6
    metric = Matrix{Float64}(I, D, D)
    q0 = _det_pos(D)
    m0 = _det_mom(D)

    # Scalar-potential boundary differentiated by Enzyme through DI — never a
    # handwritten gradient. Prepared once; value_and_gradient! fills the passed
    # (slot-owned) gradient buffer in place. There is NO shared caller buffer: each
    # CompiledReactiveState slot passes its own buffer to this closure.
    preparation = prepare_gradient(_std_pot, _NUTS_ENZYME_BACKEND, copy(q0))
    di_potential_gradient!(gradient, q) =
        first(value_and_gradient!(_std_pot, gradient, preparation,
                                  _NUTS_ENZYME_BACKEND, q))

    di_value = di_potential_gradient!(similar(q0), q0)
    @test di_value ≈ _std_pot(q0)

    # Distinct-position non-alias regression on the flat group: init/fwd/bwd
    # gradient slots are distinct array objects with distinct, correct values.
    di_group = ReactiveKernels.reactive_nuts_group(di_potential_gradient!, metric, q0, m0)
    di_group.init_pos = _det_pos(D)
    di_group.fwd_pos = [0.5 * sin(i) for i in 1:D]
    di_group.bwd_pos = [-0.3 * cos(i) for i in 1:D]
    gi = di_group.init_dpot_dpos
    gf = di_group.fwd_dpot_dpos
    gb = di_group.bwd_dpot_dpos
    @test gi !== gf && gf !== gb && gi !== gb     # three distinct array objects
    @test gi ≈ di_group.init_pos                  # grad U(q) = q, per endpoint
    @test gf ≈ di_group.fwd_pos
    @test gb ≈ di_group.bwd_pos
    @test !(gi ≈ gf) && !(gf ≈ gb)                # distinct values, not overwritten

    # After one endpoint invalidates/recomputes, the OTHER endpoints' gradients are
    # unchanged (the aliasing failure a shared buffer would cause).
    di_group.fwd_pos = 5.0 .* q0
    @test di_group.fwd_dpot_dpos ≈ 5.0 .* q0
    @test gi ≈ di_group.init_pos                  # init slot untouched
    @test gb ≈ di_group.bwd_pos                   # bwd slot untouched

    # Agreement with the analytic group everywhere.
    analytic_group = ReactiveKernels.reactive_nuts_group(_std_pot_grad!, metric, q0, m0)
    di_group2 = ReactiveKernels.reactive_nuts_group(di_potential_gradient!, metric, q0, m0)
    di_group2.fwd_mom = 2.0 .* m0
    analytic_group.fwd_mom = 2.0 .* m0
    @test di_group2.dham ≈ analytic_group.dham
    @test di_group2.fwd_ham ≈ analytic_group.fwd_ham
    @test di_group2.fwd_dpot_dpos ≈ analytic_group.fwd_dpot_dpos
end

@testset "CompiledNUTSState — transition-by-transition parity with the oracle" begin
    # The compiled-reactive transition must be BYTE-for-byte deterministically
    # identical to the ordinary-Julia NUTSState oracle under a shared RNG stream:
    # same draws in the same order, so every accepted sample and every diagnostic
    # field matches. Both are driven by the SAME analytic potential (the oracle via
    # its (value, gradient) boundary, the compiled group via the in-place boundary).
    for (label, D, metric) in (
            ("gaussian-identity", 4, Matrix{Float64}(I, 4, 4)),
            ("gaussian-dense", 6,
             (A = [1.0 / (i + j) for i in 1:6, j in 1:6]; A'A + 6 * I)),
        )
        q0 = _det_pos(D)

        oracle = ReactiveKernels._oracle_nuts_state(
            euclidean_phasepoint(_std_pot,
                                 q -> (_std_pot(q), copy(q)), metric, copy(q0),
                                 zeros(D));
            rng = Xoshiro(2024),
            step_f = partial(leapfrog!; stepsize = 0.25),
            max_depth = 5,
        )
        compiled = compiled_nuts_state(
            reactive_nuts_group(_std_pot_grad!, metric, copy(q0), zeros(D));
            rng = Xoshiro(2024),
            step_f = partial(leapfrog!; stepsize = 0.25),
            max_depth = 5,
        )

        # The public compiled state carries the flat compiled ReactiveProgram.
        @test reactive_program(compiled) isa ReactiveKernels.ReactiveProgram
        @test reactive_program(compiled) === compiled.group.state.program

        for transition in 1:40
            oracle_diag = sample!(oracle)
            compiled_diag = sample!(compiled)
            @test compiled.group.init_pos ≈ oracle.init.pos
            @test compiled_diag.depth == oracle_diag.depth
            @test compiled_diag.n_steps == oracle_diag.n_steps
            @test compiled_diag.diverged == oracle_diag.diverged
            @test compiled_diag.acceptance_rate ≈ oracle_diag.acceptance_rate
            @test compiled_diag.energy_error ≈ oracle_diag.energy_error
        end
    end
end

@testset "CompiledNUTSState — sample!(draws) parity" begin
    D = 4
    metric = Matrix{Float64}(I, D, D)
    q0 = _det_pos(D)
    oracle = ReactiveKernels._oracle_nuts_state(
        euclidean_phasepoint(_std_pot, q -> (_std_pot(q), copy(q)), metric,
                             copy(q0), zeros(D));
        rng = Xoshiro(77), step_f = partial(leapfrog!; stepsize = 0.3), max_depth = 6)
    compiled = compiled_nuts_state(
        reactive_nuts_group(_std_pot_grad!, metric, copy(q0), zeros(D));
        rng = Xoshiro(77), step_f = partial(leapfrog!; stepsize = 0.3), max_depth = 6)

    oracle_chain = sample!(oracle, 200)
    compiled_chain = sample!(compiled, 200)
    @test compiled_chain.samples ≈ oracle_chain.samples
    @test all(getfield.(compiled_chain.diagnostics, :depth) .==
              getfield.(oracle_chain.diagnostics, :depth))
    @test [d.energy_error for d in compiled_chain.diagnostics] ≈
          [d.energy_error for d in oracle_chain.diagnostics]
end

@testset "CompiledNUTSState — ca9 fixture bit-for-bit reproduction" begin
    # Direct fixed-momentum step! (no momentum refresh) on the compiled-reactive
    # group must reproduce the executable upstream ReactiveHMC ca9 fixture
    # (test/fixtures/reactivehmc_ca9_reference.jl @ ca9ea4ca) BIT-FOR-BIT: identical
    # init pos/mom/ham and every diagnostic field.
    compiled = compiled_nuts_state(
        reactive_nuts_group(_std_pot_grad!, Diagonal(ones(2)),
                            [0.1, -0.2], [0.3, -0.4]);
        rng = Xoshiro(42),
        step_f = partial(leapfrog!; stepsize = 0.25),
        max_depth = 3,
    )
    transition = step!(compiled)   # fixed momentum [0.3, -0.4], no refresh
    @test compiled.group.init_pos == [0.27957763671875, -0.4214599609375]
    @test compiled.group.init_mom == [0.15133209228515626, -0.15659484863281245]
    @test compiled.group.init_ham == 0.15160775120020845
    @test transition.depth == 3
    @test transition.n_steps == 7
    @test transition.energy_error == -0.0012290068185673575
    @test transition.acceptance_rate == 0.9985695900582436
    @test !transition.diverged
end

@testset "CompiledNUTSState — direct step! is bit-for-bit equal to the oracle" begin
    # Fixed-momentum direct step! (no refresh): the compiled state and the oracle
    # execute the same float ops in the same order, so every field is EXACTLY (==)
    # equal, including init momentum and Hamiltonian.
    for (D, metric) in ((4, Matrix{Float64}(I, 4, 4)),
                        (6, (A = [1.0 / (i + j) for i in 1:6, j in 1:6];
                             A'A + 6 * I)))
        q0 = _det_pos(D)
        m0 = _det_mom(D)
        oracle = ReactiveKernels._oracle_nuts_state(
            euclidean_phasepoint(_std_pot, q -> (_std_pot(q), copy(q)), metric,
                                 copy(q0), copy(m0));
            rng = Xoshiro(9), step_f = partial(leapfrog!; stepsize = 0.2),
            max_depth = 5)
        compiled = compiled_nuts_state(
            reactive_nuts_group(_std_pot_grad!, metric, copy(q0), copy(m0));
            rng = Xoshiro(9), step_f = partial(leapfrog!; stepsize = 0.2),
            max_depth = 5)
        for _ in 1:25
            od = step!(oracle)          # direct step!, fixed momentum
            cd = step!(compiled)
            @test compiled.group.init_pos == oracle.init.pos
            @test compiled.group.init_mom == oracle.init.mom
            @test compiled.group.init_ham == oracle.init.ham
            @test cd.depth == od.depth
            @test cd.n_steps == od.n_steps
            @test cd.diverged == od.diverged
            @test cd.energy_error == od.energy_error
            @test cd.acceptance_rate == od.acceptance_rate
            # Manually re-inject the same fixed momentum for the next direct step!
            # on both, keeping the fixed-momentum comparison deterministic.
            oracle.init.mom = copy(m0)
            compiled.group.init_mom = copy(m0)
        end
    end
end

@testset "CompiledNUTSState — constructor rejects a non-group phase point" begin
    plain = euclidean_phasepoint(_std_pot, q -> (_std_pot(q), copy(q)),
                                 Matrix{Float64}(I, 3, 3), zeros(3), zeros(3))
    @test_throws ArgumentError compiled_nuts_state(
        plain; rng = Xoshiro(1), step_f = partial(leapfrog!; stepsize = 0.1))
end

@testset "CompiledNUTSState — public nuts_state default + integrator/threshold" begin
    D = 4
    metric = Matrix{Float64}(I, D, D)
    q0 = _det_pos(D)
    group = reactive_nuts_group(_std_pot_grad!, metric, copy(q0), zeros(D))

    # Public nuts_state builds the COMPILED state for a group.
    state = nuts_state(group; rng = Xoshiro(1),
                       step_f = partial(leapfrog!; stepsize = 0.2))
    @test state isa CompiledNUTSState
    @test reactive_program(state) === group.state.program

    # A plain (non-group) phase point is rejected with actionable guidance.
    plain = euclidean_phasepoint(_std_pot, q -> (_std_pot(q), copy(q)),
                                 metric, zeros(D), zeros(D))
    @test_throws ArgumentError nuts_state(plain; rng = Xoshiro(1),
                                          step_f = partial(leapfrog!; stepsize = 0.2))

    # Non-leapfrog integrators are rejected.
    g2 = reactive_nuts_group(_std_pot_grad!, metric, copy(q0), zeros(D))
    @test_throws ArgumentError compiled_nuts_state(
        g2; rng = Xoshiro(1),
        step_f = partial(generalized_leapfrog!; stepsize = 0.2, n_fi_steps = 1))
    @test_throws ArgumentError compiled_nuts_state(
        g2; rng = Xoshiro(1), step_f = leapfrog!)

    # The min_dham threshold is a reactive HAVE source: compiled_nuts_state syncs it
    # and changing it invalidates/recomputes the group's diverged node.
    g3 = reactive_nuts_group(_std_pot_grad!, metric, copy(q0), zeros(D))
    st = compiled_nuts_state(g3; rng = Xoshiro(1),
                             step_f = partial(leapfrog!; stepsize = 0.2),
                             min_dham = -1000.0)
    @test g3.min_dham == -1000.0
    g3.fwd_mom = 50.0 .* _det_mom(D)          # large energy error, dham very negative
    @test g3.dham < -1000.0
    @test g3.diverged == true
    g3.min_dham = -1.0e9                        # loosen threshold reactively
    @test g3.diverged == false                  # diverged recomputed reactively
end

@testset "CompiledNUTSState — reactive group.diverged is CONSUMED (validity)" begin
    # Non-vacuous regression: the compiled transition must READ the reactive
    # group.diverged node, so its slot is VALID (computed) at every stats callback —
    # never hand-computed with the diverged node left stale/invalid.
    D = 4
    metric = Matrix{Float64}(I, D, D)
    group = reactive_nuts_group(_std_pot_grad!, metric, _det_pos(D), zeros(D))
    diverged_handle = group.handles.diverged
    dham_handle = group.handles.dham
    diverged_slot = ReactiveKernels._slot_index(diverged_handle)
    dham_slot = ReactiveKernels._slot_index(dham_handle)

    validities = Tuple{Bool,Bool}[]
    recorder = function (state)
        gstate = state.group.state
        push!(validities, (gstate.valid[dham_slot], gstate.valid[diverged_slot]))
    end
    state = compiled_nuts_state(group; rng = Xoshiro(3),
                                step_f = partial(leapfrog!; stepsize = 0.2),
                                stats_f = recorder, max_depth = 4)
    for _ in 1:5
        sample!(state)
    end
    @test !isempty(validities)
    @test all(v -> v == (true, true), validities)   # diverged CONSUMED, not stale
end

@testset "CompiledNUTSState — warmup! adaptation parity with the oracle" begin
    D = 4
    q0 = _det_pos(D)
    oracle = ReactiveKernels._oracle_nuts_state(
        euclidean_phasepoint(_std_pot, q -> (_std_pot(q), copy(q)),
                             Matrix{Float64}(I, D, D), copy(q0), zeros(D));
        rng = Xoshiro(123), step_f = partial(leapfrog!; stepsize = 0.5),
        max_depth = 6)
    compiled = nuts_state(
        reactive_nuts_group(_std_pot_grad!, Matrix{Float64}(I, D, D),
                            copy(q0), zeros(D));
        rng = Xoshiro(123), step_f = partial(leapfrog!; stepsize = 0.5),
        max_depth = 6)

    ow = warmup!(oracle, 200; target_accept = 0.8)
    cw = warmup!(compiled, 200; target_accept = 0.8)
    @test cw.initial_stepsize == ow.initial_stepsize
    @test cw.final_stepsize == ow.final_stepsize
    @test cw.metric == ow.metric
    @test cw.metric_window_ends == ow.metric_window_ends

    # Post-warmup sampling stays in exact lockstep.
    oc = sample!(oracle, 100)
    cc = sample!(compiled, 100)
    @test cc.samples == oc.samples
end

@testset "CompiledNUTSState — Float32 group divergence control" begin
    D = 3
    metric = Matrix{Float32}(I, D, D)
    q0 = Float32[sin(1.0f0 * i) for i in 1:D]
    m0 = Float32[cos(0.7f0 * i) for i in 1:D]
    f32_grad!(g, q) = (copyto!(g, q); 0.5f0 * sum(abs2, q))
    group = reactive_nuts_group(f32_grad!, metric, copy(q0), copy(m0))
    @test eltype(group.init_pos) === Float32
    @test group.min_dham isa Float32
    st = compiled_nuts_state(group; rng = Xoshiro(5),
                             step_f = partial(leapfrog!; stepsize = 0.2f0),
                             min_dham = -1000.0)
    @test group.min_dham == -1000.0f0
    group.fwd_mom = 100.0f0 .* m0
    @test group.diverged == true
end

@testset "CompiledNUTSState — @inferred public step! + init view inference" begin
    D = 4
    group = reactive_nuts_group(_std_pot_grad!, Matrix{Float64}(I, D, D),
                                _det_pos(D), zeros(D))
    state = nuts_state(group; rng = Xoshiro(2),
                       step_f = partial(leapfrog!; stepsize = 0.2), max_depth = 5)
    refresh_momentum!(state)
    @test (@inferred step!(state)) isa NUTSDiagnostics
    # Init endpoint view exposes phase-point fields with concrete inferred types.
    initview = state.init
    @test (@inferred((v -> v.pos)(initview)))::Vector{Float64} == group.init_pos
    @test (@inferred((v -> v.ham)(initview)))::Float64 == group.init_ham
    @test (@inferred((v -> v.metric)(initview))) == group.metric
end
