using ReactiveKernels
using LinearAlgebra
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
