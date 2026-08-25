using ReactiveKernels
using LinearAlgebra
using DifferentiationInterface
import Enzyme
using Test

# Increment 1 of the flat compiled-reactive ca9 NUTS port. Proves the flattened
# phase-point group is a genuine compiled reactive object: the active-endpoint
# selection, the energy error `dham`, and the `diverged` flag are reactive nodes
# over ONE wide ReactiveProgram spanning init/fwd/bwd — not hand-refreshed. Also
# proves the DI+Enzyme owned value/gradient bundle is the sampled gradient path
# (no handwritten gradient), matched against an analytic oracle at the floor.

const _NUTS_ENZYME_BACKEND = AutoEnzyme(;
    mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
    function_annotation = Enzyme.Const,
)

# Standard normal potential: U(q) = 0.5||q||^2, grad U(q) = q.
_std_pot(q) = 0.5 * sum(abs2, q)
_std_pot_grad(q) = (0.5 * sum(abs2, q), copy(q))
_det_pos(D) = [sin(1.0i) for i in 1:D]
_det_mom(D) = [cos(0.7i) for i in 1:D]

@testset "flat compiled-reactive NUTS group — reactive dham selection" begin
    D = 6
    metric = Matrix{Float64}(I, D, D)
    q0 = _det_pos(D)
    m0 = _det_mom(D)
    group = ReactiveKernels.reactive_nuts_group(
        _std_pot_grad, metric, q0, m0; gofwd = true)

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
    group = ReactiveKernels.reactive_nuts_group(_std_pot_grad, metric, q0, q0)
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

@testset "flat NUTS group — copy / copyto! isolation" begin
    D = 5
    metric = Matrix{Float64}(I, D, D)
    q0 = _det_pos(D)
    m0 = _det_mom(D)
    source = ReactiveKernels.reactive_nuts_group(_std_pot_grad, metric, q0, m0)
    source.fwd_mom = 2.0 .* m0
    source_dham = source.dham

    # copy: an independent compiled state sharing the (immutable) handles.
    clone = copy(source)
    @test clone.handles === source.handles
    @test clone.dham ≈ source_dham
    # Mutating the clone must not disturb the source's reactive state.
    clone.fwd_mom = 10.0 .* m0
    @test clone.dham ≈ clone.init_ham - clone.fwd_ham
    @test !(clone.dham ≈ source_dham)
    @test source.dham ≈ source_dham          # source unchanged
    @test source.fwd_mom ≈ 2.0 .* m0

    # copyto!: restore a destination's HAVE sources from another state of the
    # SAME program (the proposal-swap pattern). copyto! requires identical
    # handles, so the destination is a copy of the source's compiled state.
    destination = copy(source)
    destination.bwd_mom = 3.0 .* m0
    destination.fwd_mom = 9.0 .* m0
    @test !(destination.dham ≈ source_dham)
    copyto!(destination, source)
    @test destination.dham ≈ source_dham
    @test destination.fwd_mom ≈ source.fwd_mom
    @test destination.bwd_mom ≈ source.bwd_mom   # source's bwd (untouched m0)
    # Independence after copyto!: further source mutation does not leak.
    source.fwd_mom = 7.0 .* m0
    @test !(source.dham ≈ destination.dham)
end

@testset "flat NUTS group — engine invalidation allocation" begin
    D = 6
    metric = Matrix{Float64}(I, D, D)
    group = ReactiveKernels.reactive_nuts_group(
        _std_pot_grad, metric, _det_pos(D), _det_mom(D))

    # Isolate the reactive engine cost on the scalar selection chain: toggling the
    # gofwd source and reading active_ham drives only Bool/Float64 recipes (no
    # array numeric work), so the invalidation + recompute is allocation-free.
    select_alloc(g) = (ReactiveKernels.set!(g.state, g.handles.gofwd, true);
                       g.active_ham;
                       @allocated (ReactiveKernels.set!(g.state, g.handles.gofwd, false);
                                   ReactiveKernels.get!(g.state, g.handles.active_ham)))
    @test select_alloc(group) == 0
end

@testset "flat NUTS group — DI+Enzyme gradient is the sampled path (non-aliased)" begin
    D = 6
    metric = Matrix{Float64}(I, D, D)
    q0 = _det_pos(D)
    m0 = _det_mom(D)

    # Scalar-potential boundary differentiated by Enzyme through DI — never a
    # handwritten gradient. Prepared once; out-of-place value_and_gradient returns
    # a FRESH gradient per call, so each endpoint's dpot slot owns a distinct
    # array. (The per-slot in-place owned bundle for the 0-B floor is increment 2:
    # it must reuse each CompiledReactiveState SLOT's buffer via the nonallocating
    # cache hook — a caller-shared or program-level buffer would alias across both
    # endpoints AND copied states, breaking proposal swaps.)
    preparation = prepare_gradient(_std_pot, _NUTS_ENZYME_BACKEND, copy(q0))
    di_potential_gradient(q) =
        value_and_gradient(_std_pot, preparation, _NUTS_ENZYME_BACKEND, q)

    di_value, di_gradient = di_potential_gradient(q0)
    @test di_value ≈ _std_pot(q0)
    @test di_gradient ≈ q0                       # matches analytic oracle grad U = q
    @test di_gradient !== q0                      # fresh array, not aliasing input

    # Distinct-position non-alias regression: give init/fwd/bwd DIFFERENT positions
    # and confirm their gradient slots are distinct array objects with distinct
    # values. This is the property a shared owned buffer would violate.
    di_group = ReactiveKernels.reactive_nuts_group(di_potential_gradient, metric, q0, m0)
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

    # Agreement with the analytic group everywhere.
    analytic_group = ReactiveKernels.reactive_nuts_group(_std_pot_grad, metric, q0, m0)
    di_group2 = ReactiveKernels.reactive_nuts_group(di_potential_gradient, metric, q0, m0)
    di_group2.fwd_mom = 2.0 .* m0
    analytic_group.fwd_mom = 2.0 .* m0
    @test di_group2.dham ≈ analytic_group.dham
    @test di_group2.fwd_ham ≈ analytic_group.fwd_ham
    @test di_group2.fwd_dpot_dpos ≈ analytic_group.fwd_dpot_dpos
end
