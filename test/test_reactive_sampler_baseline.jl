using ReactiveKernels
using LinearAlgebra
using Test

# Raw in-place Euclidean phase-point oracle — the allocation FLOOR the compiled
# reactive sampler must reach. It performs exactly the Hamiltonian work of the
# reactive phase point (gradient, Cholesky velocity solve, kinetic, hamiltonian)
# but with caller-owned preallocated buffers and in-place kernels, so a leapfrog
# step and every Hamiltonian read allocate ZERO bytes after warmup. This
# supersedes the earlier "~992 B numeric floor" framing: the numeric work is not
# an irreducible allocation floor — with owned in-place storage it is 0 B/step,
# which is the target for the compiled-reactive path (its current 1328 B/step is
# a hot-path failure to be removed via the MutatingFunctions in-place getters).

mutable struct RawEuclideanPoint{G,C}
    grad!::G                 # grad!(dpot, q) -> pot; writes gradient into dpot
    chol::C                  # cached Cholesky of the (fixed) metric
    pos::Vector{Float64}
    mom::Vector{Float64}
    dpot::Vector{Float64}    # owned gradient buffer
    vel::Vector{Float64}     # owned velocity buffer (M^-1 mom)
    pot::Float64
end

function RawEuclideanPoint(grad!, metric, pos, mom)
    chol = cholesky(metric)
    dpot = similar(pos); vel = similar(mom)
    pot = grad!(dpot, pos)
    copyto!(vel, mom); ldiv!(chol, vel)
    RawEuclideanPoint(grad!, chol, copy(pos), copy(mom), dpot, vel, pot)
end

# In-place refreshers: recompute derived buffers from current pos/mom.
@inline function _refresh_pos!(p::RawEuclideanPoint)
    p.pot = p.grad!(p.dpot, p.pos)
    p
end
@inline function _refresh_mom!(p::RawEuclideanPoint)
    copyto!(p.vel, p.mom); ldiv!(p.chol, p.vel)
    p
end
@inline raw_kin(p::RawEuclideanPoint) = 0.5 * (logdet(p.chol) + dot(p.mom, p.vel))
@inline raw_ham(p::RawEuclideanPoint) = p.pot + raw_kin(p)

# One in-place Störmer–Verlet leapfrog: mom half-kick, pos drift, mom half-kick,
# refreshing exactly the buffers that changed — one gradient evaluation/step.
function raw_leapfrog!(p::RawEuclideanPoint, stepsize::Float64)
    @. p.mom -= 0.5 * stepsize * p.dpot
    _refresh_mom!(p)
    @. p.pos += stepsize * p.vel
    _refresh_pos!(p)
    @. p.mom -= 0.5 * stepsize * p.dpot
    _refresh_mom!(p)
    p
end

# Deterministic inputs (no RNG) so the oracle floor and the reactive-parity
# comparison are reproducible.
_det_pos(D) = [sin(1.0i) for i in 1:D]
_det_mom(D) = [cos(0.7i) for i in 1:D]
_det_dense_metric(D) = (A = [1.0 / (i + j) for i in 1:D, j in 1:D]; A'A + D * I)
# standard normal: U = 0.5||q||^2, grad = q
_grad!(dpot, q) = (copyto!(dpot, q); 0.5 * sum(abs2, q))
_pot(q) = 0.5 * sum(abs2, q)
_pot_grad(q) = (0.5 * sum(abs2, q), copy(q))

@testset "raw in-place Euclidean oracle — 0 B/step floor" begin
    D = 50
    metric = Matrix{Float64}(I, D, D)
    p = RawEuclideanPoint(_grad!, metric, _det_pos(D), _det_mom(D))

    # correctness: ham finite, one leapfrog is (near) energy-preserving at small h
    h0 = raw_ham(p)
    raw_leapfrog!(p, 0.01)
    @test isfinite(raw_ham(p))
    @test abs(raw_ham(p) - h0) < 1e-2

    # allocation floor — measured behind function barriers
    _lf_alloc(p, h)  = (raw_leapfrog!(p, h); @allocated raw_leapfrog!(p, h))
    _ham_alloc(p)    = (raw_ham(p); @allocated raw_ham(p))
    _refm_alloc(p)   = (_refresh_mom!(p); @allocated _refresh_mom!(p))
    _refp_alloc(p)   = (_refresh_pos!(p); @allocated _refresh_pos!(p))
    @test _ham_alloc(p)  == 0
    @test _refm_alloc(p) == 0     # Cholesky velocity solve, in place
    @test _refp_alloc(p) == 0     # gradient, in place
    @test _lf_alloc(p, 0.01) == 0 # whole leapfrog step: 0 B

    # dense (non-identity) metric: still 0 B/step
    pd = RawEuclideanPoint(_grad!, _det_dense_metric(D), _det_pos(D), _det_mom(D))
    @test _lf_alloc(pd, 0.01) == 0
    @test _ham_alloc(pd) == 0
end

@testset "raw oracle == reactive Euclidean phase point (same work)" begin
    # Prove the oracle performs the SAME Hamiltonian work as the compiled
    # reactive phase point: identical deterministic inputs, one leapfrog each,
    # then compare pos, mom, gradient (dham_dpos), velocity (dham_dmom), and ham.
    for (label, metric) in (("identity", Matrix{Float64}(I, 8, 8)),
                            ("dense", _det_dense_metric(8)))
        D = size(metric, 1)
        q0 = _det_pos(D); m0 = _det_mom(D); h = 0.05
        oracle = RawEuclideanPoint(_grad!, metric, copy(q0), copy(m0))
        point = euclidean_phasepoint(_pot, _pot_grad, metric, copy(q0), copy(m0))
        # pre-leapfrog agreement
        @test raw_ham(oracle) ≈ point.ham
        @test oracle.dpot ≈ point.dham_dpos
        @test oracle.vel ≈ point.dham_dmom
        # one leapfrog on each and compare every field
        raw_leapfrog!(oracle, h)
        leapfrog!(point; stepsize = h)
        @test oracle.pos ≈ point.pos
        @test oracle.mom ≈ point.mom
        @test oracle.dpot ≈ point.dham_dpos    # gradient at new pos
        @test oracle.vel ≈ point.dham_dmom     # velocity from new mom
        @test raw_ham(oracle) ≈ point.ham
    end
end
