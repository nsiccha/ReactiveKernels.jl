using ReactiveKernels
using Test
using Random
using LinearAlgebra

_ta_grad!(g, q) = (copyto!(g, q); sum(abs2, q) / 2)
_ta_pos(D) = [sin(1.0i) for i in 1:D]

# GAP-1a — reactive dual-averaging authored through @reactive specialize=true.
# DualAveragingState stays a public NOMINAL wrapper type over the reactive object.

# Local ordinary Nesterov recurrence — parity ORACLE for this test only.
mutable struct _OldDA{T}
    target::T; reg::T; rel::T; off::T; it::T; er::T; cn::T; lc::T; lf::T; cur::T; fin::T
end
function _old_da(x::T; target = 0.8, reg = 0.05, rel = 0.75, off = 10) where {T}
    c = log(T(10)) + log(x)
    _OldDA{T}(T(target), T(reg), T(rel), T(off), one(T), zero(T), c, c, zero(T),
              exp(c), exp(zero(T)))
end
function _old_fit!(s::_OldDA, a)
    s.it += 1
    s.er += (s.target - a - s.er) / (s.it + s.off)
    s.lc = s.cn - sqrt(s.it) / s.reg * s.er
    w = s.it^(-s.rel)
    s.lf += w * (s.lc - s.lf)
    s.cur = exp(s.lc); s.fin = exp(s.lf); s
end

@testset "reactive dual-averaging — nominal type, parity, F32/F64, inference" begin
    for initial in (0.1, 1.0, 2.5)
        oracle = _old_da(initial)
        state = dual_averaging_state(initial)
        @test state isa DualAveragingState                 # public nominal type
        @test reactive_program(state) isa ReactiveKernels.ReactiveProgram
        @test state.current ≈ oracle.cur
        for a in (0.9, 0.6, 0.85, 0.3, 0.95, 0.7, 0.8, 0.5, 0.99, 0.75, 0.1)
            _old_fit!(oracle, a)
            r = fit!(state, a)
            @test r === state                              # fit! returns the wrapper
            @test state.current == oracle.cur              # exact parity
            @test state.final == oracle.fin
            @test state.iteration == oracle.it
        end
    end

    # Generic over precision: Float32 stays Float32 (concrete).
    o32 = _old_da(0.1f0); s32 = dual_averaging_state(0.1f0)
    for a in (0.9f0, 0.4f0, 0.85f0)
        _old_fit!(o32, a); fit!(s32, a)
        @test s32.current === o32.cur && s32.final === o32.fin   # === (same Float32 bits)
    end
    @test s32.current isa Float32 && s32.final isa Float32

    # Concrete inference behind a function barrier.
    s = dual_averaging_state(0.5)
    @test (@inferred((x -> x.current)(s)))::Float64 == s.current
    @test (@inferred((x -> x.final)(s)))::Float64 == s.final
    @test :current in propertynames(s) && :iteration in propertynames(s)

    # Mutable-compat: manual public field assignment forwards to the reactive
    # source, invalidates the derived current/final, and returns the assigned value.
    m = dual_averaging_state(0.5)
    fit!(m, 0.9)                                    # make error != 0 so iteration matters
    before = m.current
    m.iteration = m.iteration + 5.0                 # mutate an accumulator source
    @test m.current != before                       # derived recomputed reactively
    @test (m.target = 0.7) == 0.7                    # Julia assignment return semantics
    @test :target in propertynames(m, true)

    # Public copy(::DualAveragingState) + bidirectional clone/source isolation.
    base = dual_averaging_state(0.5); fit!(base, 0.9)
    cl = copy(base)
    @test cl isa DualAveragingState
    @test getfield(cl, :object) !== getfield(base, :object)   # distinct wrapped objects
    @test cl.current == base.current
    # Snapshot base, advance clone, assert base unchanged (and vice-versa below).
    base_before = base.current
    fit!(cl, 0.1)
    @test base.current == base_before                # base untouched by the clone's fit!
    @test cl.current != base.current                 # clone advanced; base untouched
    c_after = cl.current
    fit!(base, 0.2); fit!(base, 0.3)                 # advancing base must not touch clone
    @test cl.current == c_after
end

@testset "reactive Welford — nominal wrapper, parity, matrix, F32/F64, ownership" begin
    w = welford_var(2)
    @test w isa WelfordVariance
    @test reactive_program(w) isa ReactiveKernels.ReactiveProgram
    step!(w, [1.0, 2.0]); step!(w, [2.0, 4.0]); step!(w, [3.0, 6.0])
    @test w.mean == [2.0, 4.0]
    @test w.var ≈ [2 / 3, 8 / 3]
    @test w.n == 3

    # matrix step! == column-wise folding.
    wm = welford_var(2); step!(wm, [1.0 2.0 3.0; 2.0 4.0 6.0])
    @test wm.mean == w.mean && wm.var ≈ w.var

    # Generic over precision.
    w32 = welford_var(2, Float32); step!(w32, Float32[1, 2]); step!(w32, Float32[3, 4])
    @test eltype(w32.var) === Float32 && eltype(w32.mean) === Float32
    @test w32.n isa Float32

    # Concrete inference of the exposed accumulator arrays.
    @test (@inferred((e -> e.var)(w)))::Vector{Float64} == w.var

    # Clone ownership: public copy(::WelfordVariance) detaches state (bidirectional).
    clone = copy(w)
    @test clone isa WelfordVariance
    @test getfield(clone, :object) !== getfield(w, :object)   # distinct wrapped objects
    @test clone.object.state !== w.object.state               # distinct reactive states
    step!(clone, [10.0, 10.0])
    @test w.var ≈ [2 / 3, 8 / 3]     # source estimate unaffected by clone
    @test w.n == 3
    # Snapshot the clone state, THEN advance the source; the clone must be unchanged.
    clone_n = clone.n
    clone_mean = copy(clone.mean)
    clone_var = copy(clone.var)
    step!(w, [0.0, 0.0]); step!(w, [5.0, 5.0])
    @test clone.n == clone_n
    @test clone.mean == clone_mean
    @test clone.var == clone_var

    # Warmed step! mutates mean/var in place — allocation-free.
    ww = welford_var(2); v = [1.0, 2.0]
    _wcyc!(e, x, n) = (for _ in 1:n; step!(e, x); end)
    _wcyc!(ww, v, 2)
    bytes = @allocated _wcyc!(ww, v, 1000)
    println("REACTIVE_WELFORD_STEP_ALLOC_BYTES\t", bytes)
    @test bytes == 0
end

@testset "reactive TrajectoryStats/SamplingStats — wrappers, program-share, copy isolation" begin
    t = trajectory_stats(2)
    @test t isa TrajectoryStats
    @test reactive_program(t) isa ReactiveKernels.ReactiveProgram

    # Drive it through one real compiled-sampler transition.
    grp = reactive_nuts_group(_ta_grad!, Matrix{Float64}(I, 2, 2), [0.1, -0.2], [0.3, -0.4])
    st = nuts_state(grp; rng = Xoshiro(42),
                    step_f = partial(leapfrog!; stepsize = 0.25),
                    stats_f = t, max_depth = 3)
    tr = sample!(st)
    @test size(t.positions) == (2, tr.n_steps + 1)     # view over the reactive storage
    @test size(t.gradients) == size(t.positions)
    @test length(t.dhams) == tr.n_steps + 1
    @test sort(t.idxs) == 0:tr.n_steps

    # copy: shared immutable program, detached state, view isolation.
    tc = copy(t)
    @test tc isa TrajectoryStats
    @test reactive_program(tc) === reactive_program(t)                  # shared program
    @test getfield(tc, :object).state !== getfield(t, :object).state    # detached state
    t_pos_snapshot = copy(Matrix(t.positions))
    reset!(tc, st.init)                                                 # mutate the clone
    @test Matrix(t.positions) == t_pos_snapshot                        # source unchanged

    # setproperty! forwarding + read-only view rejection.
    @test (t.count = t.count) == t.count                               # source assignment
    @test_throws ArgumentError (t.positions = zeros(2, 1))

    # SamplingStats: program-share + trajectory source NOT aliased on copy.
    run = sampling_stats(t)
    @test run isa SamplingStats
    @test reactive_program(run) isa ReactiveKernels.ReactiveProgram
    run(st)
    @test size(run.draws, 2) == 1
    @test length(run.n_steps) == 1
    rc = copy(run)
    @test reactive_program(rc) === reactive_program(run)               # shared program
    @test getfield(rc, :object).trajectory !== getfield(run, :object).trajectory
    run(st)                                                            # advance source
    @test size(rc.draws, 2) == 1                                       # clone unaffected
end
