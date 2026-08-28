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

@testset "reactive stats — dynamic capacity growth + prepend/append order survives resize" begin
    # Drive >16 reserved columns on BOTH sides (initial capacity is 16) so the
    # backing storage resizes, and assert positions/gradients/dhams/pots/idxs keep
    # their ordered prepend/append semantics across the resize.
    t = trajectory_stats(1)
    # Seed one column; gradient storage records -dham_dpos, so set dham_dpos=pos for
    # a self-consistent gradients == -positions check, and reset! seeds dhams with 0.
    reset!(t, (pos = [100.0], dham_dpos = [100.0], pot = 100.0))
    exp_pos = [100.0]; exp_dh = [0.0]
    function _drive!(stats, prepend, val)
        obj = getfield(stats, :object); gs = obj.state; h = obj.handles
        col = ReactiveKernels._reserve_trajectory_column!(stats, prepend)
        ReactiveKernels.mutate!(gs, h.position_storage) do s; s[:, col] .= val; s; end
        ReactiveKernels.mutate!(gs, h.gradient_storage) do s; s[:, col] .= -val; s; end
        if prepend
            ReactiveKernels.mutate!(gs, h.dhams) do d; pushfirst!(d, val); d; end
            ReactiveKernels.mutate!(gs, h.pots) do p; pushfirst!(p, val); p; end
            ReactiveKernels.mutate!(gs, h.idxs) do i; pushfirst!(i, length(i)); i; end
        else
            ReactiveKernels.mutate!(gs, h.dhams) do d; push!(d, val); d; end
            ReactiveKernels.mutate!(gs, h.pots) do p; push!(p, val); p; end
            ReactiveKernels.mutate!(gs, h.idxs) do i; push!(i, length(i)); i; end
        end
    end
    for k in 1:24
        prepend = isodd(k)
        _drive!(t, prepend, Float64(k))
        prepend ? (pushfirst!(exp_pos, Float64(k)); pushfirst!(exp_dh, Float64(k))) :
                  (push!(exp_pos, Float64(k)); push!(exp_dh, Float64(k)))
    end
    @test size(getfield(t, :object).position_storage, 2) > 16    # actually resized
    @test length(exp_pos) == 25
    @test vec(t.positions) == exp_pos                            # order survived resize
    @test vec(t.gradients) == -exp_pos
    @test t.dhams == exp_dh
    @test t.pots == exp_pos
    @test sort(t.idxs) == 0:24
end

@testset "reactive stats — Float32 construction/callback typing + SamplingStats isolation" begin
    # Float32 stats over a Float32 compiled sampler.
    D = 2
    f32g!(g, q) = (copyto!(g, q); sum(abs2, q) / 2)
    t32 = trajectory_stats(D, Float32)
    @test t32 isa TrajectoryStats
    @test eltype(getfield(t32, :object).position_storage) === Float32
    grp = reactive_nuts_group(f32g!, Matrix{Float32}(I, D, D),
                              Float32[0.1, -0.2], Float32[0.3, -0.4])
    st = nuts_state(grp; rng = Xoshiro(7),
                    step_f = partial(leapfrog!; stepsize = 0.25f0),
                    stats_f = t32, max_depth = 3)
    sample!(st)
    @test eltype(t32.positions) === Float32
    @test eltype(t32.dhams) === Float32

    # SamplingStats bidirectional clone/source isolation + setproperty! return.
    run = sampling_stats(t32); run(st)
    @test (run.n_steps = run.n_steps) == run.n_steps        # setproperty! forwards+returns
    clone = copy(run)
    n0 = size(run.draws, 2)
    run(st)                                                 # advance SOURCE
    @test size(clone.draws, 2) == n0                        # clone unaffected (source->clone)
    c0 = size(clone.draws, 2)
    clone(st)                                               # advance CLONE
    @test size(run.draws, 2) == n0 + 1                      # source unaffected (clone->source)
    @test size(clone.draws, 2) == c0 + 1
end


# --- reset! reuse regressions (deterministic-AST warmup-window fix) ------------
@testset "reset! — matches fresh, reuses program, isolated, inferred, 0-B" begin
    for T in (Float64, Float32)
        # DualAveragingState: mutate, then reset -> equals a fresh object.
        da = dual_averaging_state(T(0.5); target = 0.8)
        prog0 = reactive_program(da)
        fit!(da, T(0.9)); fit!(da, T(0.7))
        reset!(da, T(0.33); target = 0.8)
        fresh = dual_averaging_state(T(0.33); target = 0.8)
        @test da.iteration == fresh.iteration
        @test da.error == fresh.error
        @test da.current == fresh.current
        @test da.final == fresh.final
        @test da.current isa T
        @test reactive_program(da) === prog0            # SAME program, no reconstruction
        @test (@inferred reset!(da, T(0.4); target = 0.8)) === da

        # WelfordVariance: mutate, then reset -> equals fresh; owned buffers retained.
        wv = welford_var(3, T)
        prog_w = reactive_program(wv)
        mean0 = wv.mean; var0 = wv.var
        step!(wv, T[1, 2, 3]); step!(wv, T[3, 2, 1])
        reset!(wv)
        wf = welford_var(3, T)
        @test wv.n == wf.n
        @test wv.mean == wf.mean
        @test wv.var == wf.var
        @test reactive_program(wv) === prog_w           # SAME program
        @test wv.mean === mean0 && wv.var === var0      # owned array identities retained
        @test (@inferred reset!(wv)) === wv
    end

    # Dual reset covers EVERY source field == a fresh construction (Float32/64).
    for T in (Float64, Float32)
        da = dual_averaging_state(T(0.5); target = T(0.82), regularization_scale = T(0.06),
                                  relaxation_exponent = T(0.7), offset = T(12))
        fit!(da, T(0.9)); fit!(da, T(0.6))
        reset!(da, T(0.3); target = T(0.77), regularization_scale = T(0.04),
               relaxation_exponent = T(0.8), offset = T(9))
        fresh = dual_averaging_state(T(0.3); target = T(0.77), regularization_scale = T(0.04),
                                     relaxation_exponent = T(0.8), offset = T(9))
        for name in (:iteration, :error, :log_final, :center, :target,
                     :regularization_scale, :relaxation_exponent, :offset, :current, :final)
            @test getproperty(da, name) == getproperty(fresh, name)
        end
    end

    # Bidirectional detached-copy / reset isolation.
    da = dual_averaging_state(0.5); fit!(da, 0.9); fit!(da, 0.85)
    clone = copy(da)
    src_current = da.current; src_iter = da.iteration
    reset!(clone, 0.2)                                    # reset clone
    @test da.current == src_current && da.iteration == src_iter   # source untouched
    @test clone.iteration == 1
    clone_current = clone.current
    reset!(da, 0.7)                                       # reset source
    @test clone.current == clone_current                 # clone untouched (other direction)

    # 0-B over 1000 resets through warmed named barriers that return `nothing`
    # (reset! is the side effect, so the loop is not eliminated; a scalar return
    # would be boxed by @allocated and read as a spurious 16 B).
    da2 = dual_averaging_state(0.5)
    function _reset_da(d, n)
        for _ in 1:n; reset!(d, 0.3; target = 0.8); end
        nothing
    end
    _reset_da(da2, 1)
    @test @allocated(_reset_da(da2, 1000)) == 0
    wv2 = welford_var(4)
    function _reset_wv(w, n)
        for _ in 1:n; reset!(w); end
        nothing
    end
    _reset_wv(wv2, 1)
    @test @allocated(_reset_wv(wv2, 1000)) == 0
end

@testset "warmup! metric-window restart: exact oracle parity through restarts" begin
    _wr_grad!(g, q) = (copyto!(g, q); sum(abs2, q) / 2)
    _wr_valgrad(q) = (sum(abs2, q) / 2, copy(q))
    _wr_pot(q) = sum(abs2, q) / 2
    D = 4; q0 = [sin(1.0i) for i in 1:D]
    metric = Matrix{Float64}(I, D, D)
    oracle = ReactiveKernels._oracle_nuts_state(
        euclidean_phasepoint(_wr_pot, _wr_valgrad, metric, copy(q0), zeros(D));
        rng = Xoshiro(321), step_f = partial(leapfrog!; stepsize = 0.5), max_depth = 6)
    compiled = nuts_state(reactive_nuts_group(_wr_grad!, metric, copy(q0), zeros(D));
        rng = Xoshiro(321), step_f = partial(leapfrog!; stepsize = 0.5), max_depth = 6)
    # 400 iterations guarantees several metric-window restarts; the reset/reuse path
    # must stay BIT-FOR-BIT identical to the ordinary-Julia oracle through them.
    cw = warmup!(compiled, 400; target_accept = 0.8)
    ow = warmup!(oracle, 400; target_accept = 0.8)
    @test cw.initial_stepsize == ow.initial_stepsize
    @test cw.final_stepsize == ow.final_stepsize
    @test cw.metric == ow.metric
end

@testset "warmup! window branch resets in place (static: no reconstruction)" begin
    # Durable static regression: the metric-window boundary must RESET the existing
    # adaptation/variance objects, never reconstruct them (the per-window recompile).
    src = read(joinpath(pkgdir(ReactiveKernels), "examples", "nuts_runtime", "hmc.jl"), String)
    lo = findfirst("iteration == window_ends[next_window]", src)
    hi = findnext("next_window += 1", src, last(lo))
    window_branch = src[first(lo):last(hi)]
    @test occursin("reset!(adaptation", window_branch)
    @test occursin("reset!(variance", window_branch)
    @test !occursin("dual_averaging_state(", window_branch)
    @test !occursin("welford_var(", window_branch)
end
