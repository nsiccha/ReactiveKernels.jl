#!/usr/bin/env julia

# Reproducible DECOMPOSED microbenchmark of the compiled-reactive NUTS transition.
#
#   julia --startup-file=no --project=packages benchmark/nuts_microbench.jl
#
# Measures allocations + wall time for matched Hamiltonian work across in-repo arms:
#   :compiled  — public CompiledNUTSState over the reactive_nuts_group program
#   :oracle    — internal handwritten _OracleNUTSState reference (ordinary-Julia
#                recursion over reactive per-endpoint phase points)
#   :raw_floor — pure in-place handwritten point with owned buffers, NO reactivity
#                (the allocation/latency floor; low-level stages only)
#
# The public ReactiveHMC ca9 arm is a SEPARATE reproducible temp-environment
# script (benchmark/nuts_microbench_ca9.jl) that pins the actual upstream package;
# the internal _OracleNUTSState here is a faithful line-for-line port, NOT a
# substitute for that public baseline.
#
# The gradient fixture is the analytic standard-normal score (grad U(q) = q) so the
# stages isolate REACTIVE-machinery overhead from AD cost. End-to-end DI+Enzyme
# sampling is the separate benchmark/nuts_comparison.jl comparator.
#
# Every stage GATES numerical parity (and expected leaf-execution) across arms
# BEFORE timing, construction/compile is warmed and timed separately from the hot
# path, and the script ERRORS if any reactive-overhead 0-B gate or the compiled
# step! typed/LLVM evidence fails.

using ReactiveKernels
using ReactiveKernelsNUTSExamples
using LinearAlgebra
using Random
using InteractiveUtils
using Test: @inferred

const RK = ReactiveKernels

_micro_pot(q) = 0.5 * sum(abs2, q)
_micro_grad!(gradient, q) = (copyto!(gradient, q); 0.5 * sum(abs2, q))   # in-place boundary
_micro_valgrad(q) = (0.5 * sum(abs2, q), copy(q))                        # (value, gradient)
_micro_pos(D) = [sin(1.0i) for i in 1:D]
_micro_mom(D) = [cos(0.7i) for i in 1:D]

# --- Handwritten in-place floor (owned buffers, no reactivity) ----------------
mutable struct RawPoint{C}
    chol::C
    pos::Vector{Float64}
    mom::Vector{Float64}
    dpot::Vector{Float64}
    vel::Vector{Float64}
    pot::Float64
    ham::Float64
    init_ham::Float64                             # ham of the transition's start point
end
function _raw_kinetic(chol, mom, vel)
    copyto!(vel, mom); ldiv!(chol, vel)
    0.5 * (logdet(chol) + dot(mom, vel))
end
function RawPoint(metric, pos, mom)
    chol = cholesky(metric)
    dpot = copy(pos); vel = copy(mom)
    pot = _micro_pot(pos)
    ham = pot + _raw_kinetic(chol, mom, vel)
    RawPoint(chol, copy(pos), copy(mom), dpot, vel, pot, ham, ham)
end
# Recompute potential/gradient/velocity/kinetic/ham from the CURRENT pos & mom.
function _raw_refresh!(p::RawPoint)
    copyto!(p.dpot, p.pos)                        # grad U(q) = q
    p.pot = 0.5 * sum(abs2, p.pos)
    p.ham = p.pot + _raw_kinetic(p.chol, p.mom, p.vel)
    p
end
# Momentum-only refresh: recompute velocity+kinetic+ham, leaving potential/gradient
# valid — matched to a `touch!(fwd_mom)` invalidation (fwd_pos, hence pot/grad, is
# untouched, so only the kinetic/hamiltonian nodes recompute).
function _raw_refresh_mom!(p::RawPoint)
    p.ham = p.pot + _raw_kinetic(p.chol, p.mom, p.vel)
    p
end
# Exact leapfrog matching the reactive lazy-recompute order: mom half; velocity from
# the NEW mom; pos full; potential+gradient from the NEW pos; mom half; then
# velocity+kinetic+ham from the FINAL mom before ham is read.
function _raw_leapfrog!(p::RawPoint, stepsize)
    @. p.mom -= 0.5 * stepsize * p.dpot                    # 1: mom half (grad @ old pos)
    copyto!(p.vel, p.mom); ldiv!(p.chol, p.vel)            # 2: velocity from new mom
    @. p.pos += stepsize * p.vel                           # 3: pos full
    copyto!(p.dpot, p.pos); p.pot = 0.5 * sum(abs2, p.pos) # 4: grad+pot from new pos
    @. p.mom -= 0.5 * stepsize * p.dpot                    # 5: mom half (grad @ new pos)
    p.ham = p.pot + _raw_kinetic(p.chol, p.mom, p.vel)     # 6: velocity+kinetic+ham, final mom
    p
end

# --- Timing / allocation: batched inside a named nothing-returning probe -------
# (measuring `@allocated f()` on a scalar-returning closure boxes the result — a
# 16-B harness artifact; a batched nothing-returning probe reproduces the real
# per-call figure, matching the package's other 0-B receipts.)
const _SINK = Ref(0.0)
function _probe(f, n)
    @inbounds for _ in 1:n; _SINK[] += f(); end
    nothing
end
function _bench_ns(f; batch, reps, warmup = 3)
    _probe(f, warmup * batch)
    best = Inf
    for _ in 1:reps
        t0 = time_ns(); _probe(f, batch); t1 = time_ns()
        best = min(best, (t1 - t0) / batch)
    end
    best
end
# Allocation PROFILE through a named parametric barrier (arm enters as a concrete
# type parameter, so it is never a boxed reassigned caller local — that boxing is
# itself a per-call allocation). Whole-batch totals are recorded at n, 2n, 4n and
# the two marginal slopes computed: a genuine per-call leak makes BOTH slopes equal
# and positive; a one-time harness constant (JIT residue / GC granularity, ~16-48 B
# whole-batch even for a pure field read) makes both slopes zero while the totals
# share that constant. Requiring slope stability across two scales also prevents a
# PERIODIC allocation from being accidentally cancelled by a single difference.
function alloc_profile(fc::F, arm::A; n = 4000) where {F,A}
    op = let fc = fc, arm = arm
        () -> fc(arm)
    end
    _probe(op, 4n)                                # warm all three code paths
    a1 = @allocated _probe(op, n)
    a2 = @allocated _probe(op, 2n)
    a4 = @allocated _probe(op, 4n)
    s1 = (a2 - a1) / n
    s2 = (a4 - a2) / (2n)
    (; totals = (a1, a2, a4), slope1 = s1, slope2 = s2,
       percall = max(0, round(Int, s2)))
end

# 0-B iff both marginal slopes are ~0 and non-negative (stable, non-periodic).
function _is_zero_b(prof; tol = 0.5)
    prof.slope1 >= -tol && prof.slope2 >= -tol &&
        abs(prof.slope1) <= tol && abs(prof.slope2) <= tol
end

# Parametric barrier for TIMING: @inferred guards type-stability so a non-inferred
# stage fails loudly instead of silently contaminating the numbers.
function measure_stage(fc::F, arm::A; batch::Int, reps::Int) where {F,A}
    op = let fc = fc, arm = arm
        () -> fc(arm)
    end
    @inferred op()
    prof = alloc_profile(fc, arm; n = max(batch, 4000))
    (ns = _bench_ns(op; batch, reps), bytes = prof.percall, profile = prof)
end

# --- Arm construction on identical work ---------------------------------------
function build_arms(D)
    metric = Matrix{Float64}(I, D, D)
    pos = _micro_pos(D); mom = _micro_mom(D)
    group = ReactiveKernelsNUTSExamples.reactive_nuts_group(_micro_grad!, metric, copy(pos), copy(mom))
    compiled = ReactiveKernelsNUTSExamples.nuts_state(group; rng = Xoshiro(2024),
                             step_f = RK.partial(ReactiveKernelsNUTSExamples.leapfrog!; stepsize = 0.1),
                             max_depth = 6)
    point = ReactiveKernelsNUTSExamples.euclidean_phasepoint(_micro_pot, _micro_valgrad, metric,
                                    copy(pos), copy(mom))
    oracle = ReactiveKernelsNUTSExamples._oracle_nuts_state(point; rng = Xoshiro(2024),
                                   step_f = RK.partial(ReactiveKernelsNUTSExamples.leapfrog!; stepsize = 0.1),
                                   max_depth = 6)
    raw = RawPoint(metric, copy(pos), copy(mom))
    (; group, compiled, oracle, raw)
end

# Independent 0-B control mirroring the package's own named-barrier receipt
# (test/test_reactive_nuts.jl): a self-contained leapfrog-shaped cycle driven
# directly through state+handles get!/mutate!, returning nothing, so a bare
# `@allocated` gives the true figure with no difference method and no global sink.
function _leapfrog_cycle!(state::S, handles::H, stepsize, n) where {S,H}
    for _ in 1:n
        gradient = RK.get!(state, handles.fwd_dpot_dpos)
        RK.mutate!(state, handles.fwd_mom) do mom
            @. mom -= 0.5 * stepsize * gradient
            mom
        end
        velocity = RK.get!(state, handles.fwd_dham_dmom)
        RK.mutate!(state, handles.fwd_pos) do pos
            @. pos += stepsize * velocity
            pos
        end
        RK.get!(state, handles.fwd_ham)
    end
end
# Measure the control inside a concrete-typed barrier so call-site boxing from the
# surrounding driver's degraded inference cannot inflate the figure.
function _cycle_alloc(state::S, handles::H; stepsize = 0.1, n = 1000) where {S,H}
    _leapfrog_cycle!(state, handles, stepsize, 1)             # warm the steady path
    @allocated _leapfrog_cycle!(state, handles, stepsize, n)
end

# Energy error read the SAME way across arms: finite(init_ham - active_fwd_ham).
_compiled_dham(a) = a.group.dham
_oracle_dham(a)   = ReactiveKernelsNUTSExamples._finite_or_neginf(a.oracle.init.ham - a.oracle.fwd.ham)
_raw_dham(a)      = ReactiveKernelsNUTSExamples._finite_or_neginf(a.raw.init_ham - a.raw.ham)

# Advance every arm by ONE leapfrog so getter/dham/source stages read a
# non-trivial, matched state (fwd endpoint moved off init).
function _advance_one!(a)
    ReactiveKernelsNUTSExamples._group_leapfrog!(a.group, Val(:fwd), 0.1)
    ReactiveKernelsNUTSExamples.leapfrog!(a.oracle.fwd; stepsize = 0.1)
    _raw_leapfrog!(a.raw, 0.1)
    a
end

# ============================ stages ==========================================
# Stage 1: source mutation + dependent invalidation. Compiled/oracle: touch! the
# forward-momentum HAVE source (0-B invalidation) then read the downstream ENERGY
# ERROR to force one real recompute — so each repetition measures invalidate+
# recompute (a cached read alone would short-circuit once dependents stay invalid).
# Raw has NO reactive invalidation: it unconditionally recomputes the Hamiltonian;
# the row is labelled accordingly. Every arm returns the SAME energy error.
_s1_compiled(a) = (RK.touch!(a.group.state, a.group.handles.fwd_mom); _compiled_dham(a))
_s1_oracle(a)   = (RK.touch!(a.oracle.fwd.state, a.oracle.fwd.handles.mom); _oracle_dham(a))
_s1_raw(a)      = (_raw_refresh_mom!(a.raw); _raw_dham(a))   # momentum-only, matched to touch!(fwd_mom)

# Stage 2: ham getter — read the already-valid active-endpoint Hamiltonian.
_s2_compiled(a) = a.group.fwd_ham
_s2_oracle(a)   = a.oracle.fwd.ham
_s2_raw(a)      = a.raw.ham

# Stage 3: dham getter — the energy error, computed the same way per arm.
_s3_compiled(a) = _compiled_dham(a)
_s3_oracle(a)   = _oracle_dham(a)
_s3_raw(a)      = _raw_dham(a)

# Stage 4: one leapfrog on the active forward endpoint.
_s4_compiled(a) = (ReactiveKernelsNUTSExamples._group_leapfrog!(a.group, Val(:fwd), 0.1); a.group.fwd_ham)
_s4_oracle(a)   = (ReactiveKernelsNUTSExamples.leapfrog!(a.oracle.fwd; stepsize = 0.1); a.oracle.fwd.ham)
_s4_raw(a)      = (_raw_leapfrog!(a.raw, 0.1); a.raw.ham)

# Stage 5: reset + depth-1 tree unit (guarantees exactly one leaf/leapfrog per rep).
_s5_compiled(a) = (ReactiveKernelsNUTSExamples._cn_reset_transition!(a.compiled); ReactiveKernelsNUTSExamples._cn_start_tree!(a.compiled, 1);
                   a.compiled.energy_error)
_s5_oracle(a)   = (ReactiveKernelsNUTSExamples._reset_transition!(a.oracle); ReactiveKernelsNUTSExamples._start_tree!(a.oracle, 1);
                   a.oracle.energy_error)

# Stage 6: one full transition (reset + tree growth + proposal selection + RNG).
_s6_compiled(a) = (ReactiveKernelsNUTSExamples.step!(a.compiled); a.compiled.energy_error)
_s6_oracle(a)   = (ReactiveKernelsNUTSExamples.step!(a.oracle); a.oracle.energy_error)

# --- Parity gates -------------------------------------------------------------
_approx(x, y) = isapprox(x, y; rtol = 1e-10, atol = 1e-12)

function gate_low_level(D)
    a = build_arms(D); _advance_one!(a)
    @assert _approx(a.group.fwd_pos, a.oracle.fwd.pos) && _approx(a.group.fwd_pos, a.raw.pos) "leapfrog pos parity"
    @assert _approx(a.group.fwd_mom, a.oracle.fwd.mom) && _approx(a.group.fwd_mom, a.raw.mom) "leapfrog mom parity"
    @assert _approx(a.group.fwd_ham, a.oracle.fwd.ham) && _approx(a.group.fwd_ham, a.raw.ham) "leapfrog ham parity"
    @assert _approx(_compiled_dham(a), _oracle_dham(a)) "dham parity (oracle)"
    @assert _approx(_compiled_dham(a), _raw_dham(a)) "dham parity (raw)"
    @assert isfinite(_compiled_dham(a)) && !_approx(_compiled_dham(a), 0.0) "dham finite & non-trivial"
    # Source-stage parity: after touch/refresh, all arms return the same energy error.
    a = build_arms(D); _advance_one!(a)
    @assert _approx(_s1_compiled(a), _s1_oracle(a)) "source-stage energy-error parity (oracle)"
    a = build_arms(D); _advance_one!(a)
    @assert _approx(_s1_compiled(a), _s1_raw(a)) "source-stage energy-error parity (raw)"
    nothing
end

function gate_depth1(D)
    a = build_arms(D)
    ReactiveKernelsNUTSExamples._cn_reset_transition!(a.compiled); ReactiveKernelsNUTSExamples._cn_start_tree!(a.compiled, 1)
    ReactiveKernelsNUTSExamples._reset_transition!(a.oracle); ReactiveKernelsNUTSExamples._start_tree!(a.oracle, 1)
    @assert a.compiled.n_steps == 1 "compiled depth-1 executed exactly one leaf"
    @assert a.oracle.n_steps == 1 "oracle depth-1 executed exactly one leaf"
    @assert _approx(a.compiled.energy_error, a.oracle.energy_error) "depth-1 energy-error parity"
    @assert isfinite(a.compiled.energy_error) "depth-1 energy-error finite"
    nothing
end

# Full-transition parity under IDENTICAL RNG: several fixed-momentum step!s must
# agree bit-close on accepted position and every diagnostic.
function gate_full_transition(D; n = 6)
    a = build_arms(D)
    for t in 1:n
        dc = ReactiveKernelsNUTSExamples.step!(a.compiled); do_ = ReactiveKernelsNUTSExamples.step!(a.oracle)   # returned diagnostics
        @assert _approx(a.compiled.group.init_pos, a.oracle.init.pos) "transition $t accepted-position parity"
        @assert dc.depth == do_.depth "transition $t depth parity"
        @assert dc.n_steps == do_.n_steps "transition $t n_steps parity"
        @assert dc.diverged == do_.diverged "transition $t diverged parity"
        @assert _approx(dc.energy_error, do_.energy_error) "transition $t energy-error parity"
        @assert _approx(dc.acceptance_rate, do_.acceptance_rate) "transition $t acceptance-rate parity"
    end
    nothing
end

# --- code_typed / LLVM evidence for the hot compiled step! path (HARD gate) ----
# The hard evidence is: a concrete return type, NO Any-typed local SLOTS, and an
# optimized-LLVM census clean of dynamic dispatch / runtime getfield / boxing / GC
# allocation. Any-typed SSA VALUES are compiler bookkeeping on effectful `:invoke`
# statements whose results are discarded (static dispatch, zero-allocation — the
# separate full-transition profile confirms 0 B); their count is reported honestly
# but NOT asserted zero.
const _LLVM_FORBIDDEN = ("ijl_apply_generic", "jl_apply_generic", "jl_apply",
                         "jl_f_getfield", "jl_get_nth_field_checked", "jl_invoke",
                         "jl_box", "jl_gc_alloc", "ijl_gc_pool_alloc",
                         "jl_gc_pool_alloc", "ijl_gc_big_alloc", "jl_gc_big_alloc")
function assert_step_typed_llvm(compiled)
    T = typeof(compiled)
    typed = only(code_typed(ReactiveKernelsNUTSExamples.step!, Tuple{T}; optimize = true))
    ci, return_type = typed.first, typed.second
    llvm = sprint(io -> code_llvm(io, ReactiveKernelsNUTSExamples.step!, Tuple{T};
                                  optimize = true, debuginfo = :none))
    return_concrete = isconcretetype(return_type) || return_type === Union{}
    any_slots = count(t -> t === Any, ci.slottypes)
    any_ssa = count(t -> t === Any, ci.ssavaluetypes)
    dynamic_calls = count(eachindex(ci.code)) do i
        s = ci.code[i]
        s isa Expr && s.head === :call && ci.ssavaluetypes[i] === Any
    end
    forbidden = filter(p -> occursin(p, llvm), _LLVM_FORBIDDEN)
    evidence = (; inferred_return = return_type, return_concrete,
                  any_typed_slots = any_slots,
                  any_typed_ssa_bookkeeping = any_ssa,       # reported, not asserted
                  dynamic_calls, llvm_forbidden_symbols = forbidden)
    @assert return_concrete "compiled step! return type is non-concrete: $return_type"
    @assert any_slots == 0 "compiled step! has $any_slots Any-typed local slot(s)"
    @assert dynamic_calls == 0 "compiled step! has $dynamic_calls dynamic :call(s) returning Any"
    @assert isempty(forbidden) "compiled step! optimized LLVM contains forbidden runtime symbols: $forbidden"
    evidence
end

# --- Construction / setup timing (warm-excluded, per arm) ---------------------
function timed_setup(D)
    metric = Matrix{Float64}(I, D, D); pos = _micro_pos(D); mom = _micro_mom(D)
    sf() = RK.partial(ReactiveKernelsNUTSExamples.leapfrog!; stepsize = 0.1)
    build_compiled() = (g = ReactiveKernelsNUTSExamples.reactive_nuts_group(_micro_grad!, metric, copy(pos), copy(mom));
                        ReactiveKernelsNUTSExamples.nuts_state(g; rng = Xoshiro(1), step_f = sf(), max_depth = 6))
    build_oracle() = (p = ReactiveKernelsNUTSExamples.euclidean_phasepoint(_micro_pot, _micro_valgrad, metric,
                                                  copy(pos), copy(mom));
                      ReactiveKernelsNUTSExamples._oracle_nuts_state(p; rng = Xoshiro(1), step_f = sf(), max_depth = 6))
    build_raw() = RawPoint(metric, copy(pos), copy(mom))
    build_compiled(); build_oracle(); build_raw()          # warm: exclude compilation
    tc = @timed build_compiled(); to = @timed build_oracle(); tr = @timed build_raw()
    (; compiled = (seconds = tc.time, bytes = tc.bytes),
       oracle = (seconds = to.time, bytes = to.bytes),
       raw = (seconds = tr.time, bytes = tr.bytes))
end

function run_microbench(; dims = (4, 10))
    println((; julia = string(VERSION),
               gradient_fixture = "analytic std-normal score (grad U(q)=q)",
               note = "internal _OracleNUTSState is a ca9 PORT, not the public ca9 baseline"))
    for D in dims
        gate_low_level(D); gate_depth1(D); gate_full_transition(D)
        println((; dimension = D, construction_setup = timed_setup(D),
                   note = "warm/compile excluded; hot-path timings below exclude construction"))
        stages = (
            (:source_mutate_invalidate, "compiled/oracle: touch!+recompute energy error; raw: recompute (no invalidation)",
             _s1_compiled, _s1_oracle, _s1_raw, 500, 100, true),
            (:ham_getter, "read valid active-endpoint ham (advanced one leapfrog)",
             _s2_compiled, _s2_oracle, _s2_raw, 2000, 100, true),
            (:dham_getter, "energy error finite(init_ham-active_ham), matched per arm",
             _s3_compiled, _s3_oracle, _s3_raw, 2000, 100, true),
            (:one_leapfrog, "one leapfrog on the active forward endpoint",
             _s4_compiled, _s4_oracle, _s4_raw, 500, 100, false),
            (:reset_plus_depth1, "reset + depth-1 tree unit (exactly one leaf/rep)",
             _s5_compiled, _s5_oracle, nothing, 20, 500, false),
            (:full_transition, "one full transition (reset + tree growth + proposal + RNG)",
             _s6_compiled, _s6_oracle, nothing, 1, 2000, false),
        )
        # Reactive source/getter/leapfrog overhead must be 0 B; depth-1/full carry
        # ordinary recursion/RNG/proposal orchestration and are reported, not gated.
        zero_b_stages = (:source_mutate_invalidate, :ham_getter, :dham_getter, :one_leapfrog)
        for (name, semantics, fc, fo, fr, batch, reps, advance) in stages
            setup!(a) = advance ? _advance_one!(a) : a
            a = build_arms(D); setup!(a); compiled = measure_stage(fc, a; batch, reps)
            # Assert the compiled reactive overhead is 0 B through the named barrier
            # BEFORE printing, so a contaminated row can never be reported as a result.
            if name in zero_b_stages
                @assert compiled.bytes == 0 && _is_zero_b(compiled.profile) "compiled $name is not 0 B/call: $(compiled.profile)"
            end
            a = build_arms(D); setup!(a); oracle = measure_stage(fo, a; batch, reps)
            raw = nothing
            if fr !== nothing
                a = build_arms(D); setup!(a); raw = measure_stage(fr, a; batch, reps)
            end
            println((;
                dimension = D, stage = name, semantics,
                compiled_ns = round(compiled.ns; digits = 2), compiled_bytes = compiled.bytes,
                oracle_ns = round(oracle.ns; digits = 2), oracle_bytes = oracle.bytes,
                raw_ns = raw === nothing ? missing : round(raw.ns; digits = 2),
                raw_bytes = raw === nothing ? missing : raw.bytes,
            ))
        end
        # Harness-constant transparency: a pure RAW field getter and the compiled
        # getter share the SAME nonzero one-time whole-batch total but BOTH marginal
        # slopes are zero, proving the constant is harness-wide, not a per-call leak.
        gp = build_arms(D); _advance_one!(gp)
        raw_get = alloc_profile(_s2_raw, gp)
        cmp_get = alloc_profile(_s2_compiled, gp)
        println((; dimension = D, getter_alloc_totals_n_2n_4n =
                   (raw = raw_get.totals, compiled = cmp_get.totals),
                   getter_slopes = (raw = (raw_get.slope1, raw_get.slope2),
                                    compiled = (cmp_get.slope1, cmp_get.slope2)),
                   both_slopes_zero = _is_zero_b(raw_get) && _is_zero_b(cmp_get)))
        @assert _is_zero_b(raw_get) && _is_zero_b(cmp_get) "getter marginal slopes must be 0"

        # HARD 0-B gate for the reactive source/getter/leapfrog overhead, each via a
        # named parametric barrier + slope-stable profile (no boxed captured local).
        g1 = build_arms(D); _advance_one!(g1)
        profs = (source = alloc_profile(_s1_compiled, g1),
                 ham = alloc_profile(_s2_compiled, g1),
                 dham = alloc_profile(_s3_compiled, g1),
                 leapfrog = alloc_profile(_s4_compiled, build_arms(D)))
        gate = map(p -> p.percall, profs)
        passed = all(_is_zero_b, values(profs))
        println((; dimension = D, reactive_overhead_0B_gate_bytes_per_call = gate,
                   slopes_stable = passed, passed = passed && all(==(0), gate)))
        @assert passed && all(==(0), gate) "reactive source/getter/leapfrog overhead must be 0 B/call: $profs"

        # Independent control: the package's own named-barrier receipt style — a
        # self-contained leapfrog cycle over the group (no difference method) must
        # allocate exactly 0 B, corroborating the profiled gate above.
        gc = build_arms(D)
        cycle_bytes = _cycle_alloc(gc.group.state, gc.group.handles; n = 1000)
        println((; dimension = D, named_barrier_leapfrog_cycle_bytes = cycle_bytes))
        @assert cycle_bytes == 0 "named-barrier leapfrog cycle must be 0 B, got $cycle_bytes"

        # Full-transition allocation: MEASURED, not pre-attributed. If nonzero it is
        # left explicitly UNASSIGNED here pending profiling, not labelled as scratch.
        full = alloc_profile(_s6_compiled, build_arms(D); n = 2000)
        println((; dimension = D, full_transition_bytes_per_call = full.percall,
                   full_transition_slopes = (full.slope1, full.slope2),
                   attribution = full.percall == 0 ? "zero" : "UNASSIGNED pending profiling"))
        # Typed/LLVM hard evidence for the compiled step! hot path.
        a = build_arms(D)
        println((; dimension = D, step_typed_llvm = assert_step_typed_llvm(a.compiled)))
    end
    println((; microbench = :ok))
    nothing
end

# A fast smoke check (parity gates + one measurement + the typed/LLVM hard gate on a
# tiny build) for the test suite, so the harness cannot rot silently — no long
# timing loops. Returns the step! typed/LLVM evidence.
function microbench_smoke(D = 3)
    gate_low_level(D); gate_depth1(D); gate_full_transition(D; n = 3)
    a = build_arms(D); _advance_one!(a)
    prof = alloc_profile(_s4_compiled, a; n = 500)
    @assert _is_zero_b(prof) "smoke: leapfrog overhead not 0 B: $prof"
    gc = build_arms(D)
    @assert _cycle_alloc(gc.group.state, gc.group.handles; n = 100) == 0 "smoke: leapfrog cycle not 0 B"
    assert_step_typed_llvm(build_arms(D).compiled)
end

# Only execute the full (slow) benchmark when run as a script; `include`-ing the
# file (e.g. from the test smoke gate) just defines the functions.
if abspath(PROGRAM_FILE) == @__FILE__
    run_microbench()
end
