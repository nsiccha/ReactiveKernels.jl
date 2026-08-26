# Inner body of benchmark/nuts_microbench_ca9.jl — runs in the pinned temp env with
# ReactiveKernels, ReactiveHMC (ca9) and ReactiveObjects (@invalidatedependants!)
# all loaded. Three arms on identical standard-normal work:
#   :ca9       — the actual public ReactiveHMC ca9 sampler / phasepoint
#   :compiled  — ReactiveKernels CompiledNUTSState over reactive_nuts_group
#   :oracle    — ReactiveKernels internal handwritten _OracleNUTSState (a ca9 port)

const RK = ReactiveKernels

_pot(q) = 0.5 * sum(abs2, q)
_grad!(g, q) = (copyto!(g, q); 0.5 * sum(abs2, q))       # in-place boundary (RK compiled)
_valgrad(q) = (0.5 * sum(abs2, q), copy(q))              # (value, gradient) (ca9 + RK oracle)
_pos(D) = [sin(1.0i) for i in 1:D]
_mom(D) = [cos(0.7i) for i in 1:D]

# --- measurement: parametric barriers (arm concrete ⇒ no boxed-capture allocation) -
const _SINK = Ref(0.0)
function _probe(f, n)
    s = 0.0
    @inbounds for _ in 1:n; s += f(); end
    _SINK[] += s
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
# Two-slope allocation profile (same contract as benchmark/nuts_microbench.jl):
# totals at n/2n/4n and BOTH marginal slopes, so a one-time harness constant reads
# as zero slope while a genuine (even periodic) per-call leak shows a stable slope.
function alloc_profile(fc::F, arm::A; n = 4000) where {F,A}
    op = let fc = fc, arm = arm; () -> fc(arm) end
    _probe(op, 4n)
    a1 = @allocated _probe(op, n)
    a2 = @allocated _probe(op, 2n)
    a4 = @allocated _probe(op, 4n)
    s1 = (a2 - a1) / n
    s2 = (a4 - a2) / (2n)
    (; totals = (a1, a2, a4), slope1 = s1, slope2 = s2, percall = max(0, round(Int, s2)))
end
_is_zero_b(p; tol = 0.5) = abs(p.slope1) <= tol && abs(p.slope2) <= tol &&
                           p.slope1 >= -tol && p.slope2 >= -tol
function measure(fc::F, arm::A; batch, reps) where {F,A}
    op = let fc = fc, arm = arm; () -> fc(arm) end
    prof = alloc_profile(fc, arm; n = max(batch, 4000))
    (ns = _bench_ns(op; batch, reps), bytes = prof.percall, profile = prof)
end

# --- ca9 stage ops (public upstream API only) ---------------------------------
_ca9_ham(p) = p.ham
function _ca9_source(p)
    @invalidatedependants! p.mom = p.mom          # invalidate kinetic/ham dependents
    p.ham
end
function _ca9_leapfrog(p)
    ReactiveHMC.leapfrog!(p; stepsize = 0.1)
    p.ham
end
_ca9_full(state) = (ReactiveHMC.step!(state); state.init.ham)   # step!-only, no reset

# --- RK compiled / oracle stage ops -------------------------------------------
_c_ham(a) = a.group.fwd_ham
_c_source(a) = (RK.touch!(a.group.state, a.group.handles.fwd_mom); a.group.fwd_ham)
_c_leapfrog(a) = (RK._group_leapfrog!(a.group, Val(:fwd), 0.1); a.group.fwd_ham)
_c_full(a) = (RK.step!(a.compiled); a.compiled.init.pos[1])

_o_ham(a) = a.oracle.fwd.ham
_o_source(a) = (RK.touch!(a.oracle.fwd.state, a.oracle.fwd.handles.mom); a.oracle.fwd.ham)
_o_leapfrog(a) = (RK.leapfrog!(a.oracle.fwd; stepsize = 0.1); a.oracle.fwd.ham)
_o_full(a) = (RK.step!(a.oracle); a.oracle.init.pos[1])

# --- per-arm builders (setup timed individually) ------------------------------
function build_ca9(D)
    metric = Matrix{Float64}(I, D, D)
    point = ReactiveHMC.euclidean_phasepoint(_pot, _valgrad, metric, _pos(D), _mom(D))
    state = ReactiveHMC.nuts_state(point; rng = Xoshiro(2024),
                                   step_f = p -> ReactiveHMC.leapfrog!(p; stepsize = 0.1),
                                   max_depth = 6)
    (; point, state)
end
function build_compiled(D)
    metric = Matrix{Float64}(I, D, D)
    group = RK.reactive_nuts_group(_grad!, metric, _pos(D), _mom(D))
    compiled = RK.nuts_state(group; rng = Xoshiro(2024),
                             step_f = RK.partial(RK.leapfrog!; stepsize = 0.1), max_depth = 6)
    (; group, compiled)
end
function build_oracle(D)
    metric = Matrix{Float64}(I, D, D)
    point = RK.euclidean_phasepoint(_pot, _valgrad, metric, _pos(D), _mom(D))
    oracle = RK._oracle_nuts_state(point; rng = Xoshiro(2024),
                                   step_f = RK.partial(RK.leapfrog!; stepsize = 0.1), max_depth = 6)
    (; point, oracle)
end

_approx(x, y; rtol = 1e-8, atol = 1e-10) = isapprox(x, y; rtol, atol)

# One leapfrog from identical (pos, mom) is deterministic across all three arms.
function gate_leapfrog_parity(D)
    ca9 = build_ca9(D); c = build_compiled(D); o = build_oracle(D)
    ReactiveHMC.leapfrog!(ca9.point; stepsize = 0.1)
    RK._group_leapfrog!(c.group, Val(:fwd), 0.1)
    RK.leapfrog!(o.oracle.fwd; stepsize = 0.1)
    @assert _approx(ca9.point.pos, c.group.fwd_pos) "ca9/compiled pos parity"
    @assert _approx(ca9.point.mom, c.group.fwd_mom) "ca9/compiled mom parity"
    @assert _approx(ca9.point.ham, c.group.fwd_ham) "ca9/compiled ham parity"
    @assert _approx(ca9.point.ham, o.oracle.fwd.ham) "ca9/oracle ham parity"
    nothing
end

# Several step!-ONLY transitions (fixed momentum, no reset in any arm): the accepted
# position must agree across ca9/compiled/oracle. compiled↔oracle is exact (shared
# RK code path); ca9↔oracle is asserted at a tight tolerance (a faithful port sharing
# the RNG stream) so gross divergence fails loudly.
function gate_full_parity(D; n = 5)
    ca9 = build_ca9(D); c = build_compiled(D); o = build_oracle(D)
    for t in 1:n
        ReactiveHMC.step!(ca9.state); RK.step!(c.compiled); RK.step!(o.oracle)
        @assert _approx(c.compiled.group.init_pos, o.oracle.init.pos) "compiled/oracle pos parity t=$t"
        @assert c.compiled.depth == o.oracle.depth "compiled/oracle depth parity t=$t"
        @assert _approx(ca9.state.init.pos, o.oracle.init.pos; rtol = 1e-6, atol = 1e-8) "ca9/oracle pos parity t=$t"
    end
    nothing
end

function run(; dims = (4, 10))
    println((;
        candidate_sha = get(ENV, "REACTIVEKERNELS_CANDIDATE_SHA", "unknown"),
        julia = string(VERSION),
        reactive_objects_rev = "419881dcfe93fbf0c679837c5421322fbd2c6888",
        reactive_hmc_rev = "ca9ea4ca41924bb0e1fadc01c717e1333916aba6",
        arms = (:ca9_public, :compiled, :oracle),
        gradient_fixture = "analytic std-normal score"))
    for D in dims
        gate_leapfrog_parity(D)
        gate_full_parity(D)
        # per-arm construction/setup (warm-excluded, timed individually)
        build_ca9(D); build_compiled(D); build_oracle(D)
        tca9 = @timed build_ca9(D); tc = @timed build_compiled(D); to = @timed build_oracle(D)
        println((; dimension = D, setup = (
            ca9 = (seconds = tca9.time, bytes = tca9.bytes),
            compiled = (seconds = tc.time, bytes = tc.bytes),
            oracle = (seconds = to.time, bytes = to.bytes))))
        # (name, ca9_op, compiled_op, oracle_op, ca9_arm_kind, batch, reps, zero_b)
        stages = (
            (:ham_getter, _ca9_ham, _c_ham, _o_ham, :point, 2000, 100, true),
            (:source_mutate, _ca9_source, _c_source, _o_source, :point, 500, 100, true),
            (:one_leapfrog, _ca9_leapfrog, _c_leapfrog, _o_leapfrog, :point, 500, 100, true),
            (:full_transition, _ca9_full, _c_full, _o_full, :state, 1, 2000, false),
        )
        for (name, fca9, fc, fo, kind, batch, reps, zero_b) in stages
            b = build_ca9(D); ca9_arm = kind === :state ? b.state : b.point
            ca9 = measure(fca9, ca9_arm; batch, reps)
            cmp = measure(fc, build_compiled(D); batch, reps)
            orc = measure(fo, build_oracle(D); batch, reps)
            if zero_b
                @assert cmp.bytes == 0 && _is_zero_b(cmp.profile) "compiled $name not 0 B/call: $(cmp.profile)"
            end
            _pp(p) = (totals = p.totals, slope1 = round(p.slope1; digits = 3),
                      slope2 = round(p.slope2; digits = 3))
            println((; dimension = D, stage = name,
                       ca9_ns = round(ca9.ns; digits = 2), ca9_bytes = ca9.bytes,
                       compiled_ns = round(cmp.ns; digits = 2), compiled_bytes = cmp.bytes,
                       oracle_ns = round(orc.ns; digits = 2), oracle_bytes = orc.bytes,
                       alloc_profiles = (ca9 = _pp(ca9.profile), compiled = _pp(cmp.profile),
                                         oracle = _pp(orc.profile))))
        end
    end
    println((; ca9_microbench = :ok))
    nothing
end

run()
