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
function _percall_bytes(fc::F, arm::A; n = 4000) where {F,A}
    op = let fc = fc, arm = arm; () -> fc(arm) end
    _probe(op, 4n)
    a1 = @allocated _probe(op, n)
    a2 = @allocated _probe(op, 2n)
    a4 = @allocated _probe(op, 4n)
    s = (a4 - a2) / (2n)
    max(0, round(Int, s))
end
function measure(fc::F, arm::A; batch, reps) where {F,A}
    op = let fc = fc, arm = arm; () -> fc(arm) end
    (ns = _bench_ns(op; batch, reps), bytes = _percall_bytes(fc, arm; n = max(batch, 4000)))
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
function _ca9_full(nt)
    @invalidatedependants! nt.state.init.mom = nt.mom0
    ReactiveHMC.step!(nt.state)
    nt.state.init.ham
end

# --- RK compiled / oracle stage ops -------------------------------------------
_c_ham(a) = a.group.fwd_ham
_c_source(a) = (RK.touch!(a.group.state, a.group.handles.fwd_mom); a.group.fwd_ham)
_c_leapfrog(a) = (RK._group_leapfrog!(a.group, Val(:fwd), 0.1); a.group.fwd_ham)
_c_full(a) = (RK.step!(a.compiled); a.compiled.energy_error)

_o_ham(a) = a.oracle.fwd.ham
_o_source(a) = (RK.touch!(a.oracle.fwd.state, a.oracle.fwd.handles.mom); a.oracle.fwd.ham)
_o_leapfrog(a) = (RK.leapfrog!(a.oracle.fwd; stepsize = 0.1); a.oracle.fwd.ham)
_o_full(a) = (RK.step!(a.oracle); a.oracle.energy_error)

# --- builders -----------------------------------------------------------------
function build_ca9(D)
    metric = Matrix{Float64}(I, D, D)
    point = ReactiveHMC.euclidean_phasepoint(_pot, _valgrad, metric, _pos(D), _mom(D))
    state = ReactiveHMC.nuts_state(point; rng = Xoshiro(2024),
                                   step_f = p -> ReactiveHMC.leapfrog!(p; stepsize = 0.1),
                                   max_depth = 6)
    (; point, state, mom0 = _mom(D))
end
function build_rk(D)
    metric = Matrix{Float64}(I, D, D)
    group = RK.reactive_nuts_group(_grad!, metric, _pos(D), _mom(D))
    compiled = RK.nuts_state(group; rng = Xoshiro(2024),
                             step_f = RK.partial(RK.leapfrog!; stepsize = 0.1), max_depth = 6)
    point = RK.euclidean_phasepoint(_pot, _valgrad, metric, _pos(D), _mom(D))
    oracle = RK._oracle_nuts_state(point; rng = Xoshiro(2024),
                                   step_f = RK.partial(RK.leapfrog!; stepsize = 0.1), max_depth = 6)
    (; group, compiled, oracle)
end

_approx(x, y) = isapprox(x, y; rtol = 1e-8, atol = 1e-10)

# One leapfrog from identical (pos, mom) is deterministic across all three arms.
function gate_leapfrog_parity(D)
    ca9 = build_ca9(D); rk = build_rk(D)
    ReactiveHMC.leapfrog!(ca9.point; stepsize = 0.1)
    RK._group_leapfrog!(rk.group, Val(:fwd), 0.1)
    RK.leapfrog!(rk.oracle.fwd; stepsize = 0.1)
    @assert _approx(ca9.point.pos, rk.group.fwd_pos) "ca9/compiled pos parity"
    @assert _approx(ca9.point.mom, rk.group.fwd_mom) "ca9/compiled mom parity"
    @assert _approx(ca9.point.ham, rk.group.fwd_ham) "ca9/compiled ham parity"
    @assert _approx(ca9.point.ham, rk.oracle.fwd.ham) "ca9/oracle ham parity"
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
        # setup / construction (warm-excluded)
        build_ca9(D); build_rk(D)
        tca9 = @timed build_ca9(D); trk = @timed build_rk(D)
        println((; dimension = D,
                   setup = (ca9 = (tca9.time, tca9.bytes), rk_bundle = (trk.time, trk.bytes))))
        stages = (
            (:ham_getter, _ca9_ham, _c_ham, _o_ham, :point, 2000, 100),
            (:source_mutate, _ca9_source, _c_source, _o_source, :point, 500, 100),
            (:one_leapfrog, _ca9_leapfrog, _c_leapfrog, _o_leapfrog, :point, 500, 100),
            (:full_transition, _ca9_full, _c_full, _o_full, :full, 1, 2000),
        )
        for (name, fca9, fc, fo, kind, batch, reps) in stages
            ca9_arm = kind === :full ? build_ca9(D) : build_ca9(D).point
            rk = build_rk(D)
            ca9 = measure(fca9, ca9_arm; batch, reps)
            cmp = measure(fc, rk; batch, reps)
            orc = measure(fo, rk; batch, reps)
            println((; dimension = D, stage = name,
                       ca9_ns = round(ca9.ns; digits = 2), ca9_bytes = ca9.bytes,
                       compiled_ns = round(cmp.ns; digits = 2), compiled_bytes = cmp.bytes,
                       oracle_ns = round(orc.ns; digits = 2), oracle_bytes = orc.bytes))
        end
    end
    println((; ca9_microbench = :ok))
    nothing
end

run()
