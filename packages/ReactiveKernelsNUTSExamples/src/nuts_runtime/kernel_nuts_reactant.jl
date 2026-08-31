# =============================================================================
# Reactant full-depth compiler path for the UNCHANGED adaptive NUTS kernel.
#
# Ownership: ReactiveKernels:reactant (decision 19rzxhy, todo 1vwgw6r), built on
# ReactiveKernels:hmc's verified foundation (benchmark/reactant_handoff/). The
# mathematical kernel source `benchmark/nuts_kernel_authoring_fixture.jl`
# (`@kernel nuts_state`) stays BYTE-IDENTICAL — this is compiler/lowering/runtime
# emission only, a sibling to compile_nuts / compile_nuts_native.
#
# Strategy (proven feasible at Reactant 0.2.278, ReactiveKernels:reactant probes):
#   * the SoA control-stack dispatcher (compile_nuts_dispatcher) lowers to a single
#     bounded `@trace while sum(csp)>=1` — depth-independent graph, no unroll, no cap;
#   * the full mutable NUTS state becomes fixed-capacity traced-tensor SoA (below);
#   * host `mid==`/`pc==` dispatch → masked all-blocks with `active=(mid==m)&(pc==p)`;
#   * host `_CtrlFrame`/`_FrameStore` indexing → vectorized dynamic gather/scatter
#     `col[idx]` / `w=copy(col); w[idx]=x` (NEVER scalar index, NEVER raw Ops.dynamic_*);
#   * RNG effects → the pre-generated (momentum, dirs, exps) bundle consumed by counters.
#
# This file is Reactant-FREE (plain Julia); Reactant tracing happens only at the
# gate's `@compile`. Its optional compiler glue lives in the package's
# `ReactiveKernelsNUTSExamplesReactantExt` weak extension.
#
# BUILD STATE: flat tensor-state SoA + frame round-trip, tensor numerics, and the
# full masked CFG emitter are implemented below. Reactant owns only the final
# `@trace while` wrapper; all source/CFG validation and effects stay here.
# =============================================================================

# -----------------------------------------------------------------------------
# Flat fixed-capacity tensor-state SoA (increment A — verified == native).
# Mirrors benchmark/reactant_handoff/tensorize.jl's field set, re-laid flat into
# the exact tensor layout the traced interpreter carries. Phasepoint pool order:
#   1=init, 2=fwd, 3=bwd, 4..(3+#proposals)=proposals.
# Per phasepoint: pos/mom/dpot/dkin (f4/f5/f8/f10, D-vec) + pot/kin/ham
# (f7/f11/f12, scalar). Trees: log_weight(2) + 6 D-vecs each.
# -----------------------------------------------------------------------------
const _NUTS_PP_VEC = (4, 5, 8, 10)   # f4 pos, f5 mom, f8 dpot, f10 dkin
const _NUTS_PP_SCA = (7, 11, 12)     # f7 pot, f11 kin, f12 ham
@inline _nuts_ppfield(p, i) = getfield(p, Symbol("f", i))

"""
    _nuts_frame_to_tensors(fr) -> NamedTuple

Capture the full mutable NUTS state of a constructed `_NutsFrame` as a flat SoA
of dense `Float64`/`Int` arrays — the tensor layout the Reactant transition
carries. Round-trip with `_nuts_tensors_to_frame!` preserves the native
transition (verified md 2..8 against `compile_nuts_native`).
"""
function _nuts_frame_to_tensors(fr)
    D = length(getfield(fr.init, :f4))
    pps = (fr.init, fr.fwd, fr.bwd, fr.proposals...)
    NPP = length(pps); NT = length(fr.trees)
    pp_pos = zeros(D, NPP); pp_mom = zeros(D, NPP); pp_dpot = zeros(D, NPP); pp_dkin = zeros(D, NPP)
    pp_pot = zeros(NPP); pp_kin = zeros(NPP); pp_ham = zeros(NPP)
    for (j, p) in enumerate(pps)
        pp_pos[:, j] .= _nuts_ppfield(p, 4); pp_mom[:, j] .= _nuts_ppfield(p, 5)
        pp_dpot[:, j] .= _nuts_ppfield(p, 8); pp_dkin[:, j] .= _nuts_ppfield(p, 10)
        pp_pot[j] = _nuts_ppfield(p, 7); pp_kin[j] = _nuts_ppfield(p, 11); pp_ham[j] = _nuts_ppfield(p, 12)
    end
    lw = zeros(2, NT); tb_mom = zeros(D, NT); tb_dh = zeros(D, NT)
    tbf_mom = zeros(D, NT); tbf_dh = zeros(D, NT); sm_b = zeros(D, NT); sm_f = zeros(D, NT)
    for (j, t) in enumerate(fr.trees)
        lw[:, j] .= t.log_weight; tb_mom[:, j] .= t.bwd.mom; tb_dh[:, j] .= t.bwd.dham_dmom
        tbf_mom[:, j] .= t.bwd_fwd.mom; tbf_dh[:, j] .= t.bwd_fwd.dham_dmom
        sm_b[:, j] .= t.summed_mom.bwd; sm_f[:, j] .= t.summed_mom.fwd
    end
    (D = D, NPP = NPP, NT = NT,
     pp_pos = pp_pos, pp_mom = pp_mom, pp_dpot = pp_dpot, pp_dkin = pp_dkin,
     pp_pot = pp_pot, pp_kin = pp_kin, pp_ham = pp_ham,
     lw = lw, tb_mom = tb_mom, tb_dh = tb_dh, tbf_mom = tbf_mom, tbf_dh = tbf_dh, sm_b = sm_b, sm_f = sm_f,
     gofwd = fr.gofwd, may_sample = fr.may_sample, may_continue = fr.may_continue, diverged = fr.diverged,
     n_steps = fr.diag.n_steps, reached_depth = fr.diag.reached_depth,
     acceptance_rate = fr.diag.acceptance_rate, dham = fr.diag.dham)
end

"""
    _nuts_tensors_to_frame!(fr, s) -> fr

Inverse of `_nuts_frame_to_tensors`: write a flat tensor SoA back into a frame.
Used by the gate to seed a frame from tensors and to read Reactant results back.
"""
function _nuts_tensors_to_frame!(fr, s)
    pps = (fr.init, fr.fwd, fr.bwd, fr.proposals...)
    for (j, p) in enumerate(pps)
        _nuts_ppfield(p, 4) .= s.pp_pos[:, j]; _nuts_ppfield(p, 5) .= s.pp_mom[:, j]
        _nuts_ppfield(p, 8) .= s.pp_dpot[:, j]; _nuts_ppfield(p, 10) .= s.pp_dkin[:, j]
        setfield!(p, :f7, s.pp_pot[j]); setfield!(p, :f11, s.pp_kin[j]); setfield!(p, :f12, s.pp_ham[j])
    end
    for (j, t) in enumerate(fr.trees)
        t.log_weight .= s.lw[:, j]; t.bwd.mom .= s.tb_mom[:, j]; t.bwd.dham_dmom .= s.tb_dh[:, j]
        t.bwd_fwd.mom .= s.tbf_mom[:, j]; t.bwd_fwd.dham_dmom .= s.tbf_dh[:, j]
        t.summed_mom.bwd .= s.sm_b[:, j]; t.summed_mom.fwd .= s.sm_f[:, j]
    end
    fr.gofwd = s.gofwd; fr.may_sample = s.may_sample; fr.may_continue = s.may_continue; fr.diverged = s.diverged
    _diag_set_value!(fr.diag, Val(1), s.n_steps); _diag_set_value!(fr.diag, Val(2), s.reached_depth)
    _diag_set_value!(fr.diag, Val(3), s.acceptance_rate); _diag_set_value!(fr.diag, Val(4), s.dham)
    fr
end

# -----------------------------------------------------------------------------
# Tensor numerics (increment B, incremental). Plain traceable Julia over gathered
# phasepoint slices — no mutable structs, no reactive scheduler. Each op reproduces
# the fixture's authored recipe INCLUDING the reactive derived-field recomputation
# the compiler's _ensure_read seam performs at read points.
# -----------------------------------------------------------------------------

"""
    _nuts_tensor_leapfrog(pos, mom, ss, ddpos, ddmom) -> (pos', mom')

One leapfrog step (fixture `leapfrog!` L83-87) over tensor slices, with the
reactive recomputation of the derived momenta made explicit at the read points:
`ddpos(pos)` recomputes `dham_dpos` (the gradient, invalidated by a `pos` write),
`ddmom(mom)` recomputes `dham_dmom` (the metric solve, invalidated by a `mom`
write). The dependency order matches the authored kernel exactly:

    mom -= ½·ss·dham_dpos(pos)     # dham_dpos current from the incoming pos
    pos += ss·dham_dmom(mom)       # dham_dmom RECOMPUTED after the mom write
    mom -= ½·ss·dham_dpos(pos)     # dham_dpos RECOMPUTED after the pos write

`ddpos`/`ddmom` are the euclidean_phasepoint derived recipes (from grad_f and the
metric Cholesky solve); they must be Reactant-traceable in the compiled path.
Verified == the native reactive leapfrog and an independent hand-calc.
"""
@inline function _nuts_tensor_leapfrog(pos, mom, ss, ddpos, ddmom)
    mom = mom .- (0.5 * ss) .* ddpos(pos)     # reads dham_dpos(pos_in); writes mom
    dmom = ddmom(mom)                         # recompute dham_dmom after the mom write
    pos = pos .+ ss .* dmom                   # writes pos
    mom = mom .- (0.5 * ss) .* ddpos(pos)     # recompute dham_dpos after the pos write
    (pos, mom)
end

"""
    _nuts_tensor_ham(pos, mom, potf, ddmom, logdet_metric) -> ham

The euclidean_phasepoint Hamiltonian (fixture L72-77) over tensor slices:
`pot = potf(pos)`, `kin = ½·(logdet(metric) + dot(mom, dham_dmom))` with
`dham_dmom = ddmom(mom) = metric⁻¹·mom`, `ham = pot + kin`. `logdet_metric` is
the shared constant `@node(logdet(chol_metric))`. Used for the divergence check
`dham = init.ham − ep.ham`. Verified == native `ham` across configs.
"""
@inline _nuts_tensor_kin(mom, ddmom, logdet_metric) =
    oftype(logdet_metric, 0.5) * (logdet_metric + dot(mom, ddmom(mom)))
@inline _nuts_tensor_ham(pos, mom, potf, ddmom, logdet_metric) =
    potf(pos) + _nuts_tensor_kin(mom, ddmom, logdet_metric)

"""
    _nuts_tensor_uturn_depth1(summed_fwd, bwd_dhdmom, ep_dhdmom) -> Bool

The depth-1 U-turn no-turn criterion (fixture finish! L221-227): both the
backward and forward projections of the running summed momentum onto the
respective `dham_dmom` must be strictly positive to keep going. Higher depths
add the analogous sub-tree dot pairs (L228-251) — same shape, more terms.
"""
@inline function _nuts_tensor_uturn_depth1(summed_fwd, bwd_dhdmom, ep_dhdmom)
    (dot(summed_fwd, bwd_dhdmom) > zero(eltype(summed_fwd))) &
    (dot(summed_fwd, ep_dhdmom)  > zero(eltype(summed_fwd)))
end

# -----------------------------------------------------------------------------
# Full adaptive transition emitter.
#
# The current public exemplar deliberately admits the specialization proved by
# the acceptance harness: Float64 endpoints, a diagonal Euclidean metric, the
# registered leapfrog binding, and the registered diagnostics callback.  Other
# specializations reject before tracing; there is no host fallback.
# -----------------------------------------------------------------------------

struct _CompiledNutsReactant{F,P,C,R}
    transition::F
    plan::P
    config::C
    RootToken::R
end

@inline _nr_gc(A, i) = A[:, i]
@inline _nr_gv(v, i) = v[i]
@inline _nr_gset(A, i, val) = (w = copy(A); w[:, i] = val; w)
@inline _nr_gsetv(v, i, val) = (w = copy(v); w[i] = val; w)
@inline _nr_sel(active, new, old) = ifelse.(active, new, old)
@inline _nr_vdot(a, b) = vec(sum(a .* b; dims = 1))

function _nr_recompute(st, i, cfg)
    pos = _nr_gc(st.pp_pos, i)
    mom = _nr_gc(st.pp_mom, i)
    dpot = similar(pos)
    pot = cfg.grad_f(dpot, pos)
    dkin = (mom ./ cfg.diagL) ./ cfg.diagL
    kin = oftype(cfg.logdet_metric, 0.5) .* (cfg.logdet_metric .+ _nr_vdot(mom, dkin))
    potv = kin .* 0 .+ pot
    (dpot, dkin, potv, kin, potv .+ kin)
end

function _nr_set_derived(st, i, active, cfg)
    dpot, dkin, pot, kin, ham = _nr_recompute(st, i, cfg)
    merge(st, (
        pp_dpot = _nr_sel(active, _nr_gset(st.pp_dpot, i, dpot), st.pp_dpot),
        pp_dkin = _nr_sel(active, _nr_gset(st.pp_dkin, i, dkin), st.pp_dkin),
        pp_pot = _nr_sel(active, _nr_gsetv(st.pp_pot, i, pot), st.pp_pot),
        pp_kin = _nr_sel(active, _nr_gsetv(st.pp_kin, i, kin), st.pp_kin),
        pp_ham = _nr_sel(active, _nr_gsetv(st.pp_ham, i, ham), st.pp_ham),
    ))
end

function _nr_copyowned(st, dst, src, active)
    merge(st, (
        pp_pos = _nr_sel(active, _nr_gset(st.pp_pos, dst, _nr_gc(st.pp_pos, src)), st.pp_pos),
        pp_mom = _nr_sel(active, _nr_gset(st.pp_mom, dst, _nr_gc(st.pp_mom, src)), st.pp_mom),
        pp_dpot = _nr_sel(active, _nr_gset(st.pp_dpot, dst, _nr_gc(st.pp_dpot, src)), st.pp_dpot),
        pp_dkin = _nr_sel(active, _nr_gset(st.pp_dkin, dst, _nr_gc(st.pp_dkin, src)), st.pp_dkin),
        pp_pot = _nr_sel(active, _nr_gsetv(st.pp_pot, dst, _nr_gv(st.pp_pot, src)), st.pp_pot),
        pp_kin = _nr_sel(active, _nr_gsetv(st.pp_kin, dst, _nr_gv(st.pp_kin, src)), st.pp_kin),
        pp_ham = _nr_sel(active, _nr_gsetv(st.pp_ham, dst, _nr_gv(st.pp_ham, src)), st.pp_ham),
    ))
end

function _nr_swapprop(st, i, j, active)
    pi = _nr_gv(st.prop, i); pj = _nr_gv(st.prop, j)
    merge(st, (prop = _nr_sel(active,
        _nr_gsetv(_nr_gsetv(st.prop, i, pj), j, pi), st.prop),))
end

@inline _nr_overflow(st, slot, active) = merge(st, (
    overflow = _nr_sel(active, _nr_gsetv(st.overflow, slot,
        _nr_gv(st.overflow, slot) .* 0 .+ 1), st.overflow),))

function _nr_draw_dir(st, active, ONE)
    k = st.kd .+ 1
    idx = min.(k, ONE .* length(st.dirs))
    st = _nr_overflow(st, ONE .+ 1, active .& (k .> st.ndirs))
    dec = st.dirs[idx]
    st = merge(st, (kd = _nr_sel(active, k, st.kd),))
    dec, st
end

function _nr_draw_exp(st, consume, ONE)
    k = st.ke .+ 1
    idx = min.(k, ONE .* length(st.exps))
    st = _nr_overflow(st, ONE .+ 2, consume .& (k .> st.nexps))
    ex = st.exps[idx]
    st = merge(st, (ke = _nr_sel(consume, k, st.ke),))
    ex, st
end

@inline _nr_dec(v, hp, active) = _nr_sel(active,
    (w = copy(v); w[hp:hp] = v[hp:hp] .- 1; w), v)
@inline _nr_inc(v, hp, active) = _nr_sel(active,
    (w = copy(v); w[hp:hp] = v[hp:hp] .+ 1; w), v)

function _nr_branch(st, ::Val{kind}, ::Val{pg}, d, e, active, ONE, cs) where {kind,pg}
    if kind === :step
        if pg == 2
            d .<= st.mdcap, st
        elseif pg == 18
            dec, st = _nr_draw_dir(st, active, ONE)
            dec .!= 0, st
        elseif pg == 17
            d .> 1, st
        elseif pg in (15, 24, 12)
            st.gofwd .!= 0, st
        elseif pg == 9
            st.may_sample .!= 0, st
        elseif pg == 5
            st.may_continue .!= 0, st
        elseif pg == 7
            diff = st.lw1[d] .- st.lw2[d]
            # The authored `diff > 0 || -randexp(rng) < diff` consumes whenever
            # `diff > 0` is false, including an unordered NaN from `-Inf - -Inf`.
            consume = active .& .!(diff .> 0)
            ex, st = _nr_draw_exp(st, consume, ONE)
            ifelse.(diff .> 0, ONE .!= 0, (-ex) .< diff), st
        else
            active .& false, st
        end
    elseif kind === :finish
        pg in (14, 7) ? (d .== 1, st) :
        pg == 10 ? (st.may_continue .!= 0, st) : (active .& false, st)
    else
        if pg == 15
            d .== 1, st
        elseif pg == 5
            active .& false, st # stats_f is validated effectful
        elseif pg == 3
            st.diverged .!= 0, st
        elseif pg == 13
            st.may_continue .!= 0, st
        elseif pg == 9
            st.may_sample .!= 0, st
        elseif pg == 8
            diff = st.lw1[d .- 1] .- st.lw1[d]
            consume = active .& .!(diff .> 0)
            ex, st = _nr_draw_exp(st, consume, ONE)
            ifelse.(diff .> 0, ONE .!= 0, (-ex) .< diff), st
        else
            active .& false, st
        end
    end
end

@inline function _nr_callargs(st, ::Val{kind}, e, d, ONE) where {kind}
    kind === :step ? (ifelse.(st.gofwd .!= 0, ONE .+ 1, ONE .+ 2), d) :
    kind === :finish ? (e, d) : (e, d .- 1)
end

function _nr_effects(st, ::Val{kind}, ::Val{pg}, e, d, cs, NP, ONE, active, cfg) where {kind,pg}
    if kind === :step
        if pg == 26
            # `zero.(dham)`, unlike `dham .* 0`, is a true authored reset even when the
            # prior transition ended with the finite-or-negative-infinity sentinel.
            reset_dham = zero.(st.dham)
            reset_diverged = ifelse.(reset_dham .>= st.min_dham, ONE .* 0, ONE)
            st = merge(st, (gofwd = _nr_sel(active, ONE, st.gofwd),
                may_sample = _nr_sel(active, ONE, st.may_sample),
                may_continue = _nr_sel(active, ONE, st.may_continue),
                diverged = _nr_sel(active, reset_diverged, st.diverged),
                dham = _nr_sel(active, reset_dham, st.dham),
                n_steps = _nr_sel(active, st.n_steps .* 0, st.n_steps),
                reached_depth = _nr_sel(active, st.reached_depth .* 0, st.reached_depth),
                acc = _nr_sel(active, st.acc .* 0, st.acc)))
            st = _nr_copyowned(st, ONE .+ 1, ONE, active)
            st = _nr_copyowned(st, ONE .+ 2, ONE, active)
            st = _nr_copyowned(st, _nr_gv(st.prop, ONE), ONE, active)
        elseif pg == 25
            st = _nr_copyowned(st, _nr_gv(st.prop, ONE .* 0 .+ NP), ONE, active)
        elseif pg == 22
            st = merge(st, (pp_mom = _nr_sel(active,
                    _nr_gset(st.pp_mom, ONE .+ 2, -_nr_gc(st.pp_mom, ONE .+ 2)), st.pp_mom),
                pp_dkin = _nr_sel(active,
                    _nr_gset(st.pp_dkin, ONE .+ 2, -_nr_gc(st.pp_dkin, ONE .+ 2)), st.pp_dkin)))
        elseif pg == 23
            st = merge(st, (pp_mom = _nr_sel(active,
                    _nr_gset(st.pp_mom, ONE .+ 1, -_nr_gc(st.pp_mom, ONE .+ 1)), st.pp_mom),
                pp_dkin = _nr_sel(active,
                    _nr_gset(st.pp_dkin, ONE .+ 1, -_nr_gc(st.pp_dkin, ONE .+ 1)), st.pp_dkin)))
        elseif pg == 21
            st = merge(st, (lw1 = _nr_sel(active,
                _nr_gsetv(st.lw1, ONE, st.dham .* 0), st.lw1),))
        elseif pg == 20
            st = merge(st, (dep = _nr_sel(active, _nr_gsetv(st.dep, cs, ONE), st.dep),))
        elseif pg == 19
            st = merge(st, (reached_depth = _nr_sel(active, d, st.reached_depth),))
        elseif pg == 16
            st = merge(st, (gofwd = _nr_sel(active, ONE .- st.gofwd, st.gofwd),))
        elseif pg == 14
            st = merge(st, (tb_mom = _nr_sel(active,
                    _nr_gset(st.tb_mom, d, -_nr_gc(st.pp_mom, ONE .+ 1)), st.tb_mom),
                tb_dh = _nr_sel(active,
                    _nr_gset(st.tb_dh, d, -_nr_gc(st.pp_dkin, ONE .+ 1)), st.tb_dh),
                sm_f = _nr_sel(active, _nr_gset(st.sm_f, d, -_nr_gc(st.sm_f, d)), st.sm_f)))
        elseif pg == 13
            st = merge(st, (tb_mom = _nr_sel(active,
                    _nr_gset(st.tb_mom, d, -_nr_gc(st.pp_mom, ONE .+ 2)), st.tb_mom),
                tb_dh = _nr_sel(active,
                    _nr_gset(st.tb_dh, d, -_nr_gc(st.pp_dkin, ONE .+ 2)), st.tb_dh),
                sm_f = _nr_sel(active, _nr_gset(st.sm_f, d, -_nr_gc(st.sm_f, d)), st.sm_f)))
        elseif pg == 6
            st = _nr_swapprop(st, d, ONE .* 0 .+ NP, active)
        elseif pg == 3
            st = merge(st, (dep = _nr_sel(active, _nr_gsetv(st.dep, cs, d .+ 1), st.dep),))
        elseif pg == 1
            st = _nr_copyowned(st, ONE, _nr_gv(st.prop, ONE .* 0 .+ NP), active)
        end
    elseif kind === :finish
        if pg == 15
            st = merge(st, (lw2 = _nr_sel(active, _nr_gsetv(st.lw2, d, st.lw1[d]), st.lw2),))
        elseif pg == 12
            st = merge(st, (tb_mom = _nr_sel(active,
                    _nr_gset(st.tb_mom, d .+ 1, _nr_gc(st.pp_mom, e)), st.tb_mom),
                tb_dh = _nr_sel(active,
                    _nr_gset(st.tb_dh, d .+ 1, _nr_gc(st.pp_dkin, e)), st.tb_dh)))
        elseif pg == 13
            st = merge(st, (tb_mom = _nr_sel(active,
                    _nr_gset(st.tb_mom, d .+ 1, _nr_gc(st.tb_mom, d)), st.tb_mom),
                tb_dh = _nr_sel(active,
                    _nr_gset(st.tb_dh, d .+ 1, _nr_gc(st.tb_dh, d)), st.tb_dh),
                tbf_mom = _nr_sel(active,
                    _nr_gset(st.tbf_mom, d, _nr_gc(st.pp_mom, e)), st.tbf_mom),
                tbf_dh = _nr_sel(active,
                    _nr_gset(st.tbf_dh, d, _nr_gc(st.pp_dkin, e)), st.tbf_dh),
                sm_b = _nr_sel(active, _nr_gset(st.sm_b, d, _nr_gc(st.sm_f, d)), st.sm_b)))
        elseif pg == 9
            st = merge(st, (may_sample = _nr_sel(active, st.may_sample .* 0, st.may_sample),))
        elseif pg == 8
            st = merge(st, (lw1 = _nr_sel(active,
                _nr_gsetv(st.lw1, d .+ 1, LogExpFunctions.logaddexp.(st.lw1[d], st.lw2[d])), st.lw1),))
        elseif pg == 6
            st = merge(st, (sm_f = _nr_sel(active,
                _nr_gset(st.sm_f, d .+ 1, _nr_gc(st.sm_b, d) .+ _nr_gc(st.sm_f, d)), st.sm_f),))
        elseif pg == 5
            q1 = _nr_vdot(_nr_gc(st.sm_f, d .+ 1), _nr_gc(st.tb_dh, d .+ 1))
            q2 = _nr_vdot(_nr_gc(st.sm_f, d .+ 1), _nr_gc(st.pp_dkin, e))
            q3 = _nr_vdot(_nr_gc(st.sm_b, d) .+ _nr_gc(st.tb_mom, d), _nr_gc(st.tb_dh, d .+ 1))
            q4 = _nr_vdot(_nr_gc(st.sm_b, d) .+ _nr_gc(st.tb_mom, d), _nr_gc(st.tb_dh, d))
            q5 = _nr_vdot(_nr_gc(st.tbf_mom, d) .+ _nr_gc(st.sm_f, d .+ 1), _nr_gc(st.tbf_dh, d))
            q6 = _nr_vdot(_nr_gc(st.tbf_mom, d) .+ _nr_gc(st.sm_f, d .+ 1), _nr_gc(st.pp_dkin, e))
            criterion = ifelse.((q1 .> 0) .& (q2 .> 0) .& (q3 .> 0) .&
                (q4 .> 0) .& (q5 .> 0) .& (q6 .> 0), ONE, ONE .* 0)
            st = merge(st, (criterion = _nr_sel(active, criterion, st.criterion),))
        elseif pg == 4
            st = merge(st, (may_continue = _nr_sel(active, st.criterion, st.may_continue),))
        elseif pg == 3
            st = merge(st, (sm_f = _nr_sel(active,
                _nr_gset(st.sm_f, d .+ 1, _nr_gc(st.tb_mom, d .+ 1) .+ _nr_gc(st.pp_mom, e)), st.sm_f),))
        elseif pg == 2
            st = merge(st, (bd = _nr_sel(active,
                    _nr_vdot(_nr_gc(st.sm_f, d .+ 1), _nr_gc(st.tb_dh, d .+ 1)), st.bd),
                fd = _nr_sel(active,
                    _nr_vdot(_nr_gc(st.sm_f, d .+ 1), _nr_gc(st.pp_dkin, e)), st.fd)))
        elseif pg == 1
            keep = ifelse.((st.bd .> 0) .& (st.fd .> 0), ONE, ONE .* 0)
            st = merge(st, (may_continue = _nr_sel(active, keep, st.may_continue),))
        end
    else
        if pg == 6
            pos = _nr_gc(st.pp_pos, e); mom = _nr_gc(st.pp_mom, e)
            grad = similar(pos); cfg.grad_f(grad, pos)
            mom = mom .- (oftype(cfg.stepsize, 0.5) * cfg.stepsize) .* grad
            pos = pos .+ cfg.stepsize .* ((mom ./ cfg.diagL) ./ cfg.diagL)
            cfg.grad_f(grad, pos)
            mom = mom .- (oftype(cfg.stepsize, 0.5) * cfg.stepsize) .* grad
            st = merge(st, (pp_pos = _nr_sel(active, _nr_gset(st.pp_pos, e, pos), st.pp_pos),
                pp_mom = _nr_sel(active, _nr_gset(st.pp_mom, e, mom), st.pp_mom)))
            st = _nr_set_derived(st, e, active, cfg)
            raw = _nr_gv(st.pp_ham, ONE) .- _nr_gv(st.pp_ham, e)
            finite = ifelse.((raw .- raw) .== 0, raw, zero.(st.dham) .+ cfg.neginf)
            st = merge(st, (dham = _nr_sel(active, finite, st.dham),
                diverged = _nr_sel(active, ifelse.(finite .>= st.min_dham, ONE .* 0, ONE), st.diverged)))
        elseif pg == 4
            ns = st.n_steps .+ 1
            a = (1 .- 1 ./ ns) .* st.acc .+ (1 ./ ns) .* ifelse.(st.dham .>= 0, 1.0, exp.(st.dham))
            st = merge(st, (n_steps = _nr_sel(active, ns, st.n_steps), acc = _nr_sel(active, a, st.acc)))
        elseif pg == 2
            st = merge(st, (may_continue = _nr_sel(active, st.may_continue .* 0, st.may_continue),))
        elseif pg == 1
            st = merge(st, (lw1 = _nr_sel(active, _nr_gsetv(st.lw1, ONE, st.dham), st.lw1),))
            st = _nr_copyowned(st, _nr_gv(st.prop, ONE), e, active)
        elseif pg == 11
            st = _nr_swapprop(st, d .- 1, d, active)
        elseif pg == 12
            st = merge(st, (may_sample = _nr_sel(active, st.may_sample .* 0, st.may_sample),))
        elseif pg == 7
            st = _nr_swapprop(st, d .- 1, d, active)
        end
    end
    st
end

function _nr_block(st, ::Val{mid}, ::Val{kind}, ::Val{pg}, ::Val{tt},
        ::Val{ta}, ::Val{tb}, ::Val{cm}, ::Val{ce}, ::Val{resume},
        ::Val{mp}, ::Val{mpc}, m, p, e, d, cs, ONE, cfg) where
        {mid,kind,pg,tt,ta,tb,cm,ce,resume,mp,mpc}
    active = (m .== mid) .& (p .== pg)
    st = _nr_effects(st, Val(kind), Val(pg), e, d, cs, length(st.prop), ONE, active, cfg)
    if tt === :ret
        st = merge(st, (fsp = _nr_dec(st.fsp, mp, active), csp = _nr_sel(active, st.csp .- 1, st.csp)))
    elseif tt === :goto
        if ta == 0
            st = merge(st, (fsp = _nr_dec(st.fsp, mp, active), csp = _nr_sel(active, st.csp .- 1, st.csp)))
        else
            st = merge(st, (pcs = _nr_sel(active, _nr_gsetv(st.pcs, cs, ONE .* 0 .+ ta), st.pcs),))
        end
    elseif tt === :branch
        cnd, st = _nr_branch(st, Val(kind), Val(pg), d, e, active, ONE, cs)
        np = ifelse.(cnd, ONE .* 0 .+ ta, ONE .* 0 .+ tb)
        ispop = active .& (np .== 0)
        st = merge(st, (fsp = _nr_dec(st.fsp, mp, ispop), csp = _nr_sel(ispop, st.csp .- 1, st.csp)))
        st = merge(st, (pcs = _nr_sel(active .& (np .!= 0), _nr_gsetv(st.pcs, cs, np), st.pcs),))
    else
        calle, calld = _nr_callargs(st, Val(kind), e, d, ONE)
        st = merge(st, (pcs = _nr_sel(active, _nr_gsetv(st.pcs, cs, ONE .* 0 .+ resume), st.pcs),
            fsp = _nr_inc(st.fsp, mpc, active)))
        wanted = st.csp .+ 1
        over = active .& (wanted .> length(st.mids))
        st = _nr_overflow(st, ONE, over)
        ncsp = ifelse.(over, st.csp, _nr_sel(active, wanted, st.csp))
        st = merge(st, (csp = ncsp,
            mids = _nr_sel(active .& .!over, _nr_gsetv(st.mids, ncsp, ONE .* 0 .+ cm), st.mids),
            ep = _nr_sel(active .& .!over, _nr_gsetv(st.ep, ncsp, calle), st.ep),
            dep = _nr_sel(active .& .!over, _nr_gsetv(st.dep, ncsp, calld), st.dep),
            pcs = _nr_sel(active .& .!over, _nr_gsetv(st.pcs, ncsp, ONE .* 0 .+ ce), st.pcs)))
    end
    st
end

function _nr_refresh(st, cfg)
    ONE = st.csp .* 0 .+ 1
    mom = reshape(cfg.diagL .* st.momentum, :, 1)
    mids = _nr_gsetv(st.mids .* 0, ONE, ONE .* 0 .+ cfg.root_mid)
    pcs = _nr_gsetv(st.pcs .* 0, ONE, ONE .* 0 .+ cfg.root_entry)
    fsp = _nr_gsetv(st.fsp .* 0, ONE .* 0 .+ cfg.root_pos, ONE)
    st = merge(st, (pp_mom = _nr_gset(st.pp_mom, ONE, mom),
        mids, pcs, ep = st.ep .* 0, dep = st.dep .* 0, csp = ONE, fsp,
        criterion = st.criterion .* 0 .+ 1, kd = st.kd .* 0,
        ke = st.ke .* 0, overflow = st.overflow .* 0, _step = st.csp .* 0))
    _nr_set_derived(st, ONE, ONE .!= 0, cfg)
end

function _nr_finish(st)
    hit = (st.csp .>= 1) .& (st._step .>= st.stepcap)
    _nr_overflow(st, st.csp .* 0 .+ 4, hit)
end

function _nr_plan(skel)
    generic = _control_program(skel; root_name=:step!)
    named = Dict(name => mid for (mid, name) in generic.names)
    Set(keys(named)) == Set((:step!, :finish!, :start!)) || throw(ArgumentError(
        "compile_nuts_reactant supports the captured step!/finish!/start! NUTS CFG; got $(sort!(collect(keys(named))))"))
    methods = collect(generic.methods)
    midpos = generic.midpos
    entries = generic.entries
    blocks = NamedTuple[]
    kinds = Dict(named[:step!] => :step, named[:finish!] => :finish, named[:start!] => :start)
    expected = Dict(:step => 26, :finish => 15, :start => 15)
    for m in methods
        kind = kinds[m]
        count(b -> b.mid == m, generic.blocks) == expected[kind] || throw(ArgumentError(
            "compile_nuts_reactant rejects drifted $kind CFG: expected $(expected[kind]) blocks, got " *
            "$(count(b -> b.mid == m, generic.blocks))"))
    end
    for b in generic.blocks
        kind = kinds[b.mid]
        tt = b.term === :return ? :ret : b.term
        push!(blocks, (; mid=b.mid, kind, pc=b.pc, tt,
            ta=b.then_pc, tb=b.else_pc, cm=b.callee_mid,
            ce=b.callee_entry, resume=b.resume_pc,
            mp=b.midpos, mpc=b.callee_midpos))
        if tt === :call && b.callee_mid == 0
            throw(ArgumentError("compile_nuts_reactant rejects an opaque CFG call"))
        end
    end
    (; blocks=Tuple(blocks), methods=Tuple(methods), midpos, entries,
       root_mid=named[:step!], root_entry=entries[named[:step!]])
end

function _nr_config(pf, frame)
    pos = _nuts_ppfield(frame.init, 4)
    eltype(pos) === Float64 || throw(ArgumentError(
        "compile_nuts_reactant currently supports Float64 endpoints; got $(eltype(pos))"))
    metric = _canon_slot(frame.shared, kernel_plan_named_slot_val(pf.plan, Val(:metric)))
    size(metric, 1) == size(metric, 2) == length(pos) || throw(DimensionMismatch(
        "compile_nuts_reactant metric/position dimensions disagree"))
    all(i == j || iszero(metric[i,j]) for i in axes(metric,1), j in axes(metric,2)) ||
        throw(ArgumentError("compile_nuts_reactant currently supports diagonal Euclidean metrics"))
    diagmetric = [metric[i,i] for i in axes(metric,1)]
    all(>(0), diagmetric) || throw(ArgumentError("compile_nuts_reactant requires a positive diagonal metric"))
    kw = prepared_callable_kwargs(frame.step_f)
    isequal(prepared_callable_token(nuts_frame_step(frame)), kernel_prepared_token(pf)) ||
        throw(ArgumentError("compile_nuts_reactant requires the factory's captured leapfrog binding"))
    keys(kw) == (:stepsize,) || throw(ArgumentError(
        "compile_nuts_reactant requires exactly the bound leapfrog `stepsize` keyword"))
    kw.stepsize isa Float64 || throw(ArgumentError("compile_nuts_reactant requires a Float64 stepsize"))
    binding = nuts_frame_stats(frame)
    stats_binding_registration(binding) === nothing && throw(ArgumentError(
        "compile_nuts_reactant requires the registered diagnostics callback"))
    Set(stats_binding_produced(binding)) == Set((1,3)) || throw(ArgumentError(
        "compile_nuts_reactant supports the n_steps/acceptance_rate diagnostics callback"))
    grad_f = _canon_slot(frame.shared, kernel_plan_named_slot_val(pf.plan, Val(:grad_f)))
    chol = LinearAlgebra.cholesky(Matrix(metric))
    (; stepsize=kw.stepsize, grad_f,
       diagL=sqrt.(diagmetric), logdet_metric=LinearAlgebra.logdet(chol), neginf=-Inf)
end

function nuts_reactant_bundle(momentum, dirs, exps, max_depth::Integer)
    md = Int(max_depth); md >= 0 || throw(ArgumentError("max_depth must be nonnegative"))
    eltype(momentum) === Float64 || throw(ArgumentError("momentum bundle must contain Float64 values"))
    eltype(exps) === Float64 || throw(ArgumentError("exponential bundle must contain Float64 values"))
    all(isfinite, momentum) || throw(ArgumentError("momentum bundle values must be finite"))
    all(x -> x isa Bool || (x isa Integer && x in (0, 1)), dirs) ||
        throw(ArgumentError("direction bundle values must be Bool or 0/1 integers"))
    all(x -> isfinite(x) && x >= 0, exps) ||
        throw(ArgumentError("exponential bundle values must be finite and nonnegative"))
    md < 8*sizeof(Int)-1 || throw(OverflowError("max_depth is too large for the fixed exponential bundle"))
    dircap = max(md, 1); expcap = max(1, 1 << md)
    length(dirs) <= dircap || throw(DimensionMismatch("direction bundle exceeds max_depth capacity"))
    length(exps) <= expcap || throw(DimensionMismatch("exponential bundle exceeds 2^max_depth capacity"))
    d = zeros(Int, dircap); d[1:length(dirs)] .= Int.(dirs)
    e = ones(eltype(momentum), expcap); e[1:length(exps)] .= exps
    (; momentum=collect(momentum), dirs=d, exps=e,
       ndirs=[length(dirs)], nexps=[length(exps)])
end

function nuts_reactant_state(C::_CompiledNutsReactant, fr, bundle)
    s = _nuts_frame_to_tensors(fr); md = fr.max_depth
    md >= 0 || throw(ArgumentError("max_depth must be nonnegative"))
    length(bundle.momentum) == s.D || throw(DimensionMismatch("momentum bundle dimension mismatch"))
    cap = Base.checked_mul(4, Base.checked_add(md, 2))
    md < 8*sizeof(Int)-1 || throw(OverflowError("max_depth is too large for the fixed control budget"))
    leaves = 1 << md
    length(bundle.dirs) == max(md, 1) || throw(DimensionMismatch("direction bundle capacity mismatch"))
    length(bundle.exps) == max(leaves, 1) || throw(DimensionMismatch("exponential bundle capacity mismatch"))
    length(bundle.ndirs) == length(bundle.nexps) == 1 ||
        throw(DimensionMismatch("bundle draw counts must be length-one arrays"))
    0 <= only(bundle.ndirs) <= length(bundle.dirs) || throw(DimensionMismatch("invalid direction draw count"))
    0 <= only(bundle.nexps) <= length(bundle.exps) || throw(DimensionMismatch("invalid exponential draw count"))
    stepcap = Base.checked_mul(512, max(leaves, 1))
    mids=zeros(Int,cap); pcs=zeros(Int,cap); ep=zeros(Int,cap); dep=zeros(Int,cap); fsp=zeros(Int,length(C.plan.methods))
    rootpos=C.plan.midpos[C.plan.root_mid]; fsp[rootpos]=1; mids[1]=C.plan.root_mid; pcs[1]=C.plan.root_entry
    (pp_pos=s.pp_pos, pp_mom=s.pp_mom, pp_dpot=s.pp_dpot, pp_dkin=s.pp_dkin,
     pp_pot=s.pp_pot, pp_kin=s.pp_kin, pp_ham=s.pp_ham,
     lw1=copy(s.lw[1,:]), lw2=copy(s.lw[2,:]), tb_mom=s.tb_mom, tb_dh=s.tb_dh,
     tbf_mom=s.tbf_mom, tbf_dh=s.tbf_dh, sm_b=s.sm_b, sm_f=s.sm_f,
     prop=collect(4:(3+length(fr.proposals))), gofwd=[Int(s.gofwd)], may_sample=[Int(s.may_sample)],
     may_continue=[Int(s.may_continue)], diverged=[Int(s.diverged)], criterion=[1],
     mdcap=[md], n_steps=[s.n_steps], reached_depth=[s.reached_depth], acc=[s.acceptance_rate],
     dham=[s.dham], min_dham=[fr.min_dham], bd=[zero(s.dham)], fd=[zero(s.dham)], mids, pcs, ep, dep, csp=[1], fsp,
     kd=[0], ke=[0], momentum=bundle.momentum, dirs=bundle.dirs, exps=bundle.exps,
     ndirs=bundle.ndirs, nexps=bundle.nexps, overflow=zeros(Int,4), _step=[0], stepcap=[stepcap])
end

function nuts_reactant_rebundle(st, bundle)
    length(st.momentum) == length(bundle.momentum) || throw(DimensionMismatch("momentum bundle shape changed"))
    length(st.dirs) == length(bundle.dirs) || throw(DimensionMismatch("direction bundle shape changed"))
    length(st.exps) == length(bundle.exps) || throw(DimensionMismatch("exponential bundle shape changed"))
    merge(st, (momentum=bundle.momentum, dirs=bundle.dirs, exps=bundle.exps,
               ndirs=bundle.ndirs, nexps=bundle.nexps))
end

function nuts_reactant_compile(C::_CompiledNutsReactant, state; sync::Bool=false)
    ext = Base.get_extension(@__MODULE__, :ReactiveKernelsNUTSExamplesReactantExt)
    ext === nothing && throw(ArgumentError("nuts_reactant_compile requires loading Reactant"))
    ext.compile_nuts_reactant_executable(C.transition, state; sync)
end

function nuts_reactant_writeback!(fr, st)
    A(x) = x isa AbstractArray ? Array(x) : x
    prop = A(st.prop)
    pools = (fr.init, fr.fwd, fr.bwd)
    for (j,p) in enumerate(pools)
        _nuts_ppfield(p,4) .= A(st.pp_pos)[:,j]; _nuts_ppfield(p,5) .= A(st.pp_mom)[:,j]
        _nuts_ppfield(p,8) .= A(st.pp_dpot)[:,j]; _nuts_ppfield(p,10) .= A(st.pp_dkin)[:,j]
        setfield!(p,:f7,A(st.pp_pot)[j]); setfield!(p,:f11,A(st.pp_kin)[j]); setfield!(p,:f12,A(st.pp_ham)[j])
    end
    for (j,p) in enumerate(fr.proposals)
        k=prop[j]
        _nuts_ppfield(p,4) .= A(st.pp_pos)[:,k]; _nuts_ppfield(p,5) .= A(st.pp_mom)[:,k]
        _nuts_ppfield(p,8) .= A(st.pp_dpot)[:,k]; _nuts_ppfield(p,10) .= A(st.pp_dkin)[:,k]
        setfield!(p,:f7,A(st.pp_pot)[k]); setfield!(p,:f11,A(st.pp_kin)[k]); setfield!(p,:f12,A(st.pp_ham)[k])
    end
    for (j,t) in enumerate(fr.trees)
        t.log_weight[1]=A(st.lw1)[j]; t.log_weight[2]=A(st.lw2)[j]
        t.bwd.mom .= A(st.tb_mom)[:,j]; t.bwd.dham_dmom .= A(st.tb_dh)[:,j]
        t.bwd_fwd.mom .= A(st.tbf_mom)[:,j]; t.bwd_fwd.dham_dmom .= A(st.tbf_dh)[:,j]
        t.summed_mom.bwd .= A(st.sm_b)[:,j]; t.summed_mom.fwd .= A(st.sm_f)[:,j]
    end
    fr.gofwd=Bool(A(st.gofwd)[1]); fr.may_sample=Bool(A(st.may_sample)[1]);
    fr.may_continue=Bool(A(st.may_continue)[1]); fr.diverged=Bool(A(st.diverged)[1])
    _diag_set_value!(fr.diag,Val(1),A(st.n_steps)[1]); _diag_set_value!(fr.diag,Val(2),A(st.reached_depth)[1])
    _diag_set_value!(fr.diag,Val(3),A(st.acc)[1]); _diag_set_value!(fr.diag,Val(4),A(st.dham)[1])
    fr.derived_pending=UInt(1); fr.derived_committed=UInt(1)
    fr
end

function compile_nuts_reactant(pf::_PreparedFactory, skel, refresh_skel, nuts_root_skel,
                               frame::_NutsFrame; root_name::Symbol=:step!)
    root_name === :step! || throw(ArgumentError("compile_nuts_reactant supports the authored step! root"))
    native = compile_nuts(pf, skel, refresh_skel, nuts_root_skel, frame; root_name)
    plan = _nr_plan(skel)
    cfg = merge(_nr_config(pf, frame), (root_mid=plan.root_mid,
        root_entry=plan.root_entry, root_pos=plan.midpos[plan.root_mid]))
    ext = Base.get_extension(@__MODULE__, :ReactiveKernelsNUTSExamplesReactantExt)
    ext === nothing && throw(ArgumentError("compile_nuts_reactant requires loading Reactant"))
    transition = ext.compile_nuts_reactant_transition(plan, cfg)
    _CompiledNutsReactant(transition, plan, cfg, native.RootToken)
end
