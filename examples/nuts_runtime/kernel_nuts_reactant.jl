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
# gate's `@compile`. Any Reactant make_tracer/traced_type glue lives in
# ext/ReactiveKernelsReactantExt.jl.
#
# BUILD STATE (incremental): [DONE] flat tensor-state SoA + frame round-trip
# (== native, md 2..8). [NEXT] traced-while masked CFG interpreter.
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
