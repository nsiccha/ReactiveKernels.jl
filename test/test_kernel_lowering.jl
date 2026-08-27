# Increment-3 LOWERING gate — the fused leaf schedule computed over the EXACT selected Plan, keyed by
# graph identity (Value/Recipe id + owner/view path), recipe-atomic produces, an explicit entry-current
# contract. Method-local ordered kills/produces/currentness; the physical layout/codegen seam (syntax's
# factory: `_OwnerState` + path→slot map + immutable plan) is integrated separately.

using ReactiveKernels
using Test
const RK = ReactiveKernels

module _FixLower
    include(joinpath(@__DIR__, "..", "benchmark", "nuts_kernel_authoring_fixture.jl"))
end
const _LwPP = _FixLower.euclidean_phasepoint     # the phase-point endpoint (a stateless KernelSpec)
const _LwLF = _FixLower.leapfrog!                # the Mode-2 free integrator
const _LwDA = _FixLower.dual_averaging_state     # accumulator object (fit!)
const _LwWV = _FixLower.welford_var              # accumulator object (step!)
_spec(k) = k isa RK.KernelSpec ? k : RK.kernel_spec(k)

# Build the endpoint plan graph at a path (SCAFFOLDING plan; integration consumes the factory's immutable
# plan-key/template identity — this helper is not that authority).
function _pg(spec::RK.KernelSpec; path = (), have = spec.have_names)
    RK._l_plan_graph(RK._l_endpoint_plan(spec; have = have), path; name_of = RK._l_name_of(spec))
end
_field(pg, key) = get(pg.name_of, key.id, key.id)   # reporting label of a schedule step

@testset "lowering — plan-derived recompute graph keyed by Value/Recipe identity" begin
    pg = _pg(_LwPP)
    posid = RK._l_field_id(_LwPP, :pos); momid = RK._l_field_id(_LwPP, :mom)
    gid = RK._l_field_id(_LwPP, :dham_dpos); vid = RK._l_field_id(_LwPP, :dham_dmom)
    cholid = RK._l_field_id(_LwPP, :chol_metric); metricid = RK._l_field_id(_LwPP, :metric)
    @test posid in pg.sources && momid in pg.sources && metricid in pg.sources
    @test !haskey(pg.producer, posid) && !haskey(pg.producer, momid)
    @test haskey(pg.producer, gid) && haskey(pg.producer, vid)
    @test gid in pg.dependents[posid]
    @test vid in pg.dependents[momid]
    @test !(cholid in pg.dependents[posid]) && !(cholid in pg.dependents[momid])
    @test cholid in pg.dependents[metricid]
end

@testset "lowering — producer is EXACTLY plan.producer (the selected owner, not source order)" begin
    pg = _pg(_LwPP)
    potid = RK._l_field_id(_LwPP, :pot)
    # the faithful phasepoint has BOTH pot_f(pos) and grad_f(pos) as pot producers; the OWNER is the
    # plan's selected producer — assert EQUALITY to plan.producer, not mere membership in plan.recipes.
    @test haskey(pg.producer, potid)
    @test pg.producer[potid] == pg.plan.producer[potid].id     # the plan's chosen owner recipe id
    # every derived value's owner is exactly its plan.producer entry (no last-writer drift)
    for (cid, rid) in pg.producer
        @test rid == pg.plan.producer[cid].id
    end
end

@testset "lowering — COLLATERAL outputs follow plan.producer (non-owner recipe never blesses)" begin
    # a synthetic plan where TWO selected recipes emit the same canonical Value y: R1 a->(x,y) is
    # selected for x, R2 a->(y,z) for z. plan.producer picks ONE owner for y; the other's y is discarded
    # collateral (core lower() gensyms/discards duplicates). Executing the non-owner must NOT bless y.
    g = RK.Graph()
    a = RK.value!(g, :a, Float64)
    x = RK.value!(g, :x, Float64); y = RK.value!(g, :y, Float64); z = RK.value!(g, :z, Float64)
    r1 = RK.add!(g; inputs = (a,), outputs = (x, y), op = v -> (v, v), cost = 1.0)
    r2 = RK.add!(g; inputs = (a,), outputs = (y, z), op = v -> (v, v), cost = 1.0)
    p = RK.plan(g; have = (a,), want = (x, z))
    pg = RK._l_plan_graph(p, ())
    ycid = RK.canon_id(g, y.id); zcid = RK.canon_id(g, z.id)
    owner = p.producer[ycid].id                                # the SELECTED owner of y
    nonowner = owner == r1.id ? r2.id : r1.id
    @test pg.producer[ycid] == owner                           # producer = plan's chosen owner (equality)
    @test ycid in pg.recipe_owned[owner]                       # owner owns y
    @test !(ycid in pg.recipe_owned[nonowner])                 # non-owner: y is collateral, NOT owned
    # executing the non-owner recipe (via its OWNED output) never blesses/redirects y
    sched = RK._LSchedStep[]
    RK._l_ensure_current!(sched, Set{Int}(), pg, first(pg.recipe_owned[nonowner]), Set{Int}())
    @test all(s -> s.kind !== :exec || !(ycid in s.outputs) || s.recipe === owner, sched)
end

@testset "lowering — leapfrog! fused leaf schedule (ONE gradient RECIPE per leaf, atomic)" begin
    ir = RK.method_irs(_LwLF)[1]
    pg = _pg(_LwPP)
    entry = RK._l_all_produced(pg)                             # factory contract: endpoint fully current
    sched = RK.lower_leaf_schedule(ir, _LwPP, pg; entry_current = entry)
    # exactly ONE execution of the GRADIENT recipe (the pgrad producing dpot_dpos) per leaf — the count
    # is over the RECIPE, so a multi-output grad recipe (pot AND dpot_dpos) cannot be double-counted.
    @test RK._l_recipe_exec_count(sched, _LwPP, pg, :dpot_dpos) == 1
    gradrid = pg.producer[RK._l_field_id(_LwPP, :dpot_dpos)]
    @test length(pg.recipe_owned[gradrid]) >= 2               # genuinely shared multi-output recipe
    execs_of_grad = [s for s in sched if s.kind === :exec && s.recipe === gradrid]
    @test length(execs_of_grad) == 1
    @test length(execs_of_grad[1].outputs) >= 2               # atomic: all outputs produced together
    @test RK._l_recipe_exec_count(sched, _LwPP, pg, :dkin_dmom) == 1
    writes = [_field(pg, s.key) for s in sched if s.kind === :write]
    @test writes == [:mom, :pos, :mom]
    gexec = findfirst(s -> s.kind === :exec && s.recipe === gradrid, sched)
    poswrite = findlast(s -> s.kind === :write && _field(pg, s.key) === :pos, sched)
    lastmom = findlast(s -> s.kind === :write && _field(pg, s.key) === :mom, sched)
    @test poswrite < gexec < lastmom
end

@testset "lowering — entry-currentness is CONSUMED (proven), not assumed" begin
    ir = RK.method_irs(_LwLF)[1]
    pg = _pg(_LwPP)
    gradrid = pg.producer[RK._l_field_id(_LwPP, :dpot_dpos)]
    full = RK._l_all_produced(pg)
    s_full = RK.lower_leaf_schedule(ir, _LwPP, pg; entry_current = full)
    firstread = findfirst(s -> s.kind === :read, s_full)
    @test !any(i -> s_full[i].kind === :exec && s_full[i].recipe === gradrid, 1:firstread)
    # if NOTHING is current at entry, the SAME first read (dham_dpos) triggers the full gradient
    # recompute chain (dham_dpos ← dpot_dpos ← grad_f) — proving the schedule CONSUMES the entry
    # contract rather than blanket-assuming currentness.
    s_stale = RK.lower_leaf_schedule(ir, _LwPP, pg; entry_current = Set{Int}())
    firstread2 = findfirst(s -> s.kind === :read, s_stale)
    @test any(i -> s_stale[i].kind === :exec && s_stale[i].recipe === gradrid, 1:firstread2)
end

@testset "lowering — a stale, non-producible read is REJECTED actionably" begin
    pg = _pg(_LwPP)
    orphan = 9_999_999                                         # a Value id with no producer, not a source
    @test_throws RK._LLowerReject RK._l_ensure_current!(RK._LSchedStep[], Set{Int}(), pg, orphan, Set{Int}())
end

@testset "lowering — same field name on TWO endpoint paths stays DISTINCT (path-keyed)" begin
    ir = RK.method_irs(_LwLF)[1]
    fwd = _pg(_LwPP; path = (:fwd,)); bwd = _pg(_LwPP; path = (:bwd,))
    sf = RK.lower_leaf_schedule(ir, _LwPP, fwd; entry_current = RK._l_all_produced(fwd))
    sb = RK.lower_leaf_schedule(ir, _LwPP, bwd; entry_current = RK._l_all_produced(bwd))
    momid = RK._l_field_id(_LwPP, :mom)
    kf = RK._LKey((:fwd,), momid); kb = RK._LKey((:bwd,), momid)
    @test kf != kb                                            # same Value id, distinct by PATH
    @test any(s -> s.key == kf, sf) && any(s -> s.key == kb, sb)
    @test !any(s -> s.key == kb, sf)                          # no path bleed
end

@testset "lowering — NO second planner in the schedule/ensure hot paths" begin
    # The lowering CONSUMES whatever `Plan` it is handed (integration: syntax's immutable factory
    # plan-key/template identity — NOT a caller Boolean, which would be forgeable). Code-structure fact:
    # `plan(` appears ONLY in the scaffolding helper `_l_endpoint_plan`, never in `_l_plan_graph` /
    # `lower_leaf_schedule` / `_l_ensure_current!`.
    src = read(joinpath(@__DIR__, "..", "src", "kernel_lowering.jl"), String)
    codeonly = join([(i = findfirst('#', ln); i === nothing ? ln : ln[1:prevind(ln, i)])
                     for ln in split(src, '\n')], "\n")
    body = replace(codeonly, r"function _l_endpoint_plan[\s\S]*?\nend" => "GUARDED")
    @test !occursin("plan(", body)
    # the graph consumes the passed plan's producer map directly (no re-selection).
    pg = _pg(_LwPP)
    @test pg.plan isa RK.Plan
end

@testset "lowering — dual_averaging_state fit! (log_current recomputed once, mu never)" begin
    ir = only(RK.method_irs(_LwDA)); spec = _spec(_LwDA)
    writes = Symbol[]
    for s in ir.body
        s isa RK._PlaceWrite && s.root === :self && s.owner !== nothing && !isempty(s.owner) &&
            push!(writes, s.owner[1])
    end
    have = collect(union(Set(spec.have_names), Set(writes)))
    pg = _pg(spec; have = have)
    sched = RK.lower_leaf_schedule(ir, spec, pg; entry_current = RK._l_all_produced(pg))
    @test [get(pg.name_of, s.key.id, s.key.id) for s in sched if s.kind === :write] == [:m, :H, :log_final]
    @test RK._l_recipe_exec_count(sched, spec, pg, :log_current) == 1
    @test RK._l_recipe_exec_count(sched, spec, pg, :mu) == 0
end

@testset "lowering — welford_var step! (pure accumulator; no recompute)" begin
    ir = RK.method_irs(_LwWV)[1]; spec = _spec(_LwWV)
    have = collect(union(Set(spec.have_names), Set([:n, :var, :mean])))
    pg = _pg(spec; have = have)
    sched = RK.lower_leaf_schedule(ir, spec, pg; entry_current = RK._l_all_produced(pg))
    @test [get(pg.name_of, s.key.id, s.key.id) for s in sched if s.kind === :write] == [:n, :var, :mean]
    @test !any(s -> s.kind === :exec, sched)
end

@testset "lowering — registered step_f -> leapfrog! STATIC INLINE (consume factory field_regs)" begin
    # the factory's resolved field_regs map (I CONSUME it — never re-resolve a global). step_f binds to
    # the registered leapfrog! kernel; stats_f is the optional no-effect (`nothing`) callable.
    field_regs = Dict{Symbol,Union{RK._KernelRegistration,Nothing}}(
        :step_f => RK.kernel_registration(_LwLF),
        :stats_f => nothing,
    )
    reg = RK._l_resolve_field_reg(:step_f, field_regs)
    @test reg !== nothing
    # the registered inline target is leapfrog!'s method IR
    ir = RK._l_registered_method_ir(reg)
    @test ir.id.name === :leapfrog!
    # inlining step_f(ep) at the fwd endpoint == leapfrog!'s leaf schedule at that path (one gradient)
    fwd = _pg(_LwPP; path = (:fwd,))
    inl = RK.lower_registered_call(reg, _LwPP, fwd; entry_current = RK._l_all_produced(fwd))
    direct = RK.lower_leaf_schedule(RK.method_irs(_LwLF)[1], _LwPP, fwd; entry_current = RK._l_all_produced(fwd))
    @test [(s.kind, s.key, s.recipe) for s in inl] == [(s.kind, s.key, s.recipe) for s in direct]
    @test RK._l_recipe_exec_count(inl, _LwPP, fwd, :dpot_dpos) == 1
    # stats_f = nothing (optional no-effect) inlines to an EMPTY schedule, no effect
    statsreg = RK._l_resolve_field_reg(:stats_f, field_regs; optional = true)
    @test isempty(RK.lower_registered_call(statsreg, _LwPP, fwd; entry_current = RK._l_all_produced(fwd)))
    # a REQUIRED unregistered callable is a hard reject at the lowering boundary
    @test_throws RK._LLowerReject RK._l_resolve_field_reg(:stats_f, field_regs)   # required, but nothing
    bad = Dict{Symbol,Union{RK._KernelRegistration,Nothing}}()
    @test_throws RK._LLowerReject RK._l_resolve_field_reg(:step_f, bad)           # no entry at all
end

@testset "invariant 1 — a write to a plan-PRODUCED value is REJECTED (factory plans write roots HAVE)" begin
    # welford_var's n/var/mean carry construction recipes (n=0., var/mean=zeros(dim)). If they are NOT
    # planned as authoritative HAVE (write roots), they are plan-produced — a direct write to them is the
    # ambiguous "recomputed AND written" case that the factory must resolve; the lowering rejects it.
    ir = RK.method_irs(_LwWV)[1]; spec = _spec(_LwWV)
    pg_bad = _pg(spec; have = spec.have_names)                # n/var/mean NOT in HAVE -> plan-produced
    @test haskey(pg_bad.producer, RK._l_field_id(spec, :n))   # confirm n is plan-produced under this plan
    @test_throws RK._LLowerReject RK.lower_leaf_schedule(ir, spec, pg_bad;
                                                         entry_current = RK._l_all_produced(pg_bad))
    # with the factory planning the write roots as HAVE (authoritative), the same method schedules fine.
    have = collect(union(Set(spec.have_names), Set([:n, :var, :mean])))
    pg_ok = _pg(spec; have = have)
    @test !haskey(pg_ok.producer, RK._l_field_id(spec, :n))
    sched = RK.lower_leaf_schedule(ir, spec, pg_ok; entry_current = RK._l_all_produced(pg_ok))
    @test [get(pg_ok.name_of, s.key.id, s.key.id) for s in sched if s.kind === :write] == [:n, :var, :mean]
end

@testset "invariant 1 — a write EXPLICITLY makes the written value current (no stale re-read)" begin
    # a straight-line method that writes then re-reads a derived value that depends on the write: the
    # re-read must see the write's currentness for the written HAVE root itself (no spurious recompute
    # of the written value). leapfrog writes mom then (in the drift) reads the velocity derived from mom
    # — the mom write is current; only the velocity (a DERIVED dependent) recomputes.
    ir = RK.method_irs(_LwLF)[1]; pg = _pg(_LwPP)
    sched = RK.lower_leaf_schedule(ir, _LwPP, pg; entry_current = RK._l_all_produced(pg))
    momid = RK._l_field_id(_LwPP, :mom)
    # mom is a HAVE source (never in an :exec recompute); the schedule never tries to "recompute" a write root
    @test !any(s -> s.kind === :exec && momid in s.outputs, sched)
end

@testset "composition — branch-specialized fwd/bwd (two-direct-branch, no Ref)" begin
    # the fixture's direction: `gofwd ? op!(__self__, fwd, …) : op!(__self__, bwd, …)` — TWO direct
    # physical-endpoint branches over concrete owned endpoints, each a leaf schedule at its OWN path.
    ir = RK.method_irs(_LwLF)[1]
    fwd = _pg(_LwPP; path = (:fwd,)); bwd = _pg(_LwPP; path = (:bwd,))
    sf = RK.lower_leaf_schedule(ir, _LwPP, fwd; entry_current = RK._l_all_produced(fwd))
    sb = RK.lower_leaf_schedule(ir, _LwPP, bwd; entry_current = RK._l_all_produced(bwd))
    gofwd = RK._LKey((), 1)                              # a control-field read (synthetic id; Value-keyed)
    br = RK._l_branch_specialize(gofwd, sf, sb)
    # each arm operates ONLY on its own endpoint path — no path bleed between fwd and bwd
    @test all(s -> s isa RK._LSchedStep && s.key.path == (:fwd,), br.then_)
    @test all(s -> s isa RK._LSchedStep && s.key.path == (:bwd,), br.else_)
    # ONE gradient recipe per leaf on EACH taken direction (mutually exclusive arms)
    gradrid_f = fwd.producer[RK._l_field_id(_LwPP, :dpot_dpos)]
    gradrid_b = bwd.producer[RK._l_field_id(_LwPP, :dpot_dpos)]
    @test RK._l_composed_exec_count([br], gradrid_f, :then) == 1
    @test RK._l_composed_exec_count([br], gradrid_b, :else) == 1
    # the recipe id is shared across paths (same spec) — the one-gradient property is PER (Path, Recipe):
    # the :both census has exactly one gradient exec at EACH direction path (only one arm runs at runtime).
    grad_execs = [s for s in RK._l_flatten([br], :both) if s.kind === :exec && s.recipe === gradrid_f]
    @test length(grad_execs) == 2
    @test Set(e.key.path for e in grad_execs) == Set([(:fwd,), (:bwd,)])
end

@testset "composition — inline sibling/registered leaf (no residual call, path-threaded)" begin
    # inlining step_f(ep) at the fwd endpoint: the composed body IS leapfrog!'s leaf schedule at (:fwd,);
    # NO residual step_f/leapfrog call node remains — the schedule is the effect.
    reg = RK.kernel_registration(_LwLF)
    fwd = _pg(_LwPP; path = (:fwd,))
    leaf = RK.lower_registered_call(reg, _LwPP, fwd; entry_current = RK._l_all_produced(fwd))
    inl = RK._l_inline(:step_f, (:fwd,), leaf)
    @test inl.at_path == (:fwd,)
    flat = RK._l_flatten([inl])
    @test all(s -> s.key.path == (:fwd,), flat)         # all effects at the bound endpoint path
    @test !isempty(flat) && all(s -> s isa RK._LSchedStep, flat)
    # the inlined body carries the leaf's writes + one gradient, no opaque/get!/residual-call marker
    @test [s.key.id for s in flat if s.kind === :write] ==
          [RK._l_field_id(_LwPP, :mom), RK._l_field_id(_LwPP, :pos), RK._l_field_id(_LwPP, :mom)]
end

@testset "composition — sequential control concatenates ordered sub-schedules" begin
    ir = RK.method_irs(_LwLF)[1]
    fwd = _pg(_LwPP; path = (:fwd,))
    leaf = RK.lower_leaf_schedule(ir, _LwPP, fwd; entry_current = RK._l_all_produced(fwd))
    seq = RK._l_seq(leaf, leaf)                          # two sequential leaves (two integrator steps)
    flat = RK._l_flatten(seq)
    gradrid = fwd.producer[RK._l_field_id(_LwPP, :dpot_dpos)]
    @test count(s -> s.kind === :exec && s.recipe === gradrid, flat) == 2   # one gradient PER leaf
    @test count(s -> s.kind === :write && s.key.id == RK._l_field_id(_LwPP, :mom), flat) == 4  # 2 per leaf
end

@testset "copy!! strong-copy — physical closure = sources+derived; derived validity transfers" begin
    pg = _pg(_LwPP)
    # RK endpoint OWNED closure (9): pos, mom (authoritative sources) + the 7 endpoint-dependent caches.
    # SHARED (metric/chol_metric/@node(logdet)/pot_f/grad_f) is EXCLUDED from copy!!.
    owned_names = [:pos, :mom, :pot, :dpot_dpos, :dkin_dmom, :kin, :ham, :dham_dpos, :dham_dmom]
    # ALIAS COLLAPSE (b2ee0ca): dham_dpos≡dpot_dpos, dham_dmom≡dkin_dmom — 9 authored names -> 7 distinct
    # canonical physical ids. The copy!! physical closure is the DISTINCT canonical owned set.
    @test RK._l_field_id(_LwPP, :dham_dpos) == RK._l_field_id(_LwPP, :dpot_dpos)   # alias collapsed in-graph
    @test RK._l_field_id(_LwPP, :dham_dmom) == RK._l_field_id(_LwPP, :dkin_dmom)
    physical = unique([RK._l_field_id(_LwPP, n) for n in owned_names])
    sources  = [RK._l_field_id(_LwPP, n) for n in [:pos, :mom]]
    prop = (:proposals, 7); init = (:init,)
    cp = RK._LCopyStep(init, prop, physical, sources)
    # PHYSICAL-COPY CENSUS: pos/mom + the distinct owned canonical derived ids, each EXACTLY once, shared excluded
    @test length(cp.physical) == length(unique(cp.physical)) == 7
    @test Set(cp.sources) == Set([RK._l_field_id(_LwPP, :pos), RK._l_field_id(_LwPP, :mom)])
    @test !(RK._l_field_id(_LwPP, :chol_metric) in cp.physical)   # metric-only closure is SHARED, excluded
    @test !(RK._l_field_id(_LwPP, :metric) in cp.physical)
    @test Set(RK._l_copy_derived(cp)) ==
          Set(unique(RK._l_field_id(_LwPP, n) for n in [:pot, :dpot_dpos, :dkin_dmom, :kin, :ham]))
    # DERIVED validity transfers src→dest (sources are authoritative, never in the currentness set)
    gradid = RK._l_field_id(_LwPP, :dham_dpos); dpotid = RK._l_field_id(_LwPP, :dpot_dpos)
    veloid = RK._l_field_id(_LwPP, :dham_dmom); dkinid = RK._l_field_id(_LwPP, :dkin_dmom)
    current = RK._l_path_current(pg, prop)                   # accepted proposal fully current
    RK._l_copy_transfer!(current, cp)                        # copy!!(init, proposal): no Recipe
    @test RK._LKey(init, gradid) in current && RK._LKey(init, dpotid) in current   # restore -> gradient current
    # momentum refresh (write init.mom) invalidates ONLY the kinetic/momentum closure, not the gradient
    RK._l_write_kill!(current, pg, init, RK._l_field_id(_LwPP, :mom))
    @test !(RK._LKey(init, veloid) in current) && !(RK._LKey(init, dkinid) in current)
    @test RK._LKey(init, gradid) in current && RK._LKey(init, dpotid) in current   # gradient UNTOUCHED (delta 0)
end

@testset "epoch exception — executed writes' KILLS stand; retry MUST recompute pgrad (not entry)" begin
    pg = _pg(_LwPP; path = (:fwd,)); pgs = Dict{RK._LPath,RK._LPlanGraph}((:fwd,) => pg)
    entry = RK._l_path_current(pg, (:fwd,))
    posid = RK._l_field_id(_LwPP, :pos); gradid = RK._l_field_id(_LwPP, :dham_dpos)
    dpotid = RK._l_field_id(_LwPP, :dpot_dpos)
    # a body that WRITES pos then throws BEFORE the pgrad recompute (only the pos-write executed)
    prefix = RK._LComposed([RK._LSchedStep(:write, RK._LKey((:fwd,), posid), 0, ())])
    @test RK._LKey((:fwd,), gradid) in entry                 # gradient current at entry
    onexc = RK._l_epoch_on_exception(entry, prefix, pgs)
    # the pos-write's KILL stands (physical write not rolled back) — gradient is DIRTY, NOT restored to entry
    @test !(RK._LKey((:fwd,), gradid) in onexc)
    @test !(RK._LKey((:fwd,), dpotid) in onexc)
    @test onexc != entry                                     # NEVER copy(entry)
    # a retry from onexc MUST execute the pgrad Recipe (the gradient was killed by the un-rolled-back write)
    entry_ids = Set{Int}(k.id for k in onexc if k.path == (:fwd,))
    ir = RK.method_irs(_LwLF)[1]
    retry = RK.lower_leaf_schedule(ir, _LwPP, pg; entry_current = entry_ids)
    gradrid = pg.producer[dpotid]
    @test any(s -> s.kind === :exec && s.recipe === gradrid, retry)
end

@testset "epoch commit — consumes the SELECTED trace only; inactive path never blessed" begin
    fwd = _pg(_LwPP; path = (:fwd,)); bwd = _pg(_LwPP; path = (:bwd,))
    ir = RK.method_irs(_LwLF)[1]
    sf = RK.lower_leaf_schedule(ir, _LwPP, fwd; entry_current = RK._l_all_produced(fwd))
    sb = RK.lower_leaf_schedule(ir, _LwPP, bwd; entry_current = RK._l_all_produced(bwd))
    gofwd = RK._LKey((), 1)
    body = RK._l_seq([RK._l_branch_specialize(gofwd, sf, sb)])
    pgs = Dict{RK._LPath,RK._LPlanGraph}((:fwd,) => fwd, (:bwd,) => bwd)
    # bwd starts fully current; fwd taken -> the bwd endpoint's validity is UNCHANGED and no bwd write/exec fires
    entry = union(RK._l_path_current(fwd, (:fwd,)), RK._l_path_current(bwd, (:bwd,)))
    committed = RK._l_epoch_commit(entry, body, pgs; taken = RK._LTaken(gofwd => :then))
    # every bwd-path key present at entry is STILL present after commit (bwd untouched, not blessed/killed)
    bwd_entry = filter(k -> k.path == (:bwd,), entry)
    @test all(k -> k in committed, bwd_entry)
    # and no bwd Recipe was executed in the selected (fwd) trace
    flat_then = RK._l_flatten([RK._l_branch_specialize(gofwd, sf, sb)], :then)
    @test all(s -> s.key.path == (:fwd,), flat_then)         # only fwd effects in the taken trace
    # sibling/recursive calls internal to ONE root epoch — no nested epoch
    root = RK._LEpochStep(RK._l_seq([RK._l_inline(:step_f, (:fwd,), sf)]))
    @test !any(x -> x isa RK._LEpochStep, root.body)
    # COMMIT rejects a dynamic branch with NO explicit taken arm (never defaults/blesses a guessed arm)
    @test_throws RK._LLowerReject RK._l_epoch_commit(entry, body, pgs)   # MISSING trace -> reject
    # an INVALID trace (neither :then nor :else) also rejects and blesses nothing
    @test_throws RK._LLowerReject RK._l_epoch_commit(entry, body, pgs; taken = RK._LTaken(gofwd => :bogus))
end

@testset "warmed transition boundary — momentum refresh: ZERO boundary pgrad, one pgrad/leaf" begin
    # PRODUCT CONTRACT (RK): public nuts!! is a COMPLETE transition — it refreshes momentum on owned init
    # BEFORE the subject step. `refresh_momentum!!(init; rng)` is a registered free Mode-2 kernel
    # (randn!+lmul! into init.mom) inlined like leapfrog. The REAL kernel lands with the revised fixture;
    # here we pin the CURRENTNESS contract it must satisfy, modeling refresh as its owned effect: a
    # mom-write. It must kill ONLY the kinetic/momentum closure, leaving the gradient CURRENT so the
    # boundary costs ZERO pgrad, and each leaf costs exactly one.
    initpg = _pg(_LwPP; path = (:init,))
    ir = RK.method_irs(_LwLF)[1]
    gradid = RK._l_field_id(_LwPP, :dham_dpos); dpotid = RK._l_field_id(_LwPP, :dpot_dpos)
    veloid = RK._l_field_id(_LwPP, :dham_dmom); kinid = RK._l_field_id(_LwPP, :kin)
    hamid  = RK._l_field_id(_LwPP, :ham); momid = RK._l_field_id(_LwPP, :mom)
    current = RK._l_path_current(initpg, (:init,))               # init fully current at transition entry
    RK._l_write_kill!(current, initpg, (:init,), momid)          # refresh_momentum!!: the owned mom write
    # kinetic/momentum closure invalidated; GRADIENT chain untouched (pos unchanged) — zero boundary pgrad
    @test !(RK._LKey((:init,), veloid) in current) && !(RK._LKey((:init,), kinid) in current)
    @test !(RK._LKey((:init,), hamid) in current)
    @test RK._LKey((:init,), gradid) in current && RK._LKey((:init,), dpotid) in current
    # the transition's first leaf consumes the current gradient; its ONE pgrad is the leaf's, NOT boundary
    entry_ids = Set{Int}(k.id for k in current if k.path == (:init,))
    leaf = RK.lower_leaf_schedule(ir, _LwPP, initpg; entry_current = entry_ids)
    gradrid = initpg.producer[dpotid]
    @test RK._l_recipe_exec_count(leaf, _LwPP, initpg, :dpot_dpos) == 1   # exactly one pgrad per leaf
    # zero boundary pgrad: the leaf's FIRST read (the gradient) needs no recompute (current post-refresh)
    firstread = findfirst(s -> s.kind === :read, leaf)
    @test !any(i -> leaf[i].kind === :exec && leaf[i].recipe === gradrid, 1:firstread)
    # but the velocity (kinetic) WAS invalidated by refresh -> the drift's velocity read recomputes once
    @test RK._l_recipe_exec_count(leaf, _LwPP, initpg, :dkin_dmom) == 1
end

@testset "lowering — ROLE-AWARE remap: owned keys per-endpoint DISTINCT, SHARED IDENTICAL (RK 06:47)" begin
    role_of(id) = id in (20, 21, 22) ? :shared : :owned      # metric/chol/node shared; mom/dkin owned
    body = RK._LComposed([
        RK._LSchedStep(:write, RK._LKey((), 5), 0, ()),      # mom (owned)
        RK._LSchedStep(:exec,  RK._LKey((), 10), 3, (10,)),  # dkin (owned)
        RK._LSchedStep(:exec,  RK._LKey((), 21), 4, (21,)),  # chol (SHARED authority)
    ])
    fwd = RK._l_remap(body, (:fwd,), role_of); bwd = RK._l_remap(body, (:bwd,), role_of)
    fk = [s.key for s in fwd]; bk = [s.key for s in bwd]
    @test RK._LKey((:fwd,), 5) in fk && RK._LKey((:bwd,), 5) in bk && RK._LKey((:fwd,), 5) ∉ bk  # owned DISTINCT
    @test RK._LKey((), 21) in fk && RK._LKey((), 21) in bk       # SHARED chol IDENTICAL across fwd/bwd
    @test RK._LKey((:fwd,), 21) ∉ fk                            # shared never moved to the endpoint
end

@testset "lowering — SHARED metric write FAN-OUT vs OWNED write (RK 06:52)" begin
    role_of(id) = id in (20, 21, 22) ? :shared : :owned
    deps = Dict(20 => Set([21, 22, 10, 11]), 5 => Set([10, 11]))   # metric→chol/node/dkin/kin ; mom→dkin/kin
    pg = RK._LPlanGraph((), nothing, Dict{Int,Int}(), Dict{Int,Vector{Int}}(),
                        Dict{Int,Vector{Int}}(), deps, Set{Int}(), Dict{Int,Symbol}())
    eps = ((:init,), (:fwd,), (:bwd,)); sp = (:shared,)
    # SHARED metric(20) write: shared chol/node die ONCE at authority; owned dkin/kin die on ALL endpoints
    cur = Set([RK._LKey(sp, 21), RK._LKey(sp, 22)])
    for ep in eps, d in (10, 11); push!(cur, RK._LKey(ep, d)); end
    RK._l_write_kill!(cur, pg, sp, 20; role_of = role_of, shared_path = sp, endpoints = eps)
    @test RK._LKey(sp, 21) ∉ cur && RK._LKey(sp, 22) ∉ cur       # shared killed once
    @test all(RK._LKey(ep, 10) ∉ cur && RK._LKey(ep, 11) ∉ cur for ep in eps)  # owned killed on EVERY endpoint
    # OWNED mom(5) write at fwd: kills only fwd's owned dependents; init/bwd untouched
    cur2 = Set{RK._LKey}(); for ep in eps, d in (10, 11); push!(cur2, RK._LKey(ep, d)); end
    RK._l_write_kill!(cur2, pg, (:fwd,), 5; role_of = role_of, shared_path = sp, endpoints = eps)
    @test RK._LKey((:fwd,), 10) ∉ cur2 && RK._LKey((:fwd,), 11) ∉ cur2         # fwd killed
    @test RK._LKey((:init,), 10) in cur2 && RK._LKey((:bwd,), 10) in cur2      # init/bwd UNTOUCHED (no fan-out)
end

@testset "lowering — loop shape is IMMUTABLE / DETERMINISTIC Tuple model metadata only (RK 06:54)" begin
    role_of(_::Int) = :owned
    body = RK._LComposed([RK._LSchedStep(:write, RK._LKey((), 5), 0, ())])
    l3 = RK._l_loop(:p, (:proposals,), 1:3, body, role_of)
    l4 = RK._l_loop(:p, (:proposals,), 1:4, body, role_of)
    @test l3.indices === (1, 2, 3) && l3.indices isa Tuple{Vararg{Int}}    # IMMUTABLE Tuple shape, not a hot Vector
    @test length(RK._l_loop_iterations(l3)) == 3 && length(RK._l_loop_iterations(l4)) == 4  # shape-keyed count
    # distinct indexed-child paths (never collapsed): proposals[1..3].mom are three distinct keys
    paths = [s.key.path for s in RK._l_loop_iterations(l3)]
    @test Set(paths) == Set([(:proposals, 1), (:proposals, 2), (:proposals, 3)])
    # SAME shape → identical model reconstruction (DETERMINISTIC/model-only; this is NOT a no-per-instance-
    # replan proof — the real no-replan counter+poison gate is syntax's prepared factory, deferred to rebase).
    @test RK._l_loop(:p, (:proposals,), 1:3, body, role_of).iterations == l3.iterations
end

@testset "lowering — replay ROLE-RESOLVES produced outputs: shared→authority key, owned→endpoint (RK 06:57)" begin
    role_of(id) = id in (20, 21, 22) ? :shared : :owned
    # an atomic exec on fwd producing BOTH an OWNED output (dkin=10) and a SHARED output (chol=21)
    body = RK._LComposed([RK._LSchedStep(:exec, RK._LKey((:fwd,), 10), 3, (10, 21))])
    cur = Set{RK._LKey}()
    RK._l_replay!(cur, body, Dict{RK._LPath,RK._LPlanGraph}(); role_of = role_of, shared_path = (:shared,))
    @test RK._LKey((:fwd,), 10) in cur          # OWNED output blessed at the endpoint (fwd)
    @test RK._LKey((:shared,), 21) in cur       # SHARED output blessed ONLY at the authority path
    @test RK._LKey((:fwd,), 21) ∉ cur           # NOT cloned onto the endpoint — one authority key
end
