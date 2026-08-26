# MethodIR representation gate — the ReactiveHMC-STRUCTURE IMPLICIT-FIELD / NO-Ref `@kernel` NUTS
# fixture as the binding consumer. Representation/classification ONLY (V7 pivot): every authored
# method is REPRESENTED (never rejected), with three orthogonal facts. NO lowering / epochs / execution
# here (that is the later increment, oracle-gated separately).
#
# Two receiver modes are exercised: (1) IMPLICIT-FIELD object kernels (nuts_state / dual_averaging_state
# / welford_var) — no receiver formal, bare unshadowed names are owner FIELDS, `__self__` is the sibling
# receiver; (2) Mode-2 FREE kernels (leapfrog! / nuts!!) — an explicit subject formal is the receiver.
# euclidean_phasepoint is methodless (stateless) -> `method_irs == ()`.

using ReactiveKernels
using Test
const RK = ReactiveKernels

# The faithful fixture lives in benchmark/; include it inside an ISOLATED module so its @kernel globals
# (`nuts_state`/`dual_averaging_state`/`welford_var`/…) do NOT leak into `Main` and shadow the package
# constructors that later test files use.
module _FixImpl
    include(joinpath(@__DIR__, "..", "benchmark", "nuts_kernel_authoring_fixture.jl"))
end
const _NS = _FixImpl.nuts_state
const _DA = _FixImpl.dual_averaging_state
const _WV = _FixImpl.welford_var
const _PP = _FixImpl.euclidean_phasepoint
const _LF = _FixImpl.leapfrog!
const _NB = _FixImpl.nuts!!

_mir(k, nm) = (irs = RK.method_irs(k); irs[findfirst(ir -> ir.id.name === nm, irs)])
_mirs(k, nm) = [ir for ir in RK.method_irs(k) if ir.id.name === nm]
# recursive node finder over an _MExpr / _MStmt tree
function _find(pred, x)
    pred(x) && return true
    (x isa RK._MExpr || x isa RK._MStmt) || return false
    for fn in fieldnames(typeof(x))
        v = getfield(x, fn)
        (v isa RK._MExpr || v isa RK._MStmt) && _find(pred, v) && return true
        if v isa Tuple
            for e in v
                (e isa RK._MExpr || e isa RK._MStmt) && _find(pred, e) && return true
                e isa Pair && e.second isa RK._MExpr && _find(pred, e.second) && return true
            end
        end
    end
    false
end
_body_find(pred, ir) = any(s -> _find(pred, s), ir.body)
# flatten a structured access-event tree into every leaf/group (for structural assertions)
function _ae_all(evs)
    out = Any[]
    for e in evs
        push!(out, e)
        e isa RK._ACall  && append!(out, _ae_all(e.reads))
        e isa RK._ABranch && (append!(out, _ae_all(e.thn)); append!(out, _ae_all(e.els)))
        e isa RK._ALoop   && append!(out, _ae_all(e.body))
        e isa RK._AGuard  && (append!(out, _ae_all(e.cond)); append!(out, _ae_all(e.body)))
    end
    out
end

@testset "implicit-field consumer gate — every authored method REPRESENTED (no rejection)" begin
    for k in (_NS, _DA, _WV, _LF, _NB)
        for ir in RK.method_irs(k)
            @test ir.ok                                        # no faithful shape is rejected
        end
    end
    @test length(RK.method_irs(_NS)) == 9
    @test length(RK.method_irs(_DA)) == 1
    @test length(RK.method_irs(_WV)) == 2              # vector + matrix step!
    @test length(RK.method_irs(_LF)) == 1              # Mode-2 free method
    @test length(RK.method_irs(_NB)) == 1              # Mode-2 `!!` free method
    @test RK.method_irs(_PP) == ()            # methodless -> stateless, no methods
    @test isempty(RK.method_irs(_PP))
end

@testset "implicit-field consumer gate — three orthogonal facts per method" begin
    facts(ir) = (ir.control, Set(ir.effects))
    @test facts(_mir(_NS, :logadvanceprob)) == (:straight, Set())            # pure value method
    @test facts(_mir(_NS, :swapproposal!))  == (:straight, Set([:place_write]))
    @test _mir(_NS, :flip!).control          === :branch                     # `if depth > 1`
    @test :place_write in _mir(_NS, :flip!).effects
    @test facts(_mir(_NS, :flip_neg!))       == (:straight, Set([:place_write]))
    @test _mir(_NS, :step!).control          === :loop
    @test _mir(_NS, :finish!).control        === :recursive                  # finish! <-> start! recursion
    @test _mir(_NS, :start!).control         === :recursive
    @test facts(_mir(_DA, :fit!)) == (:straight, Set([:place_write]))
    wv, wm = _mirs(_WV, :step!)
    @test wv.control === :straight && :place_write in wv.effects
    @test wm.control === :loop                                               # matrix step! loops
    # `kind` is a SECONDARY convenience, never gating a direct write
    @test _mir(_NS, :flip!).kind === :segment            # direct writes legal in every shape
    @test _mir(_NS, :step!).kind === :orchestration
end

@testset "implicit-field resolution — bare fields are self paths; NO receiver formal" begin
    # a nested method declares NO `self` formal; the receiver is the synthetic `__self__`
    st = _mir(_NS, :step!)
    @test st.self === :__self__
    @test all(f -> f.name !== :self && f.name !== :__self__, st.formals)   # no receiver formal
    @test any(f -> f.name === :rng && f.kind === :pos, st.formals)         # runtime rng is a plain formal
    # bare `gofwd = !gofwd` in flip! is a SELF place write on field :gofwd (not a local rebind)
    fl = _mir(_NS, :flip!)
    @test _body_find(n -> n isa RK._PlaceWrite && n.root === :self && n.owner == (:gofwd,), fl)
    # bare `trees[depth-1].log_weight[1]` in logadvanceprob is a self field READ (implicit receiver)
    la = _mir(_NS, :logadvanceprob)
    @test _body_find(n -> n isa RK._SelfField && n.path == (:trees,), la)
    @test (:trees,) in RK.read_roots(la)
end

@testset "implicit-field — __self__ sibling calls resolve to registered siblings" begin
    st = _mir(_NS, :step!)
    # `reset!(__self__)` -> sibling _Call with a _SelfRef target
    @test _body_find(n -> n isa RK._Call && n.name === :reset! && n.target isa RK._SelfRef, st)
    # step! call edges carry the siblings (reset!/flip!/finish!/swapproposal!) with single candidates
    edges = RK.call_edges(st)
    for nm in (:reset!, :flip!, :finish!, :swapproposal!)
        @test (:sibling, nm, 1) in edges
    end
    # the matrix welford step! forwards `; kwargs...` to the sibling vector step! -> 2-candidate overload
    wm = _mirs(_WV, :step!)[2]
    @test _body_find(n -> n isa RK._Call && n.name === :step! &&
                          any(p -> p.first === RK._KMIR_KWSPLAT, n.kw), wm)
    @test (:sibling_overload, :step!) in wm.resolution_deps
end

@testset "registered-resolvable — step_f / stats_f / copy!! are NEVER opaque" begin
    # step_f(ep): a registered callable field (construction-time registered-or-REJECT), NOT opaque
    sr = _mir(_NS, :start!)
    @test _body_find(n -> n isa RK._FieldCall && n.path == (:step_f,) && n.hint === :registered, sr)
    @test (:registered_field, (:step_f,)) in sr.resolution_deps
    # stats_f(__self__) guarded by `isnothing(stats_f) ||` -> registered-OR-NOTHING
    cs = _mir(_NS, :collectstats!)
    @test _body_find(n -> n isa RK._FieldCall && n.path == (:stats_f,) &&
                          n.hint === :registered_or_nothing, cs)
    @test (:registered_or_nothing_field, (:stats_f,)) in cs.resolution_deps
    # copy!!(dest, src) resolves to a `_RegisteredCall` from the def-time SNAPSHOT (finding 2), carrying
    # the captured intrinsic `_KernelRegistration` identity (Token/!!/subject) — NOT a reread global,
    # NOT opaque. The authored slot is the bare `copy!!` binding.
    rc = let out = Ref{Any}(nothing)
        _body_find(n -> n isa RK._RegisteredCall && n.ref.slot.name === Symbol("copy!!") ?
                        (out[] = n; true) : false, sr); out[]
    end
    @test rc !== nothing
    @test rc.intrinsic && rc.registration.kind === :intrinsic && rc.registration.is_bang_bang
    @test rc.registration.token === RK.kernel_token(RK.copy!!)   # captured intrinsic Token identity
    @test rc.ref.field === nothing                               # authored bare (not qualified)
    @test any(d -> d isa Tuple && d[1] === :intrinsic_call, sr.resolution_deps)
    # NO opaque dep is falsely minted for the registered intrinsic / callable fields
    @test !any(d -> d isa Tuple && d[1] === :opaque_call &&
                    d[2] isa GlobalRef && d[2].name === Symbol("copy!!"), sr.resolution_deps)
end

@testset "Mode-2 free kernels — subject-rooted writes; delegation" begin
    lf = _mir(_LF, :leapfrog!)
    @test lf.self === :phasepoint                          # explicit subject is the receiver
    # `@. phasepoint.mom -= …` / `@. phasepoint.pos += …` -> subject-rooted place writes
    @test (:self, (:mom,)) in RK.write_roots(lf)
    @test (:self, (:pos,)) in RK.write_roots(lf)
    @test (:mom,) in RK.read_roots(lf) && (:dham_dpos,) in RK.read_roots(lf)
    # nuts!!(state; rng): `step!(state, rng)` then `return state` (mutate-owned + return SAME object)
    nb = _mir(_NB, :nuts!!)
    @test nb.self === :state
    @test _body_find(n -> n isa RK._Return && n.value isa RK._SelfRef, nb)   # `return state`
    # `step!(state, rng)` is a PENDING registered SUBJECT-METHOD call (finding 1): callee `step!`, the
    # subject `state` as receiver, resolved at construction via the object's Token/MethodId — NEVER a
    # global external/opaque callee.
    @test _body_find(n -> n isa RK._SubjectMethodCall && n.name === :step! &&
                          n.subject isa RK._SelfRef, nb)
    @test (:subject_method, :step!) in RK.call_edges(nb)
    @test (:subject_method, :step!) in nb.resolution_deps
    @test !any(e -> e[1] === :external, RK.call_edges(nb))                   # not an external GlobalRef
end

@testset "place model — indexed-nested, owned-alias, swap, set-then-return" begin
    st = _mir(_NS, :step!)
    # `trees[1].log_weight[1] = 0.` -> an INDEXED-NESTED self place write, owner (:trees,)
    @test _body_find(n -> n isa RK._PlaceWrite && n.root === :self && n.owner == (:trees,) &&
                          n.target isa RK._Index, st)
    # owned-alias: `tree = trees[depth]; @. tree.bwd.mom = -ep.mom` -> alias write on (:trees,)
    fn = _mir(_NS, :flip_neg!)
    @test _body_find(n -> n isa RK._PlaceWrite && n.root === :alias && n.alias === :tree &&
                          n.owner == (:trees,) && n.dot, fn)
    # the proposal swap: RHS-before-LHS _PlaceSwap on (:proposals,)
    sw = _mir(_NS, :swapproposal!)
    swap = sw.body[findfirst(s -> s isa RK._PlaceSwap, sw.body)]
    @test length(swap.targets) == 2
    @test all(t -> t isa RK._PlaceWrite && t.owner == (:proposals,), swap.targets)
    # set-then-return: `may_continue || return may_sample = false` -> _SetReturn writing (:may_sample,)
    ft = _mir(_NS, :finish!)
    @test _body_find(n -> n isa RK._SetReturn && n.write.owner == (:may_sample,), ft)
end

@testset "structured access events — branch-scoped + loop-carried (fault-2a SOUND)" begin
    st = _mir(_NS, :step!)
    ev = RK.access_events(st)
    # the top-level `gofwd ? … : …` is a MUTUALLY-EXCLUSIVE branch group (NOT a flat concatenation)
    @test any(e -> e isa RK._ABranch, ev)
    # the `for depth in 1:max_depth` body is a LOOP group, may-run-zero + carried
    loop = ev[findfirst(e -> e isa RK._ALoop, ev)]
    @test loop.may_run_zero
    # EVERY sibling/field/external call is a call event carrying its arg reads (every call a read)
    all_ev = _ae_all(ev)
    @test any(e -> e isa RK._ACall && e.kind === :sibling, all_ev)
    @test any(e -> e isa RK._ACall && e.kind === :intrinsic, all_ev)   # copy!!
    @test any(e -> e isa RK._ACall && e.kind === :opaque, all_ev)      # rand(rng,…) is opaque (no name-RNG)
    # a write-only scalar (`gofwd = !gofwd`) contributes a WRITE with no read of its own terminal place
    fl = RK.access_events(_mir(_NS, :flip!))
    @test any(e -> e isa RK._AWrite && e.owner == (:gofwd,), _ae_all(fl))
end

@testset "no-Ref — dynamic Ref current-views are SUPERSEDED (removed)" begin
    @test !isdefined(RK, :dynamic_views)      # the Ref current-view accessor is gone
    @test !isdefined(RK, :_DynamicView)       # …and its node type
    # the fixture authors fwd/bwd as CONCRETE owned fields (deepcopy(init)), never a Ref — a read of
    # `fwd`/`bwd` is a plain self field, never a container-deref node.
    st = _mir(_NS, :step!)
    @test (:fwd, :mom) in RK.read_roots(st) || (:bwd, :mom) in RK.read_roots(st)
end

# ---- adversarial soundness probes (representation-only; negative / hygiene / boundary) -------------
module _Probes
    using ReactiveKernels
    module PA; f(z) = z; end
    module PB; f(z) = z; end
    # implicit-field: bare field write vs a FORMAL that shadows a same-named field vs a loop-var shadow.
    @kernel pshadow(x, arr) = begin
        setx!(z) = begin; x = z; end                       # bare field write on :x
        useformal(x) = begin; arr[1] = x; end              # formal `x` SHADOWS field :x -> reads formal
        useloopvar() = for x in arr; arr[1] = x; end       # for-var `x` SHADOWS field :x -> reads loop var
    end
    @kernel pnode(v, w) = begin
        f!() = begin; v = @node(w + 1); end                # @node in a METHOD body; inner reads field :w
    end
    @kernel pqual(v) = begin; q!(z) = begin v = MissingMod.f(z) end; end   # unresolved qualifier -> reject
    @kernel pcalls(v) = begin; c!(z) = begin v = PA.f(z) + PB.f(z) end; end  # distinct-module opaque callees

    # Finding 4 — sound definite assignment: one-arm / guard binds are maybe-bound (reject later use);
    # both-arms binds are definite (valid).
    @kernel pdefassign(v) = begin
        onearm!(flag) = begin; if flag; t = v; end; v = t; end              # t maybe-bound -> REJECT read
        botharm!(flag) = begin; if flag; t = 1; else; t = 2; end; v = t; end # t definite -> OK
        guardbind!(flag) = begin; flag && (t = v); v = t; end               # guard bind maybe -> REJECT
        rebindsafe!(flag) = begin; if flag; t = 1; end; t = 2; v = t; end   # unconditional rebind -> OK
    end
    # Finding 5 — address-reading swap: RHS (reads :b) must precede ALL LHS addresses (reads :a).
    @kernel pswap(a, b) = begin
        sw!(i, j) = begin; a[i], a[j] = b[i], b[j]; end
    end
    # Finding 9 — while: the initial condition read is guaranteed OUTSIDE the loop; a retest rides inside.
    @kernel pwhile(c, v) = begin
        loop!() = begin; while c[1]; v = 0.0; end; end
    end
    # Finding 10 — RNG is NOT classified by the spelling `rng`: a formal literally named `rng` used in an
    # unregistered call is OPAQUE, not rng; a differently-named RNG formal is likewise opaque (deferred).
    @kernel prng(v) = begin
        collide!(rng) = begin; v = opaquef(rng); end        # arg spelled `rng` -> still opaque, not :rng
        renamed!(gen) = begin; v = draw(gen); end            # differently-named RNG formal -> opaque
    end
    # Finding 6 — optional callable field is registered-OR-NOTHING ONLY under a DOMINATING `isnothing(f) ||`
    # at that call site; an unguarded call or an `isnothing(f) &&` stays registered-OR-REJECT.
    @kernel poptional(f) = begin
        guarded!() = begin; isnothing(f) || f(1); end        # dominated `||` -> registered_or_nothing
        unguarded!() = begin; f(2); end                      # unguarded -> registered (reject)
        andguard!() = begin; isnothing(f) && f(3); end       # `&&` -> NOT optional -> registered (reject)
    end
    # Finding 2 — direct REGISTERED @kernel callees captured from the def-time snapshot: a bare call, a
    # qualified call, and a REBIND-drift rejection.
    module _RegMod
        using ReactiveKernels
        @kernel base_lf!(p; s) = begin; @. p.x -= s; end     # a registered Mode-2 kernel
    end
    const bare_lf! = _RegMod.base_lf!                        # a bare alias to a registered kernel
    @kernel pregbare(pp) = begin; drive!(x) = bare_lf!(pp; s = x); end        # bare registered call
    @kernel pregqual(pp) = begin; drive!(x) = _RegMod.base_lf!(pp; s = x); end # qualified registered call
    driftf = _RegMod.base_lf!                                # a REBINDABLE (non-const) registered global
    @kernel pregdrift(pp) = begin; drive!(x) = driftf(pp; s = x); end  # captures driftf's registration…
    driftf = (args...; kwargs...) -> nothing                 # …then REBIND it away (drift after emission)
end

@testset "adversarial — implicit lexical shadowing (field vs formal vs local)" begin
    ps = _Probes.pshadow
    setx = _mir(ps, :setx!)
    # `x = z` writes FIELD :x (bare unshadowed) — a self place write
    @test _body_find(n -> n isa RK._PlaceWrite && n.root === :self && n.owner == (:x,), setx)
    uf = _mir(ps, :useformal)
    # formal `x` SHADOWS field :x: `arr[1] = x` reads the FORMAL, not the field
    @test _body_find(n -> n isa RK._FormalRef && n.arg === :x, uf)
    @test !_body_find(n -> n isa RK._SelfField && n.path == (:x,), uf)
    ul = _mir(ps, :useloopvar)
    # for-loop var `x` SHADOWS field :x within the loop body: `arr[1] = x` reads the loop LOCAL
    @test _body_find(n -> n isa RK._LocalRef && n.name === :x, ul)
    @test !_body_find(n -> n isa RK._SelfField && n.path == (:x,), ul)
end

@testset "adversarial — @node is a lifted recipe (inner deps emitted, NOT double-promoted)" begin
    pn = _mir(_Probes.pnode, :f!)
    @test pn.ok
    # the @node lifts to a distinguished _NodeExpr wrapping the SOURCE-NORMALIZED inner expr...
    @test _body_find(n -> n isa RK._NodeExpr && n.inner isa RK._MExpr, pn)
    # ...and its inner dependency reads ARE emitted (finding 7): `@node(w + 1)` reads field :w.
    @test (:w,) in RK.read_roots(pn)
end

@testset "adversarial — ordinary Julia stays opaque; unresolved qualifier rejects" begin
    # a distinct-module external call is opaque with a HYGIENIC GlobalRef (PA.f and PB.f never collide)
    pc = _mir(_Probes.pcalls, :c!)
    edges = RK.call_edges(pc)
    @test (:external, GlobalRef(_Probes.PA, :f), :opaque) in edges
    @test (:external, GlobalRef(_Probes.PB, :f), :opaque) in edges
    # an unresolved qualified callee (`MissingMod.f`) is REJECTED actionably (not silently dropped)
    pq = _mir(_Probes.pqual, :q!)
    @test !pq.ok
    @test occursin("unresolved qualifier", something(pq.reason, ""))
end

@testset "compiler boundary — NO Julia IR / inference APIs in the analysis path" begin
    # gate 8 discriminator: the emission CODE must not reference any Julia-IR / inference API. Read the
    # engine source, STRIP line comments (the module header documents what it deliberately AVOIDS, which
    # would false-positive a raw grep), then assert none of the forbidden identifiers appears in code.
    raw = read(joinpath(@__DIR__, "..", "src", "kernel_methodir.jl"), String)
    codeonly = join([(i = findfirst('#', ln); i === nothing ? ln : ln[1:prevind(ln, i)])
                     for ln in split(raw, '\n')], "\n")
    for forbidden in ("code_lowered", "code_typed", "CodeInfo", "uncompressed_ir",
                      "Core.Compiler", "code_ircode", "@code_", "IRCode", "typeinf",
                      "Base.code_", "InferenceResult")
        @test !occursin(forbidden, codeonly)
    end
    # positive control: the analysis consumes the substrate's DEFINITION-TIME registration/port SNAPSHOTS
    # (the sanctioned macro-source hops), not Julia IR / a live-global reread.
    @test occursin("kernel_callee_registrations", codeonly)
    @test occursin("kernel_port_names", codeonly)
end

@testset "finding 4 — sound definite assignment (maybe-bound reject; both-arms valid)" begin
    pd = _Probes.pdefassign
    onearm = _mir(pd, :onearm!)
    @test !onearm.ok                                                    # one-arm bind -> maybe -> reject
    @test occursin("maybe-bound", something(onearm.reason, ""))
    @test _mir(pd, :botharm!).ok                                        # both-arms bind -> definite -> OK
    @test !_mir(pd, :guardbind!).ok                                     # guard bind -> maybe -> reject
    @test _mir(pd, :rebindsafe!).ok                                     # unconditional rebind -> definite
end

@testset "finding 5 — swap reads ALL RHS before ANY LHS address/store" begin
    sw = _mir(_Probes.pswap, :sw!)
    leaves = _ae_all(RK.access_events(sw))
    reads = [e for e in leaves if e isa RK._ARead]
    writes = [e for e in leaves if e isa RK._AWrite]
    @test length(writes) == 2
    # all stores come LAST (after every read)
    lastread = findlast(e -> e isa RK._ARead, leaves)
    firstwrite = findfirst(e -> e isa RK._AWrite, leaves)
    @test lastread < firstwrite
    # all RHS container reads (:b) precede ALL LHS address container reads (:a)
    bpos = findlast(e -> e isa RK._ARead && e.path == (:b,), reads)
    apos = findfirst(e -> e isa RK._ARead && e.path == (:a,), reads)
    @test bpos !== nothing && apos !== nothing && bpos < apos
end

@testset "finding 9 — while: guaranteed initial condition OUTSIDE the loop + carried retest" begin
    lp = _mir(_Probes.pwhile, :loop!)
    ev = RK.access_events(lp)
    # a top-level (guaranteed, once) read of :c precedes the loop group
    li = findfirst(e -> e isa RK._ALoop, ev)
    @test li !== nothing
    @test any(e -> e isa RK._ARead && e.path == (:c,), ev[1:li-1])       # initial cond, outside loop
    @test any(e -> e isa RK._ARead && e.path == (:c,), _ae_all(ev[li].body))  # carried retest, inside
    @test ev[li].may_run_zero
end

@testset "finding 10 — RNG not classified by the spelling `rng`" begin
    pr = _Probes.prng
    coll = _mir(pr, :collide!)
    # `opaquef(rng)` — arg spelled `rng`, callee unregistered -> OPAQUE, never an :rng effect
    @test !(:rng in coll.effects)
    @test any(e -> e[1] === :external && e[3] === :opaque, RK.call_edges(coll))
    @test !any(e -> e isa RK._ACall && e.kind === :rng, _ae_all(RK.access_events(coll)))
    ren = _mir(pr, :renamed!)                                            # differently-named RNG formal
    @test !(:rng in ren.effects)
    @test any(e -> e[1] === :external && e[3] === :opaque, RK.call_edges(ren))
end

@testset "finding 6 — optional field ONLY under a dominating `isnothing(f) ||` at that site" begin
    po = _Probes.poptional
    function fhint(ir)                                     # the _FieldCall on (:f,), or nothing
        found = Ref{Any}(nothing)
        _body_find(x -> x isa RK._FieldCall && x.path == (:f,) ? (found[] = x; true) : false, ir)
        found[]
    end
    g = fhint(_mir(po, :guarded!));   @test g !== nothing && g.hint === :registered_or_nothing
    u = fhint(_mir(po, :unguarded!)); @test u !== nothing && u.hint === :registered
    a = fhint(_mir(po, :andguard!));  @test a !== nothing && a.hint === :registered   # `&&` never optional
end

@testset "finding 2 — registered callees from the def-time SNAPSHOT (bare/qualified/rebind)" begin
    rc(ir) = let out = Ref{Any}(nothing)
        _body_find(n -> n isa RK._RegisteredCall ? (out[] = n; true) : false, ir); out[]
    end
    # bare `bare_lf!(pp; s)` -> _RegisteredCall with the captured registration identity (Token/kind)
    b = rc(_mir(_Probes.pregbare, :drive!))
    @test b !== nothing && !b.intrinsic
    @test b.registration.kind === :free_method
    @test b.registration.token === RK.kernel_token(_Probes._RegMod.base_lf!)   # captured Token identity
    @test b.ref.field === nothing                                             # authored BARE
    # qualified `_RegMod.base_lf!(pp; s)` -> the authored SLOT is the module + field
    q = rc(_mir(_Probes.pregqual, :drive!))
    @test q !== nothing && q.ref.field === :base_lf!
    @test q.registration.token === RK.kernel_token(_Probes._RegMod.base_lf!)
    # REBIND drift: `driftf` was a registered kernel at definition, rebound to a plain λ afterwards ->
    # the captured snapshot drifts from the current binding -> REJECT actionably (finding 2 rebind check).
    d = _mir(_Probes.pregdrift, :drive!)
    @test !d.ok
    @test occursin("REBOUND", something(d.reason, ""))
end

@testset "finding 11 — field/port set from the DETACHED snapshot (mutate-then-reanalyze stable)" begin
    irs1 = RK.method_irs(_NS)
    names1 = [ir.id.name for ir in irs1]
    wr1 = RK.write_roots(_mir(_NS, :step!))
    # kernel_spec RECONSTRUCTS a fresh planning copy per call; mutating one returned spec must NOT affect
    # a re-analysis (which reads the immutable kernel_port_names snapshot, never a live shared spec).
    sp = RK.kernel_spec(_NS)
    try; empty!(sp.ports); catch; end          # vandalize the returned spec's port dict
    irs2 = RK.method_irs(_NS)
    @test [ir.id.name for ir in irs2] == names1
    @test RK.write_roots(_mir(_NS, :step!)) == wr1
    # the detached port names are non-empty and carry the composed fields
    @test :gofwd in RK.kernel_port_names(_NS)
    @test :fwd in RK.kernel_port_names(_NS) && :trees in RK.kernel_port_names(_NS)
end
