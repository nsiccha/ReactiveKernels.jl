# Inc3 factory/composition substrate tests. Isolated in a module so fixtures cannot
# shadow package exports in the shared Pkg.test Main.
module TestKernelFactory
using ReactiveKernels, Test
using LinearAlgebra, Random
const RKS = ReactiveKernels

# a PURE helper reading reactive places by reference — admissible ONLY via an explicit exact
# declaration (the public `@rk_pure`), never by body inference.
crit(a, b) = a + b
@rk_pure crit 2
@kernel critowner(x, y) = begin
    go!() = begin
        z = crit(x, y)          # reads self places x,y — admissible via @rk_pure
        Base.fill!(x, z)        # a real owned write so the closure runs
    end
end
# an UNDECLARED helper on a self place stays opaque → the closure rejects by name.
uncrit(a, b) = a + b
@kernel undeclowner(x, y) = begin
    go!() = begin
        z = uncrit(x, y)
        Base.fill!(x, z)
    end
end
# same-NAME cross-module helpers get DISTINCT identity-derived tokens (no collision).
module CritA; f(a, b) = a + b; end
module CritB; f(a, b) = a; end
@rk_pure CritA.f 2
@rk_pure CritB.f 2
# borrow / rng declarations (top-level: method defs cannot live in a testset's local scope)
borrowfn(a) = a
@rk_borrows borrowfn 1
rngfn(r, x) = x
@rk_rng rngfn 2 1
# WRONG-ARITY: crit is declared arity 2; calling it with 1 positional must reject deterministically.
@kernel wrongarity(x) = begin
    go!() = begin
        z = crit(x)
        Base.fill!(x, z)
    end
end
# REBIND adversary: a declared helper captured through a NON-const module slot; rebinding the slot
# to a module without the helper must read as rebound (same detached-identity check as primitives).
module Helpers; hot(a, b) = a + b; end
@rk_pure Helpers.hot 2
HelperMod = Helpers
@kernel hotowner(x, y) = begin
    go!() = begin
        z = HelperMod.hot(x, y)
        Base.fill!(x, z)
    end
end
baremodule NoHot end
# DESCRIPTOR-DRIFT adversary: same callable identity, re-declared with different SEMANTICS.
driftfn(a, b) = a
@rk_pure driftfn 2
@kernel driftowner(x, y) = begin
    go!() = begin
        z = driftfn(x, y)
        Base.fill!(x, z)
    end
end

# Faithful euclidean_phasepoint ENDPOINT (mirrors benchmark/nuts_kernel_authoring_fixture.jl):
# methodless — owned set is the recipe closure seeded by the integrator's subject write-roots.
@kernel phasepoint_ep(pot_f, grad_f, metric, pos, mom) = begin
    pot = pot_f(pos)
    pot, dpot_dpos = grad_f(pos)
    chol_metric = cholesky(metric)
    dkin_dmom = chol_metric \ mom
    kin = 0.5 * (@node(logdet(chol_metric)) + dot(mom, dkin_dmom))
    ham = pot + kin
    dham_dpos = dpot_dpos
    dham_dmom = dkin_dmom
end
@kernel leapfrog_ep!(phasepoint; stepsize) = begin
    @. phasepoint.mom -= stepsize * phasepoint.dham_dpos
    @. phasepoint.pos +=       stepsize * phasepoint.dham_dmom
end
# a small kernel whose LIVE mutable graph we mutate after capturing a plan (recipe-input snapshot).
@kernel mutkernel(x) = begin
    y = x + 1
    z = y * 2
end
# authored input ORDER matters for execution binding: `b - a` has inputs (b, a), NOT sorted.
@kernel orderk(a, b) = begin
    z = b - a
end
# a STATEFUL external gradient functor (DI/counting): construction must retain it by IDENTITY
# (identity + counter), never deep-copy it.
mutable struct CountingGrad; n::Int; end
(c::CountingGrad)(x) = (c.n += 1; x)
# function barriers so 0-B measurements aren't polluted by a boxed loop-local (the ops themselves
# are 0-B; a loop-scoped variable captured by @allocated is what boxes).
_canonset0b(o, v) = RKS._canon_set!(o, Val(1), v)
_canoncopy0b(o, s) = RKS._canon_copy_slot!(o, s, Val(2))
# a SECOND, byte-identical endpoint DEFINITION — same slot/recipe SHAPE, DIFFERENT canonical
# Value identities — to prove the plan key distinguishes definitions (RK 04:17c).
@kernel phasepoint_ep2(pot_f, grad_f, metric, pos, mom) = begin
    pot = pot_f(pos)
    pot, dpot_dpos = grad_f(pos)
    chol_metric = cholesky(metric)
    dkin_dmom = chol_metric \ mom
    kin = 0.5 * (@node(logdet(chol_metric)) + dot(mom, dkin_dmom))
    ham = pot + kin
    dham_dpos = dpot_dpos
    dham_dmom = dkin_dmom
end

# Single-object owned/shared fixture: method `add!` writes `total`; `combined` derives
# from `total`; `scale` is a constant recipe; `seed` is an unwritten source.
@kernel accum(seed) = begin
    total = seed
    scale = 2.0
    combined = total * scale
    add!(x) = begin
        total += x
    end
end

# a registered free integrator, for callable-field/partial resolution.
@kernel leapfrog!(phasepoint; stepsize) = begin
    @. phasepoint.mom -= stepsize * phasepoint.grad
end

# adversary: a Function with a `func` field but DIFFERENT call semantics — must NOT be
# accepted as a token-preserving binder (it never opts into the binder trait).
struct EvilWrap{F} <: Function
    func::F
end
(e::EvilWrap)(args...; kwargs...) = error("evil call semantics")

# call-induced ownership: `s` is written ONLY through the intrinsic `copy!!(s, y)`, not by
# a direct place-write — local write_roots miss it; the call-writes pass catches it.
@kernel callowner(x, y) = begin
    s = x
    drive!() = copy!!(s, y)
end

# identity-bound primitive: `Base.fill!(buf, 0.0)` writes `buf` (an owner field) through
# the registered RK-core primitive — captured DETACHED at definition with its positional
# write descriptor, keyed by the authored slot, rebind-checked (fix 1).
@kernel filler(buf) = begin
    zero!() = Base.fill!(buf, 0.0)
end

# two-primitive capture: distinct primitives captured in one owner must carry DISTINCT
# registration tokens (else they collide as _RegisteredCall edge / cache keys).
@kernel dualprim(a, b) = begin
    go!() = begin
        Base.fill!(a, 0.0)
        Base.copyto!(b, a)
    end
end

# local-spoof adversary: a method-local `fill!` shadows the primitive → it is a body LOCAL,
# excluded at ref collection, NEVER captured as the registered `Base.fill!` identity.
@kernel spooffill(buf) = begin
    zero!() = begin
        fill! = (a, b) -> a
        fill!(buf, 0.0)
    end
end

# alias-module rebind adversary: `AliasMod.fill!` is captured through a NON-const module
# slot; rebinding `AliasMod` to a module without `fill!` must read as rebound.
AliasMod = Base
@kernel aliasfill(buf) = begin
    zero!() = AliasMod.fill!(buf, 0.0)
end
baremodule NoFill end

# ep-chain owner (faithful nuts_state shape): `fwd`/`bwd` become owned SOLELY through
# `step! → finish!(fwd|bwd) → start!(ep) → step_f(ep)` — never a direct write. `step_f` is
# the integrator hole (resolves to a subject-writer); `pot_f` a read-only callback whose
# only argument `metric` must therefore stay SHARED, not spuriously owned.
@kernel epowner(fwd, bwd, metric; step_f, pot_f) = begin
    start!(ep) = step_f(ep)                       # step_f writes subject → start! writes formal 1
    finish!(ep) = start!(__self__, ep)             # sibling call passes __self__ → writes formal 1
    step!() = begin
        finish!(__self__, fwd)                     # → fwd owned via the chain
        finish!(__self__, bwd)                     # → bwd owned via the chain
    end
    energy!() = pot_f(metric)                      # pot_f read-only → metric NOT owned
end

# opaque-subject adversaries: an unregistered mutator on a self place carries no descriptor.
module Qual
    evil!(x) = x
end
@kernel evilowner(buf) = begin
    go!() = evil!(buf)                       # bare unregistered → :opaque, self subject → REJECT
end
@kernel qevilowner(buf) = begin
    go!() = Qual.evil!(buf)                  # qualified unregistered → :opaque, self subject → REJECT
end

# higher-arity primitive adversary: `copyto!(dest,src,do,so,n)` resolves to the SAME
# registration as the 2-arg form but its arity does not match the descriptor → must REJECT.
@kernel bigcopy(a, b) = begin
    go!() = Base.copyto!(a, 1, b, 1, 3)
end

# FULL real-nuts discriminator (RK block pt 7): every owned endpoint reached by a DISTINCT
# mechanism, shared identities untouched, plus the overload + branch-phi + direct-formal-write
# adversaries. Owned: init (intrinsic dest) · fwd/bwd (ep chain + branch-phi + direct formal
# write) · tree/proposals (primitive dest) · control (direct scalar write). Shared: metric,
# chol (read-only reads only) · pot_f (read-only callback arg).
@kernel nutsowner(init, fwd, bwd, tree, proposals, control, metric, chol; step_f, pot_f) = begin
    start!(ep) = step_f(ep)                        # integrator hole writes its subject
    start!(ep, half) = step_f(ep)                   # OVERLOAD (arity 2) — distinct MethodId
    finish!(ep) = start!(__self__, ep)              # sibling → writes formal 1
    extend!(dir) = begin
        t = dir ? fwd : bwd                          # branch-phi alias → SET {fwd, bwd}
        finish!(__self__, t)                         # → BOTH fwd and bwd owned
    end
    flip_neg!(ep) = begin
        @. ep.mom = -ep.mom                          # DIRECT write THROUGH a formal (pt 3)
    end
    step!() = begin
        copy!!(init, proposals)                      # intrinsic dest → init owned
        Base.fill!(tree, 0.0)                        # primitive dest → tree owned
        Base.fill!(proposals, 0.0)                   # primitive dest → proposals owned
        flip_neg!(__self__, fwd)                     # direct-formal-write sibling → fwd owned
        extend!(__self__, true)                      # phi+chain → fwd, bwd owned
        control = control + 1                        # direct scalar self-write → control owned
    end
    energy!() = pot_f(init) * metric[1] * chol[1]    # read-only callback + pure reads of metric/chol
end

@testset "Inc3 factory substrate" begin
    @testset "LOCAL owned seed (not authoritative)" begin
        # the direct-write seed is exactly the mutated field
        @test RKS._kernel_factory_direct_writes(accum) == Set((:total,))
        # local seed = direct writes ∪ recipe outputs derived from them (owned-ONLY; the
        # authoritative closure over call effects + the shared complement come later).
        seed = RKS._kernel_factory_local_owned_seed(accum)
        @test :total in seed && :combined in seed          # combined derives from total
        @test !(:scale in seed) && !(:seed in seed)        # constant / unwritten source
    end

    @testset "concrete no-Ref owner storage (redirect 1)" begin
        s = RKS._OwnerState{:tok}((1, 2.0, [3.0]))
        @test RKS.owner_token(s) === :tok
        # Val-indexed reads → constant getfield on the value tuple
        @test RKS._owner_slot(s, Val(1)) === 1
        @test RKS._owner_slot(s, Val(2)) === 2.0
        # slots are a concrete VALUE tuple — NO Ref/RefValue/cell wrapper on any slot
        @test fieldtype(typeof(s), :slots) <: Tuple
        @test !any(t -> t <: Ref, fieldtypes(typeof(RKS.owner_slots(s))))
        @test fieldtypes(typeof(RKS.owner_slots(s))) == (Int, Float64, Vector{Float64})
        # scalar updates commit by ONE typed tuple replacement (same T)
        RKS._owner_commit!(s, (10, 20.0, [30.0]))
        @test RKS._owner_slot(s, Val(1)) === 10
        # a different-typed tuple does not match the typed commit (layout stability)
        @test_throws MethodError RKS._owner_commit!(s, ("x", 1, 2))
    end

    @testset "callable field / partial resolution (toward gate 4)" begin
        # `partial(leapfrog!; stepsize)` resolves to leapfrog!'s registered Token (binder
        # is token-preserving).
        reg = RKS._kernel_resolve_callable(partial(leapfrog!; stepsize = 0.1))
        @test reg !== nothing && reg.token === RKS.kernel_token(leapfrog!)
        # a bare registered kernel resolves directly
        @test RKS._kernel_resolve_callable(leapfrog!).token === RKS.kernel_token(leapfrog!)
        # an opaque Julia callable does NOT resolve → REJECT
        @test RKS._kernel_resolve_callable(sin) === nothing
        @test_throws ArgumentError RKS._kernel_resolve_callable_or_reject(:step_f, sin)
        # a partial binding a NON-kernel rejects too
        @test_throws ArgumentError RKS._kernel_resolve_callable_or_reject(:step_f, partial(sin; a = 1))
        # ADVERSARY: a Function with a `func` field but no binder-trait opt-in is REJECTED,
        # even though its `func` IS a genuine kernel (explicit-registration boundary).
        @test RKS._kernel_binder_target(EvilWrap(leapfrog!)) === nothing
        @test RKS._kernel_resolve_callable(EvilWrap(leapfrog!)) === nothing
        @test_throws ArgumentError RKS._kernel_resolve_callable_or_reject(:step_f, EvilWrap(leapfrog!))
        # the approved binder DOES opt in
        @test RKS._kernel_binder_target(partial(leapfrog!; stepsize = 0.1)) === leapfrog!
    end

    @testset "call-induced writes (transitive ownership seed)" begin
        # `s` is mutated ONLY through the intrinsic `copy!!(s, y)` — NO local place-write,
        # so `write_roots` misses it, but the call-writes pass catches it as owned.
        @test isempty(RKS._kernel_factory_direct_writes(callowner))
        @test :s in RKS._kernel_factory_call_writes(callowner)
    end

    @testset "owned SEED heuristic (PROVISIONAL, owned-candidate ONLY)" begin
        # direct-subject case only: `copy!!(s, y)` has `s` as a direct first-actual, so the
        # seed catches it. NO shared complement from a seed — `shared` is created only by
        # the authoritative API (call-graph/formal-to-actual fixed point + unresolved reject).
        owned = RKS._kernel_factory_owned_seed(callowner)
        @test owned isa Set{Symbol}
        @test :s in owned
        @test !(:s in RKS._kernel_factory_local_owned_seed(callowner))
    end

    @testset "identity-bound primitive effect registry" begin
        # exact Base.fill! → complete positional descriptor: dest(1)=write, x(2)=read,
        # result aliases positional 1 (so `x = fill!(buf,0)` preserves x===buf downstream).
        pf = RKS._kernel_primitive_effect(Base.fill!)
        @test pf.writes == (1,) && pf.reads == (2,) && pf.result_alias == 1
        @test pf.token === Symbol("__rk_primitive_Base_fill!__")
        # two-primitive collision discriminator: DISTINCT primitives carry DISTINCT tokens
        # (a shared constant token would collide in _RegisteredCall edges / plan caches).
        pc = RKS._kernel_primitive_effect(Base.copyto!)
        @test pc.token === Symbol("__rk_primitive_Base_copyto!__")
        @test pf.token !== pc.token
        # copy!! is NOT a positional primitive — it is the structural :intrinsic
        # (owned-closure copy, result===dest), carried by its own registration.
        @test RKS._kernel_primitive_effect(copy!!) === nothing
        @test RKS.kernel_registration(copy!!).kind === :intrinsic
        # a LOCAL function named `fill!` is NOT the registered identity → reject
        local myfill! = (a, b) -> a
        @test RKS._kernel_primitive_effect(myfill!) === nothing
        # foreign callables (qualified or not) are not registered → reject
        @test RKS._kernel_primitive_effect(sin) === nothing
        @test RKS._kernel_primitive_effect(println) === nothing
    end

    @testset "detached Base.fill! capture + rebind (fix 1)" begin
        caps = RKS.kernel_callee_registrations(filler)
        prims = [c for c in caps if c.registration.kind === :primitive]
        @test length(prims) == 1
        prim = only(prims)
        # captured by EXACT identity, with the detached positional descriptor
        @test prim.target === Base.fill!
        @test prim.registration.primitive_effect isa RKS._PrimitiveEffect
        @test prim.registration.primitive_effect.writes == (1,)
        # the registration token is the primitive's DISTINCT identity (no shared constant)
        @test prim.registration.token === Symbol("__rk_primitive_Base_fill!__")
        # two captured primitives in one owner carry DISTINCT tokens (edge-key collision)
        dcaps = [c for c in RKS.kernel_callee_registrations(dualprim)
                 if c.registration.kind === :primitive]
        @test length(dcaps) == 2
        @test length(unique(c.registration.token for c in dcaps)) == 2
        # keyed by the AUTHORED SLOT — an independently reconstructed ref matches (value key)
        ref2 = RKS._CapturedCalleeRef(GlobalRef(@__MODULE__, :Base), :fill!)
        @test RKS.kernel_callee_registration(filler, ref2).primitive_effect.writes == (1,)
        # the slot re-resolves to Base.fill! → NOT rebound
        @test !RKS.kernel_callee_rebound(filler, prim.ref)

        # local-spoof: a method-local `fill!` is excluded → NO primitive capture at all
        @test isempty([c for c in RKS.kernel_callee_registrations(spooffill)
                       if c.registration.kind === :primitive])

        # alias-module rebind: captured through non-const `AliasMod`; rebinding it to a
        # module without `fill!` reads as rebound (detached slot re-resolution).
        aprim = only(c for c in RKS.kernel_callee_registrations(aliasfill)
                     if c.registration.kind === :primitive)
        @test aprim.target === Base.fill!
        @test !RKS.kernel_callee_rebound(aliasfill, aprim.ref)   # AliasMod === Base still
        @eval AliasMod = NoFill                                  # rebind the module slot
        @test RKS.kernel_callee_rebound(aliasfill, aprim.ref)    # NoFill has no `fill!`
        @eval AliasMod = Base                                    # restore
    end

    @testset "authoritative interprocedural ownership fixed point (step 2/3)" begin
        # step_f resolves to the subject-writer leapfrog!; pot_f to a read-only callback.
        stepreg = RKS.kernel_registration(leapfrog!)
        @test RKS._kernel_reg_writes_subject(stepreg)            # leapfrog! writes (:mom,)
        potreg = RKS._KernelRegistration(:pot_tok, :free_method, :phasepoint,
                                         (), (:pot,), false, nothing)
        @test !RKS._kernel_reg_writes_subject(potreg)            # empty write-roots → reader
        fr = Dict(:step_f => stepreg, :pot_f => potreg)

        owned = RKS._kernel_factory_owned_authoritative(epowner; field_regs = fr)
        # fwd AND bwd owned SOLELY through the ep chain (neither is ever directly written)
        @test :fwd in owned && :bwd in owned
        # the read-only callback's argument is NOT spuriously owned
        @test !(:metric in owned)
        # shared is the COMPLEMENT, created only after the closure — excludes fwd/bwd
        shared = RKS._kernel_factory_shared(epowner; field_regs = fr)
        @test :metric in shared
        @test !(:fwd in shared) && !(:bwd in shared)

        # NON-VACUOUS: with step_f resolved to a READ-ONLY callable, the chain writes nothing,
        # so fwd/bwd are NOT owned — ownership flows solely through the resolved write-roots.
        fr_ro = Dict(:step_f => potreg, :pot_f => potreg)
        owned_ro = RKS._kernel_factory_owned_authoritative(epowner; field_regs = fr_ro)
        @test !(:fwd in owned_ro) && !(:bwd in owned_ro)

        # RK block pt 5: the AUTHORITATIVE API REJECTS an unresolved required callable field —
        # it must never let an undecided step_f be blessed shared. The TEMPLATE API defers.
        @test_throws RKS._KernelFactoryReject RKS._kernel_factory_owned_authoritative(epowner)
        @test_throws RKS._KernelFactoryReject RKS._kernel_factory_shared(epowner)
        @test isempty(RKS._kernel_factory_owned_template(epowner))     # holes → no decided write

        # req 4 REJECTS: opaque unregistered mutator on a self place (bare + qualified), and a
        # method-local `fill!` spoof (opaque at the call site) — qualification/`!` not evidence.
        @test_throws RKS._KernelFactoryReject RKS._kernel_factory_owned_authoritative(evilowner)
        @test_throws RKS._KernelFactoryReject RKS._kernel_factory_owned_authoritative(qevilowner)
        @test_throws RKS._KernelFactoryReject RKS._kernel_factory_owned_authoritative(spooffill)
        # exact Base.fill! is admitted (registered primitive) — filler closes with buf owned
        @test :buf in RKS._kernel_factory_owned_authoritative(filler)
        # higher-arity copyto! (5 positionals) mismatches the arity-2 descriptor → REJECT
        @test_throws RKS._KernelFactoryReject RKS._kernel_factory_owned_authoritative(bigcopy)
    end

    @testset "FULL real-nuts ownership discriminator (step 2/3, pt 7)" begin
        stepreg = RKS.kernel_registration(leapfrog!)
        potreg = RKS._KernelRegistration(:pot_tok, :free_method, :phasepoint,
                                         (), (:pot,), false, nothing)
        fr = Dict(:step_f => stepreg, :pot_f => potreg)
        owned = RKS._kernel_factory_owned_authoritative(nutsowner; field_regs = fr)
        shared = RKS._kernel_factory_shared(nutsowner; field_regs = fr)

        # every writable endpoint owned, each via its distinct mechanism
        for f in (:init, :fwd, :bwd, :tree, :proposals, :control)
            @test f in owned
        end
        # branch-phi: BOTH arms owned (not just the last) — fwd via `dir ? fwd : bwd`
        @test :fwd in owned && :bwd in owned
        # read-only / identity fields are SHARED, never spuriously owned
        for f in (:metric, :chol)
            @test f in shared && !(f in owned)
        end
        # owned and shared partition the ports with no overlap
        @test isempty(intersect(owned, shared))
        @test union(owned, shared) == Set{Symbol}(RKS.kernel_port_names(nutsowner))

        # OVERLOAD keying: start! has two declaration MethodIds (arity 1 and 2); the arity-1
        # call site resolves to the arity-1 candidate — both are subject-writers so fwd/bwd
        # still close, but the summary is keyed by MethodId, not the collapsed name.
        ids = [ir.id for ir in RKS.method_irs(nutsowner) if ir.id.name === Symbol("start!")]
        @test length(ids) == 2
    end

    @testset "phase-point ENDPOINT ownership + path→slot plan (RK copy!! policy, step 4)" begin
        integ = RKS.kernel_registration(leapfrog_ep!)
        @test integ.write_roots == (:mom, :pos) || integ.write_roots == (:pos, :mom)
        owned = RKS._kernel_factory_endpoint_owned(phasepoint_ep, integ)
        shared = RKS._kernel_factory_endpoint_shared(phasepoint_ep, integ)
        # OWNED: integrator-written sources + endpoint-dependent closures
        for f in (:pos, :mom, :pot, :dpot_dpos, :dkin_dmom, :kin, :ham, :dham_dpos, :dham_dmom)
            @test f in owned
        end
        # SHARED: read-only authority + metric-only closures (chol_metric, @node(logdet))
        for f in (:pot_f, :grad_f, :metric, :chol_metric)
            @test f in shared && !(f in owned)
        end
        @test any(occursin("node", String(f)) for f in shared)   # @node(logdet) shared
        @test isempty(intersect(owned, shared))

        # the DEEPLY-IMMUTABLE canonical PLAN poc consumes (never recomputes ownership)
        plan = RKS._kernel_factory_endpoint_plan(phasepoint_ep, integ)
        @test RKS.kernel_plan_slot(plan, :pos).role === :owned
        @test RKS.kernel_plan_slot(plan, :dham_dpos).role === :owned
        @test RKS.kernel_plan_slot(plan, :metric).role === :shared
        # ALIAS COLLAPSE: dham_dpos≡dpot_dpos, dham_dmom≡dkin_dmom → SAME canonical slot
        @test RKS.kernel_plan_slot(plan, :dham_dpos).canon == RKS.kernel_plan_slot(plan, :dpot_dpos).canon
        @test RKS.kernel_plan_slot(plan, :dham_dpos).slot  == RKS.kernel_plan_slot(plan, :dpot_dpos).slot
        @test RKS.kernel_plan_slot(plan, :dham_dmom).slot  == RKS.kernel_plan_slot(plan, :dkin_dmom).slot
        # 9 owned authored names, 2 alias pairs collapse → 7 distinct physical owned slots
        @test RKS.kernel_plan_nowned(plan) == 7
        # alias groups exposed (both labels retained), sorted names
        @test any(g -> Set(g) == Set((:dham_dpos, :dpot_dpos)), RKS.kernel_plan_alias_groups(plan))
        @test any(g -> Set(g) == Set((:dham_dmom, :dkin_dmom)), RKS.kernel_plan_alias_groups(plan))
        # EXACT selected-Plan identity carried (Recipe ids), not a re-derived imitation
        @test !isempty(RKS.kernel_plan_recipes(plan))
        # SELECTED PRODUCER map (canonical Value id → selected Recipe id), immutable + sorted
        prod = RKS.kernel_plan_producer(plan)
        @test prod isa Tuple && !isempty(prod) && all(e -> e isa Tuple{Int,Int}, prod)
        cdp = RKS.kernel_plan_slot(plan, :dham_dpos).canon
        @test any(e -> e[1] == cdp, prod)                 # aliased target has a selected producer
        # POST-CONSTRUCTION entry_current = HAVE ∪ producer-map KEYS (recipe-owned; no collateral)
        ec = RKS.kernel_plan_entry_current(plan)
        gphc = RKS.kernel_graph(phasepoint_ep)
        canonof(n) = RKS.canon_id(gphc, phasepoint_ep.ports[n].id)
        @test cdp in ec && canonof(:pos) in ec
        @test length(ec) == length(unique(ec))            # deduped, stable
        # every canonical HAVE is present directly
        for n in phasepoint_ep.have_names
            @test canonof(n) in ec
        end
        # ec carries ONLY HAVE ∪ producer-map keys — nothing else is blessed
        havecanon = Set(canonof(n) for n in phasepoint_ep.have_names)
        prodkeys = Set(e[1] for e in prod)
        @test all(c -> c in havecanon || c in prodkeys, ec)
        # DEF-UNIQUE key as a VALUE type parameter: fresh reads of ONE definition → SAME key;
        # a same-SHAPE DIFFERENT definition → DIFFERENT key (canonical Value identities differ).
        @test RKS.kernel_plan_key(plan) ===
              RKS.kernel_plan_key(RKS._kernel_factory_endpoint_plan(phasepoint_ep, integ))
        @test RKS.kernel_plan_key(plan) isa Tuple && RKS.kernel_plan_key(plan)[1] === integ.token
        plan2 = RKS._kernel_factory_endpoint_plan(phasepoint_ep2, integ)
        @test RKS.kernel_plan_key(plan) != RKS.kernel_plan_key(plan2)
        # the key is a VALUE type parameter → distinct keys give distinct plan TYPES (for @generated)
        @test typeof(plan) !== typeof(plan2)
        # producer tuple EXACTLY matches the actual selected Plan mapping
        havev = Value[phasepoint_ep.ports[n] for n in phasepoint_ep.have_names]
        wantv = Value[phasepoint_ep.ports[n] for n in RKS._kernel_all_ports(phasepoint_ep)
                      if !(n in phasepoint_ep.have_names)]
        plactual = RKS.plan(RKS.kernel_graph(phasepoint_ep); have = havev, want = wantv)
        @test RKS.kernel_plan_producer(plan) ==
              Tuple(sort!([(cid, r.id) for (cid, r) in plactual.producer]))
        # multi-output producer: `pot` is produced by BOTH pot_f(pos) and grad_f(pos); the producer
        # map selects EXACTLY ONE (recipe-owned), so entry_current follows it, not collateral.
        cpot = RKS.canon_id(RKS.kernel_graph(phasepoint_ep), phasepoint_ep.ports[:pot].id)
        @test count(e -> e[1] == cpot, RKS.kernel_plan_producer(plan)) == 1
        # DEEPLY IMMUTABLE: only Tuples reachable — no Vector/Dict anywhere in the seam
        @test RKS.kernel_plan_slots(plan) isa Tuple
        @test RKS.kernel_plan_alias_groups(plan) isa Tuple
        @test RKS.kernel_plan_recipes(plan) isa Tuple && RKS.kernel_plan_producer(plan) isa Tuple
        @test all(s -> s.path isa Tuple && s.canon isa Int, RKS.kernel_plan_slots(plan))
        # PATH type admits INDEXED owner children (Symbol|Int steps for trees[i])
        @test RKS._PlanSlot((:trees, 1, :mom), 1, :owned, 1).path == (:trees, 1, :mom)
        # every owner field classified exactly once
        @test length(owned) + length(shared) == length(RKS._kernel_all_ports(phasepoint_ep))
        # aliased target's PRODUCER stays canonical: dham_dpos resolves to dpot_dpos's producer
        gph = RKS.kernel_graph(phasepoint_ep)
        @test !isempty(RKS.producers_of(gph, RKS.canon_id(gph, phasepoint_ep.ports[:dham_dpos].id)))

        # DETACHED recipe-input seam (RK 04:43): each selected recipe's canonical input Value ids,
        # captured as immutable Tuples so poc never rereads the live mutable graph.
        ri = RKS.kernel_plan_recipe_inputs(plan)
        @test ri isa Tuple && !isempty(ri) && all(e -> e isa Tuple{Int,<:Tuple}, ri)
        @test all(e -> e[2] isa Tuple, ri)                       # inputs immutable (not live Vectors)
        @test any(e -> canonof(:pos) in e[2], ri)                # some recipe reads pos's canon

        # a plan over a LIVE MUTABLE graph is a SNAPSHOT: mutating the graph after capture does not
        # change the captured recipe-input seam (a fresh plan would differ; the captured one is unchanged).
        gm = RKS.kernel_graph(mutkernel)
        planm = RKS._kernel_factory_plan(mutkernel, Set((:y, :z)), Set((:x,)))
        snap = RKS.kernel_plan_recipe_inputs(planm)
        RKS.add!(gm; inputs = (mutkernel.ports[:x],), outputs = (mutkernel.ports[:z],),
                 op = identity, cost = 1.0, cse_key = nothing, effectful = false)  # alt producer of z
        @test RKS.kernel_plan_recipe_inputs(planm) === snap      # captured snapshot UNCHANGED

        # AUTHORED INPUT ORDER preserved (RK 05:00 pt1/5): `z = b - a` records inputs (b, a), the
        # exact order the execution applier binds — NOT sorted; a position swap is distinguishable.
        go = RKS.kernel_graph(orderk)
        ca = RKS.canon_id(go, orderk.ports[:a].id); cb = RKS.canon_id(go, orderk.ports[:b].id)
        plano = RKS._kernel_factory_plan(orderk, Set((:z,)), Set((:a, :b)))
        rio = RKS.kernel_plan_recipe_inputs(plano)
        @test length(rio) == 1 && rio[1][2] == (cb, ca)         # authored order (b, a)
        @test rio[1][2] != Tuple(sort([ca, cb]))                # NOT sorted → a swap is distinct
        @test cb in RKS.kernel_plan_key(plano)[6][1][2]         # ordered inputs are in the Key
    end

    @testset "bare-identity canonical alias at expansion (RK 03:57 hardening)" begin
        # SINGLE-DEF identity → alias: both names one canonical Value; both labels retained
        @kernel al_single(a) = begin
            b = a
            c = b + 1
        end
        g = RKS.kernel_graph(al_single)
        @test RKS.canon_id(g, al_single.ports[:b].id) == RKS.canon_id(g, al_single.ports[:a].id)
        @test haskey(al_single.ports, :b) && haskey(al_single.ports, :a)

        # MULTI-DEF output → NOT aliased (`b=a; b=c` are alternative producers, not `a===c`)
        @kernel al_multi(a, c) = begin
            b = a
            b = c
        end
        gm = RKS.kernel_graph(al_multi)
        @test RKS.canon_id(gm, al_multi.ports[:b].id) != RKS.canon_id(gm, al_multi.ports[:a].id)
        @test RKS.canon_id(gm, al_multi.ports[:b].id) != RKS.canon_id(gm, al_multi.ports[:c].id)

        # TYPED MISMATCH → uncertain, keep the ordinary identity recipe (no collapse)
        @kernel al_typed(a::Float64) = begin
            b::Float32 = a
        end
        gt = RKS.kernel_graph(al_typed)
        @test RKS.canon_id(gt, al_typed.ports[:b].id) != RKS.canon_id(gt, al_typed.ports[:a].id)

        # REVERSE/TRANSITIVE → no cycle; canon stable (`d=b=a` all collapse to a's class)
        @kernel al_rev(a) = begin
            b = a
            d = b
        end
        gr = RKS.kernel_graph(al_rev)
        ca = RKS.canon_id(gr, al_rev.ports[:a].id)
        @test RKS.canon_id(gr, al_rev.ports[:b].id) == ca
        @test RKS.canon_id(gr, al_rev.ports[:d].id) == ca

        # `_kernel_alias!` bumps Graph.version on a real alias mutation (like CSE/merge), and the
        # canonical target's PRODUCER index stays correct before/after the alias.
        g = ReactiveKernels.Graph()
        va = ReactiveKernels.value!(g, :a, Float64)
        vb = ReactiveKernels.value!(g, :b, Float64)
        ReactiveKernels.add!(g; inputs = (), outputs = (va,), op = () -> 0.0,
                             cost = 1.0, cse_key = nothing, effectful = false)
        prod_before = length(RKS.producers_of(g, RKS.canon_id(g, va.id)))
        ver0 = g.version
        RKS._kernel_alias!(g, vb, va, identity, 1.0)     # b ≡ a
        @test g.version == ver0 + 1                       # real canonical mutation bumped version
        @test RKS.canon_id(g, vb.id) == RKS.canon_id(g, va.id)
        # a's producer index is unchanged; b now resolves to a's producer via canon
        @test length(RKS.producers_of(g, RKS.canon_id(g, va.id))) == prod_before
        @test length(RKS.producers_of(g, RKS.canon_id(g, vb.id))) == prod_before
        # a NO-OP alias (already same class) does NOT bump the version
        ver1 = g.version
        RKS._kernel_alias!(g, vb, va, identity, 1.0)
        @test g.version == ver1
    end

    @testset "public exact-identity effect descriptors (@rk_* + built-in randn!/lmul!)" begin
        # RK-core built-in RNG/effect primitives for refresh_momentum!!
        rn = RKS._kernel_primitive_effect(Random.randn!)
        @test rn.kind === :rng && rn.order === :ordered && rn.rng_arg == 1
        @test rn.writes == (2,) && rn.reads == (1,) && rn.result_alias == 2
        lm = RKS._kernel_primitive_effect(LinearAlgebra.lmul!)
        @test lm.kind === :effect && lm.writes == (2,) && lm.reads == (1, 2) && lm.result_alias == 2

        # a DECLARED pure helper is captured detached as :declared_effect, IDENTITY-derived token
        caps = RKS.kernel_callee_registrations(critowner)
        dec = only(c for c in caps if c.registration.kind === :declared_effect)
        @test dec.target === crit
        @test dec.registration.primitive_effect.kind === :pure
        @test dec.registration.primitive_effect.token === typeof(crit)
        @test !RKS.kernel_callee_rebound(critowner, dec.ref)          # slot resolves → not rebound
        # the closure ADMITS the declared helper over self places (no opaque reject); x owned via fill!
        @test :x in RKS._kernel_factory_owned_authoritative(critowner)
        # an UNDECLARED helper on a self place REJECTS by exact name
        @test_throws RKS._KernelFactoryReject RKS._kernel_factory_owned_authoritative(undeclowner)

        # same-NAME cross-module helpers → DISTINCT identity-derived tokens (no collision)
        @test RKS._kernel_declared_effect(CritA.f).token !== RKS._kernel_declared_effect(CritB.f).token
        @test RKS._kernel_declared_effect(CritA.f).token === typeof(CritA.f)
        # @rk_borrows marks the result as borrowing ALL actuals; @rk_rng names the RNG position
        @test RKS._kernel_declared_effect(borrowfn).borrows == (1,) &&
              RKS._kernel_declared_effect(borrowfn).kind === :pure
        @test RKS._kernel_declared_effect(rngfn).kind === :rng &&
              RKS._kernel_declared_effect(rngfn).rng_arg == 1

        # WRONG ARITY: crit declared arity 2, called with 1 positional → deterministic reject
        @test_throws RKS._KernelFactoryReject RKS._kernel_factory_owned_authoritative(wrongarity)

        # REBIND: HelperMod.hot captured declared; rebinding HelperMod to a module without `hot`
        # reads as rebound (same detached-identity re-resolution as a primitive/registered callee)
        hc = only(c for c in RKS.kernel_callee_registrations(hotowner)
                  if c.registration.kind === :declared_effect)
        @test hc.target === Helpers.hot
        @test !RKS.kernel_callee_rebound(hotowner, hc.ref)
        @eval HelperMod = NoHot
        @test RKS.kernel_callee_rebound(hotowner, hc.ref)
        @eval HelperMod = Helpers

        # DESCRIPTOR DRIFT (RK 04:29/04:30): re-declaring the SAME identity with different semantics
        # (pure/arity-2 → rng) must read REBOUND via the WHOLE-descriptor comparison, even though
        # `typeof(driftfn)` is unchanged — and WITHOUT mutating the captured snapshot.
        dcap = only(c for c in RKS.kernel_callee_registrations(driftowner)
                    if c.registration.kind === :declared_effect)
        @test !RKS.kernel_rebound(dcap.registration, driftfn)     # descriptor matches
        capsnap = dcap.registration.primitive_effect
        @eval @rk_rng driftfn 2 1                                 # same identity, drifted semantics
        @test RKS.kernel_rebound(dcap.registration, driftfn)      # whole-descriptor drift → rebound
        @test dcap.registration.primitive_effect === capsnap      # captured snapshot UNMUTATED
        @test dcap.registration.primitive_effect.kind === :pure
        @eval @rk_pure driftfn 2                                  # restore

        # constructor/macro VALIDATION: rngpos out of range and duplicate/out-of-range positions reject
        @test_throws ArgumentError RKS._effect_rng(rngfn, 2, 3)   # rngpos 3 ∉ 1:2
        @test_throws ArgumentError RKS._effect_check(2, (0,), "x")   # position 0 out of range
        @test_throws ArgumentError RKS._effect_check(3, (1, 1), "x") # duplicate positions
        @test_throws ArgumentError RKS._effect_check(0, (), "x")     # zero arity rejected (positive only)
        @test_throws ArgumentError RKS._effect_check(-1, (), "x")    # negative arity rejected
        # Random.rand 2-positional ordered-RNG built-in (rand(rng, Bool) in step!)
        rd = RKS._kernel_primitive_effect(Random.rand)
        @test rd.kind === :rng && rd.rng_arg == 1 && rd.writes == () && rd.reads == (1, 2)
    end

    @testset "REAL nuts_state authoritative ownership (benchmark fixture, RK pt 5/7)" begin
        # Eval the APPROVED benchmark/nuts_kernel_authoring_fixture.jl into an isolated module
        # (selective import dodges the exported-name collision), declaring its HOT HELPERS via the
        # PUBLIC effect macros BEFORE nuts_state is captured — no fixture source change, no body
        # inference. This is the real end-to-end authoritative-ownership gate.
        RN = Module(:RealNuts)
        Core.eval(RN, :(using ReactiveKernels: @kernel, @node, partial, copy!!,
                                               @rk_pure, @rk_borrows, @rk_rng))
        Core.eval(RN, :(using LinearAlgebra, LogExpFunctions, Random))
        fixpath = normpath(joinpath(@__DIR__, "..", "benchmark", "nuts_kernel_authoring_fixture.jl"))
        fixsrc = read(fixpath, String)
        stmts = filter(s -> !(s isa LineNumberNode), Meta.parseall(fixsrc).args)
        isnuts(st) = st isa Expr && st.head === :macrocall && st.args[1] === Symbol("@kernel") &&
            (eq = st.args[end]; eq isa Expr && eq.head === :(=) && eq.args[1] isa Expr &&
             eq.args[1].head === :call && eq.args[1].args[1] === :nuts_state)
        decls = [:(@rk_pure finiteorneginf 1), :(@rk_borrows badd 2), :(@rk_rng randbernoullilog 2 1),
                 :(@rk_pure logswapprob 1), :(@rk_pure compute_criterion 3), :(@rk_pure smooth 3)]
        done = Ref(false)
        for st in stmts
            (st isa Expr && st.head === :using) && continue
            if !done[] && isnuts(st)
                for d in decls; Core.eval(RN, d); end
                done[] = true
            end
            Core.eval(RN, st)
        end
        ns = RN.nuts_state
        fr = Dict(:step_f => RKS.kernel_registration(RN.leapfrog!), :stats_f => nothing)
        owned = RKS._kernel_factory_owned_authoritative(ns; field_regs = fr)
        shared = RKS._kernel_factory_shared(ns; field_regs = fr)
        # EXACT top-level owned set: init/fwd/bwd endpoints + trees/proposals + control flags
        @test owned == Set((:init, :fwd, :bwd, :trees, :proposals,
                            :gofwd, :may_sample, :may_continue, :dham, :diverged))
        # SHARED: the callable field + scalar params (no unresolved field)
        @test shared == Set((:step_f, :max_depth, :min_dham, :stats_f))
        @test isempty(intersect(owned, shared))
        @test union(owned, shared) == Set{Symbol}(RKS.kernel_port_names(ns))
    end

    @testset "concrete no-Ref construction: isolation, independent masks, F32/F64 (step 5)" begin
        for T in (Float32, Float64)
            buf = T[1, 2, 3]
            a = RKS._kernel_construct_owned(Val(:ep), (T(0), buf))
            b = RKS._kernel_construct_owned(Val(:ep), (T(0), buf))
            # OWNED buffers isolated: two instances never alias, neither aliases the caller input
            @test RKS._owner_slot(a, Val(2)) !== RKS._owner_slot(b, Val(2))
            @test RKS._owner_slot(a, Val(2)) !== buf
            @test RKS._owner_slot(a, Val(2)) == buf               # value-equal (deepcopy)
            # F32/F64 PRESERVED; NO Ref / widening / Any
            @test eltype(RKS._owner_slot(a, Val(2))) === T && RKS._owner_slot(a, Val(1)) isa T
            @test fieldtypes(typeof(RKS.owner_slots(a))) == (T, Vector{T})
            @test !any(t -> t <: Ref, fieldtypes(typeof(RKS.owner_slots(a))))
            # INDEPENDENT currentness masks per instance (init/fwd/bwd carry their own)
            @test RKS._owner_current(a, Val(1)) && RKS._owner_current(b, Val(1))
            RKS._owner_kill!(a, Val(1))
            @test !RKS._owner_current(a, Val(1)) && RKS._owner_current(b, Val(1))
            RKS._owner_bless!(a, Val(1))
            @test RKS._owner_current(a, Val(1))
            # copy!! currentness transfer: dest inherits src's mask
            RKS._owner_kill!(b, Val(2))
            RKS._owner_copy_current!(a, b)
            @test !RKS._owner_current(a, Val(2))
            # mask ops are ALLOCATION-FREE (0-B) after warmup
            @test (@allocated RKS._owner_kill!(a, Val(1))) == 0
            @test (@allocated RKS._owner_current(a, Val(2))) == 0
        end

        # >WORD_BITS DISCRIMINATOR (RK pt1/04:50): a layout wider than one machine word uses W>1
        # words and NEVER shift-wraps — a slot in word 2 is independent of a slot in word 1.
        WB = RKS._WORD_BITS
        nbig = WB + 6
        big = RKS._kernel_construct_owned(Val(:big), ntuple(i -> Float64(i), nbig))
        @test length(RKS.owner_current_mask(big)) == 2            # W = cld(WB+6, WB) = 2
        @test RKS._owner_current(big, Val(WB + 1)) && RKS._owner_current(big, Val(1))
        RKS._owner_kill!(big, Val(WB + 1))                        # kill a word-2 slot
        @test !RKS._owner_current(big, Val(WB + 1)) && RKS._owner_current(big, Val(1))
        RKS._owner_bless!(big, Val(WB + 1))
        @test RKS._owner_current(big, Val(WB + 1))
        # cross-word kill/bless are ALLOCATION-FREE (0-B) — warm the exact Val specialization first
        RKS._owner_kill!(big, Val(WB + 2)); RKS._owner_bless!(big, Val(WB + 2))
        @test (@allocated RKS._owner_kill!(big, Val(WB + 2))) == 0
        RKS._owner_bless!(big, Val(WB + 3)); RKS._owner_kill!(big, Val(WB + 3))
        @test (@allocated RKS._owner_bless!(big, Val(WB + 3))) == 0

        # NONCONTIGUOUS entry-current mask: EXACTLY slots {1, WB+1} set (arbitrary subset across
        # words), nothing between — proves the mask is built from exact indices, not a low-prefix.
        m = RKS._owner_mask(nbig, [1, WB + 1])
        nc = RKS._OwnerState{:nc}(ntuple(i -> Float64(i), nbig), m)
        @test RKS._owner_current(nc, Val(1)) && RKS._owner_current(nc, Val(WB + 1))
        @test !RKS._owner_current(nc, Val(2)) && !RKS._owner_current(nc, Val(WB)) &&
              !RKS._owner_current(nc, Val(WB + 2))
        # copy!! currentness transfer requires IDENTICAL layout type; a different Token/slots/width
        # rejects (nc and big share a width but differ in Token; a small state differs in all).
        @test_throws MethodError RKS._owner_copy_current!(big, nc)    # different Token
        small = RKS._kernel_construct_owned(Val(:sm), (1.0, [2.0]))
        @test_throws MethodError RKS._owner_copy_current!(small, big) # different slots/width
    end

    @testset "shared authority: identity/mutable split + one-across-endpoints (RK pt4/5, 05:06)" begin
        gradf = CountingGrad(0)                 # EXTERNAL identity authority (stateful functor)
        metric = [1.0, 2.0]                      # PER-SAMPLER mutable authority
        # two independent SAMPLER instances
        s1 = RKS._kernel_construct_shared(Val(:auth), (gradf,), (metric, 3.0))
        s2 = RKS._kernel_construct_shared(Val(:auth), (gradf,), (metric, 3.0))
        # external grad_f is the SAME identity across samplers — never deep-copied (RK 05:06)
        @test RKS._shared_slot(s1, Val(1)) === gradf && RKS._shared_slot(s2, Val(1)) === gradf
        gradf(0)                                 # its counter is shared (identity, not a copy)
        @test RKS._shared_slot(s2, Val(1)).n == 1
        # per-sampler mutable metric/chol/@node are DISTINCT across samplers, and NOT the source
        @test RKS._shared_slot(s1, Val(2)) !== RKS._shared_slot(s2, Val(2))
        @test RKS._shared_slot(s1, Val(2)) !== metric

        # within ONE sampler, init/fwd/bwd owned endpoints share the SAME s1 by identity, isolate owned
        endpoints, sh = RKS._kernel_construct_group(Val(:ep), 3, (0.0, [1.0, 2.0, 3.0]), s1)
        @test endpoints[1] !== endpoints[2] && endpoints[2] !== endpoints[3]
        @test RKS._owner_slot(endpoints[1], Val(2)) !== RKS._owner_slot(endpoints[2], Val(2))
        @test sh === s1
        # a metric mutation is seen through the one shared object; its closure kills once
        RKS.shared_slots(sh)[2][1] = 99.0
        @test RKS._shared_slot(sh, Val(2))[1] == 99.0
        @test RKS._shared_current(sh, Val(2)); RKS._shared_kill!(sh, Val(2))
        @test !RKS._shared_current(sh, Val(2))
    end

    @testset "canonical-port superset storage: owned/shared roles, no duplicate, 0-B (RK 05:11/12)" begin
        # WRITER integrator: canonical slots {1,2} OWNED (scalar + array), {3} SHARED (array authority)
        ow = RKS._CanonOwned3(1.0, [2.0, 3.0], nothing, RKS._owner_mask(3, [1, 2]))
        sw = RKS._CanonShared3(nothing, nothing, [4.0, 5.0], RKS._owner_mask(3, [3]))
        # exactly ONE physical value per canonical id — the OTHER role's field is Nothing
        @test RKS._canon_slot(ow, Val(3)) === nothing && RKS._canon_slot(sw, Val(1)) === nothing
        @test RKS._canon_slot(ow, Val(1)) == 1.0 && RKS._canon_slot(sw, Val(3)) == [4.0, 5.0]
        @test (@inferred RKS._canon_slot(ow, Val(1))) == 1.0        # Val accessor type-stable
        # READ-ONLY integrator: SAME struct family, DIFFERENT selected layout — {} owned, {1,2,3} shared
        orr = RKS._CanonOwned3(nothing, nothing, nothing, RKS._owner_mask(3, Int[]))
        srr = RKS._CanonShared3(6.0, [7.0], [8.0], RKS._owner_mask(3, [1, 2, 3]))
        @test typeof(orr).name === typeof(ow).name                 # same TYPE FAMILY (no runtime emission)
        @test all(RKS._canon_slot(orr, Val(i)) === nothing for i in 1:3)   # nothing owned (read-only)
        @test RKS._canon_slot(srr, Val(1)) == 6.0                  # the physical value is in the shared role
        # current mask marks ONLY selected canonical fields
        @test RKS._canon_current_mask(ow) == RKS._owner_mask(3, [1, 2])
        @test RKS._canon_current_mask(orr) == RKS._owner_mask(3, Int[])
        # 0-B per-slot: scalar set + array copy + array identity, warmed, exact 0 B, mixed F32/F64
        for T in (Float32, Float64)
            o = RKS._CanonOwned3(T(1), T[2, 3], nothing, RKS._owner_mask(3, [1, 2]))
            src = RKS._CanonOwned3(T(9), T[7, 8], nothing, RKS._owner_mask(3, [1, 2]))
            v5 = T(5)                                    # pre-construct (dynamic `T(5)` would box)
            _canonset0b(o, v5); _canoncopy0b(o, src)                                   # warmup
            @test (@allocated _canonset0b(o, v5)) == 0                                 # scalar set 0-B
            buf = RKS._canon_slot(o, Val(2))
            @test (@allocated _canoncopy0b(o, src)) == 0                               # array copy 0-B
            @test RKS._canon_slot(o, Val(2)) === buf                                   # array identity kept
            @test RKS._canon_slot(o, Val(2)) == T[7, 8]                                # values transferred
        end
    end

    @testset "poc binder/plan seam accessors (RK 04:41)" begin
        integ = RKS.kernel_registration(leapfrog_ep!)
        plan = RKS._kernel_factory_endpoint_plan(phasepoint_ep, integ)
        # the integrator/owner Token via a dedicated accessor (no key-tuple destructuring by codegen)
        @test RKS.kernel_plan_token(plan) === integ.token
        # binder-kwargs trait: extended ONLY for PartialFunction, retrieves bound kwargs
        pf = partial(leapfrog!; stepsize = 0.1)
        @test RKS._kernel_binder_kwargs(pf) == (; stepsize = 0.1)
        @test RKS._kernel_binder_target(pf) === leapfrog!
        @test RKS._kernel_binder_kwargs(sin) === nothing               # not a binder
        @test RKS._kernel_binder_kwargs(EvilWrap(leapfrog!)) === nothing   # no duck-typing
        # bound numeric stepsize stays runtime-typed: .1 vs .2 are the same PartialFunction TYPE
        @test typeof(partial(leapfrog!; stepsize = 0.1)) === typeof(partial(leapfrog!; stepsize = 0.2))
    end
end

end # module TestKernelFactory
