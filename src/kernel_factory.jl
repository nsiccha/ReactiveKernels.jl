# Inc3 factory/composition substrate (V7). Consumes the Inc1 authoring substrate
# (`kernel_stateful.jl`: kernel_spec/kernel_methods/kernel_callee_registrations/…) and
# poc's MethodIR representation (`kernel_methodir.jl`: method_irs/write_roots/…).
#
# SCOPE: the CONSTRUCTION substrate that must exist BEFORE effect lowering — plan
# derivation, concrete no-Ref owned-state, one-time per-type planning, and per-instance
# copy/seed/bind construction. It does NOT build method schedules, epochs, or execute
# anything (that is the lowering lane). Kept UNMERGED.
#
# This file currently holds the FORK-INDEPENDENT plan CORE — the owned-vs-shared field
# derivation. The concrete no-Ref storage + one-time `@generated` planning + callable
# resolution + `deepcopy` owned-copy land on RK confirmation of the reported design forks.

# --- ownership derivation (LOCAL SEED only; NOT authoritative) ----------------
#
# ⚠ These two helpers compute only the LOCAL owned seed and are NOT the authoritative
# ownership/layout. Final ownership must additionally close TRANSITIVELY over sibling +
# subject-method + registered callable-field (`step_f`) / `partial` / intrinsic (`copy!!`)
# effects BEFORE final layout — e.g. `copy!!(init, proposals[end])` and resolved
# `step_f(ep)` induce owned writes ABSENT from local `write_roots` (RK 2026-08-27). Shared
# is the complement ONLY after that closed resolution; opaque unresolved effects REJECT
# rather than being guessed shared. The authoritative closure lands in a later increment;
# no downstream consumer may treat the local seed as final layout.

# The directly-written owner top-fields (the LOCAL owned SEED) across `skel`'s own methods.
function _kernel_factory_direct_writes(skel)
    writes = Set{Symbol}()
    for ir in method_irs(skel)
        for wr in write_roots(ir)
            # write_roots yields (root::Symbol, owner::Tuple{Vararg{Symbol}}) — `owner` is
            # the owned field-path prefix; its head is the top owned field.
            owner = wr[2]
            owner isa Tuple && !isempty(owner) && push!(writes, owner[1])
        end
    end
    writes
end

# LOCAL owned SEED: direct writes ∪ recipe outputs transitively derived from them, over
# the object's OWN recipe graph. A SEED for the authoritative transitive closure only —
# it deliberately does NOT compute a shared complement (that is unsound before closed call
# effects). Returns `Set{Symbol}` of locally-owned fields.
function _kernel_factory_local_owned_seed(skel)
    owned = _kernel_factory_direct_writes(skel)
    graph = kernel_graph(kernel_spec(skel))
    changed = true
    while changed
        changed = false
        for r in graph.recipes
            any(inp -> inp.name in owned, r.inputs) || continue
            for out in r.outputs
                out.name in owned || (push!(owned, out.name); changed = true)
            end
        end
    end
    intersect!(owned, Set{Symbol}(kernel_port_names(skel)))
end

# --- PROVISIONAL ownership heuristic (NOT authoritative; NOT for layout) ------
#
# ⚠ RK review of 39d4751: this closure and `_kernel_factory_call_writes` are only a
# PROVISIONAL SEED — they must NOT finalize a layout. They miss inter-procedural effects:
# in the real fixture `step_f(ep)` occurs in `start!(ep,…)` where `ep` is a FORMAL, and the
# physical `fwd`/`bwd` actuals reach it only through the sibling chain
# `step! → finish!(__self__, fwd|bwd,…) → start!(__self__, ep,…) → step_f(ep)`. The
# AUTHORITATIVE closure (a later increment, `_kernel_factory_owned_authoritative`) must:
# (1) resolve sibling/subject call edges; (2) map callee formal/subject effect-roots BACK
# through caller actual places (owned aliases / indexed child paths / branch arms); (3)
# resolve a `_FieldCall` through the captured registration's DECLARED subject write-roots
# (not treat every first arg as a whole-object write); (4) interpret `copy!!` as a
# destination owned-closure copy, and admit an opaque call touching a self/subject place
# ONLY when its captured identity has a detached primitive effect descriptor (else REJECT);
# (5) iterate to a FIXED POINT before shared=complement/layout. This `*_seed` is a
# heuristic ONLY — never a layout source.
# Returns ONLY the owned CANDIDATE `Set{Symbol}` — NO shared complement. `shared` is
# created by exactly ONE API, after the authoritative closure + unresolved-effect
# rejection (RK 2026-08-27), never from a seed.
function _kernel_factory_owned_seed(skel)
    owned = union(_kernel_factory_direct_writes(skel), _kernel_factory_call_writes(skel))
    graph = kernel_graph(kernel_spec(skel))
    changed = true
    while changed
        changed = false
        for r in graph.recipes
            any(inp -> inp.name in owned, r.inputs) || continue
            for out in r.outputs
                out.name in owned || (push!(owned, out.name); changed = true)
            end
        end
    end
    intersect!(owned, Set{Symbol}(kernel_port_names(skel)))
    owned
end

# --- transitive ownership: call-induced writes -------------------------------
#
# Local `write_roots` MISS owner fields mutated THROUGH a call (RK 2026-08-27):
# `copy!!(init, …)` writes `init`; a resolved callable field `step_f(fwd)` writes `fwd`
# (leapfrog! writes its subject). An owner field that is the FIRST-ARG subject of a
# registered-call / intrinsic / callable-field call is therefore OWNED. This walks poc's
# MethodIR bodies generically and collects those subjects.

# Generic recursion over a poc MethodIR node tree, calling `f` on every `_RegisteredCall`
# / `_FieldCall` node (before recursing into it).
function _kmir_walk_calls(f, x)
    (x isa _RegisteredCall || x isa _FieldCall) && f(x)
    if x isa _MExpr || x isa _MStmt
        for i in 1:nfields(x)
            _kmir_walk_calls(f, getfield(x, i))
        end
    elseif x isa Tuple
        for e in x
            _kmir_walk_calls(f, e)
        end
    elseif x isa Pair
        _kmir_walk_calls(f, x.second)
    end
    nothing
end

_kmir_call_pos(c::_RegisteredCall) = c.args
_kmir_call_pos(c::_FieldCall) = c.pos

# Owner top-fields written via a call whose first-arg subject is an owner field.
function _kernel_factory_call_writes(skel)
    fields = Set{Symbol}()
    for ir in method_irs(skel)
        _kmir_walk_calls(ir.body) do c
            pos = _kmir_call_pos(c)
            isempty(pos) && return
            subj = pos[1]
            subj isa _SelfField && !isempty(subj.path) && push!(fields, subj.path[1])
        end
    end
    fields
end

# --- concrete no-Ref owner storage (RK 2026-08-27 storage redirect) ----------
#
# ONE generic mutable wrapper around a fully concrete VALUE Tuple of the flattened owned
# slots — NOT a tuple of Ref/cell/box objects, and NOT a per-definition emitted struct
# (no runtime type emission / world-age). Val-indexed access lowers to a constant
# `getfield`; scalar updates are accumulated and committed by ONE typed tuple replacement
# at the schedule boundary. `Token` phantom-types the owner definition; layout (the slot
# order in `T`) is fixed by the plan AFTER the transitive effect closure.
mutable struct _OwnerState{Token,T<:Tuple,W}
    slots::T
    current::NTuple{W,UInt}   # allocation-free currentness BITSET over the physical owned slots
                              #   (RK 04:45/04:46): W = cld(nslots, 64) machine words, so a layout
                              #   WIDER than one word never shift-wraps into a single UInt. Bit
                              #   (I-1) set ⇔ slot I current; copy!! transfers it, a write KILLS a
                              #   dependent, a producer BLESSES it. Per-instance INDEPENDENT masks.
                              #   isbits (0-B word replacement). NOT a Dict/Set/Ref.
end
const _WORD_BITS = 8 * sizeof(UInt)                    # machine word width (RK 04:50 — not literal 64)
@inline _owner_word(i::Int) = (i - 1) ÷ _WORD_BITS + 1  # 1-based machine-word index of slot i
@inline _owner_bit(i::Int) = (i - 1) % _WORD_BITS       # bit offset within that word

# Exact-width mask (W = cld(nslots, WORD_BITS) words) with EXACTLY the physical slot indices in
# `set_indices` set (RK 04:45 pt3 / 04:50 — an ARBITRARY subset, e.g. the physical slots mapped
# from `plan.entry_current`, never an all-current/low-prefix guess). Construction-time only.
function _owner_mask(nslots::Int, set_indices = 1:nslots)
    W = max(cld(nslots, _WORD_BITS), 1)
    words = fill(UInt(0), W)
    for i in set_indices
        1 <= i <= nslots || continue
        words[_owner_word(i)] |= UInt(1) << _owner_bit(i)
    end
    ntuple(w -> @inbounds(words[w]), W)
end
_OwnerState{Token}(slots::T) where {Token,T<:Tuple} =
    (m = _owner_mask(length(slots)); _OwnerState{Token,T,length(m)}(slots, m))
_OwnerState{Token}(slots::T, current::NTuple{W,UInt}) where {Token,T<:Tuple,W} =
    _OwnerState{Token,T,W}(slots, current)

owner_token(::_OwnerState{Token}) where {Token} = Token
owner_slots(s::_OwnerState) = getfield(s, :slots)

# Val-indexed slot READ → a constant `getfield` on the value tuple (no dynamic getindex).
@inline _owner_slot(s::_OwnerState, ::Val{I}) where {I} = getfield(getfield(s, :slots), I)

# Commit a whole new concrete slot tuple in ONE TYPED replacement (same `T`). The typed
# signature enforces layout/type stability — a different-typed tuple does not match.
@inline _owner_commit!(s::_OwnerState{Token,T,W}, slots::T) where {Token,T,W} =
    (setfield!(s, :slots, slots); s)

# Currentness accessors — constant word/bit ops from WORD_BITS; the WORD index is resolved from the
# Val-indexed physical slot, so a >WORD_BITS layout addresses the correct machine word (no wrap).
@inline _owner_current(s::_OwnerState, ::Val{I}) where {I} =
    (getfield(s, :current)[_owner_word(I)] >> _owner_bit(I)) & UInt(1) == UInt(1)
@inline function _owner_kill!(s::_OwnerState, ::Val{I}) where {I}            # a write kills a slot
    c = getfield(s, :current); w = _owner_word(I)
    setfield!(s, :current, Base.setindex(c, c[w] & ~(UInt(1) << _owner_bit(I)), w)); s
end
@inline function _owner_bless!(s::_OwnerState, ::Val{I}) where {I}           # a producer blesses it
    c = getfield(s, :current); w = _owner_word(I)
    setfield!(s, :current, Base.setindex(c, c[w] | (UInt(1) << _owner_bit(I)), w)); s
end
owner_current_mask(s::_OwnerState) = getfield(s, :current)
# copy!! transfers currentness: dest inherits src's mask — restricted to an IDENTICAL layout type
# (same Token/slots/width), as the copy!! contract requires (RK 04:45 pt2); an incompatible owner
# does not match (shared-authority identity is validated by the construction seam).
@inline _owner_copy_current!(dest::_OwnerState{Tok,T,W}, src::_OwnerState{Tok,T,W}) where {Tok,T,W} =
    (setfield!(dest, :current, getfield(src, :current)); dest)

# --- compiler-owned DIAGNOSTICS storage for nuts_state (RK 08:42) ------------------------------------
#
# A `nuts_state` owns four scalar diagnostics — `n_steps`/`reached_depth` (Int) and `acceptance_rate`/
# `dham` (following `init.ham`'s type, so F32/F64 is PRESERVED — no Float64 promotion). They are the
# compiler-owned storage the registered `stats_f` (nuts_stats!) writes (n_steps + acceptance_rate per
# transition), `step!` writes (reached_depth per depth), and `start!` writes (dham per leaf). Slot order:
# 1=n_steps, 2=reached_depth, 3=acceptance_rate, 4=dham. A Val-indexed get/set/bless seam (identical shape
# to `_OwnerState`) so poc's root compiler reads/writes them literally; NO Ref, NO boxing, exact 0-B.
mutable struct _DiagnosticsStore{Tham}
    n_steps::Int
    reached_depth::Int
    acceptance_rate::Tham
    dham::Tham
    committed::UInt        # blessed at the SINGLE root commit — CROSS-epoch validity (0 mid-epoch)
    pending::UInt          # PRODUCED within the current epoch (reset + producers) — within-epoch validity
end
# Constructed from `init.ham`'s TYPE — F32/F64 PRESERVED. `n_steps`/`reached_depth` are `Int`; the fresh
# store has nothing produced or committed yet.
_diagnostics_store(::Type{Tham}) where {Tham} =
    _DiagnosticsStore{Tham}(0, 0, zero(Tham), zero(Tham), UInt(0), UInt(0))
diagnostics_ham_type(::_DiagnosticsStore{Tham}) where {Tham} = Tham
@generated _diag_slot(s::_DiagnosticsStore, ::Val{I}) where {I} = :(getfield(s, $(QuoteNode(fieldname(_DiagnosticsStore, I)))))
# A producer write marks the slot PRODUCED (within-epoch); the value change + pending set are 0-B.
@generated function _diag_set!(s::_DiagnosticsStore, ::Val{I}, v) where {I}
    fn = QuoteNode(fieldname(_DiagnosticsStore, I))
    :(setfield!(s, $fn, v); setfield!(s, :pending, getfield(s, :pending) | (UInt(1) << ($I - 1))); s)
end
# Within-epoch readability (RK 08:48): a diagnostic is READABLE once PRODUCED this epoch — a dominated
# `stats_f` read of `acceptance_rate` is valid even though the COMMITTED mask is zero mid-epoch. Poc must
# NOT recompute/reject on committed==0 alone; it consults `pending` for within-epoch reads.
@inline _diag_produced(s::_DiagnosticsStore, ::Val{I}) where {I} =
    (getfield(s, :pending) >> (I - 1)) & UInt(1) == UInt(1)
@inline _diag_committed(s::_DiagnosticsStore, ::Val{I}) where {I} =
    (getfield(s, :committed) >> (I - 1)) & UInt(1) == UInt(1)
@inline _diag_bless!(s::_DiagnosticsStore, ::Val{I}) where {I} =
    (setfield!(s, :pending, getfield(s, :pending) | (UInt(1) << (I - 1))); s)
diagnostics_pending_mask(s::_DiagnosticsStore) = getfield(s, :pending)
diagnostics_committed_mask(s::_DiagnosticsStore) = getfield(s, :committed)
# Per-transition RESET (RK 08:42/08:48): reset is itself the AUTHORITATIVE source write for all four
# diagnostics (zero in the preserved type). It ZEROS the committed mask (so an outer-epoch throw before
# the root commit leaves nothing falsely current cross-epoch) AND marks all four PENDING-PRODUCED (they
# are current within this epoch — not derived values awaiting a recipe). Exact 0-B.
@inline function _diagnostics_reset!(s::_DiagnosticsStore{Tham}) where {Tham}
    setfield!(s, :n_steps, 0); setfield!(s, :reached_depth, 0)
    setfield!(s, :acceptance_rate, zero(Tham)); setfield!(s, :dham, zero(Tham))
    setfield!(s, :committed, UInt(0))          # exception safety: nothing committed until the root commit
    setfield!(s, :pending, UInt(0x0f))         # all four PRODUCED within the epoch
    s
end
# The SINGLE root commit at epoch end (RK 08:48): bless every produced diagnostic into the committed mask.
# A throw BEFORE this leaves `committed` at its reset-zeroed value — no diagnostic is falsely current.
@inline _diagnostics_root_commit!(s::_DiagnosticsStore) =
    (setfield!(s, :committed, getfield(s, :pending)); s)

# --- callable field / partial resolution (RK callable-Token redirect) --------
#
# A callable FIELD (`step_f`, `stats_f`) must resolve to a REGISTERED kernel/intrinsic
# TOKEN identity, or be REJECTED — never an opaque runtime callable.
#
# TOKEN-PRESERVING BINDER TRAIT (RK 2026-08-27): a binder is recognized ONLY by an explicit
# opt-in — `_kernel_binder_target(binder)` returns the wrapped target, `nothing` otherwise.
# It is EXTENDED only for approved binders (RK's `PartialFunction`, in hmc.jl after that
# type is declared — include order is not a reason to duck-type). Never trust an arbitrary
# `Function` with a `.func` field: an `EvilWrap{F}<:Function` would otherwise smuggle a
# genuine Token through different call semantics, violating the explicit-registration
# compiler boundary.
_kernel_binder_target(::Any) = nothing

# Parallel binder trait (RK 04:41): the BOUND KEYWORDS of a token-preserving binder, extended ONLY
# for `PartialFunction` (in hmc.jl, after that type is declared) — the lowerer retrieves bound
# kwargs (e.g. `stepsize`) without duck-typing `.kwargs`. `nothing` = not an approved binder.
_kernel_binder_kwargs(::Any) = nothing

# Resolve a callable to its registered `_KernelRegistration`, recursing ONLY through the
# binder trait. `nothing` if it is not a registered kernel/intrinsic (or approved binder).
function _kernel_resolve_callable(v)
    reg = kernel_registration(v)
    reg === nothing || return reg
    target = _kernel_binder_target(v)
    target === nothing ? nothing : _kernel_resolve_callable(target)
end

# Resolve a callable field to its registered Token, or REJECT actionably.
function _kernel_resolve_callable_or_reject(name::Symbol, v)
    reg = _kernel_resolve_callable(v)
    reg === nothing && throw(ArgumentError(
        "callable field `$name` = $(typeof(v)) does not resolve to a registered @kernel/" *
        "intrinsic Token identity; a factory-time callable field must be a registered kernel " *
        "(or a `partial(...)` binder of one), never an opaque Julia callable."))
    reg
end

# --- identity-bound RK-core primitive effect registry -------------------------
#
# `_PrimitiveEffect` + `_kernel_primitive_effect` live in `kernel_stateful.jl` (before
# `_KernelRegistration`, whose `:primitive` kind carries the descriptor). They are
# consumed HERE by the interprocedural closure below, and captured DETACHED at owner
# definition by `_kernel_capture_callees` (rebind-checked), never reread at analysis.

# --- AUTHORITATIVE interprocedural ownership (formal-to-actual fixed point) ----
#
# The one API that produces the FINAL owned top-field set (the local `*_seed` helpers above
# are never layout). It closes ownership TRANSITIVELY over sibling / subject-method /
# registered / intrinsic / primitive / callable-field effects, mapping each callee's
# subject/formal writes BACK through the caller's actual places to a FIXED POINT
# (RK 2026-08-27). Requirements it discharges:
#   (1) resolve sibling call edges (`_Call`/`_CallExpr`, receiver `__self__`);
#   (2) map callee formal-writes back through caller actuals — owner fields (`_SelfField`),
#       owned-alias locals (`t = trees[d]`), indexed/getfield children, and a caller's OWN
#       formal (propagated up); branch/loop/guard arms all walked;
#   (3) a registered/intrinsic call writes its SUBJECT (first actual); a `_FieldCall`
#       (`step_f`) writes its subject ONLY when its RESOLVED registration is a subject-writer
#       — a pure read-only callback (`pot_f`/`stats_f`, empty write-roots) never owns its arg;
#   (4) `copy!!` = destination (first actual) owned-closure copy; an RK-core PRIMITIVE
#       (`Base.fill!`) writes its descriptor positions AFTER an EXACT arity match; an OPAQUE
#       named call (`evil!`, hint `:opaque`) touching a self place in subject position is
#       REJECTED (no detached descriptor) — qualification / `!` spelling are NOT evidence;
#   (5) iterate to a fixed point; `shared` is the complement, created by ONE API below.
#
# CALLABLE-FIELD holes (`step_f`) are FACTORY-TIME values, so this is a SPECIALIZATION-time
# closure keyed by `field_regs` (authored callable-field name → resolved `_KernelRegistration`).
# The owner-TOP-FIELD set is decidable here; the concrete sub-field LAYOUT is the later
# `@generated` specialization (RK plan-template split 02:37).

struct _KernelFactoryReject <: Exception
    reason::String
end
Base.showerror(io::IO, e::_KernelFactoryReject) = print(io, "factory ownership: ", e.reason)

# A subject-writer OWNS its first actual; a pure read-only registered callable (empty
# write-roots and not a `!!`/intrinsic/primitive strong update) does not (RK block pt 4).
_kernel_reg_writes_subject(reg::_KernelRegistration) =
    reg.kind === :intrinsic || reg.kind === :primitive ||
    !isempty(reg.write_roots) || reg.is_bang_bang

# --- identity-bound PURE/read primitives (RK block pt 6) ----------------------
#
# The locked compiler boundary: an ordinary call receiving a reactive place in ANY actual has
# UNKNOWN mutation unless a detached descriptor exists. `!`-spelling / qualification are NOT
# evidence. A call is admissible over a reactive actual iff its EXACT identity is (a) a
# registered @kernel / intrinsic / RK-core effect primitive (handled as a write), or (b) an
# identity-bound PURE primitive registered here (reads only). Everything else REJECTS.
# The exact-identity PURE primitive set + capture now live in `kernel_stateful.jl` as the
# definition-time `:pure_primitive` provenance category (`_KERNEL_PURE_PRIMS` /
# `_kernel_pure_primitive_value`, RK 06:01) — the factory NEVER does a live analysis-time spelling
# lookup: it consults the captured `_RegisteredCall` registration only.

# --- abstract place lattice + structured environment (RK block pt 1) ----------
#
# A PLACE is `(:self, top)` (owner top-field), `(:formal, pos)` (a method formal), or
# `(:escape,)` (unknown). An expression's abstract value is a SET of possible places (a phi
# from `flag ? fwd : bwd` is the UNION — never the last arm). The env maps method-locals to
# their place-sets, threaded in SOURCE ORDER with branch/loop merge (never a flat last-binding
# walk that would retroactively apply a later rebind to earlier calls).
const _Places = Set{Tuple}

function _kernel_place_of(x, env::Dict{Symbol,_Places})::_Places
    x isa _SelfField && !isempty(x.path) && return _Places([(:self, x.path[1])])
    x isa _FormalRef                      && return _Places([(:formal, x.pos)])
    x isa _LocalRef                       && return get(env, x.name, _Places([(:escape,)]))
    x isa _Index                          && return _kernel_place_of(x.base, env)
    x isa _Getfield                       && return _kernel_place_of(x.base, env)
    x isa _IfExpr                         && return union(_kernel_place_of(x.thenv, env),
                                                          _kernel_place_of(x.elsev, env))
    x isa _Short                          && return union(_kernel_place_of(x.lhs, env),
                                                          _kernel_place_of(x.rhs, env))
    x isa _BlockExpr                      && return _kernel_place_of(x.value, env)
    x isa _NodeExpr                       && return _kernel_place_of(x.inner, env)
    _Places([(:escape,)])
end

_env_merge(a::Dict{Symbol,_Places}, b::Dict{Symbol,_Places}) =
    Dict{Symbol,_Places}(k => union(get(a, k, _Places()), get(b, k, _Places()))
                         for k in union(keys(a), keys(b)))

_env_equal(a::Dict{Symbol,_Places}, b::Dict{Symbol,_Places}) =
    keys(a) == keys(b) && all(a[k] == get(b, k, _Places()) for k in keys(a))

# Mutable closure state: the growing owned top-field set + per-MethodId formal-write summary
# (RK block pt 2 — keyed by declaration MethodId, NOT name, so overloads don't collapse).
mutable struct _OwnState
    owned::Set{Symbol}
    formalw::Dict{MethodId,Set{Int}}
    field_regs::Any
    require_fields::Bool          # authoritative: an unresolved required field-call REJECTS
    byid::Dict{MethodId,MethodIR}
    changed::Bool
end

# Record a place-set as WRITES effected by method `cur`: self → owned; formal → this method's
# formalw. RK block pt 5: an (:escape,) WRITE place is an UNRESOLVED/deferred write target (a
# phi/block-alias arm that lost its place, or a `:deferred` _PlaceWrite) — in AUTHORITATIVE mode
# it must REJECT, never silently disappear from ownership; the template may defer it.
function _own_record!(st::_OwnState, cur::MethodId, places::_Places)
    for p in places
        if p[1] === :self
            p[2] in st.owned || (push!(st.owned, p[2]); st.changed = true)
        elseif p[1] === :formal
            s = get!(st.formalw, cur, Set{Int}())
            p[2] in s || (push!(s, p[2]); st.changed = true)
        elseif p[1] === :escape && st.require_fields
            throw(_KernelFactoryReject(
                "a write resolves to an UNRESOLVED/deferred place (a phi/block-alias arm that " *
                "did not resolve to an owner field or formal, or a `:deferred` target) — the " *
                "authoritative closure cannot finalize ownership over it; resolve or rewrite"))
        end
    end
end

# Resolve a callable FIELD name to its factory-time registration (`_KernelRegistration`),
# `nothing` (explicitly resolved no-effect, e.g. `stats_f=nothing`), or `:unresolved`.
function _own_field_reg(st::_OwnState, field::Symbol)
    fr = st.field_regs
    (fr isa AbstractDict && haskey(fr, field)) || return :unresolved
    fr[field]
end

# Reject an opaque/non-pure call over a reactive actual in ANY position (RK block pt 6). An exact
# pure primitive / operator is NO LONGER a live spelling exception here (RK 06:01) — it is captured
# at definition as `:pure_primitive` and emitted as a `_RegisteredCall`, so any residual `_OpCall`
# reaching this point is a genuinely opaque external and REJECTS over a reactive place.
function _own_reject_opaque!(n::_OpCall, env::Dict{Symbol,_Places})
    for a in n.args
        pl = _kernel_place_of(a, env)
        any(p -> p[1] === :self || p[1] === :formal, pl) && throw(_KernelFactoryReject(
            "call `$(n.op)` (hint $(n.hint)) receives a reactive place among its actuals but " *
            "carries NO detached effect/purity descriptor — under the locked compiler boundary " *
            "its mutation is unknowable (qualification / `!` spelling are not evidence); register " *
            "it as an effect or pure primitive, or rewrite"))
    end
end

# Does a MethodId accept `npos` positional actuals? (`npos_opt == _KMIR_POSSPLAT` ⇒ vararg.)
_own_arity_matches(mid::MethodId, npos::Int) =
    mid.npos_req <= npos <= mid.npos_req + mid.npos_opt

# SIBLING call formal-write mapping (RK block pt 2+4). Candidates come arity/kw-matched from the
# IR but NOT type-narrowed. TEMPLATE unions all candidates (over-approx). AUTHORITATIVE requires
# the candidates to AGREE on the mapped write-set — if same-arity overloads DISAGREE (a read-only
# vs a writer), it is not finalizable without concrete arg-type narrowing → REJECT.
function _own_sibling!(st::_OwnState, cur::MethodId, x, env::Dict{Symbol,_Places})
    isempty(x.candidates) && return
    perpos = [get(st.formalw, c.id, Set{Int}()) for c in x.candidates]
    if st.require_fields && length(perpos) > 1 && !all(==(perpos[1]), perpos)
        throw(_KernelFactoryReject(
            "sibling call `$(x.name)` has same-arity overloads with DIFFERENT write effects " *
            "$(perpos); the authoritative closure cannot finalize it without concrete arg-type " *
            "narrowing (template over-approximates; add typed specialization)"))
    end
    for s in perpos, p in s
        1 <= p <= length(x.pos) && _own_record!(st, cur, _kernel_place_of(x.pos[p], env))
    end
end

# Effects of ONE expression node given the current env (writes recorded; opaque calls checked).
# Record a registered/field call's SUBJECT WRITE via the callee's DECLARED subject write-ROOTS
# (RK 3rd-block pt3). When the subject actual is `__self__` (`_SelfRef`), those roots are OWNER
# top-fields → owned (e.g. `stats_f(__self__)` = the diagnostics callback writing n_steps /
# reached_depth / acceptance_rate). Otherwise the subject PLACE itself (an owner field or a formal
# endpoint) is written at top-field granularity.
function _own_subject_write!(st::_OwnState, cur::MethodId, subject, write_roots, env)
    if subject isa _SelfRef
        for r in write_roots
            _own_record!(st, cur, _Places([(:self, r)]))
        end
    else
        _own_record!(st, cur, _kernel_place_of(subject, env))
    end
end

function _own_expr_effects!(st::_OwnState, cur::MethodId, x, env::Dict{Symbol,_Places})
    if x isa _RegisteredCall
        reg = x.registration
        if reg.kind === :primitive
            pe = reg.primitive_effect
            length(x.args) == pe.arity || throw(_KernelFactoryReject(
                "primitive $(reg.token) captured for arity $(pe.arity) but called with " *
                "$(length(x.args)) positionals — unsupported arity, refusing to summarize " *
                "(higher-arity actuals would be silently dropped)"))
            for w in pe.writes
                w <= length(x.args) && _own_record!(st, cur, _kernel_place_of(x.args[w], env))
            end
        elseif reg.kind === :intrinsic
            isempty(x.args) || _own_record!(st, cur, _kernel_place_of(x.args[1], env))  # copy!! dest
        elseif reg.kind === :declared_effect
            # an author-DECLARED helper (@rk_pure/@rk_borrows/@rk_rng): admissible over reactive
            # places (NOT rejected as opaque). It owns only its DECLARED write positions (pure/rng
            # readers have none → no spurious ownership); rng_arg/borrows/order are lowering metadata.
            de = reg.primitive_effect
            length(x.args) == de.arity || throw(_KernelFactoryReject(
                "declared-effect $(de.token) captured for arity $(de.arity) but called with " *
                "$(length(x.args)) positionals — unsupported arity"))
            for w in de.writes
                w <= length(x.args) && _own_record!(st, cur, _kernel_place_of(x.args[w], env))
            end
        elseif reg.kind === :pure_primitive
            # An exact-identity RK-core PURE primitive (RK 06:01) — an ordinary/dotted/compound
            # operator or named Base/stdlib pure callable. Reads EVERY actual, writes NOTHING, any
            # arity: admissible over reactive places (NOT opaque), contributes no ownership.
        else                                       # :free_method / :object_kernel / :stateless
            # RK block pt 4: a registered call owns its subject ONLY if it writes it; a pure
            # registered reducer (empty roots) must NOT own its first actual. Its subject WRITE goes
            # to the callee's DECLARED write-roots (owner fields when the subject is __self__).
            _kernel_reg_writes_subject(reg) && !isempty(x.args) &&
                _own_subject_write!(st, cur, x.args[1], reg.write_roots, env)
        end
    elseif x isa _FieldCall
        field = isempty(x.path) ? :_ : x.path[1]
        reg = _own_field_reg(st, field)
        if reg === :unresolved
            st.require_fields && throw(_KernelFactoryReject(
                "callable field `$field` is called but UNRESOLVED — the authoritative closure " *
                "cannot decide its effect; supply its resolved registration (or an explicit " *
                "no-effect `nothing`) before ownership/shared, or use the template API"))
        elseif reg isa _KernelRegistration && _kernel_reg_writes_subject(reg) && !isempty(x.pos)
            # resolved subject-writer: its DECLARED write-roots on the subject actual (owner fields
            # when the subject is __self__, e.g. `stats_f(__self__)` writing the diagnostics).
            _own_subject_write!(st, cur, x.pos[1], reg.write_roots, env)
        end
        # reg === nothing (resolved no-effect) → no write
    elseif x isa _SubjectMethodCall
        # RK block pt 3: EXACT resolution — match owner methods by name AND arity (self-subject).
        # A matched method writes the RECEIVER iff it self-writes; its formal-writes map to the
        # actuals. A pure overload contributes nothing. An external/unresolved subject method on
        # a reactive receiver is conservatively treated as writing it (sound over-approx; the
        # receiver is a formal for a thin entry like `nuts!!`, so this propagates, never spurious).
        npos = length(x.pos)
        matched = false
        for (mid, mir) in st.byid
            (mid.name === x.name && _own_arity_matches(mid, npos)) || continue
            matched = true
            any(w -> w[2] isa Tuple && !isempty(w[2]), write_roots(mir)) &&
                _own_record!(st, cur, _kernel_place_of(x.subject, env))
            for p in get(st.formalw, mid, Set{Int}())
                1 <= p <= npos && _own_record!(st, cur, _kernel_place_of(x.pos[p], env))
            end
        end
        matched || _own_record!(st, cur, _kernel_place_of(x.subject, env))
    elseif x isa _Call || x isa _CallExpr
        _own_sibling!(st, cur, x, env)          # RK block pt 2 (expression-position siblings)
    elseif x isa _OpCall
        _own_reject_opaque!(x, env)
    end
    nothing
end

# Walk one statement in SOURCE ORDER, threading + returning the env (branch/loop/guard merged).
function _own_stmt!(st::_OwnState, cur::MethodId, s, env::Dict{Symbol,_Places})::Dict{Symbol,_Places}
    if s isa _LocalAssign
        # evaluate rhs (effects + block-threaded env + RESULT places), then (re)bind — a later
        # rebind affects only LATER reads. The rhs's abstract places become the local's alias set,
        # so `active = begin t=fwd; t end` binds `active` to {fwd} (RK block pt 1).
        env, pl = _own_eval!(st, cur, s.rhs, env)
        if length(s.lhs) == 1
            env = copy(env); env[s.lhs[1]] = pl
        else
            env = copy(env); for nm in s.lhs; delete!(env, nm); end   # destructure → escape
        end
    elseif s isa _PlaceWrite
        # RK block pt 3: a write THROUGH a formal (`@. ep.mom = …`) or self place is recorded by
        # traversing the TARGET root — write_roots alone omits formal-rooted writes.
        _own_record!(st, cur, _kernel_place_of(s.target, env))
        env, _ = _own_eval!(st, cur, s.rhs, env)
    elseif s isa _PlaceSwap
        for w in s.targets; env = _own_stmt!(st, cur, w, env); end
    elseif s isa _SetReturn
        env = _own_stmt!(st, cur, s.write, env)
    elseif s isa _Call || s isa _CallExpr
        for a in s.pos; env, _ = _own_eval!(st, cur, a, env); end
        for kv in s.kw; env, _ = _own_eval!(st, cur, kv.second, env); end
        _own_sibling!(st, cur, s, env)           # RK block pt 2+4 (statement-position siblings)
    elseif s isa _If
        env, _ = _own_eval!(st, cur, s.cond, env)
        te = env; for t in s.thenb; te = _own_stmt!(st, cur, t, te); end
        ee = env; for e in s.elseb; ee = _own_stmt!(st, cur, e, ee); end
        env = _env_merge(te, ee)                     # phi merge (RK block pt 1)
    elseif s isa _For || s isa _While
        env, _ = _own_eval!(st, cur, s isa _While ? s.cond : s.iter, env)
        be = copy(env)
        if s isa _For
            # the loop VARIABLE aliases the iterated collection's elements — `for t in trees`
            # makes a write `t.log_weight` an owned write of `trees` (RK block pt 1).
            itpl = _kernel_place_of(s.iter, env)
            for v in s.var; be[v] = itpl; end
        end
        # RK block pt 1: iterate the body abstract-env TRANSFER to a FIXED POINT (a carried
        # alias `t=fwd; …; t=bwd` must be visible on later iterations), recording under it.
        while true
            prev = be
            cb = be
            for b in s.body; cb = _own_stmt!(st, cur, b, cb); end
            be = _env_merge(be, cb)
            _env_equal(be, prev) && break
        end
        env = _env_merge(env, be)                    # may-run-zero + loop-carry: union
    elseif s isa _Guard
        env, _ = _own_eval!(st, cur, s.cond, env)
        be = env; for b in s.body; be = _own_stmt!(st, cur, b, be); end
        env = _env_merge(env, be)                    # guard may not run
    elseif s isa _Return
        s.value === nothing || ((env, _) = _own_eval!(st, cur, s.value, env))
    elseif s isa _ExprStmt
        env, _ = _own_eval!(st, cur, s.expr, env)
    end
    env
end

# Evaluate an EXPRESSION: record its effects, thread env through value-position blocks, and
# RETURN both the updated env and the abstract RESULT-place set (RK block pt 1). This is a real
# expression transfer — `active = begin t = fwd; t end` returns {fwd}, and an `_IfExpr` arm
# containing such a block keeps its phi place, instead of collapsing to :escape.
function _own_eval!(st::_OwnState, cur::MethodId, x, env::Dict{Symbol,_Places})
    if x isa _SelfField
        return env, (isempty(x.path) ? _Places([(:escape,)]) : _Places([(:self, x.path[1])]))
    elseif x isa _FormalRef
        return env, _Places([(:formal, x.pos)])
    elseif x isa _LocalRef
        return env, get(env, x.name, _Places([(:escape,)]))
    elseif x isa _Index
        env, _ = _own_eval!(st, cur, x.base, env)
        pb = _kernel_place_of(x.base, env)
        for i in x.idxs; env, _ = _own_eval!(st, cur, i, env); end
        return env, pb                                   # index of an owned place is that place
    elseif x isa _Getfield
        return _own_eval!(st, cur, x.base, env)          # field of an owned place is that place
    elseif x isa _NodeExpr
        return _own_eval!(st, cur, x.inner, env)
    elseif x isa _BlockExpr
        for s in x.stmts; env = _own_stmt!(st, cur, s, env); end   # stmts thread env + record
        return _own_eval!(st, cur, x.value, env)                    # value under the threaded env
    elseif x isa _IfExpr
        env, _ = _own_eval!(st, cur, x.cond, env)
        et, pt = _own_eval!(st, cur, x.thenv, env)
        ee, pe = _own_eval!(st, cur, x.elsev, env)
        return _env_merge(et, ee), union(pt, pe)          # phi: env merge + place union
    elseif x isa _Short
        env, _ = _own_eval!(st, cur, x.lhs, env)
        env, pr = _own_eval!(st, cur, x.rhs, env)
        return env, union(_kernel_place_of(x.lhs, env), pr)
    else
        # a call / opcall / tuple / literal / etc.: record its effects, recurse its subexpressions
        # (threading env), and yield an ESCAPE result place (a call result is not an owned place).
        _own_expr_effects!(st, cur, x, env)
        if x isa _MExpr
            for i in 1:nfields(x)
                env = _own_eval_field!(st, cur, getfield(x, i), env)
            end
        end
        return env, _Places([(:escape,)])
    end
end
function _own_eval_field!(st, cur, f, env)
    if f isa _MExpr
        env, _ = _own_eval!(st, cur, f, env)
    elseif f isa Tuple
        for e in f; env = _own_eval_field!(st, cur, e, env); end
    elseif f isa Pair
        env = _own_eval_field!(st, cur, f.second, env)
    end
    env
end

# Generic recursion over a MethodIR stmt/expr tree, calling `f` on every node.
function _kmir_walk_calls_and_assigns(f, x)
    f(x)
    if x isa _MExpr || x isa _MStmt
        for i in 1:nfields(x)
            _kmir_walk_calls_and_assigns(f, getfield(x, i))
        end
    elseif x isa Tuple
        for e in x
            _kmir_walk_calls_and_assigns(f, e)
        end
    elseif x isa Pair
        _kmir_walk_calls_and_assigns(f, x.second)
    end
    nothing
end

# Collect the distinct callable-FIELD names actually CALLED in a method (for required-field
# resolution reporting and template/authoritative gating).
function _kernel_factory_called_fields(skel)
    fields = Set{Symbol}()
    for ir in method_irs(skel), s in ir.body
        _kmir_walk_calls_and_assigns(s) do n
            n isa _FieldCall && !isempty(n.path) && push!(fields, n.path[1])
        end
    end
    fields
end

# The shared closure engine: iterate the structured per-method pass to a FIXED POINT.
function _kernel_factory_owned_closure(skel, field_regs, require_fields::Bool)
    irs = method_irs(skel)
    isempty(irs) && return Set{Symbol}()
    for ir in irs
        ir.ok || throw(_KernelFactoryReject(
            "method `$(ir.id.name)` is not compilable ($(ir.reason)); authoritative ownership " *
            "cannot be closed over it"))
    end
    st = _OwnState(Set{Symbol}(), Dict{MethodId,Set{Int}}(), field_regs, require_fields,
                   Dict{MethodId,MethodIR}(ir.id => ir for ir in irs), true)
    while st.changed
        st.changed = false
        for ir in irs
            env = Dict{Symbol,_Places}()
            for s in ir.body
                env = _own_stmt!(st, ir.id, s, env)
            end
        end
    end
    # recipe-graph closure: an owner field derived from owned inputs is owned (own graph).
    graph = kernel_graph(kernel_spec(skel))
    gchanged = true
    while gchanged
        gchanged = false
        for r in graph.recipes
            any(inp -> inp.name in st.owned, r.inputs) || continue
            for out in r.outputs
                out.name in st.owned || (push!(st.owned, out.name); gchanged = true)
            end
        end
    end
    intersect!(st.owned, Set{Symbol}(kernel_port_names(skel)))
    st.owned
end

"""
    _kernel_factory_owned_template(skel; field_regs=Dict()) -> Set{Symbol}

The DEFINITION-TIME / partial owned set — callable-field holes may stay unresolved (they
contribute no decided effect). NOT authoritative and NEVER a source of `shared` (RK block
pt 5: an unresolved required field must never be blessed shared). Use for the plan TEMPLATE.
"""
_kernel_factory_owned_template(skel; field_regs = Dict{Symbol,Any}()) =
    _kernel_factory_owned_closure(skel, field_regs, false)

"""
    _kernel_factory_owned_authoritative(skel; field_regs=Dict()) -> Set{Symbol}

The AUTHORITATIVE owned top-field set (RK 2026-08-27, hardened per the 7-point block). Every
callable field actually CALLED must be resolved in `field_regs` — a `_KernelRegistration`
(its writer status decides), or `nothing` for an explicit no-effect (`stats_f=nothing`); an
unresolved required field REJECTS. Also throws on an opaque/non-pure reactive-actual call,
an unsupported primitive arity, or an uncompilable method.
"""
_kernel_factory_owned_authoritative(skel; field_regs = Dict{Symbol,Any}()) =
    _kernel_factory_owned_closure(skel, field_regs, true)

"""
    _kernel_factory_shared(skel; field_regs=Dict()) -> Set{Symbol}

The SHARED complement — the ONE API that creates `shared`, and ONLY after the AUTHORITATIVE
owned closure (RK: shared is never a seed's `setdiff`, and an unresolved required callable
field can never be blessed shared). A port is shared iff it is not owned.
"""
_kernel_factory_shared(skel; field_regs = Dict{Symbol,Any}()) =
    setdiff(Set{Symbol}(kernel_port_names(skel)),
            _kernel_factory_owned_authoritative(skel; field_regs = field_regs))

# --- phase-point ENDPOINT ownership (RK PINNED structural-copy policy) ---------
#
# A phase-point endpoint (`euclidean_phasepoint`) is METHODLESS — its owned set is not a
# method-body effect closure but a RECIPE-graph closure SEEDED by the integrator's declared
# subject write-roots (`leapfrog!` writes `pos`,`mom`): a field DERIVED from an owned input is
# owned (endpoint-dependent closure); a field derived only from read-only authority
# (`metric`→`chol_metric`→`@node(logdet)`) stays SHARED. This is exactly copy!!'s owned set —
# the integrator-written sources + their endpoint-dependent closures, shared authority untouched.

# Port/graph accessors that work for both a stateful skeleton and a stateless `KernelSpec`
# (a phase-point endpoint like `euclidean_phasepoint` is a methodless `KernelSpec`).
_kernel_all_ports(spec::KernelSpec) = Tuple(spec.port_order)
_kernel_all_ports(skel) = kernel_port_names(skel)
_kernel_the_graph(spec::KernelSpec) = kernel_graph(spec)
_kernel_the_graph(skel) = kernel_graph(kernel_spec(skel))

# name → graph Value, and the HAVE names, for both a stateless `KernelSpec` and a skeleton.
_kernel_ports_map(spec::KernelSpec) = spec.ports
_kernel_ports_map(skel) = kernel_spec(skel).ports
_kernel_have_names(spec::KernelSpec) = spec.have_names
_kernel_have_names(skel) = kernel_spec(skel).have_names

# CANONICAL-ID recipe-graph closure (RK block pt 4): seed port names → their canonical Value ids,
# close the recipe graph BY CANONICAL id (so a bare-identity ALIAS collapses — `dham_dpos` shares
# `dpot_dpos`'s canonical Value, hence its ownership), and report the owned NAMES (a port is owned
# iff its canonical Value is owned). Never propagates by Symbol name (which would miss aliases and
# conflate collisions).
function _kernel_factory_recipe_closure(skel, seed::Set{Symbol})
    graph = _kernel_the_graph(skel)
    pmap = _kernel_ports_map(skel)
    names = _kernel_all_ports(skel)
    canonof(n) = canon_id(graph, pmap[n].id)
    owned_canon = Set{Int}(canonof(n) for n in seed if haskey(pmap, n))
    changed = true
    while changed
        changed = false
        for r in graph.recipes
            any(v -> canon_id(graph, v.id) in owned_canon, r.inputs) || continue
            for o in r.outputs
                c = canon_id(graph, o.id)
                c in owned_canon || (push!(owned_canon, c); changed = true)
            end
        end
    end
    Set{Symbol}(n for n in names if canonof(n) in owned_canon)
end

"""
    _kernel_factory_endpoint_owned(skel, integrator) -> Set{Symbol}
    _kernel_factory_endpoint_shared(skel, integrator) -> Set{Symbol}

The owned / shared top-fields of a phase-point endpoint under `integrator` (a resolved
`_KernelRegistration`, e.g. `leapfrog!`): owned = the integrator's subject write-roots closed
over the endpoint recipe graph BY CANONICAL id; shared = complement (read-only authority +
metric-only closures). Alias projections (`dham_dpos`≡`dpot_dpos`) share a canonical Value.
"""
function _kernel_factory_endpoint_owned(skel, integrator::_KernelRegistration)
    _kernel_reg_writes_subject(integrator) || throw(_KernelFactoryReject(
        "endpoint integrator $(integrator.token) declares no subject write-roots — cannot seed " *
        "endpoint ownership"))
    _kernel_factory_recipe_closure(skel, Set{Symbol}(integrator.write_roots))
end
_kernel_factory_endpoint_shared(skel, integrator::_KernelRegistration) =
    setdiff(Set{Symbol}(_kernel_all_ports(skel)),
            _kernel_factory_endpoint_owned(skel, integrator))

# --- the DEEPLY-IMMUTABLE canonical per-object PLAN (poc's codegen seam) --------------------
#
# poc CONSUMES this, never recomputes ownership (RK 3f20181 block + 03:54/03:57). It is deeply
# immutable — only Tuples / Symbols / Ints reachable, no `Vector`/`Dict` anywhere — so a consumer
# cannot mutate layout authority; every accessor returns a detached immutable value. Keyed by the
# authored PATH + CANONICAL Value id, with ALIAS projections collapsed to ONE physical slot.

# One immutable slot entry: authored owner Path — a Tuple of steps that admits INDEXED children
# (`Symbol` field steps, `Int` index steps for `trees[i]`/`proposals[i]`; top-level is `(name,)`)
# — the canonical Value id, its role, and the 1-based physical slot within that role's tuple.
# Aliased paths share the same (canon, role, slot). All fields immutable — no mutable container.
struct _PlanSlot
    path::Tuple{Vararg{Union{Symbol,Int}}}
    canon::Int
    role::Symbol            # :owned | :shared
    slot::Int
end

# The deeply-immutable plan: a def-unique selected-plan `key`; per-path `slots`; deterministic
# `alias_groups`; the exact selected `producer` map (canonical Value id → selected Recipe id);
# the selected Plan `recipes` order; and the proven post-construction `entry_current` set.
# `Key` is an actual VALUE type parameter (RK 04:17): the immutable structural selected-plan
# identity — a Tuple of the owner/integrator Token + a per-slot (path, CANONICAL Value id, role,
# slot) signature + alias groups + the (canonical Value id, Recipe id) producer signature + the
# recipe order. Because it is a TYPE parameter, two plans with distinct key VALUES have distinct
# TYPES, so an `@generated` factory dispatches to the right specialization; two graphs of the same
# slot/recipe SHAPE but different canonical Value identities (different definitions) get different
# keys, while two fresh reads of one definition get the same key. No objectid/hash/Recipe.op.
struct _KernelPlan{Key,S<:Tuple,A<:Tuple,P<:Tuple,R<:Tuple,E<:Tuple,RI<:Tuple,RO<:Tuple}
    slots::S          # Tuple{_PlanSlot...}
    alias_groups::A   # Tuple{Tuple{Vararg{Symbol}}...} — deterministic (first-occurrence order)
    producer::P       # Tuple{Tuple{Int,Int}...} — (canonical Value id, selected Recipe id), sorted
    recipes::R        # Tuple{Int...} — selected Plan Recipe ids in execution order
    entry_current::E  # Tuple{Int...} — canonical ids CURRENT after the Plan runs = HAVE ∪ the
                      #   producer-map KEYS (recipe-owned rule; collateral is NOT blessed)
    recipe_inputs::RI # Tuple{Tuple{Int,Tuple{Vararg{Int}}}...} — (selected Recipe id, its CANONICAL
                      #   input Value ids in AUTHORED positional order), a DETACHED snapshot so a later
                      #   mutation of the live KernelSpec graph cannot change dependency/kill scheduling
                      #   under the same immutable plan key (RK 04:43). poc reads inputs from HERE.
    recipe_outputs::RO # Tuple{Tuple{Int,Tuple{Vararg{Int}}}...} — (selected Recipe id, its CANONICAL
                      #   output Value ids in EXACT AUTHORED POSITIONAL ORDER, captured from `r.outputs`
                      #   during the single planning pass; RK 07:09). A returned tuple/scalar binds to
                      #   these destinations POSITIONALLY — NEVER reconstructed by sorted ids. The
                      #   producer-OWNED subset (what may be blessed) is `producer`, not this.
end
_KernelPlan(key, slots::S, groups::A, producer::P, recipes::R, ec::E, ri::RI, ro::RO) where {S,A,P,R,E,RI,RO} =
    _KernelPlan{key,S,A,P,R,E,RI,RO}(slots, groups, producer, recipes, ec, ri, ro)
kernel_plan_recipe_inputs(p::_KernelPlan) = p.recipe_inputs
# The immutable ALL-OUTPUT identities of each selected recipe in EXACT AUTHORED POSITIONAL ORDER (RK
# 07:09) — captured from `r.outputs`, so a returned tuple/scalar binds to `output[i]`'s canonical
# destination POSITIONALLY. This is the ASSIGNMENT seam; it is NOT the blessing set (a recipe may emit a
# COLLATERAL output owned by another producer — that value is assigned but must NEVER be blessed here).
kernel_plan_recipe_outputs(p::_KernelPlan) = p.recipe_outputs
# The producer-OWNED canonical output subset of each selected recipe (RK 07:09) — from `Plan.producer`,
# so BLESSING follows producer ownership only (collateral outputs excluded). In `p.recipes` order; the
# owned subset preserves the recipe's authored output order (a filtered view of `recipe_outputs`).
function kernel_plan_producer_owned(p::_KernelPlan)
    ownedof = Dict{Int,Set{Int}}()
    for (canon, rid) in p.producer; push!(get!(ownedof, rid, Set{Int}()), canon); end
    outby = Dict{Int,Any}(p.recipe_outputs)
    Tuple((rid, Tuple(c for c in get(outby, rid, ()) if c in get(ownedof, rid, Set{Int}())))
          for rid in p.recipes)
end
# The full detached EXECUTION SEAM poc consumes (RK 07:09): each selected recipe in EXECUTION order as
# `(recipe id, ordered INPUT ids, ordered ALL-OUTPUT ids, producer-OWNED output subset)`. Inputs +
# outputs bind POSITIONALLY (assignment); only the owned subset is blessed. No Recipe.op / live-graph reread.
function kernel_plan_recipe_seam(p::_KernelPlan)
    inby = Dict{Int,Any}(p.recipe_inputs)
    outby = Dict{Int,Any}(p.recipe_outputs)
    ownby = Dict{Int,Any}(kernel_plan_producer_owned(p))
    Tuple((rid, get(inby, rid, ()), get(outby, rid, ()), get(ownby, rid, ())) for rid in p.recipes)
end
kernel_plan_key(::_KernelPlan{Key}) where {Key} = Key
# The integrator/owner definition Token (RK 04:41): poc proves `binder target Token == plan
# integrator Token` through THIS accessor, never by destructuring the key tuple internals.
kernel_plan_token(::_KernelPlan{Key}) where {Key} = Key[1]
kernel_plan_slots(p::_KernelPlan) = p.slots
kernel_plan_alias_groups(p::_KernelPlan) = p.alias_groups
kernel_plan_producer(p::_KernelPlan) = p.producer
kernel_plan_recipes(p::_KernelPlan) = p.recipes
kernel_plan_entry_current(p::_KernelPlan) = p.entry_current
function kernel_plan_slot(p::_KernelPlan, field::Symbol)
    for s in p.slots
        s.path === (field,) && return s
    end
    throw(KeyError(field))
end
kernel_plan_nowned(p::_KernelPlan) = length(unique(s.canon for s in p.slots if s.role === :owned))
kernel_plan_nshared(p::_KernelPlan) = length(unique(s.canon for s in p.slots if s.role === :shared))
# The Plan owns the canonical-Value → (role, physical field index) map that the storage family and
# poc codegen consume (RK 05:11) — `(role::Symbol, slot::Int)` for a canonical Value id, else nothing.
function kernel_plan_field(p::_KernelPlan, canon::Int)
    for s in p.slots
        s.canon == canon && return (s.role, s.slot)
    end
    nothing
end
# The canonical-port SUPERSET the storage spans: the DISTINCT canonical Value ids in authored port
# order, each with its role. Position i is the Val index in BOTH the owned and shared objects.
function kernel_plan_superset(p::_KernelPlan)
    canons = Int[]; roles = Symbol[]
    for s in p.slots
        s.canon in canons || (push!(canons, s.canon); push!(roles, s.role))
    end
    (Tuple(canons), Tuple(roles))
end

# Construct the owned + shared canonical objects for ONE endpoint from the plan + a canonical
# value map. Owned canonical fields are DEEP-COPIED (per-endpoint isolation); the opposite role's
# positions are `Nothing`. SHARED is SPLIT (RK 05:06): a canonical whose id is in `external` (a
# callable authority — grad_f/pot_f/stats_f) is kept BY IDENTITY (never copied), the rest of the
# shared authority (metric/chol/@node) is deep-copied once per sampler. `values` maps canonical id
# → value. Returns `(owned::_CanonOwned, shared::_CanonShared)`.
function _kernel_construct_endpoint(::Val{Token}, plan::_KernelPlan, values,
                                    external = ()) where {Token}
    canons, roles = kernel_plan_superset(plan)
    N = length(canons)
    ext = Set{Int}(external)
    owned_vals = ntuple(N) do i
        roles[i] === :owned ? deepcopy(values[canons[i]]) : nothing
    end
    shared_vals = ntuple(N) do i
        roles[i] === :shared ?
            (canons[i] in ext ? values[canons[i]] : deepcopy(values[canons[i]])) : nothing
    end
    # current masks start with ONLY the supplied HAVE canonical slots current (RK 05:53 pt1) — NOT
    # the producer-key/derived bits. Selected recipes (incl. the pgrad applier) execute into the
    # concrete destination and atomically BLESS their outputs after success; the finalized mask then
    # equals `kernel_plan_entry_current`. HAVE = entry_current minus the producer-map keys.
    prodk = Set{Int}(c for (c, _) in kernel_plan_producer(plan))
    have = Set{Int}(c for c in kernel_plan_entry_current(plan) if !(c in prodk))
    owned_mask = _owner_mask(N, Int[i for i in 1:N if roles[i] === :owned && canons[i] in have])
    shared_mask = _owner_mask(N, Int[i for i in 1:N if roles[i] === :shared && canons[i] in have])
    (_canon_construct(Val(:owned), owned_vals, owned_mask),
     _canon_construct(Val(:shared), shared_vals, shared_mask))
end

# Construct ONLY the owned object (HAVE-only mask) — for owned CHILDREN (fwd/bwd/proposals) that share the
# group's SINGLE shared authority by identity (RK 06:59): a child must NOT build/discard its own shared
# copy. Same HAVE-only currentness as `_kernel_construct_endpoint`; the child is then seeded from init.
function _kernel_construct_owned_child(::Val{Token}, plan::_KernelPlan, values) where {Token}
    canons, roles = kernel_plan_superset(plan)
    N = length(canons)
    owned_vals = ntuple(N) do i
        roles[i] === :owned ? deepcopy(values[canons[i]]) : nothing
    end
    prodk = Set{Int}(c for (c, _) in kernel_plan_producer(plan))
    have = Set{Int}(c for c in kernel_plan_entry_current(plan) if !(c in prodk))
    owned_mask = _owner_mask(N, Int[i for i in 1:N if roles[i] === :owned && canons[i] in have])
    _canon_construct(Val(:owned), owned_vals, owned_mask)
end

function _kernel_factory_plan(skel, owned::Set{Symbol}, shared::Set{Symbol}; key_token = nothing,
                              with_ops::Bool = false)
    graph = _kernel_the_graph(skel)
    pmap = _kernel_ports_map(skel)
    names = _kernel_all_ports(skel)
    canonof(n) = canon_id(graph, pmap[n].id)
    havek = Set(_kernel_have_names(skel))
    # ONE ABSOLUTE superset field index per canonical Value (RK 05:48): distinct canonical Values in
    # authored port order → positions 1..N shared by BOTH role structs; the role only selects which
    # struct carries the real value (the other holds Nothing). A per-role index would let a shared
    # slot and an owned slot collide on the same integer and mis-address poc's Val{I}.
    superset = Int[]
    for n in names
        (n in owned || n in shared) || continue
        c = canonof(n)
        c in superset || push!(superset, c)
    end
    slots = _PlanSlot[]
    for n in names
        (n in owned || n in shared) || continue
        c = canonof(n)
        push!(slots, _PlanSlot((n,), c, n in owned ? :owned : :shared, findfirst(==(c), superset)))
    end
    # alias groups: authored names sharing a canonical Value (>1 name), DETERMINISTIC by canonical
    # first-occurrence in authored port order (RK: reproducible plan key).
    canon_order = Int[]; bycanon = Dict{Int,Vector{Symbol}}()
    for n in names
        c = canonof(n)
        haskey(bycanon, c) || push!(canon_order, c)
        push!(get!(bycanon, c, Symbol[]), n)
    end
    groups = Tuple(Tuple(bycanon[c]) for c in canon_order if length(bycanon[c]) > 1)
    # EXACT selected Plan: producer map (canonical Value id → selected Recipe id) + recipe order.
    have = Value[pmap[n] for n in _kernel_have_names(skel)]
    want = Value[pmap[n] for n in names if !(n in havek)]
    haveplan = !isempty(want)
    producer_keys = Int[]      # canonical ids that HAVE a selected producer (recipe-owned)
    if haveplan
        pl = plan(graph; have = have, want = want)
        producer = Tuple(sort!([(cid, r.id) for (cid, r) in pl.producer]))
        recipes = Tuple(r.id for r in pl.recipes)
        # DETACHED recipe-input snapshot (RK 04:43/05:00): each selected recipe's CANONICAL input
        # Value ids in EXACT authored input ORDER (never sorted — the execution applier binds
        # arguments positionally), captured now so a live-graph mutation cannot alter it.
        recipe_inputs = Tuple((r.id, Tuple(canon_id(graph, v.id) for v in r.inputs))
                              for r in pl.recipes)
        # DETACHED ordered OUTPUT snapshot (RK 07:09): each selected recipe's CANONICAL output Value ids
        # in EXACT AUTHORED POSITIONAL ORDER from `r.outputs` — a returned tuple/scalar binds positionally,
        # so this is NEVER reconstructed by inverting/sorting the producer map (which drops order AND
        # collateral outputs). The producer-OWNED subset (blessing) stays `producer`.
        recipe_outputs = Tuple((r.id, Tuple(canon_id(graph, v.id) for v in r.outputs))
                               for r in pl.recipes)
        # The selected recipe OP VALUES in exact plan order (RK 07:20) — captured from the source-built
        # Recipe objects during THIS single planning pass, for the prepared executable handle tuple.
        selected_ops = Tuple((r.id, r.op) for r in pl.recipes)
        producer_keys = Int[cid for (cid, _) in producer]
    else
        producer = (); recipes = (); recipe_inputs = (); recipe_outputs = (); selected_ops = ()
    end
    pkset = Set(producer_keys)
    # entry_current: the proven POST-construction current set = canonical HAVE ∪ the producer-map
    # KEYS (recipe-owned rule — RK 04:12). Executing a recipe does NOT bless its COLLATERAL outputs
    # owned by another producer, so this follows `keys(pl.producer)`, NOT every recipe output.
    # Stable order: authored HAVE first, then producer-owned canons in authored port order; deduped.
    ec = Int[]
    for n in _kernel_have_names(skel)
        c = canonof(n); c in ec || push!(ec, c)
    end
    for n in names
        c = canonof(n)
        (c in pkset && !(c in ec)) && push!(ec, c)
    end
    # DEF-UNIQUE STRUCTURAL key (RK 04:12/04:17): the owner/integrator Token + a per-slot
    # (path, CANONICAL Value id, role, slot) signature + alias groups + the (canonical Value id,
    # Recipe id) producer signature + recipe order. The canonical Value IDENTITIES (stable across
    # `kernel_spec` reads of one const definition, distinct across definitions) make two same-SHAPE
    # graphs under the same integrator produce DIFFERENT keys. NO objectid/hash/Recipe.op.
    slot_sig = Tuple((s.path, s.canon, s.role, s.slot) for s in slots)
    key = (key_token, slot_sig, groups, producer, recipes, recipe_inputs, recipe_outputs)
    plan_obj = _KernelPlan(key, Tuple(slots), groups, producer, recipes, Tuple(ec), recipe_inputs, recipe_outputs)
    with_ops ? (plan_obj, selected_ops) : plan_obj
end

# The endpoint plan under an integrator (the immutable canonical map poc wires codegen against).
# The def-unique key is rooted in the INTEGRATOR's definition token (endpoint slice).
_kernel_factory_endpoint_plan(skel, integrator::_KernelRegistration; with_ops::Bool = false) =
    _kernel_factory_plan(skel, _kernel_factory_endpoint_owned(skel, integrator),
                         _kernel_factory_endpoint_shared(skel, integrator);
                         key_token = integrator.token, with_ops = with_ops)

# --- concrete per-instance CONSTRUCTION (isolation + shared identity, no-Ref) ---------------
#
# Build an isolated owner instance from the physical OWNED slot values: each is DEEP-COPIED so two
# constructions never alias (endpoint isolation — init/fwd/bwd are pairwise-distinct owned buffers)
# and neither aliases the caller's input; concrete types are PRESERVED (F32/F64, no widening/box/
# Any); the currentness mask starts all-current. SHARED authority is referenced BY IDENTITY (the
# caller threads the one shared tuple; it is never copied). No per-instance planning — the layout
# and mask width come from the immutable plan (`Token` phantom-types the owner definition).
@inline _kernel_construct_owned(::Val{Token}, owned_values::Tuple) where {Token} =
    _OwnerState{Token}(map(deepcopy, owned_values))

# SHARED AUTHORITY (RK 04:45 pt4): read-only-across-endpoints closure (metric / chol_metric /
# @node(logdet)) as its OWN typed slots + validity object, held ONCE and referenced BY IDENTITY by
# every owned endpoint (init/fwd/bwd). A metric mutation therefore invalidates chol/@node EXACTLY
# ONCE, shared across endpoints — owned-only masks cannot express that. Same isbits mask machinery.
mutable struct _SharedState{Token,T<:Tuple,W}
    slots::T
    current::NTuple{W,UInt}
end
_SharedState{Token}(slots::T) where {Token,T<:Tuple} =
    (m = _owner_mask(length(slots)); _SharedState{Token,T,length(m)}(slots, m))
shared_slots(s::_SharedState) = getfield(s, :slots)
@inline _shared_slot(s::_SharedState, ::Val{I}) where {I} = getfield(getfield(s, :slots), I)
shared_current_mask(s::_SharedState) = getfield(s, :current)
@inline _shared_current(s::_SharedState, ::Val{I}) where {I} =
    (getfield(s, :current)[_owner_word(I)] >> _owner_bit(I)) & UInt(1) == UInt(1)
@inline function _shared_kill!(s::_SharedState, ::Val{I}) where {I}   # metric write kills its closure
    c = getfield(s, :current); w = _owner_word(I)
    setfield!(s, :current, Base.setindex(c, c[w] & ~(UInt(1) << _owner_bit(I)), w)); s
end

# Construct one shared-authority instance, SPLITTING the shared complement into two classes
# (RK 05:06 — never conflate them):
#   `external`  — EXTERNAL IDENTITY authorities (the callable inputs grad_f / pot_f / stats_f, even a
#                 stateful DI/counting functor) retained BY IDENTITY: shared within AND across
#                 independent sampler instances, NEVER deep-copied (its identity + counter persist).
#   `per_sampler` — the MUTABLE shared authority (metric + derived chol_metric / @node(logdet)):
#                 deep-copied ONCE per sampler, shared by init/fwd/bwd, DISTINCT across samplers.
# Slots are `external...` (identity) then the per-sampler copies.
@inline _kernel_construct_shared(::Val{Token}, external::Tuple, per_sampler::Tuple) where {Token} =
    _SharedState{Token}((external..., map(deepcopy, per_sampler)...))

# A multi-endpoint GROUP: `n` DISTINCT owned endpoint states (each isolated / deep-copied), all
# referencing the ONE shared authority object by identity. This is the concrete no-Ref shape a
# compiled `KernelObject` holds — init/fwd/bwd owned states + one shared closure.
function _kernel_construct_group(::Val{Token}, n::Int, owned_values::Tuple,
                                 shared::_SharedState) where {Token}
    (ntuple(_ -> _kernel_construct_owned(Val(Token), owned_values), n), shared)
end

# --- per-definition canonical-port SUPERSET storage (RK 05:11/05:12) -----------
#
# TWO parametric mutable templates over the SAME canonical-port superset — an OWNED object and a
# per-sampler SHARED object. Each canonical field lives in EXACTLY ONE role (the selected Plan
# role); the OTHER object's corresponding field is `Nothing`, so there is exactly ONE physical
# value per canonical id — never a duplicated chol_metric / @node / metric / grad_f. Val indices
# are plan-resolved WITHIN the selected role; `current` marks only selected canonical fields.
#
# WORLD-AGE CLEAN via a PACKAGE-LOAD PREDECLARED ARITY FAMILY (RK 05:23): `_CanonOwnedN` /
# `_CanonSharedN` for N in 1:`_CANON_MAXN` are emitted once at package load (NOT from
# @generated/runtime, NO @eval-per-definition, NO mutable registry, NO KernelSpec identity change).
# The immutable Plan chooses N = the canonical field count; a layout beyond the supported arity
# REJECTS deterministically. Same definition writer-vs-read-only → different selected layout/mask,
# same struct family.
abstract type _CanonOwned end
abstract type _CanonShared end
const _Canon = Union{_CanonOwned,_CanonShared}
const _CANON_MAXN = 32
for N in 1:_CANON_MAXN
    tp = [Symbol(:F, i) for i in 1:N]
    flds = [:($(Symbol(:f, i))::$(Symbol(:F, i))) for i in 1:N]
    for (nm, sup) in ((Symbol(:_CanonOwned, N), :_CanonOwned), (Symbol(:_CanonShared, N), :_CanonShared))
        @eval mutable struct $nm{$(tp...),W} <: $sup
            $(flds...)
            current::NTuple{W,UInt}
        end
    end
end
# Val-indexed READ / WRITE — @generated to emit a LITERAL field-SYMBOL getfield/setfield! for the
# concrete layout (RK 05:08/05:12), so per-slot mutation is exactly 0-B for mixed scalars + arrays
# (a runtime-Int `setfield!` would box). `_canon_set!` replaces a scalar/isbits field; array buffers
# are mutated in place by `_canon_copy_slot!`.
@generated _canon_slot(s::_Canon, ::Val{I}) where {I} = :(getfield(s, $(QuoteNode(fieldname(s, I)))))
@generated function _canon_set!(s::_Canon, ::Val{I}, v) where {I}
    :(setfield!(s, $(QuoteNode(fieldname(s, I))), v); s)
end
# Structural copy of ONE selected slot preserving array identity: `copyto!` for a mutable array
# buffer, `setfield!` for a scalar/isbits slot. Only ever called on a SELECTED (real) field.
@generated function _canon_copy_slot!(dest::S, src::S, ::Val{I}) where {S<:_Canon,I}
    fn = QuoteNode(fieldname(S, I))
    quote
        d = getfield(dest, $fn)
        d isa AbstractArray ? copyto!(d, getfield(src, $fn)) : setfield!(dest, $fn, getfield(src, $fn))
        dest
    end
end
# Seed an owned CHILD endpoint from `src` (init→fwd/bwd/proposals): value + VALIDITY transfer with ZERO
# additional pgrad (RK 06:53/06:56). Restricted to `_CanonOwned` — SHARED authority is uncopyable BY TYPE
# (referenced by identity, never seeded). The EPOCH contract is NO STALE BLESSED VALUE, NOT transactional
# rollback (RK 07:08): validate every buffer shape BEFORE any mutation (a shape mismatch rejects with dest
# untouched), DIRTIFY dest, copy each slot, then install `src`'s saved mask ONLY after every slot succeeds.
# So a shape mismatch leaves values unchanged, while a POST-VALIDATION `copyto!` throw leaves dest DIRTY
# (mask zero — no stale current bit) with its VALUES unspecified/partially touched (no rollback allocated).
# `dest === src` is a mask+value-preserving no-op. Scalar-vs-buffer branch is generated from the CONCRETE
# fieldtype (no runtime `isa`); exact 0-B.
@generated function _canon_copy_endpoint!(dest::S, src::S) where {S<:_CanonOwned}
    N = fieldcount(S) - 1
    W = fieldcount(fieldtype(S, N + 1))                  # `current::NTuple{W,UInt}`
    valid = Any[]; copies = Any[]; zargs = Any[]
    for i in 1:N
        fn = QuoteNode(fieldname(S, i)); ft = fieldtype(S, i)
        if ft <: AbstractArray
            push!(valid, :(size(getfield(dest, $fn)) == size(getfield(src, $fn)) || throw(
                _KernelFactoryReject("copy-endpoint buffer shape mismatch at owned slot $($i)"))))
            push!(copies, :(copyto!(getfield(dest, $fn), getfield(src, $fn))))
        elseif ft !== Nothing                            # a real scalar/isbits slot (skip Nothing role)
            push!(copies, :(setfield!(dest, $fn, getfield(src, $fn))))
        end
    end
    for _ in 1:W; push!(zargs, :(zero(UInt))); end       # LITERAL zero mask (no closure in returned AST)
    zmask = Expr(:tuple, zargs...)
    quote
        dest === src && return dest                       # no-op: dest's mask unchanged
        saved = getfield(src, :current)
        $(valid...)                                       # ALL shape checks BEFORE any mutation
        setfield!(dest, :current, $zmask)                 # DIRTIFY dest before copies
        $(copies...)
        setfield!(dest, :current, saved)                  # install validity ONLY after all slots succeed
        dest
    end
end
_canon_current_mask(s::_Canon) = getfield(s, :current)
# Currentness of a canonical field (the field index IS the mask bit) — a producer BLESSES it.
@inline _canon_current(s::_Canon, ::Val{I}) where {I} =
    (getfield(s, :current)[_owner_word(I)] >> _owner_bit(I)) & UInt(1) == UInt(1)
@inline function _canon_bless!(s::_Canon, ::Val{I}) where {I}
    c = getfield(s, :current); w = _owner_word(I)
    setfield!(s, :current, Base.setindex(c, c[w] | (UInt(1) << _owner_bit(I)), w)); s
end
@inline function _canon_kill!(s::_Canon, ::Val{I}) where {I}
    c = getfield(s, :current); w = _owner_word(I)
    setfield!(s, :current, Base.setindex(c, c[w] & ~(UInt(1) << _owner_bit(I)), w)); s
end
# ATOMIC multi-bless (RK 05:53): set BOTH bits with ONE `setfield!` on the `current` field (the whole
# NTuple is replaced once), handling the distinct-word case. So `pot`+`dpot` become current together;
# a throw/type-mismatch BEFORE this single commit leaves both dirty. Exact 0-B.
@inline function _canon_bless2!(s::_Canon, ::Val{I}, ::Val{J}) where {I,J}
    c = getfield(s, :current); wi = _owner_word(I); wj = _owner_word(J)
    c = wi == wj ?
        Base.setindex(c, c[wi] | (UInt(1) << _owner_bit(I)) | (UInt(1) << _owner_bit(J)), wi) :
        Base.setindex(Base.setindex(c, c[wi] | (UInt(1) << _owner_bit(I)), wi),
                      c[wj] | (UInt(1) << _owner_bit(J)), wj)
    setfield!(s, :current, c)   # ONE commit
    s
end

# Per-canonical-slot KIND / TYPE for poc's expression emission (RK 05:30) — derived from the CONCRETE
# field type by Val index (literal `fieldtype`), NEVER a Symbol-name heuristic or live Dict, so
# renamed storage fields (`f1`,`f2`,…) are irrelevant. `:buffer` for an AbstractArray field, `:scalar`
# otherwise (Int/Bool/Float scalars). `_canon_slot_type` returns the exact concrete field type.
@generated function _canon_slot_kind(s::_Canon, ::Val{I}) where {I}
    QuoteNode(fieldtype(s, I) <: AbstractArray ? :buffer : :scalar)
end
@generated _canon_slot_type(s::_Canon, ::Val{I}) where {I} = fieldtype(s, I)

# --- pgrad! destination-aware applier (RK 04:24/04:36/05:02) ------------------
#
# The mathematical recipe is `pot, dpot_dpos = grad_f(pos)`, but the acceptance hook supplies an
# in-place `pgrad!(dest, pos)::T` that WRITES the caller-owned gradient buffer `dest` and RETURNS
# the potential. The factory destination-binds it to the multi-output grad recipe: the owned
# canonical grad slot is the `dest`, the scalar return fills the owned `pot` slot — both canonical
# outputs seeded ATOMICALLY in ONE call, no scratch, no allocating unary wrapper, no 2nd hot eval.
# The binding holds pgrad!'s IDENTITY + the two Val slot indices (a prepared handle — NOT Recipe.op).
struct _GradBinding{F,D,P}
    pgrad!::F      # the in-place gradient callable (external authority, kept by identity)
    dest::D        # Val{I}: the owned canonical GRAD slot (destination)
    pot::P         # Val{J}: the owned canonical POT slot (scalar return)
end
_grad_binding(pgrad!, ::Val{I}, ::Val{J}) where {I,J} = _GradBinding(pgrad!, Val(I), Val(J))
grad_binding_callable(b::_GradBinding) = b.pgrad!

# Derive the destination-aware grad binding SOLELY from the selected recipe's DETACHED canonical outputs
# (RK 05:47) — NOT caller-supplied Vals. `Val{GR}` is the TYPE-LEVEL selected grad Recipe id the
# construction driver bound to the external `pgrad!` authority. TYPE-STABLE (RK 06:49): a @generated
# function keyed by the plan's structural `Key` (which carries the producer map + slot signature as a
# value-tuple type parameter) plus the CONCRETE owned type derives LITERAL `dest`/`pot` Val indices at
# compile time — so the returned `_GradBinding` field type is concrete/@inferred, never a runtime-`Val`
# abstract binding stored in the prepared object. The recipe must be the selected producer for EXACTLY
# its two OWNED outputs; the BUFFER output is the gradient `dest`, the SCALAR output is `pot`. No
# Recipe.op — only the detached plan Key + the owned struct's field types.
@generated function _grad_binding_from_plan(::_KernelPlan{Key}, ::Val{GR}, pgrad!,
                                            owned::_CanonOwned) where {Key, GR}
    slot_sig = Key[2]                                   # (path, canon, role, absolute slot)
    producer = Key[4]                                   # (canonical id, selected Recipe id) — OWNED map
    recipe_outputs = Key[7]                             # (recipe id, ORDERED output canons) — authored order
    outs = Int[]                                        # this recipe's outputs in EXACT authored order
    for (rid, os) in recipe_outputs; rid == GR && (outs = collect(os); break); end
    length(outs) == 2 || return :(throw(_KernelFactoryReject(
        "grad recipe $($GR) has $($(length(outs))) authored output(s); a destination-aware gradient " *
        "binding requires EXACTLY two (the potential + the gradient buffer)")))
    owned_by_gr = Set{Int}(c for (c, rid) in producer if rid == GR)   # blessing follows OWNERSHIP
    all(c -> c in owned_by_gr, outs) || return :(throw(_KernelFactoryReject(
        "both grad-recipe outputs must be PRODUCER-OWNED by recipe $($GR) (no collateral)")))
    slotof = Dict{Int,Tuple{Symbol,Int}}()
    for (_, canon, role, slot) in slot_sig; slotof[canon] = (role, slot); end
    fields = Union{Nothing,Tuple{Symbol,Int}}[get(slotof, c, nothing) for c in outs]
    all(f -> f !== nothing && f[1] === :owned, fields) || return :(throw(_KernelFactoryReject(
        "both grad-recipe outputs must be OWNED canonical slots")))
    # roles from TYPED prepared metadata at EXACT authored output positions (RK 07:09), never sorted ids:
    # the BUFFER-typed output is the in-place gradient `dest`, the SCALAR-typed output is `pot`.
    slots = Int[f[2] for f in fields]
    kinds = Symbol[fieldtype(owned, s) <: AbstractArray ? :buffer : :scalar for s in slots]
    (count(==(:buffer), kinds) == 1 && count(==(:scalar), kinds) == 1) || return :(throw(_KernelFactoryReject(
        "grad-recipe outputs must be one BUFFER gradient dest + one SCALAR potential (found kinds $($kinds))")))
    dest = slots[findfirst(==(:buffer), kinds)]; pot = slots[findfirst(==(:scalar), kinds)]
    :(_GradBinding(pgrad!, Val($dest), Val($pot)))      # LITERAL indices -> concrete binding type
end
# One invocation seeds construction: pgrad!(grad_dest, pos) writes the owned grad buffer in place
# (identity preserved) and returns pot, which is set into the pot slot — atomic, allocation-free.
@inline function _kernel_apply_grad!(b::_GradBinding, s::_CanonOwned, pos)
    pot = b.pgrad!(_canon_slot(s, b.dest), pos)   # writes grad dest in place, returns pot (may THROW)
    _canon_set!(s, b.pot, pot)                    # seed pot atomically from the SAME call
    # set BOTH current bits in ONE commit ONLY AFTER the call returns (RK 05:47/05:53): a throwing/
    # partially-writing/type-mismatched pgrad! leaves NEITHER output current → a retry re-runs cleanly.
    _canon_bless2!(s, b.dest, b.pot)
    s
end

# --- PREPARED factory: the plan is captured ONCE at preparation (RK 06:55) --------------------------
#
# The callable constructor must consume a DEFINITION/PREPARATION-captured immutable plan with ZERO
# planner / live-graph access per instance. `_prepare_factory` performs the SINGLE `plan(graph)`
# invocation; the per-instance `_construct_prepared` reads only the captured plan + handles — it takes NO
# graph and never calls `plan()`, so poisoning the live graph after preparation cannot change any
# constructed instance. It self-discovers the destination recipe + external identity from source shape.

# --- sanctioned graph-recipe BARE identities (RK 07:20) ---------------------------------------------
# The EXACT LinearAlgebra graph-recipe primitives that appear as RAW recipe ops in the real phasepoint —
# MINIMAL (RK 07:30): only `cholesky` and `logdet` (`\` is :ldiv, `+`/operators are in the pure set, and
# `dot` lives INSIDE the trusted fused source op, never a raw recipe). Do not widen without a consumer.
const _KERNEL_RECIPE_PRIMS = (LinearAlgebra.cholesky, LinearAlgebra.logdet)
_recipe_bare_op_ok(@nospecialize(op)) =
    _kernel_pure_primitive_value(op) || any(x -> x === op, _KERNEL_RECIPE_PRIMS)

# Per-callee SAFE-DOMAIN admission for a RAW recipe identity at CONCRETE binding (RK 07:30) — exact
# identity is necessary but NOT sufficient (`cholesky`/`logdet`/`\`/`+` are extensible; a custom overload
# type could carry arbitrary effects). Validate the concrete argument types from the prepared slots; a
# custom-overload domain REJECTS. No IR inference — a load-bearing predicate poc calls at binding.
_recipe_dom_chol(::Type{T}) where {T} =
    T <: LinearAlgebra.Cholesky && T isa DataType && length(T.parameters) >= 2 &&
        _kernel_dom_num_matrix(T.parameters[2])
"""
    kernel_recipe_op_domain_ok(op, argtypes) -> Bool

`true` iff the RAW recipe identity `op` is safe over the CONCRETE `argtypes` (RK 07:30): `cholesky`
(builtin numeric Matrix), `logdet` (a Cholesky over a builtin Matrix, or a builtin Matrix), `\` (a
Cholesky/structured/dense builtin matrix + builtin numeric Array), and the pure primitives via their
per-callee domain. A custom-overload type over the same identity REJECTS.
"""
function kernel_recipe_op_domain_ok(@nospecialize(op), argtypes)
    isempty(argtypes) && return false
    op === LinearAlgebra.cholesky && return length(argtypes) == 1 && _kernel_dom_num_matrix(argtypes[1])
    op === LinearAlgebra.logdet && return length(argtypes) == 1 &&
        (_recipe_dom_chol(argtypes[1]) || _kernel_dom_num_matrix(argtypes[1]))
    op === Base.:\ && return length(argtypes) == 2 &&
        (_recipe_dom_chol(argtypes[1]) || _kernel_dom_lmul_lhs(argtypes[1])) && _kernel_dom_num_array(argtypes[2])
    _kernel_pure_primitive_value(op) && return _kernel_pure_callee_domain_ok(op, argtypes)
    false
end

# The compile-time execution MODE of a selected recipe op, from exact identity/registration + source Form
# + BOTH the all-output and producer-owned output counts (RK 07:20/07:24/07:30): `:destination` (a
# `:portcall` source op with EXACTLY TWO all-outputs, BOTH producer-owned → `f(dest, args…)::scalar`;
# their buffer/scalar roles are resolved later from typed slots), `:assign` (a single-output `:portcall`,
# any `:fused` source op, or a sanctioned bare identity — call the op, assign its return), `:ldiv` (the
# exact `\` built-in — in-place velocity reuse). A two-output port-call with a COLLATERAL output (not both
# owned), any other port-call arity, a raw unwrapped closure, or an unregistered named op REJECTS.
function _recipe_op_mode(@nospecialize(op), n_owned_out::Int, n_all_out::Int)
    if op isa _KernelSourceOp
        if kernel_sourceop_form(op) === :portcall
            n_all_out == 1 && return :assign                          # single-output port-call: generic
            (n_all_out == 2 && n_owned_out == 2) && return :destination
            throw(_KernelFactoryReject("port-call recipe with $n_all_out all-outputs / $n_owned_out owned " *
                "is not a valid destination (need exactly two, both producer-owned) or single-output call"))
        end
        return :assign                                               # :fused source op — generic assign
    end
    op === Base.:\ && return :ldiv
    _recipe_bare_op_ok(op) && return :assign
    throw(_KernelFactoryReject("recipe op $(op) is opaque/unregistered — only compiler-source ops " *
        "(`_KernelSourceOp`) and sanctioned exact identities are admissible as prepared handles"))
end

# One captured EXECUTABLE recipe handle (RK 07:20): the concrete op value + its `MODE` (a type parameter,
# so poc dispatches on it at compile time with no runtime lookup) + ordered canonical input/all-output ids
# + the producer-owned output subset. Immutable — a live-graph mutation after preparation cannot change it.
struct _RecipeHandle{OP,MODE,IN,OUT,OWN}
    op::OP
    inputs::IN
    outputs::OUT
    owned::OWN
end
recipe_handle_mode(::_RecipeHandle{OP,MODE}) where {OP,MODE} = MODE
recipe_handle_op(h::_RecipeHandle) = h.op
function _recipe_handle(op, ins, outs, owned)
    mode = _recipe_op_mode(op, length(owned), length(outs))
    _RecipeHandle{typeof(op),mode,typeof(ins),typeof(outs),typeof(owned)}(op, ins, outs, owned)
end

struct _PreparedFactory{Token,GR,P<:_KernelPlan,H<:Tuple,E<:Tuple}
    plan::P
    handles::H      # concrete Tuple of `_RecipeHandle` in EXACT plan.recipes order (NOT a Dict/Any)
    external::E     # the identity-kept (never deep-copied) shared canonical ids (callable authorities)
end
kernel_prepared_plan(pf::_PreparedFactory) = pf.plan
kernel_prepared_grad_recipe(::_PreparedFactory{Token,GR}) where {Token,GR} = GR
kernel_prepared_token(::_PreparedFactory{Token}) where {Token} = Token
kernel_prepared_handles(pf::_PreparedFactory) = pf.handles
kernel_prepared_external(pf::_PreparedFactory) = pf.external

# SELF-DISCOVERING preparation (RK 07:24): the single plan pass captures the plan + selected ops; the
# handle for each recipe carries its provenance-validated op + mode + ids. The DESTINATION recipe is the
# UNIQUE `:destination`-mode handle; the EXTERNAL identities are the callable (first) inputs of every
# `:portcall` recipe. NO caller-supplied grad_recipe/external — those remain only test scaffolding.
function _prepare_factory(skel, integrator::_KernelRegistration)
    plan, ops = _kernel_factory_endpoint_plan(skel, integrator; with_ops = true)
    seam = kernel_plan_recipe_seam(plan)
    # `ops` and `seam` are BOTH in exact plan.recipes order (RK 07:30): ZIP them positionally — assert
    # rid equality — so the handle tuple stays CONCRETE/inferred (no Dict{Int,Any} erasing types).
    handles = map(ops, seam) do o, s
        o[1] == s[1] || throw(_KernelFactoryReject("prepared op/seam order mismatch: $(o[1]) vs $(s[1])"))
        _recipe_handle(o[2], s[2], s[3], s[4])
    end
    dests = Int[s[1] for (h, s) in zip(handles, seam) if recipe_handle_mode(h) === :destination]
    length(dests) <= 1 || throw(_KernelFactoryReject("ambiguous: $(length(dests)) destination recipes"))
    grad_recipe = isempty(dests) ? 0 : dests[1]
    # external = the callable (FIRST) input canonical of every port-call recipe (the retained authorities)
    ext = Int[]
    for (o, s) in zip(ops, seam)
        op = o[2]
        (op isa _KernelSourceOp && kernel_sourceop_form(op) === :portcall && !isempty(s[2])) &&
            (s[2][1] in ext || push!(ext, s[2][1]))
    end
    external = Tuple(ext)
    _PreparedFactory{integrator.token, grad_recipe, typeof(plan), typeof(handles), typeof(external)}(
        plan, handles, external)
end

# Private TEST SCAFFOLDING only (RK 07:24) — never the public author-facing constructor: a prepared
# factory with EXPLICIT external ids + grad recipe, bypassing self-discovery.
function _prepare_factory_scaffold(skel, integrator::_KernelRegistration, external::Tuple, grad_recipe::Int)
    plan = _kernel_factory_endpoint_plan(skel, integrator)
    _PreparedFactory{integrator.token, grad_recipe, typeof(plan), Tuple{}, typeof(external)}(
        plan, (), external)
end

# Per-instance construction from the captured plan — NO graph, NO plan(). Builds the HAVE-only owned +
# shared endpoint, then seeds the gradient via the type-stable binding (ONE pgrad, atomic pot+dpot).
function _construct_prepared(pf::_PreparedFactory{Token,GR}, values, pgrad!, pos) where {Token,GR}
    ow, sh = _kernel_construct_endpoint(Val(Token), pf.plan, values, pf.external)
    _kernel_apply_grad!(_grad_binding_from_plan(pf.plan, Val(GR), pgrad!, ow), ow, pos)
    (ow, sh)
end

# Seed the owned CHILD endpoints (fwd/bwd/proposals) from a constructed init — value + validity transfer,
# ZERO additional pgrad (RK 06:53). Each child is an independently-constructed HAVE-only owned object of
# the SAME layout; copy transfers init's producer-blessed state exactly.
_seed_children!(init::_CanonOwned, children::Vararg{_CanonOwned}) =
    (for c in children; _canon_copy_endpoint!(c, init); end; children)

# The concrete family member for a given arity — a COMPILE-TIME type lookup (no runtime Symbol
# dispatch, no runtime emission). A layout wider than the predeclared family throws a deterministic
# reject at the @generated boundary (per-N specialization → type-stable / @inferred, RK 05:24).
@generated function _canon_owned_type(::Val{N}) where {N}
    N <= _CANON_MAXN ? Symbol(:_CanonOwned, N) :
        :(throw(_KernelFactoryReject("canonical layout arity $($N) exceeds family max $_CANON_MAXN")))
end
@generated function _canon_shared_type(::Val{N}) where {N}
    N <= _CANON_MAXN ? Symbol(:_CanonShared, N) :
        :(throw(_KernelFactoryReject("canonical layout arity $($N) exceeds family max $_CANON_MAXN")))
end
# Construct an owned / shared canonical object from its selected field values + mask. `role` is a
# `Val{:owned}`/`Val{:shared}` (NOT a runtime Symbol), and the arity is compile-time from the Tuple
# TYPE — so construction is fully type-stable / @inferred.
@inline _canon_construct(::Val{:owned}, values::Tuple, mask::NTuple) =
    _canon_owned_type(Val(length(values)))(values..., mask)
@inline _canon_construct(::Val{:shared}, values::Tuple, mask::NTuple) =
    _canon_shared_type(Val(length(values)))(values..., mask)
