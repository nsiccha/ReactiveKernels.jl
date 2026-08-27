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
mutable struct _OwnerState{Token,T<:Tuple}
    slots::T
end
_OwnerState{Token}(slots::T) where {Token,T<:Tuple} = _OwnerState{Token,T}(slots)

owner_token(::_OwnerState{Token}) where {Token} = Token
owner_slots(s::_OwnerState) = getfield(s, :slots)

# Val-indexed slot READ → a constant `getfield` on the value tuple (no dynamic getindex).
@inline _owner_slot(s::_OwnerState, ::Val{I}) where {I} = getfield(getfield(s, :slots), I)

# Commit a whole new concrete slot tuple in ONE TYPED replacement (same `T`). The typed
# signature enforces layout/type stability — a different-typed tuple does not match.
@inline _owner_commit!(s::_OwnerState{Token,T}, slots::T) where {Token,T} =
    (setfield!(s, :slots, slots); s)

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

# A subject-writer OWNS its first actual; a pure read-only callback (empty write-roots and
# not a `!!`/intrinsic/primitive strong update) does not.
_kernel_reg_writes_subject(reg::_KernelRegistration) =
    reg.kind === :intrinsic || reg.kind === :primitive ||
    !isempty(reg.write_roots) || reg.is_bang_bang

# Classify a write PLACE `_MExpr` to an owner top-field or a caller formal position, else
# `nothing` (escapes ownership). `aliases` maps owned-alias locals to their classification.
function _kernel_factory_place_target(a, aliases)
    a isa _SelfField && !isempty(a.path) && return (:self, a.path[1])
    a isa _FormalRef                      && return (:formal, a.pos)
    a isa _Index                          && return _kernel_factory_place_target(a.base, aliases)
    a isa _Getfield                       && return _kernel_factory_place_target(a.base, aliases)
    (a isa _LocalRef && haskey(aliases, a.name)) && return aliases[a.name]
    nothing
end

# Per-method owned-alias map: a method-local `t = <owned place>` makes `t` an alias of that
# place (RK req 2). Built once per method; later local rebinds keep the LAST binding.
function _kernel_factory_aliases(ir::MethodIR)
    al = Dict{Symbol,Any}()
    for s in ir.body
        _kmir_walk_calls_and_assigns(s) do n
            if n isa _LocalAssign && length(n.lhs) == 1
                cls = _kernel_factory_place_target(n.rhs, al)
                cls === nothing ? delete!(al, n.lhs[1]) : (al[n.lhs[1]] = cls)
            end
        end
    end
    al
end

# Generic recursion over a MethodIR stmt/expr tree, calling `f` on every node (used for both
# assign scanning and call collection — the methodir `_kmir_walk` is private to that file).
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

# The WRITE-ACTUAL `_MExpr`s a single call node effects, given resolved `field_regs` and the
# current `formalw` (callee-name → set of formal positions it mutates). REJECTS on an
# unsupported primitive arity or an opaque self-place subject call. Returns `_MExpr[]`.
function _kernel_factory_call_writes_of(n, field_regs, formalw_byname)
    out = _MExpr[]
    if n isa _RegisteredCall
        reg = n.registration
        if reg.kind === :primitive
            pe = reg.primitive_effect
            length(n.args) == pe.arity || throw(_KernelFactoryReject(
                "primitive $(reg.token) captured for arity $(pe.arity) but called with " *
                "$(length(n.args)) positionals — unsupported arity, refusing to summarize " *
                "(higher-arity actuals would be silently dropped)"))
            for w in pe.writes
                w <= length(n.args) && push!(out, n.args[w])
            end
        else                                   # :intrinsic (copy!!) / :free_method / :object_kernel
            isempty(n.args) || push!(out, n.args[1])   # writes its SUBJECT/dest = first actual
        end
    elseif n isa _FieldCall
        reg = get(field_regs, isempty(n.path) ? :_ : n.path[1], nothing)
        # unresolved hole → not decidable as a writer here (specialization supplies it); a
        # RESOLVED subject-writer owns its subject actual.
        if reg isa _KernelRegistration && _kernel_reg_writes_subject(reg) && !isempty(n.pos)
            push!(out, n.pos[1])
        end
    elseif n isa _SubjectMethodCall
        push!(out, n.subject)                  # a subject method mutates its receiver object
    end
    out
end

# For a SIBLING call, map the CALLEE's transitively-written formal positions back onto the
# caller's actuals at those positions (RK req 1+2).
function _kernel_factory_sibling_writes_of(n, formalw_byname)
    out = _MExpr[]
    if n isa _Call || n isa _CallExpr
        for p in get(formalw_byname, n.name, Int[])
            1 <= p <= length(n.pos) && push!(out, n.pos[p])
        end
    end
    out
end

# REJECT an opaque named call (`evil!`, hint `:opaque`) whose SUBJECT (first actual) is a
# self/owner place — no detached effect descriptor, so its footprint on an owned place is
# unknowable (RK req 4). Operators (`:operator_candidate`) are value reads, admitted.
function _kernel_factory_reject_opaque_self!(ir::MethodIR, aliases)
    for s in ir.body
        _kmir_walk_calls_and_assigns(s) do n
            if n isa _OpCall && n.hint === :opaque && !isempty(n.args)
                cls = _kernel_factory_place_target(n.args[1], aliases)
                cls !== nothing && cls[1] === :self && throw(_KernelFactoryReject(
                    "opaque call `$(n.op)` receives self place `$(cls[2])` in subject " *
                    "position but carries NO detached effect descriptor (not a registered " *
                    "@kernel / intrinsic / RK-core primitive); its effect on an owned field " *
                    "is unknowable — register it or rewrite (qualification/`!` are not evidence)"))
            end
        end
    end
end

"""
    _kernel_factory_owned_authoritative(skel; field_regs=Dict()) -> Set{Symbol}

The AUTHORITATIVE owned top-field set (RK 2026-08-27). `field_regs` maps each authored
callable-field name (`:step_f`, `:stats_f`, …) to its factory-time resolved
`_KernelRegistration`; an absent field stays an undecided hole. Throws `_KernelFactoryReject`
on an opaque self-place subject call or an unsupported primitive arity.
"""
function _kernel_factory_owned_authoritative(skel; field_regs = Dict{Symbol,Any}())
    irs = method_irs(skel)
    isempty(irs) && return Set{Symbol}()
    # An uncompilable method (a rejected grammar/dynamic-callee — e.g. a method-local `fill!`
    # spoof resolves to a dynamic local callable, not the registered identity) makes the
    # ownership closure unsound; REJECT deterministically rather than closing over a hole.
    for ir in irs
        ir.ok || throw(_KernelFactoryReject(
            "method `$(ir.id.name)` is not compilable ($(ir.reason)); authoritative ownership " *
            "cannot be closed over it"))
    end
    aliases = Dict{Any,Dict{Symbol,Any}}(ir.id => _kernel_factory_aliases(ir) for ir in irs)

    # req 4: reject inadmissible opaque self-place calls up front (deterministic).
    for ir in irs
        _kernel_factory_reject_opaque_self!(ir, aliases[ir.id])
    end

    owned = Set{Symbol}()
    formalw_byname = Dict{Symbol,Set{Int}}()   # callee name → formal positions it mutates
    changed = true
    while changed
        changed = false
        for ir in irs
            al = aliases[ir.id]
            name = ir.id.name
            fw = get!(formalw_byname, name, Set{Int}())
            # every write actual reached in this method (own calls + sibling formal-mapping)
            sites = _MExpr[]
            for s in ir.body
                _kmir_walk_calls_and_assigns(s) do n
                    append!(sites, _kernel_factory_call_writes_of(n, field_regs, formalw_byname))
                    append!(sites, _kernel_factory_sibling_writes_of(n, formalw_byname))
                end
            end
            # direct place-writes to self/alias owner fields (the LOCAL seed, folded in)
            for wr in write_roots(ir)
                owner = wr[2]
                if owner isa Tuple && !isempty(owner)
                    owner[1] in owned || (push!(owned, owner[1]); changed = true)
                end
            end
            for a in sites
                cls = _kernel_factory_place_target(a, al)
                cls === nothing && continue
                if cls[1] === :self
                    cls[2] in owned || (push!(owned, cls[2]); changed = true)
                else                                    # :formal — this method mutates formal p
                    cls[2] in fw || (push!(fw, cls[2]); changed = true)
                end
            end
        end
    end

    # recipe-graph closure: an owner field derived from owned inputs is owned (own graph).
    graph = kernel_graph(kernel_spec(skel))
    gchanged = true
    while gchanged
        gchanged = false
        for r in graph.recipes
            any(inp -> inp.name in owned, r.inputs) || continue
            for out in r.outputs
                out.name in owned || (push!(owned, out.name); gchanged = true)
            end
        end
    end
    intersect!(owned, Set{Symbol}(kernel_port_names(skel)))
    owned
end

"""
    _kernel_factory_shared(skel; field_regs=Dict()) -> Set{Symbol}

The SHARED complement — the ONE API that creates `shared`, and ONLY after the authoritative
owned closure (RK: shared is never a seed's `setdiff`). A port is shared iff it is not owned.
"""
_kernel_factory_shared(skel; field_regs = Dict{Symbol,Any}()) =
    setdiff(Set{Symbol}(kernel_port_names(skel)),
            _kernel_factory_owned_authoritative(skel; field_regs = field_regs))
