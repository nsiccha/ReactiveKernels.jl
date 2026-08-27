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
# TOKEN identity, or be REJECTED — never an opaque runtime callable. A `partial(kernel;…)`
# is a static TOKEN-PRESERVING binder that wraps its target in a `func` slot; unwrap it
# generically (independent of the binder's concrete type / include order) and resolve the
# target's registration. Returns the `_KernelRegistration` (Token/effects) or `nothing`.
function _kernel_resolve_callable(v)
    reg = kernel_registration(v)
    reg === nothing || return reg
    (v isa Function && hasfield(typeof(v), :func)) &&
        return _kernel_resolve_callable(getfield(v, :func))
    nothing
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
