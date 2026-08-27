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

# --- owned-vs-shared field derivation ----------------------------------------
#
# OWNED fields = the mutable per-instance state: every field DIRECTLY written by one of
# the object's own methods (the write-root top-fields from poc's `write_roots`), plus
# every recipe output transitively DERIVED from an owned field (recomputed when the
# owned state changes). SHARED authority = the complement — read-only, unified by value
# identity at construction. (Cross-object owned-copy of `deepcopy(child)` endpoints is a
# later increment; this derives a single object's own owned/shared split.)

# The directly-written owner top-fields (the owned SEED) across all of `skel`'s methods.
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

# Fixpoint closure: owned = direct writes ∪ every recipe output whose ANY input is owned.
# Returns `(owned::Set{Symbol}, shared::Set{Symbol})` over the object's fields/ports.
function _kernel_factory_owned_shared(skel)
    owned = _kernel_factory_direct_writes(skel)
    graph = kernel_graph(kernel_spec(skel))
    changed = true
    while changed
        changed = false
        for r in graph.recipes
            any(inp -> inp.name in owned, r.inputs) || continue
            for out in r.outputs
                if !(out.name in owned)
                    push!(owned, out.name)
                    changed = true
                end
            end
        end
    end
    fields = Set{Symbol}(kernel_port_names(skel))
    intersect!(owned, fields)                 # keep only real fields/ports
    (owned = owned, shared = setdiff(fields, owned))
end
