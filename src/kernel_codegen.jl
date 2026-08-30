# Executable prepared-endpoint codegen over the factory's captured
# recipe-handle seam (kernel_factory.jl @ 4c6ed92). Consumes ONLY the immutable `_PreparedFactory` (plan +
# `_RecipeHandle` tuple + external) and the canonical-slot storage ABI (`_canon_slot`/`_canon_set!`/
# `_canon_bless!`) — NEVER the live `Graph`, NEVER a hand table of recipe/field identities. Product: RGF-
# compiled functions that mutate the REAL `_CanonOwned`/`_CanonShared` storage in place.
#   * `compile_prepared_initialization` — the full captured-handle COLD initialization pass.
#   * `compile_prepared_schedule`       — the PlanKey-verified, lowering-selected POST-WRITE RECOMPUTE trace.
#   * `prepared_transition_trace`        — the write-kill closure over PRODUCER-OWNED edges → type-level trace.
# The interim compile_leaf / `_OwnerState` / `_seam_*` / Core.eval scaffold was RETIRED on the 4c6ed92 seam
# (superseded by this path); domain-specific write composition belongs to external exemplars.

using LinearAlgebra: ldiv!


# The ordered subject `_PlaceWrite`s of a straight-line method body (same filter/order as `_l_write_steps`).
# DETERMINISTIC hot-emitter locals (RK): `gensym` uses a GLOBAL counter, so two compiles of the SAME immutable
# Plan+MethodIR+config types produce RGF expressions with DIFFERENT local names -> different RGF TYPES ->
# unstable prepared-callable type identity + compile churn. `_dsym` draws from a PER-COMPILATION counter Ref
# (created fresh at each top-level emitter entry and threaded through the emit context — NEVER a mutable global,
# so nested/concurrent compiles do not race); since the emit order is a deterministic function of the inputs,
# identical inputs yield byte-identical Exprs (and identical RGF types). Not a persistent cache.
_dsym(ctr::Base.RefValue{Int}, p) = Symbol("__e_", p, "_", (ctr[] += 1))

function _exec_place_writes(ir::MethodIR)
    ws = _PlaceWrite[]
    for s in ir.body
        s isa _PlaceWrite || continue
        (s.root === :self && s.owner !== nothing && !isempty(s.owner)) || continue
        push!(ws, s)
    end
    ws
end

# ============================================================================================
# EXECUTABLE PREPARED-ENDPOINT captured-handle executor.
# Consumes ONLY the immutable `_PreparedFactory` (plan + captured `_RecipeHandle` tuple in plan.recipes
# order + external), NEVER the live graph, NEVER a hand table of recipe/field identities. Each handle is
# literal-indexed; its typed MODE (`:destination`/`:assign`/`:ldiv`) drives the invocation shape; input/
# output canonical ids map to (role, absolute slot) via `kernel_plan_field`; the Val-ABI storage
# (`_canon_slot`/`_canon_set!`/`_canon_bless!`) is the only physical access. Bare recipe ops are
# DOMAIN-VALIDATED at prepared-binding (this compile) against the concrete owned/shared field types
# BEFORE any hot invocation (RK 07:53); the warmed executor is exact 0-B / @inferred.
# ============================================================================================

# The expr reading canonical id `c` from its role's constructed object (owned/shared), by ABSOLUTE slot.
function _pp_read(plan::_KernelPlan, c::Int)
    rs = kernel_plan_field(plan, c)
    rs === nothing && _l_reject("prepared-endpoint: canonical id $c has no plan slot")
    role, slot = rs
    role === :owned ? :(_canon_slot(owned, Val($slot))) : :(_canon_slot(shared, Val($slot)))
end
# The concrete field type of canonical id `c` in the constructed storage (for compile-time domain checks).
function _pp_fieldtype(plan::_KernelPlan, c::Int, ::Type{OW}, ::Type{SH}) where {OW,SH}
    role, slot = kernel_plan_field(plan, c)
    fieldtype(role === :owned ? OW : SH, slot)
end
# The one representation-specific solve lowering admitted here: a Cholesky over a concrete builtin
# `Diagonal{Float32/Float64,Vector}`. Julia's generic Cholesky solve wraps the same diagonal factors as two
# triangular solves. For this self-adjoint diagonal backing those are exactly two in-place divisions by the
# stored factor. Keep the predicate entirely on the concrete prepared-slot TYPE — never the field name/value —
# and leave dense/custom/other-eltype Cholesky lowering on the generic `ldiv!(dest, factor, rhs)` path.
function _pp_diag_cholesky_ldiv_type(::Type{FT}, ::Type{RT}, ::Type{DT}) where {FT,RT,DT}
    FT <: LinearAlgebra.Cholesky || return false
    (FT isa DataType && length(FT.parameters) >= 2) || return false
    ET, BT = FT.parameters[1], FT.parameters[2]
    (ET === Float32 || ET === Float64) || return false
    BT isa Type && _kernel_dom_diag(BT) && eltype(BT) === ET && RT === Vector{ET} && DT === Vector{ET}
end

# The two-factor shortcut is byte-equivalent to generic Cholesky ldiv only in the ordinary finite/nonzero,
# non-aliasing Vector domain. Keep the original rhs untouched until the result has also been checked: any IEEE
# edge (signed/ordinary zero, Inf/NaN, intermediate overflow/underflow), shape issue, or alias delegates to the
# generic three-argument method, reproducing its value, exception, and mutation-prefix semantics exactly.
@inline function _pp_vector_overlaps(a::Vector{ET}, b::Vector{ET}) where {ET}
    (isempty(a) || isempty(b)) && return false
    # `Base.mightalias` misses distinct `unsafe_wrap`ped Vectors whose contiguous ranges overlap but begin at
    # different addresses.  The fast helper's ABI is exact builtin Vector{ET}, so a GC-preserved range check is
    # authoritative.  Compare address differences rather than forming an end pointer, which could overflow.
    GC.@preserve a b begin
        pa = UInt(pointer(a)); pb = UInt(pointer(b)); bytes = UInt(sizeof(ET))
        pa <= pb ? pb - pa < UInt(length(a)) * bytes : pa - pb < UInt(length(b)) * bytes
    end
end

@inline function _pp_diag_cholesky_ldiv!(dest::Vector{ET},
        factor::LinearAlgebra.Cholesky{ET,BT}, rhs::Vector{ET}) where {
        ET<:Union{Float32,Float64},BT<:LinearAlgebra.Diagonal{ET,Vector{ET}}}
    factors = getfield(factor, :factors); diag = getfield(factors, :diag)
    (length(dest) == length(rhs) == length(diag) &&
     !_pp_vector_overlaps(dest, rhs) && !_pp_vector_overlaps(dest, diag) &&
     !_pp_vector_overlaps(rhs, diag)) ||
        return ldiv!(dest, factor, rhs)
    @inbounds for i in eachindex(diag, rhs)
        d = diag[i]; x = rhs[i]
        (isfinite(d) && !iszero(d) && isfinite(x) && !iszero(x)) || return ldiv!(dest, factor, rhs)
    end
    copyto!(dest, rhs)                       # rhs remains intact for a possible result-domain fallback
    ldiv!(factors, dest)                     # BOTH Cholesky factors; one division is wrong for scaled mass
    ldiv!(factors, dest)
    @inbounds for x in dest
        (isfinite(x) && !iszero(x)) || return ldiv!(dest, factor, rhs)
    end
    dest
end
# Atomically bless a handle's PRODUCER-OWNED output subset (only — never a shared authority not produced,
# never a collateral) after its invocation succeeds. 1 or 2 → the atomic Val ops; >2 → a bless per slot.
function _pp_bless!(stmts, plan::_KernelPlan, owned_canons)
    slots = Tuple{Symbol,Int}[]
    for c in owned_canons
        role, slot = kernel_plan_field(plan, c)
        push!(slots, (role === :owned ? :owned : :shared, slot))
    end
    if length(slots) == 2
        (o1, s1), (o2, s2) = slots[1], slots[2]
        o1 === o2 ? push!(stmts, :(_canon_bless2!($o1, Val($s1), Val($s2)))) :
                    (push!(stmts, :(_canon_bless!($o1, Val($s1)))); push!(stmts, :(_canon_bless!($o2, Val($s2)))))
    else
        for (o, s) in slots; push!(stmts, :(_canon_bless!($o, Val($s)))); end
    end
end

# Emit ONE handle (literal-indexed `handles[i]`) into `stmts` — shared by initialization + schedule so the
# two paths cannot drift. MODE drives the invocation; producer-owned outputs are blessed after success.
function _pp_emit_handle!(stmts, plan::_KernelPlan, h, i::Int, ::Type{OW}, ::Type{SH};
                          recipe_hook=nothing) where {OW,SH}
    mode = recipe_handle_mode(h)
    ins = collect(h.inputs); outs = collect(h.outputs); op = recipe_handle_op(h)
    hi = :(recipe_handle_op(handles[$i]))                      # literal-indexed captured op (type-stable)
    if mode === :destination
        # buffer output = in-place gradient dest; scalar output = returned potential. The FIRST input is
        # the destination-aware callable authority (external grad_f); the rest are its args.
        bcs = Int[c for c in outs if _pp_fieldtype(plan, c, OW, SH) <: AbstractArray]
        scs = Int[c for c in outs if !(_pp_fieldtype(plan, c, OW, SH) <: AbstractArray)]
        (length(bcs) == 1 && length(scs) == 1) || _l_reject(
            "destination handle $i must have exactly one buffer + one scalar output (got $outs)")
        destc, potc = bcs[1], scs[1]
        role_pot, slot_pot = kernel_plan_field(plan, potc)
        potobj = role_pot === :owned ? :owned : :shared
        callee = _pp_read(plan, ins[1]); argreads = Any[_pp_read(plan, c) for c in ins[2:end]]
        push!(stmts, :(__pot__ = $callee($(_pp_read(plan, destc)), $(argreads...))))
        push!(stmts, :(_canon_set!($potobj, Val($slot_pot), __pot__)))
    elseif mode === :ldiv
        # destination-reusing solve into the owned buffer slot (RK 07:20): ldiv!(dest, factor, rhs).
        length(ins) == 2 && length(outs) == 1 || _l_reject("ldiv handle $i must be 2-in/1-out (got $ins/$outs)")
        factorT = _pp_fieldtype(plan, ins[1], OW, SH); rhsT = _pp_fieldtype(plan, ins[2], OW, SH)
        destT = _pp_fieldtype(plan, outs[1], OW, SH)
        _pp_domain_ok(op, (factorT, rhsT), i)
        if _pp_diag_cholesky_ldiv_type(factorT, rhsT, destT)
            # The inline helper owns the conservative IEEE/alias guards and generic fallback. The emitted call
            # is selected solely from concrete prepared types; there is no name/value guess or dynamic dispatch.
            dest = _pp_read(plan, outs[1]); factor = _pp_read(plan, ins[1]); rhs = _pp_read(plan, ins[2])
            push!(stmts, :(_pp_diag_cholesky_ldiv!($dest, $factor, $rhs)))
        else
            push!(stmts, :(ldiv!($(_pp_read(plan, outs[1])), $(_pp_read(plan, ins[1])), $(_pp_read(plan, ins[2])))))
        end
    elseif mode === :assign
        length(outs) == 1 || _l_reject("assign handle $i must be single-output (got $outs)")
        # domain-validate a BARE op (a sanctioned recipe primitive); a fused `_KernelSourceOp` is a
        # definition-unique sanctioned closure — literal-called generically, body never inspected.
        op isa _KernelSourceOp || _pp_domain_ok(op, Tuple(_pp_fieldtype(plan, c, OW, SH) for c in ins), i)
        role_o, slot_o = kernel_plan_field(plan, outs[1])
        outobj = role_o === :owned ? :owned : :shared
        argreads = Any[_pp_read(plan, c) for c in ins]
        push!(stmts, :(_canon_set!($outobj, Val($slot_o), $hi($(argreads...)))))
    else
        _l_reject("prepared-endpoint handle $i has unsupported mode $mode")
    end
    _pp_bless!(stmts, plan, collect(h.owned))                  # bless ONLY the producer-owned subset
    # A recipe is complete only after its real call, destination assignment, and every producer-owned bless.
    # The instrumentation compiler supplies a site hook here; production supplies `nothing`.  Keeping this
    # seam after `_pp_bless!` is load-bearing for exception semantics: a throwing or half-written recipe is
    # never counted as completed.
    if recipe_hook !== nothing
        hook = recipe_hook(i, h)
        hook === nothing || push!(stmts, hook)
    end
end

# Compile-time domain admission for a BARE recipe op against concrete argument types (RK 07:53) — reject
# BEFORE the executor is emitted, so no unvalidated bare op ever reaches a hot invocation.
function _pp_domain_ok(op, argtypes::Tuple, i::Int)
    kernel_recipe_op_domain_ok(op, argtypes) || _l_reject(
        "prepared-endpoint handle $i bare op $(op) rejected for argument domain $(argtypes)")
    nothing
end

# The immutable, PlanKey-tagged recipe-id trace — COMPILER-INTERNAL provenance (NOT a security boundary:
# same-`Key`/arbitrary-`Rids` is internally constructible, RK 08:37). BOTH the plan `Key` and the selected
# recipe-id tuple `Rids` are TYPE parameters with NO runtime field, so the private schedule dispatches on
# the literal `Rids` (a value cannot carry a mutable recipe list). The PUBLIC production API takes the leaf
# MethodIR and DERIVES this trace itself, so no ordinary compiler caller supplies arbitrary `Rids`.
struct _SelectedTrace{Key,Rids} end

# Read-only accessors (RK 08:13) — acceptance censuses the exact emitted recipe identities + plan binding
# WITHOUT reaching into type internals.
selected_trace_key(::_SelectedTrace{Key}) where {Key} = Key
selected_trace_recipes(::_SelectedTrace{Key,Rids}) where {Key,Rids} = Rids

# Write-kill closure → the selected recipe-id tuple, as the type-level trace (RK 08:10 soundness):
#  * stale starts at the transition WRITE canons and propagates ONLY through `kernel_plan_producer_owned`
#    (a recipe's AUTHORITATIVE owned outputs) — NEVER `recipe_outputs`, whose collateral ids are owned by a
#    DIFFERENT recipe and must not be treated as this recipe's propagation/blessing authority;
#  * a recipe is selected iff any of its inputs is stale; recipes fed only by untouched inputs (a fixed
#    metric → chol/@node) are proven current and omitted;
#  * tagged with the EXACT `kernel_plan_key(plan)`.
# PRIVATE (raw canonical ids) — the PUBLIC production entry `prepared_transition_trace(plan, ir)` derives
# `write_canons` from a real MethodIR write trace through the plan map; raw canons stay test-only (RK 08:10).
function _prepared_trace_from_canons(plan::_KernelPlan, write_canons)
    rin   = Dict{Int,Vector{Int}}(rid => collect(ins) for (rid, ins) in kernel_plan_recipe_inputs(plan))
    owned = Dict{Int,Vector{Int}}(rid => collect(os)  for (rid, os)  in kernel_plan_producer_owned(plan))
    stale = Set{Int}(write_canons)
    changed = true
    while changed
        changed = false
        for rid in kernel_plan_recipes(plan)
            any(c -> c in stale, get(rin, rid, Int[])) || continue
            for c in get(owned, rid, Int[]); c in stale || (push!(stale, c); changed = true); end
        end
    end
    sel = Tuple(rid for rid in kernel_plan_recipes(plan) if any(c -> c in stale, get(rin, rid, Int[])))
    _SelectedTrace{kernel_plan_key(plan), sel}()
end

"""
    prepared_transition_trace(plan, leaf_ir::MethodIR) -> _SelectedTrace

PRODUCTION derivation of the fixed-input transition schedule from the real transition/leaf MethodIR
(RK 08:06/08:10): the leaf's authored place-writes are mapped to canonical ids through the plan's slot map,
then `_prepared_trace_from_canons` runs the write-kill closure (propagating ONLY through producer-owned
outputs). Under a fixed metric the returned type-level trace omits chol/@node. NO names, NO caller set.
"""
function prepared_transition_trace(plan::_KernelPlan, leaf_ir::MethodIR)
    fc = Dict{Symbol,Int}(s.path[end] => s.canon for s in kernel_plan_slots(plan))
    wc = Set{Int}()
    for pw in _exec_place_writes(leaf_ir)
        t = pw.target
        t isa _SelfField || _l_reject("transition trace: unsupported leaf write target $(typeof(t))")
        f = t.path[end]
        haskey(fc, f) || _l_reject("transition trace: leaf write field `$f` has no plan canonical slot")
        push!(wc, fc[f])
    end
    _prepared_trace_from_canons(plan, wc)
end

"""
    compile_prepared_initialization(pf, OW, SH) -> fn

The COLD initialization / non-hot metric-mutation repair executor: emits EVERY captured handle in
`plan.recipes` order — destination grad → chol → @node logdet → ldiv → fused kin → ham — over the concrete
storage `OW`/`SH`. `cholesky(metric)` allocates its fresh factorization exactly once (replacing the
uncomputed typed placeholder) + `logdet` once — an HONEST, non-hot allocation (RK 08:04 (b)). Bare ops are
domain-validated at this bind. Returns `fn(owned, shared, handles)`.
"""
function compile_prepared_initialization(pf::_PreparedFactory, ::Type{OW}, ::Type{SH}) where {OW,SH}
    plan = kernel_prepared_plan(pf); hs = kernel_prepared_handles(pf)
    stmts = Any[]
    for (i, h) in enumerate(hs); _pp_emit_handle!(stmts, plan, h, i, OW, SH); end
    compile(:((owned, shared, handles) -> $(Expr(:block, stmts..., :(return owned)))))
end

"""
    compile_prepared_ensure(pf, OW, SH, field) -> fn

Compile a demand-driven repair for one named field of an arbitrary prepared
kernel. The field and its producer chain come entirely from the captured plan;
the returned callable repairs the field only when its currentness mask is dirty
and then returns the current value.
"""
function compile_prepared_ensure(pf::_PreparedFactory, ::Type{OW}, ::Type{SH},
                                 field::Symbol) where {OW,SH}
    plan = kernel_prepared_plan(pf)
    handles = kernel_prepared_handles(pf)
    fields = _exec_canon_map(plan)
    haskey(fields, field) || _l_reject(
        "prepared ensure: field `$field` has no canonical slot")

    producer = Dict{Int,Int}(c => r for (c, r) in kernel_plan_producer(plan))
    recipes = kernel_plan_recipes(plan)
    handle_index = Dict{Int,Tuple{Any,Int}}(
        recipes[i] => (handles[i], i) for i in eachindex(handles))
    canon = fields[field]
    statements = Any[]
    _exec_ensure!(statements, canon, Set{Int}(), Set{Int}(), plan,
                  producer, handle_index, OW, SH)
    compile(:((owned, shared, handles) ->
        $(Expr(:block, statements..., :(return $(_pp_read(plan, canon)))))))
end

"""
    compile_prepared_schedule(pf, OW, SH, leaf_ir::MethodIR) -> fn

The PUBLIC production POST-WRITE RECOMPUTE executor (RK 08:04 (c) / 08:37): takes the real transition/leaf
MethodIR and DERIVES the selected trace itself (`prepared_transition_trace`), so a compiler caller can NOT
supply an arbitrary same-plan recipe list. Under a fixed metric the derived trace omits chol/@node, so the
warmed call re-runs only grad/velocity/kin/ham and is exact 0-B / @inferred.
"""
compile_prepared_schedule(pf::_PreparedFactory, ::Type{OW}, ::Type{SH}, leaf_ir::MethodIR) where {OW,SH} =
    _compile_prepared_schedule(pf, OW, SH, prepared_transition_trace(kernel_prepared_plan(pf), leaf_ir))

# PRIVATE trace-taking core (test-only): compile from an already-derived type-level trace. Not the public
# compiler entry — the `_SelectedTrace` Key/Rids are compiler-internal provenance, not a security boundary
# (RK 08:37), so this overload is reachable only by the derivation above and by provenance tests. The
# handles it emits are the acceptance schedule census; a trace whose `Key` is not this plan's is rejected.
function _compile_prepared_schedule(pf::_PreparedFactory, ::Type{OW}, ::Type{SH},
                                    ::_SelectedTrace{Key,Rids}) where {OW,SH,Key,Rids}
    plan = kernel_prepared_plan(pf)
    Key === kernel_plan_key(plan) || _l_reject(
        "selected trace PlanKey does not match this prepared plan — schedule not provably tied to it")
    hs = kernel_prepared_handles(pf); recs = kernel_plan_recipes(plan)
    sel = Set{Int}(Rids)                                       # literal recipe-id tuple (type parameter)
    stmts = Any[]
    for (i, h) in enumerate(hs)
        recs[i] in sel || continue                             # literal-index ONLY lowering-selected handles
        _pp_emit_handle!(stmts, plan, h, i, OW, SH)
    end
    compile(:((owned, shared, handles) -> $(Expr(:block, stmts..., :(return owned)))))
end
