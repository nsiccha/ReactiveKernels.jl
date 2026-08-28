# External NUTS compiler-acceptance factory/runtime extension.
#
# Loaded only by examples/nuts_runtime.jl. Nothing in this file is package API;
# the generic package owns KernelSpec/MethodIR/planning and this example owns its
# sampler frame, diagnostics, certificates, and domain-specific attachment seam.

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
# G7: RAW diagnostic value write — value ONLY, no pending update. Valid where `_diagnostics_reset!` at
# root-begin already set pending=0x0f (all four produced) and DOMINATES the whole epoch (nothing clears
# pending until the root commit), so a per-leaf pending OR is redundant. Same value/throw semantics: the
# value is written; committed is untouched (a throw before the root commit still leaves nothing committed).
@generated function _diag_set_value!(s::_DiagnosticsStore, ::Val{I}, v) where {I}
    :(setfield!(s, $(QuoteNode(fieldname(_DiagnosticsStore, I))), v); s)
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

# --- concrete nuts_state owned TREE / FRAME construction (RK 08:55) ----------------------------------
#
# One owned TREE's data buffers, byte-matching the fixture `tree(phasepoint)`: `log_weight` (a 2-vector in
# the ham type — the `oftype(ham,-Inf)` sentinel), and `bwd`/`bwd_fwd` (each `mv` = mom+dham_dmom) +
# `summed_mom` (a `trajectory` = bwd+fwd), all zeroed buffers in `pos`'s eltype. A concrete typed NamedTuple
# (typed fields, no Any/boxing); F32/F64 PRESERVED from pos's eltype + the ham type. Each call allocates
# DISTINCT buffers (cold construction — per-tree isolation).
_nuts_mv(v::AbstractVector) = (; mom = zero(v), dham_dmom = zero(v))
_nuts_trajectory(v::AbstractVector) = (; bwd = zero(v), fwd = zero(v))
_nuts_tree(pos::AbstractVector, ham) =
    (; log_weight = fill(oftype(ham, -Inf), 2),
       bwd = _nuts_mv(pos), bwd_fwd = _nuts_mv(pos), summed_mom = _nuts_trajectory(pos))
_nuts_trees(pos::AbstractVector, ham, n::Int) = [_nuts_tree(pos, ham) for _ in 1:n]

# The concrete owned SAMPLER FRAME: fixed owned endpoints (init/fwd/bwd), the tree + proposal Vectors, the
# control scalars, the compiler-owned diagnostics store, and the ONE per-sampler shared authority every
# endpoint references BY IDENTITY (metric/chol/@node — built once, never per-endpoint). No Ref, no Any:
# every field is concrete/typed (F32/F64 preserved through EP/TREE/Tham); control + diagnostics scalar
# updates and the per-transition reset are exact 0-B on the concrete mutable field ABI.
mutable struct _NutsFrame{EP,TREE,SH,Tham,M,STEP,STATS}
    init::EP              # owned endpoint (isolated buffers), refers to `shared` by identity
    fwd::EP
    bwd::EP
    trees::Vector{TREE}   # max_depth+1 owned trees
    proposals::Vector{EP} # max_depth+2 owned endpoints (isolated)
    gofwd::Bool
    may_sample::Bool
    may_continue::Bool
    diverged::Bool        # the authored DERIVED recipe `!(dham >= min_dham)` — a CONCRETE owned Bool, not
                          #   recomputed by name (bit 0 of the derived masks)
    derived_pending::UInt # PRODUCED within the current epoch (a dham write invalidates it) — RK 09:19
    derived_committed::UInt # blessed at the SINGLE root commit — a post-produce root THROW leaves this
                          #   uncommitted, so the derived value never survives an exception falsely current
    diag::_DiagnosticsStore{Tham}
    shared::SH            # the ONE per-sampler shared authority (identity across all endpoints)
    entry_mask::M        # the proven post-init owned currentness mask (entry_current) — init MUST match
                         #   this before children may be seeded (RK 09:10)
    step_f::STEP         # the detached prepared step binding (registered leapfrog! token + bound kwargs)
    stats::STATS         # the registered stats_f scalar-effect binding (or the no-effect `nothing` variant)
    max_depth::Int       # frozen config
    min_dham::Tham       # frozen config (in the preserved ham type)
end
nuts_frame_shared(f::_NutsFrame) = getfield(f, :shared)
nuts_frame_ham_type(::_NutsFrame{EP,TREE,SH,Tham}) where {EP,TREE,SH,Tham} = Tham
nuts_frame_entry_mask(f::_NutsFrame) = getfield(f, :entry_mask)
nuts_frame_step(f::_NutsFrame) = getfield(f, :step_f)
nuts_frame_stats(f::_NutsFrame) = getfield(f, :stats)
nuts_frame_max_depth(f::_NutsFrame) = getfield(f, :max_depth)
nuts_frame_min_dham(f::_NutsFrame) = getfield(f, :min_dham)
@inline nuts_frame_diverged_pending(f::_NutsFrame) = getfield(f, :derived_pending) & UInt(1) == UInt(1)
@inline nuts_frame_diverged_committed(f::_NutsFrame) = getfield(f, :derived_committed) & UInt(1) == UInt(1)
# `diverged` is PRODUCED from dham + min_dham (RK 09:11) — a real derived write, marked PENDING (produced
# within the epoch); a `dham`/reset write INVALIDATES it (poc recomputes). 0-B.
@inline function _nuts_produce_diverged!(f::_NutsFrame)
    setfield!(f, :diverged, !(_diag_slot(getfield(f, :diag), Val(4)) >= getfield(f, :min_dham)))
    setfield!(f, :derived_pending, getfield(f, :derived_pending) | UInt(1)); f
end
# Root-begin / dham invalidation clears BOTH pending AND committed (RK 09:25): after a prior successful
# root committed diverged, a new epoch's reset must not leave the OLD committed bit live — otherwise
# reset+produce+later-throw could expose it as current. Both cleared → the derived value is fully stale.
@inline _nuts_invalidate_diverged!(f::_NutsFrame) =
    (setfield!(f, :derived_pending, getfield(f, :derived_pending) & ~UInt(1));
     setfield!(f, :derived_committed, getfield(f, :derived_committed) & ~UInt(1)); f)
# The SINGLE root commit blesses the produced derived value (RK 09:19); a post-produce root THROW before
# this leaves `derived_committed` at its prior value — the derived value is never falsely current.
@inline _nuts_derived_root_commit!(f::_NutsFrame) =
    (setfield!(f, :derived_committed, getfield(f, :derived_pending)); f)
# Per-transition control+diagnostics RESET (fixture `reset!`) — control back to `true`, diagnostics through
# `_diagnostics_reset!` (authoritative source write; committed zeroed, all four pending-produced). The tree/
# endpoint buffer resets (copy!!/fill!) are separate visible producer writes poc lowers. Exact 0-B here.
@inline function _nuts_frame_reset_control!(f::_NutsFrame)
    setfield!(f, :gofwd, true); setfield!(f, :may_sample, true); setfield!(f, :may_continue, true)
    _diagnostics_reset!(getfield(f, :diag))
    _nuts_invalidate_diverged!(f)      # reset writes dham=0 → the derived `diverged` is stale (poc recomputes)
    f
end

# (`_construct_nuts_frame` is defined after `_PreparedFactory` / `_construct_prepared`, below.)


# PHASE 1 of the two-phase sampler construction (RK 09:00/09:02): build the HAVE-only-current init endpoint
# + the ONE per-sampler shared authority via `_kernel_construct_endpoint` — NO recipe execution here (no
# pgrad, no chol) — and the fwd/bwd/proposals owned children UNSEEDED (freshly built, each an isolated
# owned-only endpoint referencing the single `shared` by identity, NOT yet copied from init). POC runs the
# FULL captured six-handle initialization on `frame.init` EXACTLY ONCE (the one destination pgrad ×1 +
# chol/logdet ×1); only then `_seed_nuts_children!` copies the COMPLETE values+mask. Seeding children now
# would propagate an incomplete init. `pos`/`ham` are init's source position + hamiltonian value; F32/F64
# PRESERVED via trees/diag. (The single-path construction means no other code path double-evaluates init.)
function _construct_nuts_frame(pf::_PreparedFactory{Token}, endpoint_values, max_depth::Int;
                               step_f, stats_f, min_dham) where {Token}
    # REJECT an out-of-range max_depth BEFORE any construction/mutation (RK 09:10) — the tree/proposal
    # counts are `max_depth+1`/`max_depth+2`, so a negative depth is a shape error, not a silent empty.
    # REJECT an out-of-range / overflowing max_depth BEFORE any allocation (RK 09:10/09:27): the tree/
    # proposal counts are `max_depth+1`/`max_depth+2`, so guard against a negative depth AND Int overflow.
    (max_depth >= 0 && max_depth <= typemax(Int) - 2) || throw(_KernelFactoryReject(
        "nuts_state max_depth must be in 0:$(typemax(Int)-2) (got $max_depth) — incompatible/overflowing shape"))
    # the step binding MUST be the SAME registered integrator the endpoint Plan was prepared under — by
    # Token IDENTITY, not name/== (RK 09:27); else the frame carries a different writer than its layout.
    stepc = _prepare_callable(:step_f, step_f)              # validated record (registration + bound kwargs)
    prepared_callable_token(stepc) === kernel_plan_token(pf.plan) || throw(_KernelFactoryReject(
        "step_f resolves to a DIFFERENT registered writer than the endpoint Plan's integrator Token — the " *
        "frame layout was prepared for a different kernel"))
    statsb = _stats_binding(stats_f)                        # registration DERIVED internally
    init_ow, shared = _kernel_construct_endpoint(Val(Token), pf.plan, endpoint_values, pf.external)
    # DERIVE the pos template + ham type from init's phasepoint slots via COMPILE-TIME Val-name accessors
    # (RK 09:19/09:27) — no per-instance Symbol lookup, no runtime-Val; `pos` is read as a TEMPLATE (the
    # trees allocate fresh `zero(pos)` buffers), so no deepcopy.
    pos = _canon_slot(init_ow, kernel_plan_named_slot_val(pf.plan, Val(:pos)))
    Tham = typeof(_canon_slot(init_ow, kernel_plan_named_slot_val(pf.plan, Val(:ham))))
    entry_mask = kernel_plan_entry_owned_mask(pf.plan)      # COMPILE-TIME literal owned mask (plan-once)
    mkchild() = _kernel_construct_owned_child(Val(Token), pf.plan, endpoint_values)   # UNSEEDED, dirty
    fwd = mkchild(); bwd = mkchild()
    proposals = typeof(init_ow)[mkchild() for _ in 1:(max_depth + 2)]
    trees = _nuts_trees(pos, zero(Tham), max_depth + 1)
    # diverged starts DIRTY (neither pending nor committed) — POC produces it from dham+min_dham during init.
    _NutsFrame(init_ow, fwd, bwd, trees, proposals, true, true, true, false, UInt(0), UInt(0),
               _diagnostics_store(Tham), shared, entry_mask, stepc, statsb, max_depth, oftype(zero(Tham), min_dham))
end

# PHASE 2 (RK 09:00/09:10): AFTER POC has executed the full six-handle init on `frame.init`, seed the
# children — copy the COMPLETE init values + validity mask to fwd/bwd/proposals with ZERO extra pgrad/chol
# (a plain owned-endpoint copy). REJECTS unless init EXACTLY matches entry_current: a caller must never
# seed an INCOMPLETE init (an init exception — forced destination-grad or a later handle — leaves init's
# mask below entry_current, so every child stays DIRTY, never falsely blessed). Exception-safe per
# `_canon_copy_endpoint!`. Returns the frame.
function _seed_nuts_children!(frame::_NutsFrame)
    _canon_current_mask(getfield(frame, :init)) == getfield(frame, :entry_mask) || throw(_KernelFactoryReject(
        "cannot seed children from an INCOMPLETE init — its currentness mask does not match entry_current " *
        "(POC must run the full six-handle initialization to completion before seeding)"))
    _canon_copy_endpoint!(getfield(frame, :fwd), getfield(frame, :init))
    _canon_copy_endpoint!(getfield(frame, :bwd), getfield(frame, :init))
    for p in getfield(frame, :proposals); _canon_copy_endpoint!(p, getfield(frame, :init)); end
    frame
end

# SINGLE-PASS sampler frame from a COLD-BOOTSTRAP value tuple (RK 13:44 ergonomic path): the bootstrap has
# already executed the six handles ONCE (one gradient + one cholesky), so the init endpoint is FULLY current
# from the start — no separate POC six-handle init, no two-phase seed. The init RETAINS the bootstrap buffers
# by identity; fwd/bwd/proposals are COMPLETE isolated deep-copies of it sharing the ONE shared authority.
# Same config/validation as `_construct_nuts_frame` (checked max_depth, step_f Token identity, stats binding,
# F32/F64 via ham type). Returns a runnable frame ready for either control `compile_nuts` or production
# `compile_nuts_native` — no further init/seed step.
function _construct_nuts_frame_bootstrapped(pf::_PreparedFactory{Token}, cvals::Tuple, max_depth::Int;
                                            step_f, stats_f, min_dham) where {Token}
    (max_depth >= 0 && max_depth <= typemax(Int) - 2) || throw(_KernelFactoryReject(
        "nuts_state max_depth must be in 0:$(typemax(Int)-2) (got $max_depth) — incompatible/overflowing shape"))
    stepc = _prepare_callable(:step_f, step_f)
    prepared_callable_token(stepc) === kernel_plan_token(pf.plan) || throw(_KernelFactoryReject(
        "step_f resolves to a DIFFERENT registered writer than the endpoint Plan's integrator Token — the " *
        "frame layout was prepared for a different kernel"))
    statsb = _stats_binding(stats_f)
    init_ow, shared = _construct_endpoint_from_values(pf.plan, pf.handles, cvals)   # COMPLETE init (isolation-correct)
    pos = _canon_slot(init_ow, kernel_plan_named_slot_val(pf.plan, Val(:pos)))
    Tham = typeof(_canon_slot(init_ow, kernel_plan_named_slot_val(pf.plan, Val(:ham))))
    entry_mask = kernel_plan_entry_owned_mask(pf.plan)
    mkchild() = _construct_owned_child_from_values(pf.plan, cvals)                  # COMPLETE + isolated
    fwd = mkchild(); bwd = mkchild()
    proposals = typeof(init_ow)[mkchild() for _ in 1:(max_depth + 2)]
    trees = _nuts_trees(pos, zero(Tham), max_depth + 1)
    _NutsFrame(init_ow, fwd, bwd, trees, proposals, true, true, true, false, UInt(0), UInt(0),
               _diagnostics_store(Tham), shared, entry_mask, stepc, statsb, max_depth, oftype(zero(Tham), min_dham))
end

# --- author-facing prepared SAMPLER: a callable KernelObject over the frame (RK 12:26/12:32/12:37 / poc) --
#
# HARD GATE (RK 12:32): the compiled `root!` + `scratch` are CONCRETE TYPE PARAMETERS of the Handles before
# the public KernelObject ever escapes — NO mutable empty slot, Ref, Any, `Function`-typed field, or Union.
# So there is NO unattached-placeholder object: construction returns the prepared FRAME, POC compiles its
# root!+scratch from that concrete frame, and `nuts_sampler` builds the FINAL concrete KernelObject in one
# shot. Repeated same-signature constructions yield identical concrete object/root/scratch types (0-B).
#
# IDENTITY GATE (RK 12:37/12:39): the KernelObject keeps the OWNER `nuts_state` Token (needed for subject-
# method MethodIds), and the Handles separately carry the exact compiled public Mode-2 `nuts!!` `RootToken`
# as their FIRST type parameter (POC compiles/validates the real `nuts!!` MethodIR against it and hands it
# back to construction). A legacy sampler is `KernelObject{OwnerToken,Frame,_NutsHandles{...,
# _UnsealedNutsCertificate}}`; the production builder instead retains the compiler-derived zero-field
# `_NutsCertificate` plus the exact frame provenance in those handles.
# OwnerToken need NOT equal RootToken. Dispatch is the Mode-2 SKELETON CALL below, admitted ONLY when the
# skeleton's own Token === the handle `RootToken` — a pure-type gate, no name special-case. An unrelated
# Mode-2 skeleton has NO method and is rejected.
"""Zero-field compiler certificate retained by a sealed production NUTS sampler.

Every parameter is derived while compiling the real native root.  In particular, `ProgramT` is the
actual immutable `_NativeProgram` consumed by that root; neither an adapter nor a benchmark supplies
any of these facts.  `RootT`/`ScratchT`/`FrameT` make detached evidence fail before it can be observed.
"""
abstract type _NutsEmittedOp end
struct _NutsRealOp{Id,Kind,Authority} <: _NutsEmittedOp end
struct _NutsInstrumentWrite{Id,AdjacentTo,Counter} <: _NutsEmittedOp end
struct _NutsRecipeDescriptor{Owner,Recipe,OpT,Mode,Inputs,Outputs,Owned} end
struct _NutsLeafWriteDescriptor{Integrator,Ordinal,Canon,Dot} end
struct _NutsRootManifest{RefreshToken,RefreshBodyMarker,Ops} end
struct _NutsLeafManifest{Integrator,StepSlot,BodyMarker,Ops,Nodes} end

struct _NutsCertificate{Mode,OwnerToken,RootToken,PlanT,PlanKey,ProgramT,ControlFingerprint,
                        RecipeManifest,RootManifest,LeafManifest,EmittedOps,SelectedRecipes,Roles,
                        Integrator,RootT,ScratchT,FrameT} end
struct _NutsControlFingerprint{ProgramT,RootMid,NodeCount} end
struct _UnsealedNutsCertificate end

function _nuts_certificate_parts(::Type{<:_NutsCertificate{Mode,OwnerToken,RootToken,PlanT,
        PlanKey,ProgramT,ControlFingerprint,RecipeManifest,RootManifest,LeafManifest,EmittedOps,
        SelectedRecipes,Roles,Integrator,RootT,ScratchT,FrameT}}) where {Mode,OwnerToken,RootToken,
        PlanT,PlanKey,ProgramT,ControlFingerprint,RecipeManifest,RootManifest,LeafManifest,EmittedOps,
        SelectedRecipes,Roles,Integrator,RootT,ScratchT,FrameT}
    (mode=Mode, owner=OwnerToken, root_token=RootToken, plan=PlanT, plan_key=PlanKey,
     program=ProgramT, control=ControlFingerprint, recipe_manifest=RecipeManifest,
     root_manifest=RootManifest, leaf_manifest=LeafManifest, emitted_ops=EmittedOps,
     recipes=SelectedRecipes, roles=Roles,
     integrator=Integrator, root_type=RootT, scratch_type=ScratchT, frame_type=FrameT)
end
_nuts_certificate_parts(c::_NutsCertificate) = _nuts_certificate_parts(typeof(c))

struct _NutsHandles{RootToken,Root,Scratch,Frame,Certificate}
    root::Root
    scratch::Scratch
    frame::Frame
    certificate::Certificate
end
_NutsHandles(::Val{RootToken}, root::Root, scratch::Scratch, frame::Frame,
             certificate::Certificate) where {RootToken,Root,Scratch,Frame,Certificate} =
    _NutsHandles{RootToken,Root,Scratch,Frame,Certificate}(root, scratch, frame, certificate)
nuts_handles_root(h::_NutsHandles) = getfield(h, :root)
nuts_handles_scratch(h::_NutsHandles) = getfield(h, :scratch)
nuts_handles_root_token(::_NutsHandles{RootToken}) where {RootToken} = RootToken
nuts_sampler_frame(k::KernelObject) = getfield(k, :state)

# The one seal traversal.  All public evidence accessors below start from the concrete KernelObject and
# traverse its own handles; no root/frame/metric/counter argument can be coordinated alongside it.  The
# frame identity check is deliberately a VALUE identity check, not merely a same-type check.
function _nuts_sealed_handles(k::K) where {K<:KernelObject}
    isconcretetype(K) || throw(ArgumentError("sealed NUTS evidence requires a concrete KernelObject"))
    h = getfield(k, :handles)
    h isa _NutsHandles || throw(ArgumentError("KernelObject does not carry NUTS handles"))
    c = getfield(h, :certificate)
    c isa _NutsCertificate || throw(ArgumentError(
        "legacy/control NUTS sampler is explicitly unsealed; compile native production evidence first"))
    p = _nuts_certificate_parts(c)
    p.mode in (:production,:instrumented) || throw(ArgumentError(
        "unsupported sealed NUTS mode $(p.mode)"))
    p.owner === kernel_token(k) || throw(ArgumentError("sealed NUTS owner-token mismatch"))
    p.root_token === nuts_handles_root_token(h) || throw(ArgumentError("sealed NUTS root-token mismatch"))
    getfield(h, :frame) === getfield(k, :state) || throw(ArgumentError(
        "sealed NUTS frame provenance is detached from the KernelObject state"))
    typeof(getfield(h, :frame)) === p.frame_type || throw(ArgumentError("sealed NUTS frame-type mismatch"))
    typeof(getfield(h, :root)) === p.root_type || throw(ArgumentError("sealed NUTS root-type mismatch"))
    typeof(getfield(h, :scratch)) === p.scratch_type || throw(ArgumentError("sealed NUTS scratch-type mismatch"))
    p.plan <: _KernelPlan || throw(ArgumentError("sealed NUTS certificate carries no concrete Plan type"))
    p.plan.parameters[1] === p.plan_key || throw(ArgumentError("sealed NUTS Plan type/key mismatch"))
    p.program <: _NativeProgram || throw(ArgumentError(
        "sealed NUTS certificate must carry the actual native Program type"))
    np = _native_program_parts(p.program)
    np.owner === p.owner || throw(ArgumentError("sealed NUTS Program/owner mismatch"))
    np.plan === p.plan || throw(ArgumentError("sealed NUTS Program/Plan mismatch"))
    _native_root_program(typeof(getfield(h, :root))) === p.program || throw(ArgumentError(
        "sealed NUTS root does not execute the certified Program"))
    rm,rootm,leafm,ops=_native_root_manifests(typeof(getfield(h,:root)))
    (p.recipe_manifest===rm && p.root_manifest===rootm && p.leaf_manifest===leafm &&
     p.emitted_ops===ops) || throw(ArgumentError(
        "sealed NUTS certificate is detached from the root's compiler manifests"))
    if p.mode === :production
        getfield(h, :root) isa _CompiledNutsRootNative || throw(ArgumentError(
            "production certificate is detached from the production native root family"))
        getfield(h, :scratch) === () || throw(ArgumentError(
            "production certificate is detached from the production no-instrumentation scratch"))
    else
        getfield(h, :root) isa _CompiledNutsRootInstrumented || throw(ArgumentError(
            "instrumented certificate is detached from the instrumented native root family"))
        getfield(h, :scratch) isa _NutsInstrumentationScratch || throw(ArgumentError(
            "instrumented certificate is detached from compiler-owned instrumentation scratch"))
        getfield(getfield(h, :root), :scratch) === getfield(h, :scratch) || throw(ArgumentError(
            "instrumented root/scratch value provenance mismatch"))
        q=typeof(getfield(h,:scratch)).parameters
        (q[1]===p.owner && q[2]===p.root_token && q[3]===p.plan && q[4]===p.program &&
         q[5]===p.emitted_ops && q[6]===p.recipe_manifest) || throw(ArgumentError(
            "instrumented scratch token/plan/program/manifest provenance mismatch"))
    end
    p.control === _NutsControlFingerprint{p.program,np.root,_native_program_node_count(p.program)} ||
        throw(ArgumentError("sealed NUTS control fingerprint mismatch"))
    _nuts_validate_emitted_ops(p.mode, p.emitted_ops) || throw(ArgumentError(
        "sealed NUTS emitted-op stream is not physically adjacent/unique"))
    p.recipes === p.plan_key[5] || throw(ArgumentError("sealed NUTS selected-recipe mismatch"))
    expected_roles = Tuple((t[1], t[2], t[3], t[4]) for t in p.plan_key[2])
    p.roles === expected_roles || throw(ArgumentError("sealed NUTS physical-role mismatch"))
    prepared_callable_token(nuts_frame_step(getfield(h, :frame))) === p.integrator ||
        throw(ArgumentError("sealed NUTS integrator/frame-binding mismatch"))
    h
end

nuts_sealed_certificate(k::KernelObject) = getfield(_nuts_sealed_handles(k), :certificate)
nuts_sealed_root(k::KernelObject) = getfield(_nuts_sealed_handles(k), :root)
nuts_sealed_scratch(k::KernelObject) = getfield(_nuts_sealed_handles(k), :scratch)
nuts_sealed_frame(k::KernelObject) = getfield(_nuts_sealed_handles(k), :frame)
nuts_sealed_shared(k::KernelObject) = getfield(nuts_sealed_frame(k), :shared)

function _nuts_sealed_shared_slot_expr(K, name::Symbol)
    H = K.parameters[3]
    (H isa DataType && H <: _NutsHandles) || return :(_nuts_sealed_handles(k); nothing)
    Cert = H.parameters[5]
    (Cert isa DataType && Cert <: _NutsCertificate) || return :(_nuts_sealed_handles(k); nothing)
    PlanKey = Cert.parameters[5]
    sig = PlanKey[2]
    i = findfirst(t -> t[1] == (name,), sig)
    i === nothing && return :(throw(ArgumentError($("sealed NUTS plan has no `$name` slot"))))
    sig[i][3] === :shared || return :(throw(ArgumentError(
        $("sealed NUTS `$name` slot is not shared authority"))))
    slot = sig[i][4]
    :(begin
        h = _nuts_sealed_handles(k)
        _canon_slot(getfield(getfield(h, :frame), :shared), Val($slot))
    end)
end
@generated nuts_sealed_metric(k::K) where {K<:KernelObject} =
    _nuts_sealed_shared_slot_expr(K, :metric)
@generated nuts_sealed_chol_metric(k::K) where {K<:KernelObject} =
    _nuts_sealed_shared_slot_expr(K, :chol_metric)
@generated nuts_sealed_gradient(k::K) where {K<:KernelObject} =
    _nuts_sealed_shared_slot_expr(K, :grad_f)

# Prepare the sampler FRAME from source inputs (RK 12:26): build the frame, run POC's full six-handle
# INITIALIZATION on `frame.init` exactly once, and SEED the children — returning the ready `_NutsFrame`
# (concrete). POC compiles its root!+scratch from this frame; then `nuts_sampler` wraps it with the tokens.
function _prepare_nuts_frame(pf::_PreparedFactory, endpoint_values, max_depth::Int; step_f, stats_f, min_dham)
    frame = _construct_nuts_frame(pf, endpoint_values, max_depth; step_f, stats_f, min_dham)
    init = getfield(frame, :init); shared = getfield(frame, :shared)
    compile_prepared_initialization(pf, typeof(init), typeof(shared))(init, shared, kernel_prepared_handles(pf))
    _seed_nuts_children!(frame)
    frame
end

# Build an explicitly UNSEALED legacy sampler in one shot: a concrete `KernelObject` over the prepared frame
# plus compiled root!+scratch. It remains for control-path regression tests. The production builder below uses
# `_sealed_nuts_sampler` with the native compiler certificate. `OwnerToken`
# is the authoring `nuts_state` owner Token (kept for subject-method MethodIds); `RootToken` is the exact
# Mode-2 `nuts!!` skeleton Token POC compiled the root against — carried in the Handles so the skeleton-call
# gate below admits ONLY that skeleton. Handle types are concrete from the start; nothing untyped escapes.
nuts_sampler(::Val{OwnerToken}, ::Val{RootToken}, frame::_NutsFrame, root, scratch) where {OwnerToken,RootToken} =
    KernelObject{OwnerToken,typeof(frame),
                 _NutsHandles{RootToken,typeof(root),typeof(scratch),typeof(frame),_UnsealedNutsCertificate}}(
        frame, _NutsHandles(Val(RootToken), root, scratch, frame, _UnsealedNutsCertificate()))

function _sealed_nuts_sampler(::Val{OwnerToken}, ::Val{RootToken}, frame::_NutsFrame, root, scratch,
        certificate::_NutsCertificate{Mode,OwnerToken,RootToken}) where {Mode,OwnerToken,RootToken}
    Mode in (:production,:instrumented) || throw(ArgumentError("unsupported sealed sampler mode `$Mode`"))
    p = _nuts_certificate_parts(certificate)
    typeof(frame) === p.frame_type || throw(ArgumentError("certificate/frame type mismatch"))
    typeof(root) === p.root_type || throw(ArgumentError("certificate/root type mismatch"))
    typeof(scratch) === p.scratch_type || throw(ArgumentError("certificate/scratch type mismatch"))
    H = _NutsHandles{RootToken,typeof(root),typeof(scratch),typeof(frame),typeof(certificate)}
    KernelObject{OwnerToken,typeof(frame),H}(
        frame, _NutsHandles(Val(RootToken), root, scratch, frame, certificate))
end

# --- runnable prepare→compile→attach CORE (RK 13:18 ergonomic gate, fork-free half) ------------------------
#
# Given a prepared endpoint factory `pf`, the endpoint slot values, the three authored skeletons (the
# `nuts_state` construction owner, the `refresh_momentum!!` source, and the public `nuts!!` transition), and
# the frozen config, this does the WHOLE internal chain — prepare frame (build + POC six-handle init +
# seed) → registry-free `compile_nuts_native` → sealed sampler — and returns the FINAL concrete callable
# `KernelObject`, on
# which `nuts!!(sampler; rng)` runs end-to-end. NO manual attach; OwnerToken = the `nuts_state` owner Token,
# RootToken = the compiled public `nuts!!` Token (`C.RootToken`), distinct and both preserved (RK 12:39).
#
# This is the fork-free portion of the ergonomic gate: it still takes `endpoint_values` keyed by canonical
# slot. Eliminating that last Dict — synthesizing the endpoint storage from the AUTHORED NAMED source inputs
# (`euclidean_phasepoint(grad_f, metric, pos, mom)` → init) — is the remaining architecture crux (POC owns
# init EXECUTION, so the derived-slot storage must be typed-allocated, not executed) surfaced to RK.
# `nuts_state_skel` is the authoring OWNER (a multi-method `_StatefulKernelSkeleton`); `refresh_skel`/
# `nuts_skel` are the free `_Mode2KernelSkeleton`s POC's compiler consumes + validates.
function _build_nuts_sampler(pf::_PreparedFactory, endpoint_values, nuts_state_skel,
                             refresh_skel::_Mode2KernelSkeleton, nuts_skel::_Mode2KernelSkeleton;
                             step_f, max_depth::Int, min_dham, stats_f)
    frame = _prepare_nuts_frame(pf, endpoint_values, max_depth; step_f, stats_f, min_dham)
    C = compile_nuts_native(pf, nuts_state_skel, refresh_skel, nuts_skel, frame)
    _sealed_nuts_sampler(Val(kernel_token(nuts_state_skel)), Val(C.RootToken), frame,
                         C.root!, C.scratch, C.certificate)
end

# The authored `nuts!!(state; rng)` dispatch (RK 12:37/12:39 / poc contract): the Mode-2 `nuts!!` SKELETON
# is itself the callable, and its captured signature contract — subject positional + REQUIRED KEYWORD `rng`
# — is reused verbatim as this method's own signature (`(k; rng)`), so a missing / extra / positional `rng`
# finds NO method and is rejected. Admitted ONLY when the skeleton's own `RootToken` === the sampler handle's
# `RootToken` (a single shared type parameter); OwnerToken is free — so an unrelated Mode-2 skeleton finds NO
# method, with no token-free path to bypass the gate. It calls the compiled `root!(frame, scratch, rng)`
# DIRECTLY on the concrete handle fields — which mutates the frame in place and returns the SAME frame — and
# returns the SAME sampler, so `result === state`. @inferred / 0-B.
@inline function (::_Mode2KernelSkeleton{RootToken})(
        k::KernelObject{OwnerToken,State,<:_NutsHandles{RootToken}}; rng) where {RootToken,OwnerToken,State}
    h = getfield(k, :handles)
    getfield(h, :root)(getfield(k, :state), getfield(h, :scratch), rng)
    k
end


# --- registered stats_f scalar-effect binding (RK 08:42/08:55) ---------------------------------------
#
# The diagnostic slot index of each fixture diagnostic field (1=n_steps, 2=reached_depth, 3=acceptance_
# rate, 4=dham) — the stable map the stats binding + poc use, never a name heuristic on the store.
_diagnostic_slot_of(name::Symbol) =
    name === :n_steps ? 1 : name === :reached_depth ? 2 : name === :acceptance_rate ? 3 :
    name === :dham ? 4 : 0

# The registered `stats_f` scalar-effect binding: the registered callback (by identity) + the diagnostic
# slot indices its DETACHED write-roots produce (nuts_stats! writes n_steps + acceptance_rate → slots
# {1,3}). `stats_f=nothing` → the NO-EFFECT binding (the fixture `collectstats!` is `isnothing(stats_f) ||
# stats_f(__self__)`). An OPAQUE/unregistered stats_f REJECTS — a diagnostics callback must be a registered
# @kernel, never an opaque runtime callable. POC executes stats_f (writing the values via `_diag_set!`,
# which marks them pending); the binding is the detached which-slots contract poc consumes.

# The stats binding holds the captured `_KernelRegistration` (RK 09:25), not the raw stats_f, plus the
# diagnostic slots its write-roots produce.
struct _StatsBinding{R,Slots<:Tuple}
    registration::R    # the captured _KernelRegistration, or `nothing` for the no-effect variant
    produced::Slots
end
stats_binding_produced(b::_StatsBinding) = b.produced
stats_binding_registration(b::_StatsBinding) = b.registration
stats_binding_token(b::_StatsBinding) = b.registration === nothing ? nothing : b.registration.token
stats_binding_source(b::_StatsBinding) = b.registration === nothing ? nothing : b.registration.source
# Derive the registration INTERNALLY (RK 09:19) — NEVER trust a caller-supplied value+reg pair. `stats_f=
# nothing` is the no-effect variant. An opaque/unregistered stats_f REJECTS. Every declared write-root MUST
# map to a diagnostic slot — an UNMAPPABLE root REJECTS (no silent filter; never understate effects).
function _stats_binding(stats_f)
    stats_f === nothing && return _StatsBinding{Nothing,Tuple{}}(nothing, ())
    reg = _kernel_resolve_callable(stats_f)
    reg === nothing && throw(_KernelFactoryReject(
        "stats_f = $(typeof(stats_f)) is opaque/unregistered — the diagnostics callback must be a " *
        "registered @kernel, not an opaque runtime callable"))
    _validate_binder!(:stats_f, stats_f, reg.source)   # a partial wrapper cannot silently reduce semantics
    slots = Int[]
    for r in reg.write_roots
        s = _diagnostic_slot_of(r)
        s == 0 && throw(_KernelFactoryReject(
            "stats_f writes an UNMAPPABLE root `$r` — refusing to understate its effects; a diagnostics " *
            "callback may write only n_steps / reached_depth / acceptance_rate / dham"))
        push!(slots, s)
    end
    _StatsBinding(reg, Tuple(sort!(unique(slots))))
end
# Mark the diagnostic slots stats_f produced as PENDING on the store (RK 08:48) — poc calls this after
# executing stats_f. The no-effect (`nothing`) binding produces nothing.
_stats_produced!(diag::_DiagnosticsStore, ::_StatsBinding{Nothing}) = diag
function _stats_produced!(diag::_DiagnosticsStore, b::_StatsBinding)
    for s in b.produced; _diag_bless!(diag, Val(s)); end
    diag
end


