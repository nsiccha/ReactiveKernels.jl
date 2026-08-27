# Lowering an AUTHORED free stateful @kernel (dual_averaging_state / welford_var) to a RUNNABLE object —
# the AUTHORED recurrence, NOT the package @reactive type (HMC acceptance G3/G4). A stateful @kernel is a
# _StatefulKernelSkeleton: field-initializer recipes (m=one(init), H=zero(init), mu=..., log_current=...)
# captured through the stateless graph, plus mutating methods (fit!(x), step!(x;dn)) captured as MethodIRs.
#
# It reuses the SAME substrate proven on nuts_state:
#   * `_kernel_factory_plan` (generic owned/shared, NOT the integrator-gated `_prepare_factory`) for the plan;
#   * `_kernel_construct_endpoint` + `compile_prepared_initialization` (the assign-mode recipe handles) to
#     build+initialize the concrete state (field values get their concrete types from `init`);
#   * the leapfrog machinery `_lf_ensure!`/`_lf_kill_closure`/`_lf_mask!` for a mutating method: a method is
#     a transition — execute its place-writes, DEMAND-ENSURE a derived self-field read in authored order
#     (fit!'s statement 3 reads `log_current`, made stale by the m/H writes → recompute with the new m/H),
#     and KILL downstream deriveds (current/final) so a later field read demand-ensures.

# ---- generic prepared factory for a FREE stateful kernel (no integrator, no destination grad) --------------
# `_prepare_factory` is endpoint/integrator-gated (requires the integrator's subject write-roots to seed
# ownership); a free stateful kernel has neither. Build the plan directly from the method-derived owned set
# (fields written by a method ∪ fields transitively derived from them) and the have/derived complement as
# shared, then ZIP the selected ops with the recipe seam into concrete handles (exactly `_prepare_factory`'s
# handle construction, minus the destination/external self-discovery — a free stateful kernel has no
# `:destination`/`:portcall` recipe, which we ASSERT rather than silently permit).
function _prepare_stateful(skel)
    owned = _kernel_factory_local_owned_seed(skel)
    shared = setdiff(Set{Symbol}(kernel_port_names(skel)), owned)
    plan, ops = _kernel_factory_plan(skel, owned, shared; key_token = kernel_token(skel), with_ops = true)
    seam = kernel_plan_recipe_seam(plan)
    handles = map(ops, seam) do o, s
        o[1] == s[1] || throw(_KernelFactoryReject("prepared op/seam order mismatch: $(o[1]) vs $(s[1])"))
        _recipe_handle(o[2], s[2], s[3], s[4])
    end
    for h in handles
        recipe_handle_mode(h) === :destination && throw(_KernelFactoryReject(
            "a free stateful kernel must not carry a :destination (external-grad) recipe"))
    end
    _PreparedFactory{kernel_token(skel), 0, typeof(plan), typeof(handles), Tuple{}}(plan, handles, ())
end

# Resolve the have-port VALUES for a construction call, evaluating each unsupplied keyword default closure
# against the already-resolved earlier args (the exact `_kernel_signature_invoke` binding order): the
# positional `init` is required; each keyword (target/regularization_scale/relaxation_exponent/offset) is the
# user override or its authored default `oftype(init, …)`. Returns a Dict{canon => value} the constructor
# indexes. NO have port may be left unbound.
function _stateful_have_values(skel, pf, init, kwargs::NamedTuple)
    P, K, M, defaults = _stateful_signature(skel)
    plan = kernel_prepared_plan(pf)
    fc = Dict{Symbol,Int}(s.path[end] => s.canon for s in kernel_plan_slots(plan))
    resolved = Any[]; byname = Dict{Symbol,Any}()
    # positional (just `init` here) — required, supplied
    for (i, nm) in enumerate(P)
        val = i == 1 ? init : throw(_KernelFactoryReject("stateful kernel expects one positional (init); got $(length(P))"))
        push!(resolved, val); byname[nm] = val
    end
    # keywords — user override or default closure evaluated against resolved-so-far
    for (j, nm) in enumerate(K)
        idx = length(P) + j
        val = haskey(kwargs, nm) ? kwargs[nm] :
              M[idx] ? getfield(defaults, idx)(resolved...) :
              throw(UndefKeywordError(nm))
        push!(resolved, val); byname[nm] = val
    end
    d = Dict{Int,Any}()
    for (nm, val) in byname
        haskey(fc, nm) && (d[fc[nm]] = val)
    end
    d
end

# The (positional-names, keyword-names, default-mask, defaults) of a stateful skeleton's call signature
# (read off the immutable spec snapshot's captured `_KernelCallSignature`).
function _stateful_signature(skel)
    sig = getfield(getfield(skel, :spec_snapshot), :call_signature)
    sig isa _KernelCallSignature || throw(_KernelFactoryReject("stateful kernel has no keyword call signature"))
    P, K, M = typeof(sig).parameters[1:3]
    (P, K, M, getfield(sig, :defaults))
end

# `_kernel_construct_endpoint` indexes a value for EVERY canon (produced fields get a typed placeholder the
# init recipes then overwrite — exactly like the NUTS `_nuts_mkvals` placeholders). For a free stateful
# kernel every recipe is a single-output `:assign`, so derive the produced placeholders GENERICALLY by
# interpreting the handles once (cold) over the have-values in plan order — each output is the op applied to
# its canonical inputs. This yields type-correct values for every field; the compiled init below then
# re-runs the same recipes into the concrete storage and blesses them (the authoritative, mask-correct path).
function _stateful_seed_values(pf, have::Dict{Int,Any})
    vals = copy(have)
    for (i, h) in enumerate(kernel_prepared_handles(pf))
        recipe_handle_mode(h) === :assign || throw(_KernelFactoryReject(
            "free stateful init handle $i has mode $(recipe_handle_mode(h)); only :assign is admissible"))
        length(h.outputs) == 1 || throw(_KernelFactoryReject("stateful init handle $i must be single-output"))
        vals[h.outputs[1]] = recipe_handle_op(h)((vals[c] for c in h.inputs)...)
    end
    vals
end

# Build + INITIALIZE the concrete state (owned, shared) from a construction call. The have ports (init +
# resolved params) plus interpreted produced placeholders seed the constructor; `compile_prepared_-
# initialization` then runs the assign-mode field recipes (m=one(init), …) into the concrete storage and
# blesses them, giving every field its `init`-derived concrete type + the correct all-current entry mask.
function _construct_stateful(skel, pf, init; kwargs...)
    have = _stateful_have_values(skel, pf, init, NamedTuple(kwargs))
    vals = _stateful_seed_values(pf, have)
    owned, shared = _kernel_construct_endpoint(Val(kernel_prepared_token(pf)), kernel_prepared_plan(pf), vals, ())
    compile_prepared_initialization(pf, typeof(owned), typeof(shared))(owned, shared, kernel_prepared_handles(pf))
    (owned, shared)
end
