struct SMCSelectiveTarget{Q,B,S,R,C,D} <: AbstractSMCTarget
    query::Q
    backend::B
    signature::S
    prior::R
    conditional::C
    selection::D
end

function _prepare_smc_selective(built, binding, u0; previous, backend, model_id,
        binding_id, warm=false)
    isempty(model_id) && throw(ArgumentError("model_id must identify frozen model semantics"))
    isempty(binding_id) && throw(ArgumentError("binding_id must identify the new window"))
    isempty(previous.calibration_id) && throw(ArgumentError("carried calibration identity is empty"))
    hasproperty(built, :program) || throw(ArgumentError("prepare_smc_window needs a build_kernel result"))
    previous.frame.model_id == model_id || throw(ArgumentError("selective SMC model/preprocessing identity differs"))
    hasproperty(previous.frame, :coordinates) || throw(ArgumentError(
        "selective SMC needs a carry with authored coordinate identities; rebuild the calibration carry"))
    length(u0) == built.layout.total || throw(DimensionMismatch("exemplar and authored frame"))
    records = PPL._smc_coordinate_records(built.layout)
    match = PPL._smc_match_coordinates(previous.frame.coordinates, records)
    # This graph includes only new authored factors, along with the window
    # likelihood. Its structural read support closes over their hyperparameters.
    terms = PPL._smc_window_terms(built, binding, match.fresh, u0)
    indices = PPL._smc_read_support(terms, u0)
    # A new latent must enter its authored prior even if the likelihood omits it.
    indices = sort!(union(indices, match.fresh))
    active_slots = findall(i -> !iszero(match.old[i]), indices)
    active_old = match.old[indices[active_slots]]
    fresh_slots = findall(i -> iszero(match.old[i]), indices)
    old = _gaussian(previous.reference)
    conditional = ReactiveKernels._gaussian_conditional(old, active_old)
    marginal = ReactiveKernels._gaussian_marginal(old, active_old)
    compact_kernel = PPL._smc_selected_kernel(terms, indices, u0)
    u = Vector{Float64}(u0[indices])
    ad = Base.invokelatest(prepare_ad, compact_kernel, backend, u;
        active=:_ppl_smc_active, warm)
    query = SamplerQuery(compact_kernel, ad, (; total=length(indices)))
    prior = _compact_reference(backend, length(active_old), _gaussian_reference(marginal))
    union_coordinates = vcat(previous.frame.coordinates, records[match.fresh])
    joint_indices = copy(match.old)
    joint_indices[match.fresh] = length(old.mean) .+ collect(eachindex(match.fresh))
    present = falses(length(union_coordinates))
    present[joint_indices] .= true
    joint_order = vcat(joint_indices, findall(!, present))
    joint_coordinates = union_coordinates[joint_order]
    inverse_order = invperm(joint_order)
    selection = deepcopy((; indices, active_old, active_slots, fresh_slots,
        joint_indices=inverse_order[joint_indices[indices]],
        union_indices=joint_indices[indices], joint_order, joint_coordinates,
        current_coordinates=records, previous=old))
    signature = deepcopy((; model_id=String(model_id), binding_id=String(binding_id),
        program=built.program, layout=(length(indices), records[indices]),
        selection=(; indices, active_old, active_slots, fresh_slots),
        carried=(; previous.reference, previous.frame, previous.calibration_id)))
    SMCSelectiveTarget(query, backend, signature, prior, conditional, selection)
end

function _density(t::SMCSelectiveTarget, u)
    t.query(u) + _density(t.prior, u[t.selection.active_slots])
end
function _gradient(t::SMCSelectiveTarget, g, u)
    value = first(sampler_value_and_gradient!(t.query, g, u))
    slots = t.selection.active_slots
    h = zeros(length(slots))
    value += _gradient(t.prior, h, u[slots])
    for (i, slot) in enumerate(slots)
        g[slot] += h[i]
    end
    value
end

function _default_reference(t::SMCSelectiveTarget)
    d = t.query.layout.total
    active = _gaussian(t.prior.reference)
    mean, diagonal = zeros(d), ones(d)
    vectors = zeros(d, length(active.values))
    slots = t.selection.active_slots
    mean[slots], diagonal[slots], vectors[slots,:] = active.mean, active.diagonal, active.vectors
    _gaussian_reference(CompactGaussian(mean, diagonal, vectors, copy(active.values)))
end

function PPL.smc_window_coordinates(t::SMCSelectiveTarget)
    deepcopy((; compact=t.selection.current_coordinates[t.selection.indices],
        current_indices=t.selection.indices, active_old=t.selection.active_old,
        active_slots=t.selection.active_slots, fresh_slots=t.selection.fresh_slots,
        joint=t.selection.joint_coordinates, joint_indices=t.selection.joint_indices))
end

function PPL.smc_recover(t::SMCSelectiveTarget, u::AbstractVector; rng::Xoshiro)
    length(u) == t.query.layout.total || throw(DimensionMismatch("compact SMC particle"))
    result = zeros(length(t.selection.joint_coordinates))
    result[t.selection.union_indices] = u
    result[t.conditional.inactive] = ReactiveKernels._gaussian_recover(rng,
        t.conditional, u[t.selection.active_slots])
    result[t.selection.joint_order]
end

PPL.smc_window_coordinates(state::SMCState) = PPL.smc_window_coordinates(state.target)
PPL.smc_recover(state::SMCState, u::AbstractVector; rng::Xoshiro) =
    PPL.smc_recover(state.target, u; rng)

function PPL.carry_smc(target::SMCSelectiveTarget, reference::SMCReference;
        calibration_id::AbstractString, maxrank::Integer=16)
    isempty(calibration_id) && throw(ArgumentError("calibration_id must identify this summary"))
    length(reference.location) == target.query.layout.total ||
        throw(DimensionMismatch("compact posterior reference"))
    posterior, diagnostics = ReactiveKernels._gaussian_replace_marginal(
        target.selection.previous, target.selection.active_old, _gaussian(reference);
        new_active=target.selection.active_slots, maxrank)
    posterior = ReactiveKernels._gaussian_marginal(posterior, target.selection.joint_order)
    frame = deepcopy((; model_id=target.signature.model_id,
        layout=(length(posterior.mean), target.selection.joint_coordinates),
        coordinates=target.selection.joint_coordinates))
    SMCCarriedReference(_gaussian_reference(posterior), frame, String(calibration_id), diagnostics)
end

function _carry_summary(target::SMCSelectiveTarget, reference;
        calibration_id, diagnostics, max_reflections)
    carried = carry_smc(target, reference; calibration_id, maxrank=max_reflections)
    SMCCarriedReference(carried.reference, carried.frame, carried.calibration_id,
        deepcopy((; marginal=diagnostics, joint=carried.fit_diagnostics)))
end

_parameter_projection(::SMCSelectiveTarget, parameters) = throw(ArgumentError(
    "parameters selection requires a full-frame SMC fit; omit parameters for " *
    "a selective window to integrate its inactive conditional into the full joint carry"))

_worker_target(t::SMCSelectiveTarget) = SMCSelectiveTarget(deepcopy(t.query), t.backend,
    t.signature, _worker_target(t.prior), t.conditional, t.selection)
