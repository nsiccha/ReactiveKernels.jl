# The observation query can add key columns to the factor catalogue's binding.
# Drop only those extra keys before checking the original program and its data.
function _jax_partition_binding(built, plan, catalogue, by, expected_layout)
    _check_factor_binding(catalogue.built, catalogue.plan, catalogue)
    coordinates = coordinate_identities(expected_layout)
    for layout in (built.layout, catalogue.built.layout)
        isequal(coordinate_identities(layout), coordinates) &&
            _same_layout(layout, expected_layout) &&
            all(e.kind === k.kind && e.transform === k.transform &&
                isequal(e.lo, k.lo) && isequal(e.hi, k.hi)
                for (e, k) in zip(layout.entries, expected_layout.entries)) || throw(ContractValidationError(
                "[jax] partition and factor selections must have the expected packed layout"))
    end
    extra = setdiff(keys(plan.columns), keys(catalogue.plan.columns))
    all(k -> k in values(by), extra) || throw(ContractValidationError(
        "[jax] the observation binding has extra columns other than partition keys"))
    projected = isempty(extra) ? plan : _with(plan; columns = filter(
        pair -> first(pair) ∉ extra, plan.columns))
    _check_factor_binding(catalogue.built, projected, catalogue)
    projected
end

# The same authored identity can occur in several recorded segments of one
# site. Entries are local to the segment in its plate, global to the site in
# the public catalogue. A whole joint event has exactly one owner.
function _jax_factor_owners(catalogue, selections)
    owners = Dict{DensityFactor,Int}()
    for (bucket, selection) in enumerate(selections)
        selection.catalogue === catalogue || throw(ContractValidationError(
            "[jax] global and unit selections must refer to the supplied factor catalogue"))
        for event in select_factors(catalogue, selection)
            haskey(owners, event) && throw(ContractValidationError(
                "[jax] density factor $event is assigned to more than one bucket"))
            owners[event] = bucket
        end
    end
    [[get(owners, DensityFactor(s.kind, s.site, s.offset + row), 0)
        for row in 1:s.count] for s in catalogue.sites]
end

function _jax_partition_ownership(built, plan, observations, by, partitioned,
        global_factors, unit_factors, expected_layout)
    built = _binding_build(built, plan)
    _check_built_program(built, plan)
    catalogue = global_factors.catalogue
    frozen = plan.frozen
    plan = _factor_integer_plan(built, plan, catalogue)
    names = _query_observations(built, plan, observations)
    _refuse_absent_cells(plan, names)
    built, plan, by = _partition_label_keys(built, plan, by)
    key_names = _partition_keys(plan, names, by)
    projected = _jax_partition_binding(built, plan, catalogue, key_names, expected_layout)
    observation_labels = _partition_labels(plan, names, key_names)
    labels = unique([collect(keys(unit_factors)); observation_labels])
    all(label -> label isa Integer, labels) || throw(ArgumentError(
        "the JAX partition label argument requires integer keys"))
    labels = sort!(collect(Int, labels))
    empty_selection = select_factors(catalogue, DensityFactor[])
    by_unit = [get(unit_factors, label, empty_selection) for label in labels]
    event_owners = _jax_factor_owners(catalogue, Iterators.flatten(((global_factors,), by_unit)))
    # The global bucket has position 1. Label values (which need not be
    # contiguous or positive) cross the public boundary without renumbering.
    bucket = Dict(label => i + 1 for (i, label) in enumerate(labels))
    observation_owners = [[bucket[label] for label in vec(plan.columns[key_names[name]])]
        for name in names]
    if !isempty(observation_labels)
        _prepare_partition(built, plan, key_names, names; locality = :error,
            on_error = nothing, partitioned, u0 = nothing)
    end
    source_plan = projected.frozen === frozen ? projected : _with(projected; frozen)
    (; names, labels, observation_owners, event_owners, source_plan)
end

"""
    jax_source(built, plan, observations; by, partitioned = (), global_factors,
               unit_factors, expected_layout, values = (), doc = nothing, float_type = Float64)

Export one model module and the numeric ownership data for this binding.
`global_factors` and each integer label's selection in `unit_factors` must
select disjoint atomic events of the same density catalogue. The observation
keys and `partitioned` have the meaning they have in [`prepare_partition`](@ref).
An explicitly named unit may own no observations or factors.

Write `out.source` as usual. Pass `out.partition_data` alongside the matching
model data. In Python, `q = prepare_partition(prepare_data(**raw), partition_data)`
prepares the two targets `global_log_density(u, **q)` and
`unit_log_density(u, label, **q)`. Both take the full packed point.
"""
function ReactiveKernels.jax_source(built::NamedTuple, plan::StructuralPlan, observations;
        by, partitioned = (), global_factors::FactorSelection, unit_factors,
        expected_layout::LayoutTable, values = (), doc = nothing, float_type = Float64)
    ReactiveKernels._translate_in_latest_world(NamedTuple) do
        ReactiveKernels._jax_with_float_type(float_type) do
            _jax_partition_module(built, plan, observations; by, partitioned,
                global_factors, unit_factors, expected_layout, values, doc, float_type)
        end
    end
end

function _jax_partition_module(built, plan, observations; by, partitioned,
        global_factors, unit_factors, expected_layout, values, doc, float_type)
    ownership = _jax_partition_ownership(built, plan, observations, by, partitioned,
        global_factors, unit_factors, expected_layout)
    plan = ownership.source_plan
    built = _binding_build(global_factors.catalogue.built, plan)
    sites = _DensitySite[]
    spec = ReactiveKernels._within_kernel_names(:ppl_model) do
        _density_factor_spec(plan, built.layout, sites)
    end
    obs = _observation_plates(built, plan, ownership.names)
    ports = Symbol[[obs[n] for n in ownership.names]; [something(s.plate, s.total) for s in sites]]
    events = (; (Symbol(:rk_event_, i) => p for (i, p) in enumerate(ports))...)
    base = Base.invokelatest(_jax_module_of, merge(built, (; spec)), plan;
        values, doc, float_type, events)::ReactiveKernels.JAXSource
    event_tuple = "(" * join(string.(keys(events)), ", ") * (isempty(events) ? ")" : ",)")
    arguments = ["prepared" => Dict, "labels" => Vector{Int}, "sites" => Tuple]
    point = "unconstrained" => Vector{float_type}
    source = ReactiveKernels.JAXSource(base.text * "\n" * ReactiveKernels._JAX_INDEXED *
        "\n_RK_EVENTS = " * event_tuple * "\n_RK_DENSITY_TYPE = jnp." *
        ReactiveKernels._jax_dtype_name(float_type) * "\n" * _JAX_PARTITION,
        [base.functions;
         ReactiveKernels.JAXFunction("prepare_partition",
            ["prepared" => Dict, "partition_data" => Dict], ["partition" => Dict]);
         ReactiveKernels.JAXFunction("global_log_density",
            [point; arguments], ["score" => float_type]);
         ReactiveKernels.JAXFunction("unit_log_density",
            [point; "label" => Int; arguments],
            ["score" => float_type])], base.prepared)
    discrete = get(global_factors.catalogue.plan.frozen.values, :_ppl_discrete, nothing)
    state = discrete === nothing ? (;) : (; discrete)
    partition_data = (; ownership.labels, owners = vcat(ownership.observation_owners,
        ownership.event_owners), state...)
    (; source, partition_data)
end

Base.include_dependency(joinpath(@__DIR__, "jax_partition.py"))
const _JAX_PARTITION = read(joinpath(@__DIR__, "jax_partition.py"), String)
