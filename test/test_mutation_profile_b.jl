using Test
using LinearAlgebra
using Random
import TOML
import ReactiveKernels as RK
import ReactiveKernelsNUTSExamples as NEX
import ReactiveKernelsStreamingStats as RSS

module _MutationProfileBFixtures
include(joinpath(@__DIR__, "..", "benchmark",
                 "nuts_kernel_authoring_fixture_b.jl"))
include(joinpath(@__DIR__, "..", "benchmark",
                 "walnuts_kernel_authoring_fixture_b.jl"))
include(joinpath(@__DIR__, "..", "benchmark",
                 "reactivehmc_statistics_kernel_fixture.jl"))
end

include(joinpath(@__DIR__, "fixtures",
                 "reactivehmc_hmc_compiler_support.jl"))
isdefined(@__MODULE__, :MutationProfileBGenericControl) || include(joinpath(
    @__DIR__, "fixtures", "mutation_profile_b_generic_control.jl"))

const _MPB_NUTS = _MutationProfileBFixtures.NUTSBMutationAuthoringFixture
const _MPB_WALNUTS =
    _MutationProfileBFixtures.WALNUTSBMutationAuthoringFixture
const _MPB_HMC_SUPPORT = ReactiveHMCHMCCompilerSupport
const _MPB_STATISTICS =
    _MutationProfileBFixtures.ReactiveHMCStatisticsFixture
const _MPB_GENERIC_CONTROL = MutationProfileBGenericControl
const _MPB_TESTSET = get(ENV, "RK_MPB_TESTSET", "all")
_mpb_enabled(name) = _MPB_TESTSET == "all" || _MPB_TESTSET == name

module _MutationProfileBFormalProbe
using ReactiveKernels

@kernel formal_assignment_probe(seed) = begin
    rebind!(endpoint) = begin
        endpoint = seed
    end
    mutate!(endpoint) = begin
        endpoint .= seed
    end
end
end

function _mpb_ir_by_name(spec)
    Dict(ir.id.name => ir for ir in RK.method_irs(spec))
end

function _mpb_calls(ir)
    calls = Any[]
    for statement in ir.body
        RK._kmir_walk(statement) do node
            node isa Union{RK._Call,RK._CallExpr} && push!(calls, node)
        end
    end
    calls
end

_mpb_callee_mid(call) = only(call.candidates).id.decl

_mpb_is_formal(node, name::Symbol) =
    node isa RK._FormalRef && node.arg === name

function _mpb_is_decrement(node, name::Symbol)
    node isa RK._RegisteredCall || return false
    getfield(node.registration, :kind) === :pure_primitive || return false
    getfield(node.registration, :source) === (-) || return false
    length(node.args) == 2 || return false
    _mpb_is_formal(node.args[1], name) || return false
    node.args[2] isa RK._Lit && node.args[2].value == 1
end

function _mpb_recursive_edges(irs, recursive_mids, argument_position)
    edges = NamedTuple[]
    for ir in irs, call in _mpb_calls(ir)
        callee = _mpb_callee_mid(call)
        ir.id.decl in recursive_mids && callee in recursive_mids || continue
        push!(edges, (; caller=ir.id.decl, callee,
            argument=call.pos[argument_position]))
    end
    edges
end

function _mpb_rank_decreases(edges, recursive_name::Symbol, phase)
    all(edges) do edge
        delta = _mpb_is_formal(edge.argument, recursive_name) ? 0 :
            _mpb_is_decrement(edge.argument, recursive_name) ? -1 : nothing
        delta === nothing && return false
        # Any positive depth has the same comparison; use two to keep both the
        # same-depth phase edge and the decrementing edges away from the base.
        2 * 2 + phase[edge.caller] >
            2 * (2 + delta) + phase[edge.callee]
    end
end

function _mpb_control_shape(program)
    [(
        midpos=block.midpos,
        pc=block.pc,
        term=block.term,
        then_pc=block.then_pc,
        else_pc=block.else_pc,
        callee_midpos=block.callee_midpos,
        resume_pc=block.resume_pc,
        effects=Tuple(typeof(effect) for effect in block.effects),
    ) for block in program.blocks]
end

function _mpb_bound_metadata(bounds::RK.StatefulControlBounds)
    parameters = typeof(bounds).parameters
    (argument_types=parameters[3], iterations=parameters[4],
     recursion_depth=parameters[5], control_steps=parameters[6],
     recursion_path=parameters[7], state_type=parameters[8])
end

function _mpb_rejection_reason(f)
    try
        f()
    catch error
        error isa RK._LLowerReject || rethrow()
        return getfield(error, :reason)
    end
    nothing
end

function _mpb_source(path)
    read(joinpath(@__DIR__, "..", path), String)
end

if _mpb_enabled("source")
@testset "mutation profile B — explicit source surface and MethodIR" begin
    sources = Dict(
        :hmc => _mpb_source("benchmark/reactivehmc_hmc_kernel_fixture_b.jl"),
        :nuts => _mpb_source("benchmark/nuts_kernel_authoring_fixture_b.jl"),
        :walnuts => _mpb_source("benchmark/walnuts_kernel_authoring_fixture_b.jl"),
        :integrators =>
            _mpb_source("benchmark/reactivehmc_integrator_kernel_fixture.jl"),
        :statistics =>
            _mpb_source("benchmark/reactivehmc_statistics_kernel_fixture.jl"),
    )
    @test all(!occursin("copy!!(", source) for source in values(sources))
    @test !occursin("fill!(attempted_micro_steps", sources[:walnuts])
    authored_dot_assignments(source) =
        count(r"^\s*[^#\n].* \.= "m, source)
    @test authored_dot_assignments(sources[:walnuts]) == 13
    @test authored_dot_assignments(sources[:nuts]) == 6
    @test occursin("phasepoint.mom = phasepoint.chol_metric.L * phasepoint.mom",
                   sources[:nuts])
    @test occursin("proposals[i], proposals[j] = proposals[j], proposals[i]",
                   sources[:nuts])
    @test occursin("@. phasepoint.mom", sources[:integrators])
    @test occursin("positions[index, column] = pos[index]", sources[:statistics])
    @test !occursin(r"^\s*(push!|append!|fill!|copy!!)\("m,
                    sources[:statistics])

    for spec in (_MPB_HMC_SUPPORT.ReactiveHMCHMCBMutationAuthoringFixture.hmc_state,
                 _MPB_NUTS.nuts_state, _MPB_WALNUTS.walnuts_state,
                 _MPB_NUTS.dual_averaging_state, _MPB_NUTS.welford_var,
                 _MPB_HMC_SUPPORT.ReactiveHMCIntegratorFixture.generalized_leapfrog!,
                 _MPB_HMC_SUPPORT.ReactiveHMCIntegratorFixture.implicit_midpoint!,
                 _MPB_STATISTICS.statistics_state, RSS.welford_var)
        irs = RK.method_irs(spec)
        @test !isempty(irs)
        @test all(ir -> ir.ok, irs)
    end

    rebind_probe = only(ir for ir in RK.method_irs(
        _MutationProfileBFormalProbe.formal_assignment_probe)
        if ir.id.name === :rebind!)
    mutate_probe = only(ir for ir in RK.method_irs(
        _MutationProfileBFormalProbe.formal_assignment_probe)
        if ir.id.name === :mutate!)
    @test only(rebind_probe.body) isa RK._LocalAssign
    @test only(mutate_probe.body) isa RK._PlaceWrite
    @test only(mutate_probe.body).dot
    @test only(mutate_probe.body).target isa RK._FormalRef

    macro_step = only(ir for ir in RK.method_irs(_MPB_WALNUTS.walnuts_state)
                      if ir.id.name === :macro_step!)
    formal_copies = RK._PlaceWrite[]
    for statement in macro_step.body
        RK._kmir_walk(statement) do node
            node isa RK._PlaceWrite && node.dot &&
                node.target isa RK._FormalRef && push!(formal_copies, node)
        end
    end
    @test length(formal_copies) == 1
    @test only(formal_copies).target.arg === :ep
end
end

if _mpb_enabled("control")
@testset "mutation profile B — generic bounded-control evidence" begin
    # Two alpha-distinct recursive kernels carry the same bounded-control
    # structure. Their names and state fields deliberately differ, so the
    # discovered SCC/rank/control program cannot rely on consumer identity.
    probe_irs = RK.method_irs(_MPB_GENERIC_CONTROL.bounded_counter)
    probe_by_name = _mpb_ir_by_name(
        _MPB_GENERIC_CONTROL.bounded_counter)
    probe_recursive = RK.recursive_mids(probe_irs)
    phase_a_mid = probe_by_name[:phase_a!].id.decl
    phase_b_mid = probe_by_name[:phase_b!].id.decl
    drive_mid = probe_by_name[:drive!].id.decl
    @test probe_recursive == Set((phase_a_mid, phase_b_mid))
    @test RK.defunctionalized_mids(probe_irs) ==
          Set((phase_a_mid, phase_b_mid, drive_mid))
    probe_edges = _mpb_recursive_edges(probe_irs, probe_recursive, 1)
    @test length(probe_edges) == 2
    @test _mpb_rank_decreases(probe_edges, :n,
        Dict(phase_b_mid => 1, phase_a_mid => 0))
    probe_program = RK._control_program_from_irs(
        probe_irs; root_mid=drive_mid)
    @test Set(probe_program.methods) ==
          Set((phase_a_mid, phase_b_mid, drive_mid))
    @test all(block -> block.term !== :call ||
        block.callee_mid in probe_recursive, probe_program.blocks)

    renamed_irs = RK.method_irs(
        _MPB_GENERIC_CONTROL.renamed_machine)
    renamed_by_name = _mpb_ir_by_name(
        _MPB_GENERIC_CONTROL.renamed_machine)
    renamed_recursive = RK.recursive_mids(renamed_irs)
    descend_mid = renamed_by_name[:descend!].id.decl
    ascend_mid = renamed_by_name[:ascend!].id.decl
    run_mid = renamed_by_name[:run!].id.decl
    renamed_edges = _mpb_recursive_edges(renamed_irs, renamed_recursive, 1)
    @test renamed_recursive == Set((descend_mid, ascend_mid))
    @test _mpb_rank_decreases(renamed_edges, :n,
        Dict(ascend_mid => 1, descend_mid => 0))
    renamed_program = RK._control_program_from_irs(
        renamed_irs; root_mid=run_mid)
    @test Set(values(probe_program.names)) !=
          Set(values(renamed_program.names))
    @test _mpb_control_shape(renamed_program) ==
          _mpb_control_shape(probe_program)

    ambiguous_irs = RK.method_irs(
        _MPB_GENERIC_CONTROL.ambiguous_counter)
    @test length(RK.recursive_mids(ambiguous_irs)) == 1
    ambiguous_loops = RK._For[]
    for ir in ambiguous_irs, statement in ir.body
        RK._kmir_walk(statement) do node
            node isa RK._For && push!(ambiguous_loops, node)
        end
    end
    @test Set(loop.iter.args[2].path for loop in ambiguous_loops) ==
          Set(((:left_extent,), (:right_extent,)))

    while_irs = RK.method_irs(
        _MPB_GENERIC_CONTROL.uncertified_while_counter)
    @test any(ir -> any(statement -> begin
            found = Ref(false)
            RK._kmir_walk(node -> node isa RK._While && (found[] = true),
                          statement)
            found[]
        end, ir.body), while_irs)
end
end

if _mpb_enabled("bounds")
@testset "mutation profile B — public bounded-control certificate" begin
    probe_kernel = RK.compile_stateful(
        _MPB_GENERIC_CONTROL.bounded_counter, 0, 2)
    probe_state = RK.stateful_snapshot(probe_kernel(0, 2))
    probe_bounds = RK.stateful_control_bounds(
        probe_kernel, Val(:drive!), probe_state;
        argument_types=Tuple{})
    probe_metadata = _mpb_bound_metadata(probe_bounds)
    @test probe_metadata.argument_types === Tuple{}
    @test probe_metadata.iterations == 3
    @test probe_metadata.recursion_depth == 2
    @test probe_metadata.control_steps > 0
    @test probe_metadata.recursion_path == (:ceiling,)
    @test probe_metadata.state_type === typeof(probe_state)

    probe_transition = RK.functionalize_stateful(
        probe_kernel, Val(:drive!), probe_bounds)
    probe_result = probe_transition(probe_state)
    @test !probe_result.control_overflow
    @test probe_result.state.total == 3

    renamed_kernel = RK.compile_stateful(
        _MPB_GENERIC_CONTROL.renamed_machine, 0, 2)
    renamed_state = RK.stateful_snapshot(renamed_kernel(0, 2))
    renamed_bounds = RK.stateful_control_bounds(
        renamed_kernel, Val(:run!), renamed_state;
        argument_types=Tuple{})
    renamed_metadata = _mpb_bound_metadata(renamed_bounds)
    @test renamed_metadata.iterations == probe_metadata.iterations
    @test renamed_metadata.recursion_depth == probe_metadata.recursion_depth
    @test renamed_metadata.control_steps == probe_metadata.control_steps
    @test renamed_metadata.recursion_path == (:horizon,)
    renamed_result = RK.functionalize_stateful(
        renamed_kernel, Val(:run!), renamed_bounds)(renamed_state)
    @test !renamed_result.control_overflow
    @test renamed_result.state.tally == 3

    ambiguous_kernel = RK.compile_stateful(
        _MPB_GENERIC_CONTROL.ambiguous_counter, 0, 3, 2)
    ambiguous_state = RK.stateful_snapshot(ambiguous_kernel(0, 3, 2))
    @test _mpb_rejection_reason(() -> RK.stateful_control_bounds(
        ambiguous_kernel, Val(:step!), ambiguous_state;
        argument_types=Tuple{})) ==
        "recursive state-machine bounds require one unambiguous positive " *
        "integer state authority; pass recursion_bound as its field path"
    explicit_bounds = RK.stateful_control_bounds(
        ambiguous_kernel, Val(:step!), ambiguous_state;
        argument_types=Tuple{}, recursion_bound=:left_extent)
    explicit_metadata = _mpb_bound_metadata(explicit_bounds)
    @test explicit_metadata.iterations == 4
    @test explicit_metadata.recursion_depth == 3
    @test explicit_metadata.recursion_path == (:left_extent,)
    explicit_result = RK.functionalize_stateful(
        ambiguous_kernel, Val(:step!), explicit_bounds)(ambiguous_state)
    @test !explicit_result.control_overflow
    @test explicit_result.state.total == 7

    while_kernel = RK.compile_stateful(
        _MPB_GENERIC_CONTROL.uncertified_while_counter, 0, 3)
    while_state = RK.stateful_snapshot(while_kernel(0, 3))
    @test _mpb_rejection_reason(() -> RK.stateful_control_bounds(
        while_kernel, Val(:step!), while_state;
        argument_types=Tuple{})) ==
        "a structured while loop requires an explicit max_iterations " *
        "when constructing StatefulControlBounds"

    short_bounds = RK.stateful_control_bounds(
        while_kernel, Val(:step!), while_state;
        argument_types=Tuple{}, max_iterations=2)
    short_result = RK.functionalize_stateful(
        while_kernel, Val(:step!), short_bounds)(while_state)
    @test short_result.control_overflow
    @test short_result.state == while_state

    while_bounds = RK.stateful_control_bounds(
        while_kernel, Val(:step!), while_state;
        argument_types=Tuple{}, max_iterations=3)
    while_result = RK.functionalize_stateful(
        while_kernel, Val(:step!), while_bounds)(while_state)
    @test !while_result.control_overflow
    @test while_result.state.total == 3

    backing = [0]
    payload = (values=backing, mirror=backing)
    structured_kernel = RK.compile_stateful(
        _MPB_GENERIC_CONTROL.structured_counter, payload, 0, 2)
    structured_state = RK.stateful_snapshot(
        structured_kernel(payload, 0, 2))
    @test structured_state.payload.values === structured_state.payload.mirror
    structured_bounds = RK.stateful_control_bounds(
        structured_kernel, Val(:drive!), structured_state;
        argument_types=Tuple{})
    structured_transition = RK.functionalize_stateful(
        structured_kernel, Val(:drive!), structured_bounds)
    structured_result = structured_transition(structured_state)
    @test !structured_result.control_overflow
    @test structured_result.state.payload.values ===
          structured_result.state.payload.mirror
    @test structured_result.state.payload.values == [0]
    @test structured_result.state.total == 3

    wrong_backing = [0, 0]
    wrong_shape = merge(structured_state,
        (payload=(values=wrong_backing, mirror=wrong_backing),))
    @test_throws ArgumentError structured_transition(wrong_shape)
    broken_alias = merge(structured_state, (payload=(
        values=structured_state.payload.values,
        mirror=copy(structured_state.payload.mirror)),))
    @test_throws ArgumentError structured_transition(broken_alias)
end
end

function _mpb_nuts_pf()
    RK._prepare_factory(_MPB_NUTS.euclidean_phasepoint,
                        RK.kernel_registration(_MPB_NUTS.leapfrog!))
end

function _mpb_nuts_values(pf, ::Type{T}) where {T}
    plan = RK.kernel_prepared_plan(pf)
    metric = T[2 0; 0 2]
    values = Dict{Int,Any}()
    for slot in RK.kernel_plan_slots(plan)
        name = String(slot.path[1])
        values[slot.canon] = name == "pot_f" ? (p -> sum(abs2, p)) :
            name == "grad_f" ? ((dst, p) -> (dst .= 2 .* p; sum(abs2, p))) :
            name == "metric" ? metric :
            name == "chol_metric" ? cholesky(metric) :
            startswith(name, "##node") ? zero(T) :
            name == "pos" ? T[1, 2] :
            name == "mom" ? T[3, 4] :
            name in ("dpot_dpos", "dham_dpos", "dkin_dmom", "dham_dmom") ?
                T[0, 0] : zero(T)
    end
    values
end

function _mpb_nuts_frame(pf, ::Type{T}, max_depth) where {T}
    frame = NEX._construct_nuts_frame(
        pf, _mpb_nuts_values(pf, T), max_depth;
        step_f=RK.partial(_MPB_NUTS.leapfrog!; stepsize=T(0.1)),
        stats_f=nothing, min_dham=T(-1000))
    RK.compile_prepared_initialization(
        pf, typeof(frame.init), typeof(frame.shared))(
            frame.init, frame.shared, RK.kernel_prepared_handles(pf))
    NEX._seed_nuts_children!(frame)
    frame
end

mutable struct _MPBIndexCounter
    count::Vector{Int}
end
function (counter::_MPBIndexCounter)()
    counter.count[1] += 1
    1
end

if _mpb_enabled("nuts")
@testset "mutation profile B — specialized NUTS native lowering" begin
    control = NEX._nr_plan(_MPB_NUTS.nuts_state)
    @test count(block -> block.mid == control.root_mid, control.blocks) == 26

    pf = _mpb_nuts_pf()
    frames = [_mpb_nuts_frame(pf, Float64, 3) for _ in 1:2]
    compiled = (
        NEX.compile_nuts(pf, _MPB_NUTS.nuts_state,
                         _MPB_NUTS.refresh_momentum!!,
                         _MPB_NUTS.nuts!!, frames[1]),
        NEX.compile_nuts_native(pf, _MPB_NUTS.nuts_state,
                                _MPB_NUTS.refresh_momentum!!,
                                _MPB_NUTS.nuts!!, frames[2]),
    )
    for (transition, frame) in zip(compiled, frames)
        @test transition.root!(frame, transition.scratch,
                               Random.Xoshiro(777)) === frame
    end
    slot(field) = RK.kernel_plan_named_slot_val(
        RK.kernel_prepared_plan(pf), Val(field))
    endpoint_values(frame) = (
        pos=copy(RK._canon_slot(frame.init, slot(:pos))),
        mom=copy(RK._canon_slot(frame.init, slot(:mom))),
        velocity=copy(RK._canon_slot(frame.init, slot(:dkin_dmom))),
        diagnostics=(frame.diag.n_steps, frame.diag.reached_depth,
                     frame.diag.acceptance_rate, frame.diag.dham),
    )
    @test endpoint_values(frames[1]) == endpoint_values(frames[2])

    # Dotted write-and-return is destination-valued for value methods. The
    # synthetic parent makes its child a value method in the encoded program;
    # calling the child directly exposes that public return contract.
    PlanT = typeof(RK.kernel_prepared_plan(pf))
    proposals = NEX._NNSelfField{(:proposals,)}
    direct_write = NEX._NNPlaceWrite{
        proposals, :self, (:proposals,), nothing, proposals, true}
    direct_child = NEX._NNMethod{
        902, :direct_dot_return, Tuple{},
        Tuple{NEX._NNWriteReturn{direct_write}}}
    direct_call = NEX._NNCallExpr{
        :direct_dot_return, 902, NEX._NNSelf, Tuple{}, Tuple{}}
    direct_parent = NEX._NNMethod{
        901, :direct_dot_parent, Tuple{}, Tuple{NEX._NNReturn{direct_call}}}
    DirectProgram = NEX._NativeProgram{
        :mpb_direct_dot, PlanT, 901, Tuple{direct_parent, direct_child}, (),
        NEX._NNNoStats}
    direct_result = NEX._nn_method0(
        DirectProgram, Val(902), (;), frames[2])
    @test direct_result === frames[2].proposals

    # A computed endpoint target is evaluated exactly once, and the returned
    # value is the exact mutated leaf rather than its enclosing endpoint.
    index_calls = [0]
    indexer = _MPBIndexCounter(index_calls)
    index_call = NEX._NNRegistered{
        1, false, :mpb_index_counter, Tuple{}, Tuple{}, false}
    indexed_endpoint = NEX._NNIndex{proposals, Tuple{index_call}}
    destination = NEX._NNGetfield{indexed_endpoint, :mom}
    source = NEX._NNSelfField{(:init, :mom)}
    leaf_write = NEX._NNPlaceWrite{
        destination, :self, (:proposals,), nothing, source, true}
    leaf_child = NEX._NNMethod{
        904, :leaf_dot_return, Tuple{}, Tuple{NEX._NNWriteReturn{leaf_write}}}
    leaf_call = NEX._NNCallExpr{
        :leaf_dot_return, 904, NEX._NNSelf, Tuple{}, Tuple{}}
    leaf_parent = NEX._NNMethod{
        903, :leaf_dot_parent, Tuple{}, Tuple{NEX._NNReturn{leaf_call}}}
    LeafProgram = NEX._NativeProgram{
        :mpb_leaf_dot, PlanT, 903, Tuple{leaf_parent, leaf_child}, (),
        NEX._NNNoStats}
    mom_slot = RK.kernel_plan_named_slot_val(
        RK.kernel_prepared_plan(pf), Val(:mom))
    destination_leaf = RK._canon_slot(frames[2].proposals[1], mom_slot)
    source_value = copy(RK._canon_slot(frames[2].init, mom_slot))
    leaf_result = NEX._nn_method0(
        LeafProgram, Val(904), (; callees=(indexer,)), frames[2])
    @test leaf_result === destination_leaf
    @test leaf_result == source_value
    @test index_calls == [1]

    # Warmed repeated execution uses the same preallocated endpoint storage.
    root, frame, scratch = compiled[1].root!, frames[1], compiled[1].scratch
    root(frame, scratch, Random.Xoshiro(1))
    rng = Random.Xoshiro(2) # keep RNG construction outside the measured transition
    @test @allocated(root(frame, scratch, rng)) == 0
end
end

if _mpb_enabled("hmc")
@testset "mutation profile B — fixed-step HMC parity" begin
    receipt = TOML.parsefile(joinpath(@__DIR__, "..", "benchmark",
                                      "receipts", "reactivehmc-hmc-ca9-v1.toml"))
    case = first(receipt["cases"])
    legacy = _MPB_HMC_SUPPORT.build_case(case)
    profile_b = _MPB_HMC_SUPPORT.build_case(
        case;
        hmc_spec=_MPB_HMC_SUPPORT.
            ReactiveHMCHMCBMutationAuthoringFixture.hmc_state)
    legacy_result = _MPB_HMC_SUPPORT.result_values(
        legacy, legacy.transition(legacy.snapshot, legacy.replay))
    b_result = _MPB_HMC_SUPPORT.result_values(
        profile_b, profile_b.transition(profile_b.snapshot, profile_b.replay))
    @test b_result == legacy_result
end
end
