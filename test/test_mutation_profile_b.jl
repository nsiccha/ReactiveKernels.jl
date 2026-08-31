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

const _MPB_NUTS = _MutationProfileBFixtures.NUTSBMutationAuthoringFixture
const _MPB_WALNUTS =
    _MutationProfileBFixtures.WALNUTSBMutationAuthoringFixture
const _MPB_HMC_SUPPORT = ReactiveHMCHMCCompilerSupport
const _MPB_STATISTICS =
    _MutationProfileBFixtures.ReactiveHMCStatisticsFixture
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
