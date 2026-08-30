using LinearAlgebra
using ReactiveKernels
using Test
import TOML

if !isdefined(@__MODULE__, :ReactiveHMCExamples)
    include(joinpath(@__DIR__, "..", "examples", "preexisting_reactivehmc.jl"))
end
include(joinpath(@__DIR__, "..", "benchmark",
                 "reactivehmc_integrator_kernel_fixture.jl"))
include(joinpath(@__DIR__, "..", "benchmark", "receipts",
                 "validate_reactivehmc_integrators.jl"))

const RHMC_INTEGRATORS = ReactiveHMCIntegratorFixture

module _IntegratorCompilerEndpoint
using ReactiveKernels

@kernel endpoint(pos::Vector{Float64}, mom::Vector{Float64}) = begin
    dham_dpos::Vector{Float64} = 2 .* pos
    dham_dmom::Vector{Float64} = 3 .* mom
    ham::Float64 = sum(abs2, pos) + sum(abs2, mom)
end

end

function _integrator_observables(point, kernels)
    (; pos=point.pos, mom=point.mom, ham=point.ham,
       dham_dpos=kernels.dham_dpos(point.geometry, point.mom),
       dham_dmom=kernels.dham_dmom(point.geometry, point.mom))
end

@testset "generic functional integrator compiler" begin
    pos = [0.25, -0.5]
    mom = [0.4, 0.1]
    stepsize = 0.06
    n_fi_steps = 2
    kernels = (;
        geometry = p -> (; pos=p),
        dham_dpos = (geometry, _) -> 2 .* geometry.pos,
        dham_dmom = (_, p) -> 3 .* p,
        hamiltonian = (geometry, p) ->
            sum(abs2, geometry.pos) + sum(abs2, p),
    )

    for (source, oracle) in (
            RHMC_INTEGRATORS.generalized_leapfrog! =>
                ReactiveHMCExamples.generalized_leapfrog!,
            RHMC_INTEGRATORS.implicit_midpoint! =>
                ReactiveHMCExamples.implicit_midpoint!,
        )
        transition = compile_state_transition(
            _IntegratorCompilerEndpoint.endpoint,
            partial(source; stepsize, n_fi_steps),
            (pos, mom),
        )
        state = initial_transition_state(transition)
        independent = initial_transition_state(transition)
        @test propertynames(state) ==
              (:pos, :mom, :dham_dpos, :dham_dmom, :ham)
        @test state.pos !== independent.pos
        @test state.mom !== independent.mom
        actual = transition(state)
        expected = oracle(
            copy(pos), copy(mom), kernels; stepsize, n_fi_steps)
        @test actual.pos ≈ expected.pos atol=2e-15 rtol=2e-13
        @test actual.mom ≈ expected.mom atol=2e-15 rtol=2e-13
        @test actual.ham ≈ expected.ham atol=2e-15 rtol=2e-13
        @test pos == [0.25, -0.5]
        @test mom == [0.4, 0.1]
    end

    @test_throws ReactiveKernels._KernelFactoryReject compile_state_transition(
        _IntegratorCompilerEndpoint.endpoint,
        RHMC_INTEGRATORS.generalized_leapfrog!,
        (pos, mom),
    )
    @test_throws ReactiveKernels._LLowerReject compile_state_transition(
        _IntegratorCompilerEndpoint.endpoint,
        partial(RHMC_INTEGRATORS.generalized_leapfrog!;
                stepsize, n_fi_steps=1025),
        (pos, mom),
    )
end

@testset "ReactiveHMC integrator source and independent receipt" begin
    generalized = ReactiveKernels.kernel_registration(
        RHMC_INTEGRATORS.generalized_leapfrog!)
    midpoint = ReactiveKernels.kernel_registration(
        RHMC_INTEGRATORS.implicit_midpoint!)
    @test (generalized.kind, generalized.subject, generalized.write_roots) ==
          (:free_method, :phasepoint, (:mom, :pos))
    @test (midpoint.kind, midpoint.subject, midpoint.write_roots) ==
          (:free_method, :phasepoint, (:pos, :mom))
    @test ReactiveKernels.kernel_port_names(
        RHMC_INTEGRATORS.generalized_leapfrog!) ==
        (:phasepoint, :stepsize, :n_fi_steps)
    @test ReactiveKernels.kernel_port_names(
        RHMC_INTEGRATORS.implicit_midpoint!) ==
        (:phasepoint, :stepsize, :n_fi_steps)

    generalized_ir = only(ReactiveKernels.method_irs(
        RHMC_INTEGRATORS.generalized_leapfrog!))
    copied = generalized_ir.body[1]
    @test copied isa ReactiveKernels._LocalAssign
    @test copied.lhs == (:pos0, :mom0) && copied.style === :tuple
    @test copied.rhs isa ReactiveKernels._RegisteredCall
    @test copied.rhs.registration.source === Base.map
    @test copied.rhs.args[1] isa ReactiveKernels._CallableRef
    @test copied.rhs.args[1].registration.source === Base.copy
    @test copied.rhs.args[2] isa ReactiveKernels._TupleExpr

    midpoint_ir = only(ReactiveKernels.method_irs(
        RHMC_INTEGRATORS.implicit_midpoint!))
    midpoint_loop = only(statement for statement in midpoint_ir.body
                         if statement isa ReactiveKernels._For)
    named = midpoint_loop.body[1]
    @test named isa ReactiveKernels._LocalAssign
    @test named.lhs == (:dham_dmom, :dham_dpos)
    @test named.style === :named && named.rhs isa ReactiveKernels._SelfRef

    source_path = joinpath(@__DIR__, "..", "benchmark",
                           "reactivehmc_integrator_kernel_fixture.jl")
    source = read(source_path, String)
    @test occursin("pos0, mom0 = map(copy, (phasepoint.pos, phasepoint.mom))", source)
    @test occursin("(; dham_dmom, dham_dpos) = phasepoint", source)
    @test occursin("f(args...; stepsize=stepsize / n_steps, kwargs...)", source)

    receipt_path = joinpath(@__DIR__, "..", "benchmark", "receipts",
                            "reactivehmc-integrators-ca9-v1.toml")
    @test isempty(validate_reactivehmc_integrator_receipt(receipt_path))
    receipt = TOML.parsefile(receipt_path)
    source_digests = Dict(ReactiveHMCAlgorithmCorpus.UPSTREAM.source_sha256)
    @test receipt["pins"]["reactivehmc_revision"] ==
          ReactiveHMCAlgorithmCorpus.UPSTREAM.revision
    @test receipt["pins"]["integrators_sha256"] ==
          source_digests["src/integrators.jl"]
    @test receipt["pins"]["phasepoints_sha256"] ==
          source_digests["src/phasepoints.jl"]

    euclidean = ReactiveHMCExamples.euclidean_examples()
    riemannian = ReactiveHMCExamples.riemannian_examples()
    multistep = ReactiveHMCExamples.multistep!(
        ReactiveHMCExamples.generalized_leapfrog!, [0.25, -0.5], [0.4, 0.1],
        riemannian.gaussian;
        stepsize=0.06, n_fi_steps=2, n_steps=3)
    actual = Dict(
        "leapfrog" => _integrator_observables(
            euclidean.euclidean_phasepoint, euclidean.gaussian),
        "generalized_leapfrog" =>
            _integrator_observables(
                riemannian.generalized_leapfrog, riemannian.gaussian),
        "implicit_midpoint" =>
            _integrator_observables(
                riemannian.implicit_midpoint, riemannian.gaussian),
        "multistep" => _integrator_observables(multistep, riemannian.gaussian),
    )
    for expected in receipt["cases"]
        observed = actual[expected["name"]]
        for field in (:pos, :mom, :ham, :dham_dpos, :dham_dmom)
            @test isapprox(getproperty(observed, field), expected[string(field)];
                           rtol=2e-13, atol=2e-15)
        end
    end
end
