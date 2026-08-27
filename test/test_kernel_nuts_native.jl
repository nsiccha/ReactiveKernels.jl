using Test, ReactiveKernels
const RK = ReactiveKernels

module _NativeNutsFix
    include(joinpath(@__DIR__, "..", "benchmark", "nuts_kernel_authoring_fixture.jl"))
end

@testset "native NUTS — registry-free total type-tree encoder" begin
    irs = RK.method_irs(_NativeNutsFix.nuts_state)
    E = RK._native_encode_program(irs, Nothing, RK.kernel_token(_NativeNutsFix.nuts_state))
    @test E.program <: RK._NativeProgram
    @test RK._native_program_node_count(E.program) == 444
    @test Tuple(ir.id.decl for ir in irs) == (1,2,3,4,5,6,7,8,9)
    @test Tuple(ir.id.name for ir in irs) ==
          (:reset!, :collectstats!, :logadvanceprob, :swapproposal!, :step!, :flip!, :flip_neg!, :finish!, :start!)
    @test length(E.refs) == length(E.callees) == length(E.registrations)
    @test length(E.refs) > 10
    @test all(r -> r isa RK._CapturedCalleeRef, E.refs)
    @test !any(v -> v isa Dict || v isa Set || v isa Base.RefValue, E.callees)
    # Re-encoding one definition is deterministic and yields the identical program/callee tuple types.
    E2 = RK._native_encode_program(RK.method_irs(_NativeNutsFix.nuts_state), Nothing,
                                   RK.kernel_token(_NativeNutsFix.nuts_state))
    @test E2.program === E.program
    @test typeof(E2.callees) === typeof(E.callees)
end

@testset "native NUTS — totality rejects unsupported/ambiguous authority" begin
    b = RK._NativeCalleeBuilder()
    @test_throws RK._NativeEncodeReject RK._native_encode(
        RK._OpCall(GlobalRef(Base, :identity), (RK._Lit(1),), (), false), b)
    @test_throws RK._NativeEncodeReject RK._native_encode(
        RK._NodeExpr(RK._Lit(1), :(1), @__MODULE__), b)
    @test_throws RK._NativeEncodeReject RK._native_encode(
        RK._ExtRef(GlobalRef(@__MODULE__, :not_structural)), b)
end
