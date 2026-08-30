using ReactiveKernels
using Test
import LambertW
import TOML

isdefined(@__MODULE__, :ReactiveHMCRKEFixture) ||
    include(joinpath(@__DIR__, "..", "benchmark",
                     "reactivehmc_rke_kernel_fixture.jl"))
isdefined(@__MODULE__, :ReactiveHMCRKEFunctionalLowering) ||
    include(joinpath(@__DIR__, "..", "benchmark",
                     "reactivehmc_rke_functional_lowering.jl"))

const _RKEC = ReactiveKernels
const _RKE_FIXTURE = ReactiveHMCRKEFixture
const _RKE_FUNCTIONAL = ReactiveHMCRKEFunctionalLowering

_rke_bindings(callable=LambertW.lambertw;
              arguments=Tuple{Float64,Int}, result=Float64,
              functional_lowering=callable) =
    _RKEC.stateful_compiler_bindings(
        lambertw=_RKEC.pure_callable_port(callable, arguments, result;
            functional_lowering))

@testset "ReactiveHMC RKE generic stateful compiler" begin
    receipt = TOML.parsefile(joinpath(@__DIR__, "..", "benchmark", "receipts",
                                      "reactivehmc-rke-ca9-v1.toml"))
    for case in receipt["cases"]
        bindings = _rke_bindings()
        kernel = _RKEC.compile_stateful(_RKE_FIXTURE.rke, bindings,
            LambertW.lambertw; m=case["m"], c=case["c"])
        state = kernel(LambertW.lambertw; m=case["m"], c=case["c"])
        @test _RKEC.stateful_get(state, Val(:c1)) ≈ case["c1"]
        @test _RKEC.stateful_get(state, Val(:c2)) ≈ case["c2"]
        @test _RKEC.stateful_get(state, Val(:P0_sq)) ≈ case["P0_sq"]

        for (method, key, inputs) in (
                (:e_sq, "e_sq", case["x_sq"]),
                (:p_sq, "p_sq", case["x_sq"]),
                (:cdf_sq, "cdf_sq", case["x_sq"]),
                (:quantile_sq, "quantile_sq", case["q"]))
            actual = [_RKEC.stateful_call(state, Val(method), input)
                      for input in inputs]
            @test actual ≈ case[key]
        end
        quantiles = [_RKEC.stateful_call(state, Val(:quantile_sq), q)
                     for q in case["q"]]
        roundtrip = [_RKEC.stateful_call(state, Val(:cdf_sq), x)
                     for x in quantiles]
        @test roundtrip ≈ case["roundtrip_cdf"]

        functional_bindings = _rke_bindings(;
            functional_lowering=_RKE_FUNCTIONAL.lambertw_minus_one)
        functional_kernel = _RKEC.compile_stateful(
            _RKE_FIXTURE.rke, functional_bindings, LambertW.lambertw;
            m=case["m"], c=case["c"])
        functional_state = functional_kernel(
            LambertW.lambertw; m=case["m"], c=case["c"])
        functional_quantile = _RKEC._functionalize_stateful(
            functional_kernel, Val(:quantile_sq);
            argument_types=Tuple{Float64})
        snapshot = _RKEC._stateful_snapshot(functional_state)
        @test [functional_quantile(snapshot, q) for q in case["q"]] ≈
              case["quantile_sq"] atol=128eps(Float64) rtol=0
    end

    for x in (-exp(-1.0), -0.35, -0.25, -0.1, -0.01, -1e-6)
        @test _RKE_FUNCTIONAL.lambertw_minus_one(x, -1) ≈
              LambertW.lambertw(x, -1) atol=128eps(Float64) rtol=0
    end
    @test_throws ArgumentError _RKE_FUNCTIONAL.lambertw_minus_one(-0.1, 0)

    # A runtime callable is never silently assumed pure. The exact field set,
    # callable identity, argument tuple, and result contract are all fail-closed.
    @test_throws _RKEC._KernelFactoryReject _RKEC.compile_stateful(
        _RKE_FIXTURE.rke, LambertW.lambertw)
    @test_throws _RKEC._KernelFactoryReject _RKEC.compile_stateful(
        _RKE_FIXTURE.rke,
        _RKEC.stateful_compiler_bindings(lambertw=:opaque),
        LambertW.lambertw)
    @test_throws ArgumentError _RKEC.compile_stateful(
        _RKE_FIXTURE.rke,
        _RKEC.stateful_compiler_bindings(
            lambertw=_RKEC.pure_callable_port(
                LambertW.lambertw, Tuple{Float64,Int}, Float64),
            typo=nothing),
        LambertW.lambertw)

    identity_bound = _RKEC.compile_stateful(
        _RKE_FIXTURE.rke, _rke_bindings(), LambertW.lambertw)
    @test_throws ArgumentError identity_bound((x, branch) -> x)

    wrong_result_callable = (x, branch) -> 1
    wrong_result = _RKEC.compile_stateful(
        _RKE_FIXTURE.rke, _rke_bindings(wrong_result_callable),
        wrong_result_callable)
    @test_throws ArgumentError _RKEC.stateful_call(
        wrong_result(wrong_result_callable), Val(:quantile_sq), 0.5)

    wrong_domain = _RKEC.compile_stateful(
        _RKE_FIXTURE.rke,
        _rke_bindings(; arguments=Tuple{Float32,Int}),
        LambertW.lambertw)
    @test_throws ArgumentError _RKEC.stateful_call(
        wrong_domain(LambertW.lambertw), Val(:quantile_sq), 0.5)
end
