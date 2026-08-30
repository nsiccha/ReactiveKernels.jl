using ReactiveKernels
using Reactant
using LinearAlgebra
using LogExpFunctions: log1pexp
using Random
using Test
import Enzyme
import LambertW
import TOML
import Reactant: @compile, @jit

module _ReactantStatefulFix
include(joinpath(@__DIR__, "..", "benchmark", "nuts_kernel_authoring_fixture.jl"))
end

include(joinpath(@__DIR__, "..", "benchmark",
                 "reactivehmc_rke_kernel_fixture.jl"))
include(joinpath(@__DIR__, "..", "benchmark",
                 "reactivehmc_rke_functional_lowering.jl"))
if !isdefined(@__MODULE__, :ReactiveHMCIntegratorFixture)
    include(joinpath(@__DIR__, "..", "benchmark",
                     "reactivehmc_integrator_kernel_fixture.jl"))
end

using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    NORMAL_LOGDENSITY, CAUCHY_LOGDENSITY,
    EXPONENTIAL_SOURCE, GEOMETRIC_SOURCE, UNIFORM_SOURCE,
    MVNORMAL_SOURCE, AR1_SOURCE
using ReactiveKernelsPPLExamples.EightSchoolsExample:
    EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA, evaluate_eight_schools_source

function _evaluate_distribution_kernel_source(source::AbstractString)
    sandbox = Module(gensym(:ReactantDistributionKernel), true, true)
    Core.eval(sandbox, :(using ReactiveKernels))
    parsed = Meta.parseall(source; filename = "reactant-distribution-kernel.jl")
    expressions = parsed.head === :toplevel ? parsed.args : Any[parsed]
    for expression in expressions
        expression isa LineNumberNode && continue
        Core.eval(sandbox, expression)
    end
    Core.eval(sandbox, :docs_example)
end

@kernel reactant_normal_logscale(
        x::Float64, μ::Float64 = 0.0, logσ::Float64 = 0.0) = begin
    σ::Float64 = exp(logσ)
    z::Float64 = (x - μ) / σ
    ld::Float64 = -0.5 * log(2π) - logσ - 0.5 * z^2
end

@kernel reactant_normal_scale(x::Float64, μ::Float64, σ::Float64) = begin
    logσ::Float64 = log(σ)
    z::Float64 = (x - μ) / σ
    ld::Float64 = -0.5 * log(2π) - logσ - 0.5 * z^2
end

@kernel reactant_bernoulli_logit(observed::Bool, logit::Float64) = begin
    ld::Float64 = observed ? -log1pexp(-logit) : -log1pexp(logit)
end

@kernel reactant_replica_transition(
        q::Vector{Float64}, p0::Vector{Float64}, u::Float64,
        stepsize::Float64) = begin
    proposal::Vector{Float64} = q .+ stepsize .* p0
    kinetic::Float64 = 0.5 * dot(p0, p0)
    accept::Bool = log(u) < -0.05 * kinetic
    q_next::Vector{Float64} = accept ? proposal : q
    return (q_next, accept, kinetic)
end

@kernel reactant_hmc_transition(
        grad_U::Function, pot::Function, q::Vector{Float64},
        p0::Vector{Float64}, u::Float64, stepsize::Float64, L::Int) = begin
    integrated::Tuple{Vector{Float64},Vector{Float64}} = let ε = stepsize
        qq = q
        pp = p0 .- (ε / 2) .* grad_U(q)
        for _ in 1:(L - 1)
            qq = qq .+ ε .* pp
            pp = pp .- ε .* grad_U(qq)
        end
        qq = qq .+ ε .* pp
        pp = pp .- (ε / 2) .* grad_U(qq)
        (qq, pp)
    end
    qL::Vector{Float64} = integrated[1]
    pL::Vector{Float64} = integrated[2]
    H0::Float64 = pot(q) + 0.5 * dot(p0, p0)
    HL::Float64 = pot(qL) + 0.5 * dot(pL, pL)
    accept::Bool = log(u) < (H0 - HL)
    q_next::Vector{Float64} = accept ? qL : q
end

@kernel reactant_integrator_endpoint(
        pos::Vector{Float64}, mom::Vector{Float64}) = begin
    dham_dpos::Vector{Float64} = 2 .* pos
    dham_dmom::Vector{Float64} = 3 .* mom
    ham::Float64 = sum(abs2, pos) + sum(abs2, mom)
end

@kernel reactant_cauchy_logscale(
        x::Float64, μ::Float64, logσ::Float64) = begin
    σ::Float64 = exp(logσ)
    z::Float64 = (x - μ) / σ
    ld::Float64 = -log(π) - logσ - log1p(z^2)
end

@kernel reactant_laplace_logscale(
        x::Float64, μ::Float64, logb::Float64) = begin
    b::Float64 = exp(logb)
    z::Float64 = (x - μ) / b
    ld::Float64 = -log(2) - logb - abs(z)
end

@kernel reactant_lognormal_logscale(
        x::Float64, μ::Float64, logσ::Float64) = begin
    ld::Float64 = x > 0 ? begin
        logx = log(x)
        z = (logx - μ) / exp(logσ)
        -0.5 * log(2π) - logσ - logx - 0.5 * z^2
    end : -Inf
end

_rk_call(k, x, μ, scale) = k(x, μ, scale)
_rk_allocated(k, x, μ, scale) = @allocated k(x, μ, scale)
_rk_output_allocated(x) = @allocated similar(x, Float64)

function _normal_reference(x, μ, logσ)
    -0.5 * log(2π) - logσ - 0.5 * ((x - μ) / exp(logσ))^2
end

const REACTANT_SOURCE_NORMAL_LOGSCALE = prepare(NORMAL_LOGDENSITY;
    have = (:x, :location, :log_scale), want = :logdensity)

@kernel reactant_embedded_source_normal(
        x::Float64, location::Float64, log_scale::Float64) = begin
    logdensity::Float64 =
        REACTANT_SOURCE_NORMAL_LOGSCALE(x, location, log_scale)
end

const REACTANT_NESTED_PLATE = plate(reactant_normal_logscale;
    have = (:x, :μ, :logσ), want = :ld, batched = :x)

@kernel reactant_nested_plate_middle(
        x::Vector{Float64}, μ::Float64, logσ::Float64) = begin
    total::Float64 = REACTANT_NESTED_PLATE(x, μ, logσ)
end

const REACTANT_NESTED_MIDDLE = prepare(reactant_nested_plate_middle;
    have = (:x, :μ, :logσ), want = :total)

@kernel reactant_nested_plate_outer(
        x::Vector{Float64}, μ::Float64, logσ::Float64) = begin
    total::Float64 = REACTANT_NESTED_MIDDLE(x, μ, logσ)
end

@testset "Reactant optional compiler integration" begin
    @test Base.get_extension(ReactiveKernels, :ReactiveKernelsReactantExt) !== nothing

    @testset "source-derived functional state transition" begin
        kernel = ReactiveKernels.compile_stateful(
            _ReactantStatefulFix.dual_averaging_state, 0.65)
        native_state = kernel(0.65)
        transition = ReactiveKernels._functionalize_stateful(kernel, Val(:fit!))
        host_state = ReactiveKernels._stateful_snapshot(native_state)
        traced_state = map(
            value -> Reactant.to_rarray(value; track_numbers = true), host_state)
        traced_x = Reactant.to_rarray(0.91; track_numbers = true)
        compiled = @compile transition(traced_state, traced_x)
        for x_host in (0.91, 0.31, 0.63, 0.79)
            traced_x = Reactant.to_rarray(x_host; track_numbers = true)
            actual = compiled(traced_state, traced_x)
            expected = transition(host_state, x_host)
            for name in propertynames(expected)
                @test getfield(actual, name) ≈ getfield(expected, name)
            end
            traced_state = actual
            host_state = expected
        end
    end

    @testset "source-derived RKE callable and sibling lowering" begin
        receipt = TOML.parsefile(joinpath(@__DIR__, "..", "benchmark",
            "receipts", "reactivehmc-rke-ca9-v1.toml"))
        bindings = ReactiveKernels.stateful_compiler_bindings(
            lambertw=ReactiveKernels.pure_callable_port(
                LambertW.lambertw, Tuple{Float64,Int}, Float64;
                functional_lowering=
                    ReactiveHMCRKEFunctionalLowering.lambertw_minus_one))

        for case in receipt["cases"]
            kernel = ReactiveKernels.compile_stateful(
                ReactiveHMCRKEFixture.rke, bindings, LambertW.lambertw;
                m=case["m"], c=case["c"])
            native_state = kernel(
                LambertW.lambertw; m=case["m"], c=case["c"])
            host_state = ReactiveKernels._stateful_snapshot(native_state)
            traced_state = map(host_state) do value
                value isa Number ?
                    Reactant.to_rarray(value; track_numbers=true) : value
            end

            p_sq = ReactiveKernels._functionalize_stateful(
                kernel, Val(:p_sq))
            traced_x = Reactant.to_rarray(case["x_sq"][3]; track_numbers=true)
            compiled_p_sq = @compile p_sq(traced_state, traced_x)
            @test compiled_p_sq(traced_state, traced_x) ≈ case["p_sq"][3]

            quantile = ReactiveKernels._functionalize_stateful(
                kernel, Val(:quantile_sq))
            traced_q = Reactant.to_rarray(case["q"][2]; track_numbers=true)
            compiled_quantile = @compile quantile(traced_state, traced_q)
            for (q, expected) in zip(case["q"], case["quantile_sq"])
                traced_q = Reactant.to_rarray(q; track_numbers=true)
                @test compiled_quantile(traced_state, traced_q) ≈ expected atol=128eps(Float64) rtol=0
            end
        end
    end


    @testset "generic static-loop state transition" begin
        transition = compile_state_transition(
            reactant_integrator_endpoint,
            partial(ReactiveHMCIntegratorFixture.generalized_leapfrog!;
                    stepsize=0.06, n_fi_steps=2),
            ([0.25, -0.5], [0.4, 0.1]),
        )
        host_state = initial_transition_state(transition)
        traced_state = map(host_state) do value
            Reactant.to_rarray(value; track_numbers=true)
        end
        compiled = @compile transition(traced_state)
        actual = compiled(traced_state)
        expected = transition(host_state)
        @test Array(actual.pos) ≈ expected.pos atol=2e-15 rtol=2e-13
        @test Array(actual.mom) ≈ expected.mom atol=2e-15 rtol=2e-13
        @test only(actual.ham) ≈ expected.ham atol=2e-15 rtol=2e-13
    end

    @testset "eight-schools PPL extraction and plate boundaries compile" begin
        artifact = evaluate_eight_schools_source()
        q_host = [1.5, log(2.0), (0.25 .* (1:8))...]
        μ = Reactant.to_rarray(q_host[1]; track_numbers = true)
        log_τ = Reactant.to_rarray(q_host[2]; track_numbers = true)
        effects = Reactant.to_rarray(q_host[3:end])
        observations = Reactant.to_rarray(EIGHT_SCHOOLS_Y)
        scales = Reactant.to_rarray(EIGHT_SCHOOLS_SIGMA)

        # This is the intended PPL -> RK -> Reactant path: the PPL policy becomes
        # one static named-node selection, then Reactant compiles that prepared
        # mathematical kernel without seeing accumulator objects or graph state.
        compiled = @compile artifact.kernel(
            μ, log_τ, effects, observations, scales)
        parameters, prior, likelihood = compiled(
            μ, log_τ, effects, observations, scales)
        reference_parameters, reference_prior, reference_likelihood = artifact.output
        @test parameters.μ ≈ reference_parameters.μ
        @test parameters.τ ≈ reference_parameters.τ
        @test Array(parameters.θ) ≈ reference_parameters.θ
        @test prior ≈ reference_prior
        @test likelihood ≈ reference_likelihood

        # The full posterior is another named-node selection over the same
        # graph. Supplying the named latent HAVE boundary avoids scalar indexing
        # into a traced packed vector without defining a second model path.
        posterior_kernel = prepare(artifact.model;
            have = (:μ, :log_τ, :θ, :observations, :observation_scales),
            want = :posterior)
        posterior_compiled = @compile posterior_kernel(
            μ, log_τ, effects, observations, scales)
        @test posterior_compiled(μ, log_τ, effects, observations, scales) ≈
              posterior_kernel(
                  q_host[1], q_host[2], q_host[3:end],
                  EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA)

        pointwise_compiled = @compile artifact.pointwise_kernel(
            observations, effects, scales)
        @test Array(pointwise_compiled(observations, effects, scales)) ≈
              artifact.pointwise

        likelihood_compiled = @compile artifact.likelihood_kernel(
            observations, effects, scales)
        @test likelihood_compiled(observations, effects, scales) ≈
              reference_likelihood
    end

    @testset "direct prepared scalar kernels keep program metadata static" begin
        # Defaulted named authoring returns the signature wrapper, so this also
        # covers direct compilation of the wrapper and its default providers.
        kernel = prepare(reactant_normal_logscale)
        x, μ, logσ = Reactant.to_rarray.((0.2, 0.3, log(1.2));
                                                track_numbers = true)
        compiled = @compile kernel(x, μ, logσ)
        @test compiled(x, μ, logσ) ≈ _normal_reference(0.2, 0.3, log(1.2))
    end

    @testset "reusable Normal and Cauchy sources compile and plate" begin
        x_host = 0.4
        location_host = -0.2
        scale_host = 1.3
        log_scale_host = log(scale_host)
        x, location, scale, log_scale = Reactant.to_rarray.(
            (x_host, location_host, scale_host, log_scale_host);
            track_numbers = true)

        for spec in (NORMAL_LOGDENSITY, CAUCHY_LOGDENSITY)
            scale_kernel = prepare(spec;
                have = (:x, :location, :scale), want = :logdensity)
            logscale_kernel = prepare(spec;
                have = (:x, :location, :log_scale), want = :logdensity)
            both_kernel = prepare(spec;
                have = (:x, :location, :scale, :log_scale),
                want = :logdensity)
            scale_compiled = @compile scale_kernel(x, location, scale)
            logscale_compiled = @compile logscale_kernel(x, location, log_scale)
            both_compiled = @compile both_kernel(
                x, location, scale, log_scale)
            reference = scale_kernel(x_host, location_host, scale_host)
            @test scale_compiled(x, location, scale) ≈ reference
            @test logscale_compiled(x, location, log_scale) ≈ reference
            @test both_compiled(x, location, scale, log_scale) ≈ reference
        end

        observations_host = [28.0, 8.0, -3.0, 7.0]
        effects_host = [1.0, 1.5, -0.5, 0.25]
        scales_host = [15.0, 10.0, 16.0, 11.0]
        likelihood = plate(NORMAL_LOGDENSITY;
            have = (:x, :location, :scale),
            want = :logdensity, batched = (:x, :location, :scale))
        observations = Reactant.to_rarray(observations_host)
        effects = Reactant.to_rarray(effects_host)
        scales = Reactant.to_rarray(scales_host)
        likelihood_compiled = @compile likelihood(observations, effects, scales)
        @test likelihood_compiled(observations, effects, scales) ≈
              likelihood(observations_host, effects_host, scales_host)

        embedded = prepare(reactant_embedded_source_normal)
        embedded_compiled = @compile embedded(x, location, log_scale)
        @test embedded_compiled(x, location, log_scale) ≈
              REACTANT_SOURCE_NORMAL_LOGSCALE(
                  x_host, location_host, log_scale_host)
    end

    @testset "dynamic Bool selection is tensorized without changing native semantics" begin
        kernel = prepare(reactant_bernoulli_logit)
        observed = Reactant.to_rarray(true; track_numbers = true)
        logit = Reactant.to_rarray(0.7; track_numbers = true)
        compiled = @compile kernel(observed, logit)
        @test compiled(observed, logit) ≈ -log1pexp(-0.7)
        observed_false = Reactant.to_rarray(false; track_numbers = true)
        @test compiled(observed_false, logit) ≈ -log1pexp(0.7)
        @test kernel(true, 0.7) == -log1pexp(-0.7)
        @test kernel(false, 0.7) == -log1pexp(0.7)
    end

    @testset "whole scalar kernel maps over a trailing replica axis" begin
        kernel = replica(
            reactant_replica_transition; batched = (:q, :p0, :u))
        q_host = reshape(collect(1.0:12.0), 3, 4)
        p0_host = [0.1 0.5 0.2 0.8;
                   0.2 0.4 0.3 0.7;
                   0.3 0.3 0.4 0.6]
        u_host = [0.1, 0.99, 0.2, 0.95]
        stepsize_host = 0.25
        reference = kernel(q_host, p0_host, u_host, stepsize_host)

        q = Reactant.to_rarray(q_host)
        p0 = Reactant.to_rarray(p0_host)
        u = Reactant.to_rarray(u_host)
        stepsize = Reactant.to_rarray(stepsize_host; track_numbers = true)
        compiled = @compile kernel(q, p0, u, stepsize)
        q_next, accept, kinetic = compiled(q, p0, u, stepsize)

        @test Array(q_next) == reference[1]
        @test Array(accept) == reference[2]
        @test Array(kinetic) ≈ reference[3]
    end

    @testset "scalar-source HMC compiles once and replicas without a rewrite" begin
        inverse_mass = [2.0 0.2 0.1;
                        0.2 1.5 0.3;
                        0.1 0.3 1.0]
        grad_U = q -> inverse_mass * q
        potential = q -> 0.5 * dot(q, inverse_mass * q)
        q_host = reshape(collect(0.1:0.1:0.9), 3, 3)
        p0_host = reverse(q_host; dims = 1)
        u_host = [0.2, 0.8, 0.4]
        stepsize_host = 0.05
        leapfrog_steps = 3

        scalar = prepare(reactant_hmc_transition)
        q1 = Reactant.to_rarray(q_host[:, 1])
        p1 = Reactant.to_rarray(p0_host[:, 1])
        u1 = Reactant.to_rarray(u_host[1]; track_numbers = true)
        stepsize = Reactant.to_rarray(stepsize_host; track_numbers = true)
        scalar_compiled = @compile scalar(
            grad_U, potential, q1, p1, u1, stepsize, leapfrog_steps)
        scalar_reference = scalar(
            grad_U, potential, q_host[:, 1], p0_host[:, 1], u_host[1],
            stepsize_host, leapfrog_steps)
        @test Array(scalar_compiled(
            grad_U, potential, q1, p1, u1, stepsize, leapfrog_steps)) ≈
              scalar_reference

        replicated = replica(
            reactant_hmc_transition; batched = (:q, :p0, :u))
        reference = replicated(
            grad_U, potential, q_host, p0_host, u_host,
            stepsize_host, leapfrog_steps)
        q = Reactant.to_rarray(q_host)
        p0 = Reactant.to_rarray(p0_host)
        u = Reactant.to_rarray(u_host)
        compiled = @compile replicated(
            grad_U, potential, q, p0, u, stepsize, leapfrog_steps)
        @test Array(compiled(
            grad_U, potential, q, p0, u, stepsize, leapfrog_steps)) ≈ reference
    end

    @testset "distribution gallery operations and support selection" begin
        cauchy = prepare(reactant_cauchy_logscale)
        laplace = prepare(reactant_laplace_logscale)
        lognormal = prepare(reactant_lognormal_logscale)
        x, μ, logscale = Reactant.to_rarray.((1.4, 0.2, log(0.9));
                                                    track_numbers = true)

        cauchy_compiled = @compile cauchy(x, μ, logscale)
        z = (1.4 - 0.2) / 0.9
        @test cauchy_compiled(x, μ, logscale) ≈
              -log(π) - log(0.9) - log1p(z^2)

        laplace_compiled = @compile laplace(x, μ, logscale)
        @test laplace_compiled(x, μ, logscale) ≈
              -log(2) - log(0.9) - abs(z)

        lognormal_compiled = @compile lognormal(x, μ, logscale)
        logx = log(1.4)
        lognormal_reference = -0.5 * log(2π) - log(0.9) - logx -
                              0.5 * ((logx - 0.2) / 0.9)^2
        @test lognormal_compiled(x, μ, logscale) ≈ lognormal_reference
        unsupported_x = Reactant.to_rarray(-1.0; track_numbers = true)
        @test lognormal_compiled(unsupported_x, μ, logscale) == -Inf
    end

    @testset "exact scalar gallery sources compile directly and through plate" begin
        exponential = _evaluate_distribution_kernel_source(EXPONENTIAL_SOURCE)
        ex_x_host, ex_logθ_host = Tuple(exponential.inputs)
        ex_x, ex_logθ = Reactant.to_rarray.(
            (ex_x_host, ex_logθ_host); track_numbers = true)
        ex_compiled = @compile exponential.kernel(ex_x, ex_logθ)
        @test ex_compiled(ex_x, ex_logθ) ≈ exponential.output
        ex_bad_x = Reactant.to_rarray(-1.0; track_numbers = true)
        @test ex_compiled(ex_bad_x, ex_logθ) == -Inf
        ex_plate_x_host, _ = Tuple(exponential.plate_inputs)
        ex_plate_x = Reactant.to_rarray(ex_plate_x_host)
        ex_plate_compiled = @compile exponential.plated(ex_plate_x, ex_logθ)
        @test ex_plate_compiled(ex_plate_x, ex_logθ) ≈ exponential.plate_output

        geometric = _evaluate_distribution_kernel_source(GEOMETRIC_SOURCE)
        geo_observed_host, geo_logitp_host = Tuple(geometric.inputs)
        geo_observed = Reactant.to_rarray(geo_observed_host; track_numbers = true)
        geo_logitp = Reactant.to_rarray(geo_logitp_host; track_numbers = true)
        geo_compiled = @compile geometric.kernel(geo_observed, geo_logitp)
        @test geo_compiled(geo_observed, geo_logitp) ≈ geometric.output
        geo_bad = Reactant.to_rarray(-1; track_numbers = true)
        @test geo_compiled(geo_bad, geo_logitp) == -Inf
        geo_plate_host, _ = Tuple(geometric.plate_inputs)
        geo_plate = Reactant.to_rarray(geo_plate_host)
        geo_plate_compiled = @compile geometric.plated(geo_plate, geo_logitp)
        @test geo_plate_compiled(geo_plate, geo_logitp) ≈ geometric.plate_output

        uniform = _evaluate_distribution_kernel_source(UNIFORM_SOURCE)
        uni_x_host, uni_lower_host, uni_upper_host = Tuple(uniform.inputs)
        uni_x, uni_lower, uni_upper = Reactant.to_rarray.(
            (uni_x_host, uni_lower_host, uni_upper_host); track_numbers = true)
        uni_compiled = @compile uniform.kernel(uni_x, uni_lower, uni_upper)
        @test uni_compiled(uni_x, uni_lower, uni_upper) ≈ uniform.output
        uni_bad_x = Reactant.to_rarray(3.0; track_numbers = true)
        @test uni_compiled(uni_bad_x, uni_lower, uni_upper) == -Inf
        uni_plate_host, _, _ = Tuple(uniform.plate_inputs)
        uni_plate = Reactant.to_rarray(uni_plate_host)
        uni_plate_compiled = @compile uniform.plated(
            uni_plate, uni_lower, uni_upper)
        @test uni_plate_compiled(
            uni_plate, uni_lower, uni_upper) ≈ uniform.plate_output
    end

    @testset "non-scalar MVN and AR(1) kernels compile and replica-map" begin
        mvnormal = _evaluate_distribution_kernel_source(MVNORMAL_SOURCE)
        mvn_replica_x_host = mvnormal.replica_inputs.x
        mvn_replica_x = Reactant.to_rarray(mvn_replica_x_host)
        for name in propertynames(mvnormal.kernels)
            host_inputs = Tuple(getproperty(mvnormal.parametrization_inputs, name))
            mvn_x = Reactant.to_rarray(host_inputs[1])
            mvn_μ = Reactant.to_rarray(host_inputs[2])
            representation = Reactant.to_rarray(host_inputs[3])

            mvn_kernel = getproperty(mvnormal.kernels, name)
            mvn_compiled = @compile mvn_kernel(mvn_x, mvn_μ, representation)
            @test mvn_compiled(mvn_x, mvn_μ, representation) ≈
                  getproperty(mvnormal.parametrization_outputs, name)

            mvn_replicated = getproperty(mvnormal.replicated_kernels, name)
            mvn_replica_compiled = @compile mvn_replicated(
                mvn_replica_x, mvn_μ, representation)
            expected = mvn_replicated(
                mvn_replica_x_host, host_inputs[2], host_inputs[3])
            @test Array(mvn_replica_compiled(
                mvn_replica_x, mvn_μ, representation)) ≈ expected
        end

        ar1 = _evaluate_distribution_kernel_source(AR1_SOURCE)
        ar_x_host, ar_μ_host, ar_ϕ_host, ar_logσ_host = Tuple(ar1.inputs)
        ar_x = Reactant.to_rarray(ar_x_host)
        ar_μ, ar_ϕ, ar_logσ = Reactant.to_rarray.(
            (ar_μ_host, ar_ϕ_host, ar_logσ_host); track_numbers = true)
        ar1_kernel = ar1.kernel
        ar1_compiled = @compile ar1_kernel(ar_x, ar_μ, ar_ϕ, ar_logσ)
        @test ar1_compiled(ar_x, ar_μ, ar_ϕ, ar_logσ) ≈ ar1.output

        ar_replica_x_host, _, _, _ = Tuple(ar1.replica_inputs)
        ar_replica_x = Reactant.to_rarray(ar_replica_x_host)
        ar1_replicated = ar1.replicated
        ar1_replica_compiled = @compile ar1_replicated(
            ar_replica_x, ar_μ, ar_ϕ, ar_logσ)
        @test Array(ar1_replica_compiled(
            ar_replica_x, ar_μ, ar_ϕ, ar_logσ)) ≈ ar1.replica_output
    end

    logscale_plan = plan(reactant_normal_logscale;
                         have = (:x, :μ, :logσ), want = :ld)

    @testset "plate tensor body: fixed and traced shared parameters" begin
        kernel = ReactiveKernels._prepare_batched(
            logscale_plan; batched = (:x,), reduce = :+)
        x_host = collect(range(-1.4, 1.3; length = 16))
        x = Reactant.to_rarray(x_host)

        fixed_μ = 0.3
        fixed_logσ = log(1.2)
        fixed_compiled = @compile kernel(x, fixed_μ, fixed_logσ)
        fixed_reference = sum(_normal_reference(xi, fixed_μ, fixed_logσ)
                              for xi in x_host)
        @test fixed_compiled(x, fixed_μ, fixed_logσ) ≈ fixed_reference

        traced_μ = Reactant.to_rarray(fixed_μ; track_numbers = true)
        traced_logσ = Reactant.to_rarray(fixed_logσ; track_numbers = true)
        traced_compiled = @compile kernel(x, traced_μ, traced_logσ)
        @test traced_compiled(x, traced_μ, traced_logσ) ≈ fixed_reference
    end

    @testset "two-level prepared composition retains the tensorized plate" begin
        outer = prepare(reactant_nested_plate_outer;
            have = (:x, :μ, :logσ), want = :total)
        x_host = collect(range(-1.4, 1.3; length = 16))
        μ_host = 0.3
        logσ_host = log(1.2)
        reference = sum(_normal_reference(xi, μ_host, logσ_host)
                        for xi in x_host)
        @test outer(x_host, μ_host, logσ_host) ≈ reference
        @test outer.f isa ReactiveKernels._EmbeddedFunctionPair
        @test !occursin("for ", string(outer.f.tensorized_ast))

        x = Reactant.to_rarray(x_host)
        μ = Reactant.to_rarray(μ_host; track_numbers = true)
        logσ = Reactant.to_rarray(logσ_host; track_numbers = true)
        compiled = @compile outer(x, μ, logσ)
        @test compiled(x, μ, logσ) ≈ reference
    end

    @testset "plate tensor body: multiple batched inputs and collection" begin
        x_host = collect(range(-1.0, 1.0; length = 12))
        μ_host = collect(range(0.4, -0.5; length = 12))
        logσ_host = log(0.9)
        x = Reactant.to_rarray(x_host)
        μ = Reactant.to_rarray(μ_host)
        logσ = Reactant.to_rarray(logσ_host; track_numbers = true)
        reference = _normal_reference.(x_host, μ_host, logσ_host)

        reduced = ReactiveKernels._prepare_batched(
            logscale_plan; batched = (:x, :μ), reduce = :+)
        reduced_compiled = @compile reduced(x, μ, logσ)
        @test reduced_compiled(x, μ, logσ) ≈ sum(reference)

        collected = ReactiveKernels._prepare_batched(
            logscale_plan; batched = (:x, :μ), reduce = nothing)
        collected_compiled = @compile collected(x, μ, logσ)
        @test Array(collected_compiled(x, μ, logσ)) ≈ reference
    end

    @testset "native plate path keeps its allocation contract" begin
        x = collect(range(-2.0, 2.0; length = 256))
        μ = 0.3
        logσ = log(1.2)
        reduced = ReactiveKernels._prepare_batched(
            logscale_plan; batched = (:x,), reduce = :+)
        collected = ReactiveKernels._prepare_batched(
            logscale_plan; batched = (:x,), reduce = nothing)
        _rk_call(reduced, x, μ, logσ)
        _rk_call(collected, x, μ, logσ)
        _rk_output_allocated(x)
        @test _rk_allocated(reduced, x, μ, logσ) == 0
        @test _rk_allocated(collected, x, μ, logσ) ==
              _rk_output_allocated(x)
    end

    @testset "Enzyme gradient through the compiled reduced Normal" begin
        scale_plan = plan(reactant_normal_scale;
                          have = (:x, :μ, :σ), want = :ld)
        kernel = ReactiveKernels._prepare_batched(
            scale_plan; batched = (:x,), reduce = :+)
        x_host = collect(range(-1.25, 1.5; length = 10))
        x = Reactant.to_rarray(x_host)
        μ_host = 0.3
        σ_host = 1.2
        μ = Reactant.to_rarray(μ_host; track_numbers = true)
        σ = Reactant.to_rarray(σ_host; track_numbers = true)

        gradient = @jit Enzyme.gradient(
            Enzyme.Reverse, kernel, Enzyme.Const(x), μ, σ)
        reference_μ = sum((x_host .- μ_host) ./ σ_host^2)
        reference_σ = sum(-1 / σ_host .+
                          (x_host .- μ_host).^2 ./ σ_host^3)
        @test gradient[1] === nothing
        @test gradient[2] ≈ reference_μ
        @test gradient[3] ≈ reference_σ
    end
end

include("test_nutpie_reactant.jl")
include("test_reactivehmc_statistics_reactant.jl")
include("test_reactivehmc_hmc_reactant.jl")
include("test_kernel_nuts_reactant.jl")
include("test_pathfinder_reactant.jl")
