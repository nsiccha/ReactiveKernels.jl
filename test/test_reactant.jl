using ReactiveKernels
using Reactant
using LinearAlgebra
using LogExpFunctions: log1pexp
using Random
using Test
import Enzyme
import Reactant: @compile, @jit

include(joinpath(@__DIR__, "..", "examples", "distribution_kernel_sources.jl"))
using .DistributionKernelSources: MVNORMAL_SOURCE, AR1_SOURCE

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

@testset "Reactant optional compiler integration" begin
    @test Base.get_extension(ReactiveKernels, :ReactiveKernelsReactantExt) !== nothing

    @testset "direct prepared scalar kernels keep program metadata static" begin
        # Defaulted named authoring returns the signature wrapper, so this also
        # covers direct compilation of the wrapper and its default providers.
        kernel = prepare(reactant_normal_logscale)
        x, μ, logσ = Reactant.to_rarray.((0.2, 0.3, log(1.2));
                                                track_numbers = true)
        compiled = @compile kernel(x, μ, logσ)
        @test compiled(x, μ, logσ) ≈ _normal_reference(0.2, 0.3, log(1.2))
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

    @testset "non-scalar MVN and AR(1) kernels compile and replica-map" begin
        mvnormal = _evaluate_distribution_kernel_source(MVNORMAL_SOURCE)
        mvn_x_host, mvn_μ_host, chol_host = Tuple(mvnormal.inputs)
        mvn_x = Reactant.to_rarray(mvn_x_host)
        mvn_μ = Reactant.to_rarray(mvn_μ_host)
        chol = Reactant.to_rarray(chol_host)
        mvn_kernel = mvnormal.kernel
        mvn_compiled = @compile mvn_kernel(mvn_x, mvn_μ, chol)
        @test mvn_compiled(mvn_x, mvn_μ, chol) ≈ mvnormal.output

        mvn_replica_x_host, _, _ = Tuple(mvnormal.replica_inputs)
        mvn_replica_x = Reactant.to_rarray(mvn_replica_x_host)
        mvn_replicated = mvnormal.replicated
        mvn_replica_compiled = @compile mvn_replicated(
            mvn_replica_x, mvn_μ, chol)
        @test Array(mvn_replica_compiled(mvn_replica_x, mvn_μ, chol)) ≈
              mvnormal.replica_output

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
