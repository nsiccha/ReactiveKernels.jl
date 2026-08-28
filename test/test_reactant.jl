using ReactiveKernels
using Reactant
using LogExpFunctions: log1pexp
using Random
using Test
import Enzyme
import Reactant: @compile, @jit

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
