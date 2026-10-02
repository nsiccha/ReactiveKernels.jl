using DifferentiationInterface
using Distributions
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Test

module ArrayDefinitionGatherModels
passthrough(x) = x
firstcolumn(x) = x[:, 1]
end

_adg_definition(::Val{:linear}) = (:(b = (z * sd)'), :(b[g]))
_adg_definition(::Val{:linear_matrix}) = (:(b = z .* sd'), :(b[g]))
_adg_definition(::Val{:opaque}) = (:(b = passthrough(z)), :(b[g, 1]))
_adg_definition(::Val{:opaque_vector}) = (:(b = firstcolumn(z)), :(b[g]))
_adg_definition(::Val{:opaque_rows}) = (:(b = passthrough(z)), :(b[g, :] * sd))

function _adg_plan(kind)
    definition, read = _adg_definition(Val(kind))
    return lower_rkppl(quote
        a ~ Normal(0, 5)
        sd[1:2] .~ Normal.(0, 1)
        z[levels(h), 1:2] .~ Normal.(0, 1)
        $definition
        mu = a .+ $read
        y .~ Normal.(mu, 0.7)
    end, (:y, :g, :h); mod = ArrayDefinitionGatherModels, conditioned = (:y, :g, :h))
end

function _adg_columns(kind, G, n)
    # String level labels belong to h; g contains positions, including
    # repeated and reversed positions. Matrix linear reads cross columns.
    K = kind === :linear_matrix ? 2G : G
    return Dict(:h => ["level_$(mod1(i, G))" for i in 1:n],
        :g => [mod1(K - i, K) for i in 1:n],
        :y => [0.4 * sin(0.9i) for i in 1:n])
end

function _adg_build(kind, G, n)
    cols = _adg_columns(kind, G, n)
    bound = bind_data(_adg_plan(kind), cols)
    built = build_kernel(bound)
    return (; cols, bound, built, kind)
end

_adg_read(::Val{:linear}, nt, g) = (nt.z * nt.sd)'[g]
_adg_read(::Val{:linear_matrix}, nt, g) = (nt.z .* nt.sd')[g]
_adg_read(::Val{:opaque}, nt, g) = nt.z[g, 1]
_adg_read(::Val{:opaque_vector}, nt, g) = nt.z[:, 1][g]
_adg_read(::Val{:opaque_rows}, nt, g) = nt.z[g, :] * nt.sd

# Independent Julia indexing and Distributions density. Every coordinate
# in this fixture is unconstrained, so no generated density or Jacobian
# is reused in either the value oracle or its finite differences.
function _adg_oracle(fx, u)
    nt = constrain(fx.built.layout, u)
    mu = nt.a .+ _adg_read(Val(fx.kind), nt, fx.cols[:g])
    likelihood = sum(logpdf.(Normal.(mu, 0.7), fx.cols[:y]))
    prior = logpdf(Normal(0, 5), nt.a) + sum(logpdf.(Normal(), nt.sd)) +
        sum(logpdf.(Normal(), nt.z))
    return (; likelihood, prior, posterior = likelihood + prior)
end

function _adg_findiff(f, u; h = 1e-6)
    return map(eachindex(u)) do i
        up, down = copy(u), copy(u)
        up[i] += h
        down[i] -= h
        (f(up) - f(down)) / (2h)
    end
end

function _adg_native(fx, u)
    q = prepare_sampler(fx.built, fx.bound, u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    value, grad = sampler_value_and_gradient!(q, similar(u), u)
    oracle = _adg_oracle(fx, u)
    @test value ≈ oracle.posterior rtol = 1e-12
    @test grad ≈ _adg_findiff(w -> _adg_oracle(fx, w).posterior, u) rtol = 1e-5 atol = 1e-7
    for port in (:likelihood, :prior)
        kernel = prepare_query(fx.built, fx.bound, port)
        @test Base.invokelatest(kernel, u) ≈ getproperty(oracle, port) rtol = 1e-12
    end
    return (; q, value, grad)
end

@testset "array definition gathers: Julia positions and independent AD oracle" begin
    # Empty observations retain the plan's existing explicit n_obs > 0
    # limitation; this gather extension does not broaden that contract.
    @testset "$kind" for kind in (:linear, :linear_matrix, :opaque, :opaque_vector, :opaque_rows)
        structures = Int[]
        for (G, n) in ((2, 6), (5, 17))
            fx = _adg_build(kind, G, n)
            u = [0.3 * sin(1.2i) for i in 1:fx.built.layout.total]
            before = deepcopy(fx.cols)
            Base.invokelatest(_adg_native, fx, u)
            @test fx.cols == before
            # The graph has a fixed number of statements as bound level
            # and observation counts grow; compiled regions are checked
            # separately by test_array_definition_gathers_reactant.jl.
            push!(structures, length(fx.built.spec.graph.recipes))
        end
        @test structures[1] == structures[2]
    end
end

@testset "array definition gathers: positional index validation" begin
    @testset "$kind" for kind in (:linear, :linear_matrix, :opaque, :opaque_vector, :opaque_rows)
        plan = _adg_plan(kind)
        cols = _adg_columns(kind, 2, 6)
        for bad in (fill("level_1", 6), fill(1.5, 6), fill(true, 6), fill(0, 6), fill(-1, 6))
            # refused: positional indices must be positive integers;
            # strings are level labels only on a tracked levels axis.
            @test_throws ContractValidationError bind_data(plan, merge(cols, Dict(:g => bad)))
        end
        if kind in (:linear, :linear_matrix)
            K = kind === :linear_matrix ? 4 : 2
            # refused: known positional bounds exclude K+1 (Julia bounds).
            @test_throws ContractValidationError bind_data(plan, merge(cols, Dict(:g => fill(K + 1, 6))))
        else
            # Unknown result sizes cannot be inferred from the input's
            # levels. Ordinary Julia checks bounds after the call runs.
            bound = bind_data(plan, merge(cols, Dict(:g => fill(3, 6))))
            built = build_kernel(bound)
            kernel = prepare_query(built, bound, :sampler)
            # refused: position 3 is outside this function result's size.
            @test_throws BoundsError Base.invokelatest(kernel, zeros(built.layout.total))
        end
    end
end
