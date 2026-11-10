module VectorDensityFactorTests
using Test, Random, LinearAlgebra, ReactiveKernels, ReactiveKernelsPPL
using Distributions: Exponential, MvNormal, Normal, logpdf
import DifferentiationInterface as DI
import Enzyme

include(joinpath(pkgdir(ReactiveKernels), "test", "jax_python.jl"))

# Density factors on vector definitions (snag rkppl-vector-val-c101b992):
# `d .~ D.(…)` / `d ~ MvNormal(…)` over a definition reading a sampled value
# adds the definition's density to `prior`, with no coordinate or transform,
# exactly like the scalar factor `d ~ D(…)` on a scalar definition.

# Two connected components {1, 2, 3} and {4, 5, 6} of an ICAR graph: the
# field needs one soft sum-to-zero constraint per component.
const NODE1 = [1, 2, 4, 5]
const NODE2 = [2, 3, 5, 6]
const COMPONENTS = [1.0 1 1 0 0 0; 0 0 0 1 1 1]
const SCALES = [0.003, 0.004]
const DATA = (; y = [0.2, -0.1, 0.3, 0.1, -0.2, 0.05], node1 = NODE1,
    node2 = NODE2, C = COMPONENTS, scales = SCALES,
    comp_start = [1, 4], comp_size = [3, 3], Sigma = [1.0 0.3; 0.3 2.0])

icar(phi) = -0.5 * sum(abs2, phi[NODE1] .- phi[NODE2])
sum_to_zero(phi) = sum(logpdf.(Normal.(0, SCALES), COMPONENTS * phi))
likelihood(phi) = sum(logpdf.(Normal.(phi, 1), DATA.y))

model(body; conditioned = (:y,)) = bind_data(lower_rkppl(body, keys(DATA);
    mod = @__MODULE__, conditioned), DATA)

function fd_check(q, u; step = 1e-6)
    g = similar(u)
    sampler_value_and_gradient!(q, g, u)
    for i in eachindex(u)
        up, down = copy(u), copy(u)
        up[i] += step
        down[i] -= step
        @test g[i] ≈ (q(up) - q(down)) / 2step atol = 1e-5 rtol = 1e-6
    end
end

# The densities of a model whose only coordinates are the six field entries.
function check_field_model(plan; prior = phi -> icar(phi) + sum_to_zero(phi))
    built = build_kernel(plan)
    @test built.layout.total == 6
    @test all(startswith("phi"), string.(coordinate_names(built.layout)))
    for seed in 1:3
        u = 0.3 .* randn(Xoshiro(seed), 6)
        phi = constrain(built.layout, u).phi
        @test Base.invokelatest(prepare_query(built, plan, :prior), u) ≈
            prior(phi) rtol = 1e-12
        @test Base.invokelatest(prepare_query(built, plan, :likelihood), u) ≈
            likelihood(phi) rtol = 1e-12
        q = prepare_sampler(built, plan, u;
            backend = DI.AutoEnzyme(; mode = Enzyme.Reverse))
        @test q(u) ≈ prior(phi) + likelihood(phi) rtol = 1e-12
        seed == 1 && fd_check(q, u)
    end
    return built
end

@testset "elementwise factor on a parameter-derived vector" begin
    plan = model(quote
        phi[1:6] ~ ICAR(node1, node2)
        sums = C * phi
        sums .~ Normal.(0, scales)
        y .~ Normal.(phi, 1)
    end)
    factor = only(plan.factors)
    @test factor isa ArrayParameter
    @test factor.name === :sums && factor.family === :normal
    @test isempty(plan.responses[2:end]) && plan.responses[1].response === :y
    built = check_field_model(plan)
    source = sprint(show, Base.remove_linenums!(deepcopy(kernel_expr(plan, built.layout))))
    @test occursin("_ppl_factor_sums", source)
end

@testset "computed arguments, forward statements and per-component sums" begin
    # The reporter's body: component sums through a cumulative sum, the
    # factor stated before its definition, a computed per-entry scale.
    plan = model(quote
        phi[1:6] ~ ICAR(node1, node2)
        phi_sums .~ Normal.(0, scales ./ comp_size .* 3)
        comp_end = comp_start .+ comp_size
        cs = cumsum(vcat(0.0, phi))
        phi_sums = cs[comp_end] .- cs[comp_start]
        y .~ Normal.(phi, 1)
    end)
    check_field_model(plan)
end

@testset "joint factor on a vector definition" begin
    # A diagonal MvNormal is the elementwise factor.
    check_field_model(model(quote
        phi[1:6] ~ ICAR(node1, node2)
        sums = C * phi
        sums ~ MvNormal(zeros(2), Diagonal(scales .^ 2))
        y .~ Normal.(phi, 1)
    end))
    # A correlated one is Distributions.jl's density of the value.
    S = [1.0 0.3; 0.3 2.0]
    plan = model(quote
        phi[1:6] ~ ICAR(node1, node2)
        sums = C * phi
        sums ~ MvNormal([0.1, -0.2], [1.0 0.3; 0.3 2.0])
        y .~ Normal.(phi, 1)
    end)
    @test only(plan.factors).family === :mvnormal_vector
    check_field_model(plan;
        prior = phi -> icar(phi) + logpdf(MvNormal([0.1, -0.2], S), COMPONENTS * phi))
end

# The reporter's spelling: the gather index computed inline from data.
@rkppl vector_factor_car(node1, node2, comp_start, comp_size) = begin
    phi[1:6] ~ ICAR(node1, node2)
    cs = cumsum(vcat(0.0, phi))
    phi_sums = cs[comp_start .+ comp_size] .- cs[comp_start]
    phi_sums .~ Normal.(0, 0.001 .* comp_size)
    return phi
end

@testset "a submodel states one factor per component of its data" begin
    plan = model(quote
        r ~ vector_factor_car(node1, node2, comp_start, comp_size)
        y .~ Normal.(r, 1)
    end)
    built = build_kernel(plan)
    @test string.(coordinate_names(built.layout)) == ["r.phi.$i" for i in 1:6]
    u = 0.2 .* randn(Xoshiro(4), 6)
    phi = u
    expected = icar(phi) + sum(logpdf.(Normal.(0, 0.003), COMPONENTS * phi))
    @test Base.invokelatest(prepare_query(built, plan, :prior), u) ≈ expected rtol = 1e-12
end

@testset "one soft sum-to-zero per field of an ICAR field array" begin
    # K independent fields over one graph (the rows of `phi`), each
    # identified by one entry of an elementwise factor on the field sums.
    K, N = 3, 4
    node1, node2 = [1, 2, 3], [2, 3, 4]
    data = (; K, N, node1, node2, y = [0.1, -0.2, 0.3, 0.05])
    plan = bind_data(lower_rkppl(quote
        eachrow(phi[1:K, 1:N]) .~ ICAR(node1, node2)
        phi_sums = vec(sum(phi; dims = 2))
        phi_sums .~ Normal.(0, 0.001 * N)
        y .~ Normal.(phi[1, :], 1)
    end, keys(data); mod = @__MODULE__, conditioned = (:y,)), data)
    @test only(plan.factors).name === :phi_sums
    built = build_kernel(plan)
    @test built.layout.total == K * N
    for seed in 1:2
        u = 0.3 .* randn(Xoshiro(10 + seed), K * N)
        phi = constrain(built.layout, u).phi
        prior = sum(-0.5 * sum(abs2, phi[k, node1] .- phi[k, node2]) for k in 1:K) +
            sum(logpdf.(Normal(0, 0.001 * N), vec(sum(phi; dims = 2))))
        @test Base.invokelatest(prepare_query(built, plan, :prior), u) ≈ prior rtol = 1e-12
        q = prepare_sampler(built, plan, u;
            backend = DI.AutoEnzyme(; mode = Enzyme.Reverse))
        @test q(u) ≈ prior + sum(logpdf.(Normal.(phi[1, :], 1), data.y)) rtol = 1e-12
        seed == 1 && fd_check(q, u)
    end
end

@testset "support is checked without a transform" begin
    plan = model(quote
        a ~ Normal(0, 1)
        shifted = a .+ [0.5, 1.5]
        shifted .~ Exponential.(1.0)
        y .~ Normal.(a, 1)
    end)
    built = build_kernel(plan)
    @test built.layout.total == 1
    prior = prepare_query(built, plan, :prior)
    @test Base.invokelatest(prior, [0.2]) ≈ logpdf(Normal(), 0.2) +
        sum(logpdf.(Exponential(1.0), 0.2 .+ [0.5, 1.5])) rtol = 1e-12
    @test Base.invokelatest(prior, [-1.0]) == -Inf
end

@testset "a conditioned vector factor enters the likelihood" begin
    plan = model(quote
        phi[1:6] ~ ICAR(node1, node2)
        sums = C * phi
        sums .~ Normal.(0, scales)
        y .~ Normal.(phi, 1)
    end; conditioned = (:y, :sums))
    built = build_kernel(plan)
    u = 0.3 .* randn(Xoshiro(5), 6)
    @test Base.invokelatest(prepare_query(built, plan, :prior), u) ≈ icar(u) rtol = 1e-12
    @test Base.invokelatest(prepare_query(built, plan, :likelihood), u) ≈
        likelihood(u) + sum_to_zero(u) rtol = 1e-12
end

@testset "data-only definitions stay observations; refusals" begin
    # `.~` over a definition of data alone observes it (a derived response).
    plan = model(quote
        m ~ Normal(0, 1)
        ys = y .* 2
        ys .~ Normal.(m, 1)
    end; conditioned = ())
    @test isempty(plan.factors)
    @test any(r -> r.response === :ys, plan.responses)
    # A value reading observed (conditioned) data and a sampled value is a
    # transformed observation, not a prior factor: the derived-response
    # contract refuses it as before (test_derived_response.jl).
    err = try
        model(quote
            a ~ Normal(0, 1)
            scaled = y .* a
            scaled .~ Normal.(0, 1)
        end)
        nothing
    catch e
        e
    end
    @test err isa SurfaceLoweringError
    @test occursin("response scaled", sprint(showerror, err))
    # So is a joint law on a data-only vector definition.
    @test_throws SurfaceLoweringError model(quote
        m ~ Normal(0, 1)
        ys = y .* 2
        ys ~ MvNormal(fill(m, 6), Diagonal(ones(6)))
    end; conditioned = ())
    # One density per definition.
    @test_throws SurfaceLoweringError model(quote
        phi[1:6] ~ ICAR(node1, node2)
        sums = C * phi
        sums .~ Normal.(0, scales)
        sums ~ MvNormal(zeros(2), Diagonal(scales .^ 2))
        y .~ Normal.(phi, 1)
    end)
    # A univariate law over a whole vector is the crossed spelling.
    @test_throws SurfaceLoweringError model(quote
        phi[1:6] ~ ICAR(node1, node2)
        sums = C * phi
        sums ~ Normal(0, 1)
        y .~ Normal.(phi, 1)
    end)
end

@testset "jax_source: vector density factors" begin
    for body in (
            quote
                phi[1:6] ~ ICAR(node1, node2)
                sums = C * phi
                sums .~ Normal.(0, scales)
                y .~ Normal.(phi, 1)
            end,
            quote
                phi[1:6] ~ ICAR(node1, node2)
                sums = C * phi
                sums ~ MvNormal([0.1, -0.2], Sigma)
                y .~ Normal.(phi, 1)
            end)
        plan = model(body)
        built = build_kernel(plan)
        src = jax_source(built, plan)
        @test occursin("UNTRANSLATED = {}", string(src))
        if !rk_jax_available()
            @test_skip false
            continue
        end
        names = first.(only(f for f in src.functions if f.name == "prepare_data").inputs)
        raw = Tuple(getproperty(DATA, Symbol(n)) for n in names)
        u = 0.2 .* randn(Xoshiro(6), built.layout.total)
        for preset in (:prior, :likelihood, :sampler)
            name = preset === :prior ? "log_prior" : preset === :likelihood ?
                "log_likelihood" : "log_density"
            r = rk_jax_evaluate(src, name, (u, raw...); prepare = true)
            @test r.value ≈ Base.invokelatest(prepare_query(built, plan, preset), u) rtol = 1e-10
        end
        query = prepare_query(built, plan, :sampler)
        r = rk_jax_evaluate(src, "log_density", (u, raw...); grad = true, prepare = true)
        reference = rk_fd_gradient(query, u)
        @test maximum(abs.(r.grad .- reference) ./ max.(1, abs.(reference))) < 1e-5
    end
end

end # module
