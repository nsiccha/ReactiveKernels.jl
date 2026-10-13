using ReactiveKernels, ReactiveKernelsPPL, Distributions, Test
using DifferentiationInterface, Enzyme, Reactant

# Compiled companion of `test_glm_object_values.jl`: GLM objects over bound
# data matrices, literal and sampled intercepts and vector-literal
# coefficients keep native density and gradient parity, and their emitted
# trace does not grow with the observation count.
function _gv_reactant_case(kind, n)
    X, y, counts = _gv_X(n), _gv_y(n), _gv_counts(n)
    if kind === :zero_intercept
        m = @rkppl begin
            beta[axes(X, 2)] .~ Normal.(0, 10)
            sigma ~ Exponential(1)
            y ~ NormalIDGLM(X, 0.0, beta, sigma)
        end
        return m(; X) | (; y)
    elseif kind === :named_product
        m = @rkppl begin
            beta[axes(X, 2)] .~ Normal.(0, 10)
            sigma ~ Exponential(1)
            mu = X * beta
            y .~ Normal.(mu, sigma)
        end
        return m(; X) | (; y)
    elseif kind === :bound_design
        m = @rkppl begin
            alpha ~ Normal(0, 10)
            beta[axes(lags, 2)] .~ Normal.(0, 10)
            sigma ~ Exponential(1)
            y ~ NormalIDGLM(lags, alpha, beta, sigma)
        end
        return m(; lags = X) | (; y)
    else
        m = @rkppl begin
            alpha ~ Normal(0, 10)
            beta1 ~ Normal(0, 1)
            beta2 ~ Normal(0, 1)
            beta3 ~ Normal(0, 1)
            counts ~ PoissonLogGLM(X, alpha, [beta1, beta2, beta3])
        end
        return m(; X) | (; counts)
    end
end

@testset "GLM objects over matrix values: compiled parity and retained structure" begin
    for kind in (:zero_intercept, :named_product, :bound_design, :vector_coefficients)
        structures = Dict{String,Int}[]
        for n in (4, 9)
            plan = _gv_reactant_case(kind, n)
            built = build_kernel(plan)
            u = _gv_u(built)
            kernel = prepare_query(built, plan, :sampler)
            query = prepare_sampler(built, plan, u;
                backend = AutoEnzyme(mode = Enzyme.Reverse,
                    function_annotation = Enzyme.Const))
            value, gradient = sampler_value_and_gradient!(query, similar(u), u)
            ru = Reactant.to_rarray(u)
            hlo = repr(Reactant.@code_hlo optimize = false kernel(ru))
            operations = Dict{String,Int}()
            for m in eachmatch(r"stablehlo\.[a-z_]+", hlo)
                operations[m.match] = get(operations, m.match, 0) + 1
            end
            push!(structures, operations)
            compiled = Reactant.@compile kernel(ru)
            @test Float64(compiled(ru)) ≈ value rtol = 1e-8
            cad = compile_ad_value_and_gradient(query.ad, ru)
            rvalue, rgradient = cad(ru)
            @test Float64(rvalue) ≈ value rtol = 1e-8
            @test Array(rgradient) ≈ gradient rtol = 1e-6
        end
        @test structures[1] == structures[2]
    end
end
