using DifferentiationInterface
using Distributions
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using ReactiveKernelsDistributionKernels
using Test

function _embedded_native_source_ops(kernel)
    found = Any[]
    function visit(ex)
        if ex isa QuoteNode && ex.value isa ReactiveKernels._KernelSourceOp
            push!(found, ex.value)
        elseif ex isa Expr
            foreach(visit, ex.args)
        end
    end
    for op in kernel.ops
        op isa ReactiveKernels._KernelSourceOp || continue
        op.f isa ReactiveKernels._KernelSourceFunction || continue
        for method in methods(op.f.f), statement in Base.uncompressed_ast(method).code
            visit(statement)
        end
    end
    found
end

# The GLM rule and half-prior helpers are created when their package loads.
# A model created later may be called from an older world, so check both the
# public latest-world sampler and raw queries with their documented barrier.
function _latest_source_fresh_model()
    mod = Module(gensym(:LatestSource))
    Core.eval(mod, :(using ReactiveKernelsPPL, Distributions))
    Core.eval(mod, quote
        generated_regression = @rkppl begin
            beta[axes(X, 2)] .~ Normal.(0, 10)
            sigma ~ HalfNormal(10)
            mu = X * beta
            y .~ Normal.(mu, sigma)
        end
        nested_endpoint = @rkppl begin
            mu ~ Normal(0, 5)
            tau ~ HalfCauchy(5)
            @plate for j in eachindex(y)
                latent[j] ~ Normal(mu, tau)
                y[j] ~ Normal(latent[j], sigma[j])
            end
        end
        guarded_endpoint = @rkppl begin
            a ~ Normal(0, 1)
            c ~ Normal(0, 1)
            eta = a .+ x
            y .~ NegativeBinomial2.(exp.(eta), exp(c))
        end
    end)
    X = [1.0 0.5; 1.0 -0.4; 1.0 1.2; 1.0 -0.7]
    y = [0.3, -0.5, 1.1, -0.2]
    model = Core.eval(mod, :generated_regression)
    plan = model(; X) | (; y)
    sources = ReactiveKernelsDistributionKernels.DistributionKernelSources
    @test ReactiveKernelsPPL.PPLGeneratedModels.normal_id_glm_lpdf(X, zeros(2), 1.0, y) ≈
          sources.normal_id_glm_lpdf(X, zeros(2), 1.0, y)
    @test sources._half_normal_logpdf_on_support(1.0, 10.0) ≈
          logpdf(Normal(0.0, 10.0), 1.0) + log(2)
    built = build_kernel(plan)
    u = zeros(built.layout.total)
    raw = prepare_query(built, plan, :sampler)
    q = prepare_sampler(built, plan, u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    value = Base.invokelatest(raw, u)
    @test q(u) ≈ value
    likelihood = Base.invokelatest(prepare_query(built, plan, :likelihood), u)
    @test likelihood isa Float64
    pw = Base.invokelatest(prepare_query(built, plan, :pointwise), u).y
    @test length(pw) == length(y)
    @test sum(pw) ≈ likelihood
    @test Base.invokelatest(prepare_cell_query(built, plan, :y), u, 2) ≈ pw[2]
    gradient = zeros(length(u))
    gradient_value, returned = sampler_value_and_gradient!(q, gradient, u)
    @test returned === gradient
    @test gradient_value ≈ value
    h = 1e-5
    for i in eachindex(u)
        step = zeros(length(u))
        step[i] = h
        finite = (Base.invokelatest(raw, u .+ step) -
                  Base.invokelatest(raw, u .- step)) / (2h)
        @test gradient[i] ≈ finite rtol = 1e-5 atol = 1e-6
    end

    # The public sampler promises a latest-world entry, including operations
    # inlined into a source body under the Normal observation plate. A bare
    # raw query still uses its documented invokelatest call boundary.
    nested = Core.eval(mod, :nested_endpoint)
    nested_plan = nested(; sigma = [1.0, 1.3, 0.7]) | (; y = [0.2, -0.3, 0.4])
    nested_built = build_kernel(nested_plan)
    nested_u = zeros(nested_built.layout.total)
    nested_raw = prepare_query(nested_built, nested_plan, :sampler)
    embedded = _embedded_native_source_ops(nested_raw)
    @test !isempty(embedded)
    @test all(op -> op.f isa ReactiveKernels._KernelSourceFunction &&
                    typeof(op.f).parameters[4] === Val{:latest_world}, embedded)
    nested_q = prepare_sampler(nested_built, nested_plan, nested_u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    @test Base.invokelatest(nested_raw, nested_u) ≈ nested_q(nested_u)
    nested_g = similar(nested_u)
    nested_value, _ = sampler_value_and_gradient!(nested_q, nested_g, nested_u)
    @test nested_value ≈ Base.invokelatest(nested_raw, nested_u)
    for i in eachindex(nested_u)
        step = zeros(length(nested_u))
        step[i] = 1e-5
        finite = (Base.invokelatest(nested_raw, nested_u .+ step) -
                  Base.invokelatest(nested_raw, nested_u .- step)) / 2e-5
        @test nested_g[i] ≈ finite rtol = 1e-5 atol = 1e-6
    end

    # Cutting an endpoint's retained arm graph must preserve the same caller
    # guarantee as its original inline body. Exercise the public barrier from
    # this older world after preparation caches the data-only log-gamma terms.
    # Select ordinary reverse explicitly: the baseline's default batched
    # forward path hits an Enzyme LLVM duplicate-name error for this model.
    guarded = Core.eval(mod, :guarded_endpoint)
    guarded_plan = guarded(; x = [-0.4, 0.2, 0.7]) | (; y = [0, 2, 5])
    guarded_built = build_kernel(guarded_plan)
    guarded_u = [0.3, 0.7]
    guarded_raw = prepare_query(guarded_built, guarded_plan, :sampler)
    guarded_q = prepare_sampler(guarded_built, guarded_plan, guarded_u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse), native_forward_batch = false)
    @test guarded_q(guarded_u) ≈ Base.invokelatest(guarded_raw, guarded_u)
    guarded_g = similar(guarded_u)
    guarded_value, _ = sampler_value_and_gradient!(guarded_q, guarded_g, guarded_u)
    @test guarded_value ≈ Base.invokelatest(guarded_raw, guarded_u)
    for i in eachindex(guarded_u)
        step = zeros(length(guarded_u))
        step[i] = 1e-5
        finite = (Base.invokelatest(guarded_raw, guarded_u .+ step) -
                  Base.invokelatest(guarded_raw, guarded_u .- step)) / 2e-5
        @test guarded_g[i] ≈ finite rtol = 1e-5 atol = 1e-6
    end
    nothing
end

@testset "scoped native source calls across a fresh-model world age" begin
    captured = ReactiveKernels._KernelSourceFunction(identity, nothing, (; kept = 2.0))
    latest = ReactiveKernels._kernel_latest_source(captured)
    @test latest.captures === captured.captures
    @test latest(2.0) == captured(2.0)
    _latest_source_fresh_model()
end
