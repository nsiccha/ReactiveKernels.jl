# Parse executable syntax only: comments may legitimately mention forbidden calls.
function _has_executable_call(source, name)
    parsed = Meta.parseall(source; filename = "source")
    function contains_call(ex)
        ex isa Expr || return false
        ex.head === :call && !isempty(ex.args) && ex.args[1] === name && return true
        any(contains_call, ex.args)
    end
    any(contains_call, parsed.args)
end

using ReactiveKernelsPPLExamples.NormalMixtureExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
using LogExpFunctions: log1pexp, logaddexp

# Graph-independent reference oracle: the marginalized two-component normal
# mixture with KNOWN unit variance, free means (Normal(0,10)), and a flat
# theta prior (uniform(0,1) contributes 0). Fully normalized.
function _nm_reference(q, y)
    u_theta, mu1, mu2 = q[1], q[2], q[3]
    log_theta = -log1pexp(-u_theta)
    log1m_theta = -log1pexp(u_theta)
    log_jacobian = log_theta + log1m_theta
    nld(x, loc, sc) = -0.5 * log(2π) - log(sc) - 0.5 * ((x - loc) / sc)^2
    prior = nld(mu1, 0.0, 10.0) + nld(mu2, 0.0, 10.0)
    likelihood = 0.0
    for yj in y
        likelihood += logaddexp(log_theta + nld(yj, mu1, 1.0),
                                log1m_theta + nld(yj, mu2, 1.0))
    end
    (; prior, log_jacobian, likelihood, posterior = prior + likelihood + log_jacobian)
end

@testset "PPL graph — normal_mixture (posteriordb, known variance)" begin
    artifact = evaluate_normal_mixture_source()
    @test artifact.source == strip(NORMAL_MIXTURE_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [-0.8, -10.0, 10.0]
    reference = _nm_reference(q, NORMAL_MIXTURE_Y)

    @testset "authored on the current baseline surface" begin
        @test occursin("logaddexp(", NORMAL_MIXTURE_SOURCE)
        @test occursin("pointwise = plate(", NORMAL_MIXTURE_SOURCE)
        @test occursin("normal(m1, 1.0).logpdf", NORMAL_MIXTURE_SOURCE)
        @test occursin("log_mix", NORMAL_MIXTURE_SOURCE)  # explanatory comment is not executable syntax
        @test !_has_executable_call(NORMAL_MIXTURE_SOURCE, :log_mix)
        @test _has_executable_call(replace(NORMAL_MIXTURE_SOURCE, "logaddexp(lt +" => "log_mix(lt +"; count = 1), :log_mix)
        @test !occursin("struct ", NORMAL_MIXTURE_SOURCE)
        @test artifact.normal_object === normal
    end

    @testset "constrain-only prunes the density work" begin
        p = plan(model.graph; have = (model.unconstrained,), want = (model.parameters,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.prior.id) in produced)
        parameters = prepare(p)(q)
        @test parameters isa NamedTuple
        @test 0.0 < parameters.theta < 1.0
    end

    @testset "marginalized posterior vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.y),
                 want = (model.prior, model.log_jacobian, model.pointwise,
                         model.likelihood, model.posterior))
        prior, log_jacobian, pointwise, likelihood, posterior =
            prepare(p)(q, NORMAL_MIXTURE_Y)
        @test all(isfinite, pointwise)
        @test prior ≈ reference.prior
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test log_jacobian ≈ reference.log_jacobian
        @test posterior ≈ reference.posterior
    end

    @testset "one authored plate exposes a buffer-free total" begin
        pointwise_kernel = prepare(model;
            have = (:unconstrained, :y), want = :pointwise)
        likelihood_kernel = prepare(model;
            have = (:unconstrained, :y), want = :likelihood)
        pw = pointwise_kernel(q, NORMAL_MIXTURE_Y)
        @test likelihood_kernel(q, NORMAL_MIXTURE_Y) ≈ sum(pw)
        @test occursin("similar", string(code_expr(pointwise_kernel)))
        @test !occursin("similar", string(code_expr(likelihood_kernel)))
    end
end
