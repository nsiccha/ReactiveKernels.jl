# Regression controls for the model-only template initialization contract.
#
# Each example module builds its `_<MODEL>_GRAPH_TEMPLATE` eagerly in `__init__`
# (a world-age boundary at package load), but MODEL-ONLY: `_evaluate_ppl_source`
# stops as soon as the authored source defines `model`, skipping the source's
# trailing demo/self-check (`prepare(...)` + kernel execution + `@assert`). These
# tests pin that contract:
#   * import builds a GRAPH template, never a prepared/executed demo;
#   * the default (full) `evaluate_*_source()` still runs the demo/self-check;
#   * `build_*_graph()` + `prepare` + evaluate is safe INSIDE ONE ordinary
#     function (the benchmark run_one shape) — no world-age hazard, because the
#     source eval already happened at load, not in the caller;
#   * repeated `build_*_graph()` yields independent composed graphs;
#   * the model-only graph is behaviorally identical to the full-source graph.

const _RK = ReactiveKernels
const _M = ReactiveKernelsPPLExamples

@testset "startup initialization contract (model-only templates)" begin
    E = _M.EightSchoolsExample

    @testset "import builds a graph template, not a prepared/executed demo" begin
        # The model-only artifact carries the graph but NOT a prepared kernel or
        # a demo output — i.e. no `prepare`/execute ran to build the template.
        mo = E.evaluate_eight_schools_source(; model_only = true)
        @test mo.model isa _RK.KernelSpec
        @test !hasproperty(mo, :kernel)
        @test !hasproperty(mo, :output)
        # The cached template `__init__` built is a KernelSpec (a graph), never a
        # PreparedKernel — so package load did no eager preparation.
        @test isassigned(E._EIGHT_SCHOOLS_GRAPH_TEMPLATE)
        @test E._EIGHT_SCHOOLS_GRAPH_TEMPLATE[] isa _RK.KernelSpec
        @test !(E._EIGHT_SCHOOLS_GRAPH_TEMPLATE[] isa _RK.PreparedKernel)
    end

    @testset "the default full source still runs its demo/self-check" begin
        full = E.evaluate_eight_schools_source()
        @test full.model isa _RK.KernelSpec
        @test full.kernel isa _RK.PreparedKernel          # prepare ran
        @test hasproperty(full, :output)                  # kernel executed
    end

    @testset "no eager preparation across representative modules" begin
        for (mod, fn) in (
            (_M.EightSchoolsExample, :evaluate_eight_schools_source),
            (_M.LinearRegressionExample, :evaluate_linear_regression_source),
            (_M.BetaBinomialExample, :evaluate_beta_binomial_source),
            (_M.PoissonGammaExample, :evaluate_poisson_gamma_source),
            (_M.GLMPoissonExample, :evaluate_glm_poisson_source),
            (_M.Rate2Example, :evaluate_rate_2_source),
            (_M.ARMA11Example, :evaluate_arma11_source),
            (_M.MNISTLogisticExample, :evaluate_mnist_logistic_source),
        )
            mo = getproperty(mod, fn)(; model_only = true)
            @test mo.model isa _RK.KernelSpec
            @test !hasproperty(mo, :kernel)
        end
        # mnist carries a SECOND template (the optimized variant); both are
        # model-only and both are built eagerly at load.
        @test _M.MNISTLogisticExample.evaluate_mnist_logistic_optimized_source(;
            model_only = true).model isa _RK.KernelSpec
        @test isassigned(_M.MNISTLogisticExample._MNIST_LOGISTIC_GRAPH_TEMPLATE)
        @test isassigned(_M.MNISTLogisticExample._MNIST_LOGISTIC_OPTIMIZED_GRAPH_TEMPLATE)
    end

    @testset "first use is world-age-safe inside one ordinary function" begin
        # build + prepare + evaluate all in a SINGLE compiled function call — the
        # exact shape of the posteriordb benchmark's per-model run and the other
        # example tests. Eager `build_*_graph()` (a pure `compose`) makes this
        # safe; a lazy build_*_graph() that Core.eval'd here would throw
        # `method too new`.
        function first_use_one_call()
            g = E.build_eight_schools_graph()
            k = prepare(g; have = (:unconstrained, :observations, :observation_scales),
                           want = :posterior)
            q = [0.0, log(5.0), zeros(E.NSCHOOLS)...]
            k(q, E.EIGHT_SCHOOLS_Y, E.EIGHT_SCHOOLS_SIGMA)
        end
        v = first_use_one_call()
        @test isfinite(v)

        # Repeated use: each build yields an INDEPENDENT composed graph object,
        # and every one evaluates to the same value.
        g1 = E.build_eight_schools_graph()
        g2 = E.build_eight_schools_graph()
        @test g1 !== g2
        pv(g) = prepare(g; have = (:unconstrained, :observations, :observation_scales),
                           want = :posterior)(
            [0.0, log(5.0), zeros(E.NSCHOOLS)...], E.EIGHT_SCHOOLS_Y, E.EIGHT_SCHOOLS_SIGMA)
        @test pv(g1) ≈ v
        @test pv(g2) ≈ v
    end

    @testset "model-only graph is identical to the full-source graph" begin
        q = [0.0, log(5.0), zeros(E.NSCHOOLS)...]
        pv(g) = prepare(g; have = (:unconstrained, :observations, :observation_scales),
                           want = :posterior)(q, E.EIGHT_SCHOOLS_Y, E.EIGHT_SCHOOLS_SIGMA)
        @test pv(E.build_eight_schools_graph()) ≈
              pv(compose(E.evaluate_eight_schools_source().model))

        R = _M.Rate2Example
        pr(g) = prepare(g; have = (:unconstrained, :n1, :n2, :k1, :k2), want = :posterior,
                           bound = (; n1 = R.RATE2_N1, n2 = R.RATE2_N2,
                                      k1 = R.RATE2_K1, k2 = R.RATE2_K2))([0.1, -0.1])
        @test pr(R.build_rate_2_graph()) ≈
              pr(compose(R.evaluate_rate_2_source().model))
    end
end
