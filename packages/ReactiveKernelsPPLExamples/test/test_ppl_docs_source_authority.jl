using Test

const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))

const PPL_SOURCE_CASES = (
    ("eight-schools.md", "eight_schools.jl", EightSchoolsExample,
     :EIGHT_SCHOOLS_SOURCE, :evaluate_eight_schools_source),
    ("sum-to-zero.md", "sum_to_zero.jl", SumToZeroExample,
     :SUM_TO_ZERO_SOURCE, :evaluate_sum_to_zero_source),
    ("linear-regression.md", "linear_regression.jl", LinearRegressionExample,
     :LINEAR_REGRESSION_SOURCE, :evaluate_linear_regression_source),
    ("beta-binomial.md", "beta_binomial.jl", BetaBinomialExample,
     :BETA_BINOMIAL_SOURCE, :evaluate_beta_binomial_source),
    ("poisson-gamma.md", "poisson_gamma.jl", PoissonGammaExample,
     :POISSON_GAMMA_SOURCE, :evaluate_poisson_gamma_source),
    ("dugongs-growth.md", "dugongs_growth.jl", DugongsGrowthExample,
     :DUGONGS_SOURCE, :evaluate_dugongs_source),
    ("arma11.md", "arma11.jl", ARMA11Example,
     :ARMA11_SOURCE, :evaluate_arma11_source),
    ("gaussian-mixture.md", "gaussian_mixture.jl", GaussianMixtureExample,
     :GAUSSIAN_MIXTURE_SOURCE, :evaluate_gaussian_mixture_source),
    ("mnist-logistic.md", "mnist_logistic.jl", MNISTLogisticExample,
     :MNIST_LOGISTIC_SOURCE, :evaluate_mnist_logistic_source),
    ("mnist-logistic.md", "mnist_logistic.jl", MNISTLogisticExample,
     :MNIST_LOGISTIC_OPTIMIZED_SOURCE, :evaluate_mnist_logistic_optimized_source),
)

# One displayed/executed authority kernel per registered case: a file carrying
# two cases (the idiomatic and optimized MNIST models) declares exactly two.
const _KERNELS_PER_EXAMPLE_FILE = Dict{String,Int}()
for case in PPL_SOURCE_CASES
    _KERNELS_PER_EXAMPLE_FILE[case[2]] =
        get(_KERNELS_PER_EXAMPLE_FILE, case[2], 0) + 1
end

@testset "PPL docs and tests share one source authority" begin
    for (page_name, example_name, owner, source_name, evaluator_name) in
        PPL_SOURCE_CASES
        page = read(joinpath(REPOSITORY_ROOT, "docs", "src", page_name), String)
        example = read(joinpath(@__DIR__, "..", "src", example_name), String)
        source = getfield(owner, source_name)
        evaluator = getfield(owner, evaluator_name)

        @test occursin("execute_ppl_example(", page)
        @test occursin(":$(nameof(owner)), :$source_name", page)
        @test !occursin("raw\"\"\"", page)
        @test occursin("packages/ReactiveKernelsPPLExamples/src/$example_name", page)
        @test length(findall("@kernel model(", example)) ==
              _KERNELS_PER_EXAMPLE_FILE[example_name]
        @test occursin("@kernel model(", source)
        @test occursin(r"\w+_kernel = prepare\(model;", source)
        @test occursin(r"inputs = \(;[\s\S]*?\),\n    model,\n    kernel", source)

        artifact = evaluator()
        @test artifact.source == strip(source, '\n')
        @test parentmodule(artifact.sandbox) === owner
        @test Base.PkgId(artifact.sandbox) == Base.PkgId(owner)
        # `evaluator()` defines the authored recipe closures in a fresh sandbox.
        # Cross that dynamic-evaluation boundary in the latest world, exactly as
        # the docs renderer does, so inlined source operations remain reusable on
        # Julia versions with stricter world-age enforcement.
        observed = Base.invokelatest(
            artifact.kernel, Tuple(artifact.inputs)...,
        )
        @test artifact.output == observed
        raw_generated = code_expr(artifact.kernel)
        readable_generated = sprint(
            Base.show_unquoted,
            ReactiveKernels._readable_expr(raw_generated, artifact.kernel);
            context = :limit => false,
        )
        @test raw_generated === code_expr(artifact.kernel)
        @test !occursin(r"__ops__\[\d+\]", readable_generated)
        @test !occursin(r"\boperation\(", readable_generated)
    end

    make_source = read(joinpath(REPOSITORY_ROOT, "docs", "make.jl"), String)
    helper_source = read(
        joinpath(REPOSITORY_ROOT, "docs", "kernel_examples.jl"), String,
    )
    @test occursin("warnonly = false", make_source)
    @test !occursin("warnonly = Documenter.except(:eval_block)", make_source)
    @test occursin("assert_ppl_examples_executed!()", make_source)
    @test occursin("ReactiveKernels._readable_expr", helper_source)
    @test occursin("readable generated view retained an opaque operation slot",
                   helper_source)
    for name in (
        :eight_schools_extraction,
        :sum_to_zero_logdensity,
        :linear_regression_density,
        :beta_binomial_density,
        :poisson_gamma_density,
        :dugongs_density,
        :arma11_density,
        :gaussian_mixture_density,
        :mnist_logistic_density,
        :mnist_logistic_optimized_density,
    )
        @test occursin(":" * string(name), helper_source)
    end
end
