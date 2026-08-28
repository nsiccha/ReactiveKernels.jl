using Test

const PPL_SOURCE_CASES = (
    ("eight-schools.md", "eight_schools.jl", EightSchoolsExample,
     :EIGHT_SCHOOLS_SOURCE, :evaluate_eight_schools_source),
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
)

@testset "PPL docs and tests share one source authority" begin
    for (page_name, example_name, owner, source_name, evaluator_name) in
        PPL_SOURCE_CASES
        page = read(joinpath(@__DIR__, "..", "docs", "src", page_name), String)
        example = read(joinpath(@__DIR__, "..", "examples", example_name), String)
        source = getfield(owner, source_name)
        evaluator = getfield(owner, evaluator_name)

        @test occursin("execute_ppl_example(", page)
        @test occursin(":$(nameof(owner)), :$source_name", page)
        @test !occursin("raw\"\"\"", page)
        @test occursin("examples/$example_name", page)
        @test length(findall("@kernel model(", example)) == 1
        @test occursin("@kernel model(", source)
        @test occursin("density_kernel = prepare(model;", source)
        @test occursin(r"inputs = \(;[\s\S]*?\),\n    model,\n    kernel", source)

        artifact = evaluator()
        @test artifact.source == strip(source, '\n')
        @test artifact.output == artifact.kernel(Tuple(artifact.inputs)...)
    end

    make_source = read(joinpath(@__DIR__, "..", "docs", "make.jl"), String)
    helper_source = read(
        joinpath(@__DIR__, "..", "docs", "kernel_examples.jl"), String,
    )
    @test occursin("warnonly = Documenter.except(:eval_block)", make_source)
    @test occursin("assert_ppl_examples_executed!()", make_source)
    for name in (
        :eight_schools_density,
        :linear_regression_density,
        :beta_binomial_density,
        :poisson_gamma_density,
        :dugongs_density,
        :arma11_density,
        :gaussian_mixture_density,
    )
        @test occursin(":" * string(name), helper_source)
    end
end
