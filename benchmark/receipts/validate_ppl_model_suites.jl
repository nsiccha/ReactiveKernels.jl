#!/usr/bin/env julia

module MNISTSuite
include(joinpath(@__DIR__, "validate_mnist_logistic_suite.jl"))
end
module EightSchoolsSuite
include(joinpath(@__DIR__, "validate_eight_schools_suite.jl"))
end

function validate_ppl_model_suites(
    mnist_paths = MNISTSuite.DEFAULT_MNIST_SUITE_PATHS,
    eight_schools_paths = EightSchoolsSuite.DEFAULT_EIGHT_SCHOOLS_SUITE_PATHS,
)
    errors = String[]
    append!(errors, ("MNIST: " * error for error in
        MNISTSuite.validate_mnist_logistic_suite(mnist_paths)))
    append!(errors, ("Eight Schools: " * error for error in
        EightSchoolsSuite.validate_eight_schools_suite(eight_schools_paths)))

    mnist_configs = MNISTSuite.MNISTLogisticMatrixSpec.MNIST_RK_CONFIGURATIONS
    eight_configs = EightSchoolsSuite.EightSchoolsMatrixSpec.
        EIGHT_SCHOOLS_RK_CONFIGURATIONS
    mnist_signature = Tuple((configuration.id, configuration.differentiation,
        configuration.compiler, configuration.allocation, configuration.data)
        for configuration in mnist_configs)
    eight_signature = Tuple((configuration.id, configuration.differentiation,
        configuration.compiler, configuration.allocation, configuration.data)
        for configuration in eight_configs)
    mnist_signature == eight_signature || push!(errors,
        "MNIST and Eight Schools do not use the same configuration vocabulary")
    exclusions = MNISTSuite.MNISTLogisticMatrixSpec.ModelBenchmarkMatrixSpec.
        RK_UNAVAILABLE_CONFIGURATION_COMBINATIONS
    length(exclusions) == 3 && all(row -> !isempty(row.reason), exclusions) ||
        push!(errors, "shared unavailable modifier combinations are not explicit")
    errors
end

function main()
    errors = validate_ppl_model_suites()
    isempty(errors) &&
        (println("VALIDATE OK — both PPL model suites use the same complete " *
                 "configuration contract"); return 0)
    foreach(println, errors)
    1
end

if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) || (println("usage: validate_ppl_model_suites.jl"); exit(2))
    exit(main())
end
