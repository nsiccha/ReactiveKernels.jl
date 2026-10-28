using Pkg

const EXAMPLE_PACKAGES = (
    "ReactiveKernelsDistributionKernels",
    "ReactiveKernelsKernelExamples",
    "ReactiveKernelsBatchingExamples",
    "ReactiveKernelsPPLExamples",
    "ReactiveKernelsCompatibilityExamples",
    "ReactiveKernelsNUTSExamples",
    "ReactiveKernelsStreamingStats",
    "ReactiveKernelsHMCDiagnostics",
    "ReactiveKernelsPPL",
    "ReactiveKernelsReactantODESolvers",
)

# No arguments test every package. Package names restrict the run to those
# packages and `--exclude=<name>` removes one; hosted `Run tests` tests
# ReactiveKernelsPPL in its own sharded jobs.
function selected_packages(args)
    excluded = [chopprefix(arg, "--exclude=") for arg in args if startswith(arg, "--exclude=")]
    named = [arg for arg in args if !startswith(arg, "--exclude=")]
    unknown = setdiff(vcat(excluded, named), EXAMPLE_PACKAGES)
    isempty(unknown) || error("unknown example packages: $(join(unknown, ", "))")
    return [package for package in (isempty(named) ? EXAMPLE_PACKAGES : named)
            if package ∉ excluded]
end

for package in selected_packages(ARGS)
    Pkg.test(package; coverage = false)
end
